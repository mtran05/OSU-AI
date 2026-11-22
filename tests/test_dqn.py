"""Tests for the frame simulator, the training environment, the replay buffer, map splits and DQN training."""

from __future__ import annotations

import bisect
import json
from pathlib import Path

import numpy as np
import torch

from osuai.beatmap import KEY_HIT_MASK, KEY_K1, KEY_K2, parse_osu
from osuai.dqn import (
    DQNConfig,
    QNetwork,
    ReplayBuffer,
    load_checkpoint,
    play_map,
    resolve_checkpoint,
    split_maps,
    train_dqn,
)
from osuai.env import EnvConfig, MapData, OsuEnv, build_action_table
from osuai.judge import TAIL_LENIENCY, judge_replay
from tests.helpers import COMPLEX_MAP, perfect_inputs

SIMPLE_MAP = """osu file format v14
[General]
Mode: 0
[Difficulty]
CircleSize: 4
OverallDifficulty: 5
ApproachRate: 5
[TimingPoints]
0,500,4,1,0,100,1,0
[HitObjects]
100,100,1000,1,0
400,300,2000,1,0
"""


def _drive_perfectly(env: OsuEnv, bm) -> float:
    ofr = perfect_inputs(bm, rate=240)
    ts = np.array([f.time for f in ofr])
    xs = np.array([f.x for f in ofr])
    ys = np.array([f.y for f in ofr])
    ks = [f.keys & KEY_HIT_MASK for f in ofr]
    total = 0.0
    done = False
    while not done:
        t = env.t + env.cfg.dt_ms
        i = max(bisect.bisect_right(ts, t) - 1, 0)
        _, r, done, _ = env.step_raw(float(np.interp(t, ts, xs)), float(np.interp(t, ts, ys)), ks[i] != 0)
        total += r
    return total


def test_renderer_shows_objects_cursor_and_key_overlay():
    cfg = EnvConfig()
    bm = parse_osu(SIMPLE_MAP)
    env = OsuEnv(MapData(bm, cfg), cfg)
    env.reset(mode="full")
    env.t = 990.0  # first circle fully faded in, approach circle almost closed
    env.x, env.y = 400.0, 300.0
    frame = env._render()
    assert frame.shape == cfg.obs_shape and frame.dtype == np.uint8

    s = cfg.render.scale
    fh = cfg.render.field_height
    cu, cv = int(100 * s), int(100 * s)
    assert frame[cv, cu] > 50  # hit circle drawn where the object is
    assert frame[int(300 * s), int(400 * s)] == 255  # cursor is the brightest thing on screen
    assert frame[fh + 1 :, -10:].max() < 255  # no key held -> key overlay dark

    _, _, _, _ = env.step_raw(400.0, 300.0, True)
    pressed = env._render()
    assert pressed[fh + 1 :, :].max() == 255  # key overlay lights up while a key is held


def test_env_and_judge_agree_on_perfect_play():
    cfg = EnvConfig()
    bm = parse_osu(COMPLEX_MAP)
    env = OsuEnv(MapData(bm, cfg), cfg, record=True)
    env.reset(mode="full")
    total = _drive_perfectly(env, bm)
    assert env.stats.n_miss == 0 and env.stats.ticks_missed == 0
    assert env.stats.accuracy > 95.0
    assert total > 0.0
    res = judge_replay(bm, env.frames)
    assert res.n_miss == 0
    assert abs(res.accuracy - env.stats.accuracy) < 1e-6


def test_env_penalizes_early_spam_and_unpressed_objects():
    cfg = EnvConfig(aim_shaping=0.0, skip_breaks=False)
    bm = parse_osu(SIMPLE_MAP)

    # Pressing on the circle 300ms early (inside the miss window) is a miss, not a hit
    env = OsuEnv(MapData(bm, cfg), cfg)
    env.reset(mode="full")
    while env.t + cfg.dt_ms < 700.0:
        env.step_raw(100.0, 100.0, False)
    _, r, _, _ = env.step_raw(100.0, 100.0, True)
    assert env.stats.n_miss == 1 and env.stats.n300 == 0
    assert r < 0.0

    # Never pressing misses everything and ends the episode at the end of the map
    env.reset(mode="full")
    done = False
    while not done:
        _, _, done, _ = env.step_raw(256.0, 192.0, False)
    assert env.stats.n_miss == 2


def test_spatial_action_table_and_network_shapes():
    cfg = EnvConfig()
    table = build_action_table(cfg)
    assert cfg.n_keys == 3 and table.shape == (cfg.n_actions, 3) and cfg.n_actions == cfg.grid_w * cfg.grid_h * 3
    assert set(np.unique(table[:, 2])) == {0.0, 1.0, 2.0}  # no key / K1 / K2
    # cell = row * grid_w + col, action = cell * 3 + key; cell centres tile the playfield
    cw, ch = 512.0 / cfg.grid_w, 384.0 / cfg.grid_h
    assert np.allclose(table[0], (cw / 2, ch / 2, 0)) and np.allclose(table[2], (cw / 2, ch / 2, 2))
    r, c = 5, 7
    assert np.allclose(table[(r * cfg.grid_w + c) * 3 + 1], ((c + 0.5) * cw, (r + 0.5) * ch, 1))

    net = QNetwork(4, cfg.obs_shape, cfg.render.field_height, (cfg.grid_h, cfg.grid_w))
    obs = torch.zeros(3, 4, *cfg.obs_shape, dtype=torch.uint8)
    q = net(obs)
    assert q.shape == (3, cfg.n_actions) and net.n_actions == cfg.n_actions
    assert torch.equal(net.greedy(obs), q.argmax(1))


def test_custom_aim_grid():
    import pytest

    from osuai.cli import _check_grid, _parse_grid

    assert _parse_grid("32x24") == (32, 24) and _parse_grid("16") == (16, 12)
    for bad in ("", "a", "0x5", "1x2x3"):
        with pytest.raises(ValueError):
            _parse_grid(bad)
    # the default grid is exactly the network's feature map at width 96; finer grids need a wider frame
    assert QNetwork.max_grid(EnvConfig().obs_shape, EnvConfig().render.field_height) == (18, 24)
    _check_grid(16, 12, 96, 4)
    with pytest.raises(ValueError, match="--width 128"):
        _check_grid(32, 24, 96, 4)
    _check_grid(32, 24, 128, 4)

    cfg = EnvConfig(width=128, grid_w=32, grid_h=24)
    table = build_action_table(cfg)
    assert table.shape == (cfg.n_actions, 3) and np.allclose(table[-1, :2], (512 - 8, 384 - 8))
    net = QNetwork(4, cfg.obs_shape, cfg.render.field_height, (cfg.grid_h, cfg.grid_w))
    assert net(torch.zeros(2, 4, *cfg.obs_shape, dtype=torch.uint8)).shape == (2, cfg.n_actions)


def test_spatial_action_glides_toward_cell_at_capped_speed():
    cfg = EnvConfig(action_repeat=2, aim_shaping=0.0, skip_breaks=False, max_speed=2400.0)
    env = OsuEnv(MapData(parse_osu(SIMPLE_MAP), cfg), cfg, record=True)
    env.reset(mode="full")
    t0, x0, y0 = env.t, env.x, env.y
    table = build_action_table(cfg)
    far_right = int(np.flatnonzero((table[:, 0] > 500) & np.isclose(table[:, 1], y0, atol=11) & (table[:, 2] == 0))[0])
    step_px = cfg.max_speed * cfg.dt_ms / 1000.0
    env.step(far_right)
    assert np.isclose(env.t, t0 + 2 * cfg.dt_ms)  # two simulation ticks per decision
    assert env.stats.steps == 1 and len(env.frames) == 3  # 1 decision, 1 recorded replay frame per tick
    moved = np.hypot(env.x - x0, env.y - y0)
    assert np.isclose(moved, 2 * step_px, atol=1e-6)  # far target: moves exactly max_speed for both ticks
    assert np.isclose(np.hypot(env.frames[1].x - x0, env.frames[1].y - y0), step_px)

    # a target closer than one tick of travel is reached exactly and the cursor stays there
    near = int(np.argmin(np.hypot(table[:, 0] - env.x, table[:, 1] - env.y) + table[:, 2]))
    env.step(near)
    assert np.isclose(env.x, table[near, 0]) and np.isclose(env.y, table[near, 1])


def _stream_map(n: int, spacing_ms: int, start_ms: int = 2000) -> str:
    objs = "\n".join(f"256,192,{start_ms + i * spacing_ms},1,0" for i in range(n))
    return f"""osu file format v14
[General]
Mode: 0
[Difficulty]
CircleSize: 4
OverallDifficulty: 5
ApproachRate: 9
[TimingPoints]
0,500,4,1,0,100,1,0
[HitObjects]
{objs}
"""


def _play_stream(key_pattern: list[int]) -> OsuEnv:
    """Stand on a 30-taps/s stream and cycle through key states (0 none, 1 K1, 2 K2), one per decision."""
    cfg = EnvConfig(aim_shaping=0.0, skip_breaks=False)
    bm = parse_osu(_stream_map(12, 33))
    env = OsuEnv(MapData(bm, cfg), cfg)
    env.reset(mode="full")
    table = build_action_table(cfg)
    cell = int(np.argmin(np.hypot(table[:, 0] - 256, table[:, 1] - 192))) // cfg.n_keys
    first = bm.objects[0].time
    while env.t + 2 * cfg.dt_ms < first - 8:  # walk to the stream without pressing
        env.step(cell * cfg.n_keys)
    done, i = False, 0
    while not done:
        _, _, done, _ = env.step(cell * cfg.n_keys + key_pattern[i % len(key_pattern)])
        i += 1
    return env


def test_two_keys_allow_a_tap_every_decision():
    alternating = _play_stream([1, 2])  # K1, K2, K1, ... : a fresh tap on every decision
    assert alternating.stats.n300 == 12 and alternating.stats.accuracy == 100.0
    one_key = _play_stream([1, 0])  # K1, release, K1, ... : at most one tap per two decisions
    assert one_key.stats.n_miss > 0 and one_key.stats.accuracy < 50.0
    held = _play_stream([1])  # holding K1 is a single tap, not a stream
    assert held.stats.presses == 1


def test_replay_buffer_stacks_and_nstep_returns():
    buf = ReplayBuffer(n_envs=1, capacity=64, obs_hw=(2, 2), stack=3, n_step=3, gamma=0.5)
    f = lambda v: np.full((2, 2), v, dtype=np.uint8)  # noqa: E731
    buf.begin_episode(0, f(1))
    buf.add_step(0, action=0, reward=1.0, terminal=False, next_frame=f(2))
    buf.add_step(0, action=1, reward=1.0, terminal=False, next_frame=f(3))
    buf.add_step(0, action=2, reward=1.0, terminal=True, next_frame=f(4))
    buf.begin_episode(0, f(10))
    buf.add_step(0, action=3, reward=5.0, terminal=False, next_frame=f(11))
    buf.add_step(0, action=4, reward=0.0, terminal=False, next_frame=f(12))

    rng = np.random.default_rng(0)
    seen = {}
    for _ in range(20):
        b = buf.sample(16, rng)
        for k in range(len(b["action"])):
            seen[int(b["action"][k])] = {key: b[key][k] for key in b}
    assert set(seen) == {0, 1, 2, 3, 4}
    # first transition: 3-step return 1 + .5 + .25 ending in a terminal -> no bootstrap
    s0 = seen[0]
    assert np.isclose(s0["ret"], 1.75) and s0["disc"] == 0.0
    assert [int(x[0, 0]) for x in s0["obs"]] == [1, 1, 1]  # padded with the episode's first frame
    s1 = seen[1]
    assert np.isclose(s1["ret"], 1.5) and s1["disc"] == 0.0
    assert [int(x[0, 0]) for x in s1["obs"]] == [1, 1, 2]
    s2 = seen[2]
    assert np.isclose(s2["ret"], 1.0) and s2["disc"] == 0.0
    assert [int(x[0, 0]) for x in s2["obs"]] == [1, 2, 3]
    # stacks never cross the episode boundary; the return is cut at the newest frame and bootstraps from it
    s3 = seen[3]
    assert [int(x[0, 0]) for x in s3["obs"]] == [10, 10, 10]
    assert np.isclose(s3["ret"], 5.0) and np.isclose(s3["disc"], 0.25)
    assert [int(x[0, 0]) for x in s3["next_obs"]] == [10, 11, 12]
    s4 = seen[4]
    assert np.isclose(s4["ret"], 0.0) and np.isclose(s4["disc"], 0.5)
    assert [int(x[0, 0]) for x in s4["next_obs"]] == [10, 11, 12]


def test_split_maps_is_disjoint_and_stable_across_pools():
    import hashlib

    ids = [hashlib.md5(str(i).encode()).hexdigest() for i in range(400)]
    train, val = split_maps(ids, val_ratio=0.1)
    assert not set(train) & set(val) and len(train) + len(val) == 400
    assert 25 <= len(val) <= 55  # about 10%
    # a map's role does not depend on which other maps are in the pool (e.g. a different star filter)
    sub_train, sub_val = split_maps(ids[:150], val_ratio=0.1)
    assert set(sub_val) <= set(val) and set(sub_train) <= set(train)


def test_split_maps_keeps_roles_from_earlier_rounds():
    import hashlib

    ids = [hashlib.md5(str(i).encode()).hexdigest() for i in range(300)]
    _, fresh_val = split_maps(ids, val_ratio=0.1)
    earlier_train = set(fresh_val[:3])  # pretend an earlier round trained on some of these
    earlier_val = {i for i in ids[:50] if i not in fresh_val}  # ...and validated on some fresh training maps
    train, val = split_maps(ids, val_ratio=0.1, forced_train=earlier_train, forced_val=earlier_val)
    assert earlier_train <= set(train) and earlier_val <= set(val) and not set(train) & set(val)
    # a tiny pool still holds out one map that was never trained on
    tr, va = split_maps(ids[:3], val_ratio=0.1, forced_train={ids[0]})
    assert len(va) == 1 and va[0] != ids[0]


def test_train_save_load_and_play(tmp_path: Path):
    env_cfg = EnvConfig(episode_seconds=4.0)
    bm = parse_osu(COMPLEX_MAP)
    md = MapData(bm, env_cfg)
    cfg = DQNConfig(
        total_steps=1200,
        n_envs=4,
        buffer_size=2000,
        batch_size=16,
        learning_starts=400,
        target_update=50,
        eps_decay_steps=1000,
        eval_interval=600,
        eval_maps=1,
        eval_seconds=3.0,
        log_interval=600,
    )
    round1 = tmp_path / "round1"
    result = train_dqn([md], [md], round1, env_cfg, cfg, log=lambda *_: None)
    for name in ("best.pt", "last.pt", "log.jsonl", "round.json"):
        assert (round1 / name).exists(), name
    assert result["updates"] > 0 and result["round_dir"].endswith("round1")
    summary = json.loads((round1 / "round.json").read_text(encoding="utf-8"))
    assert summary["status"] == "finished" and summary["round"] == 1 and summary["total_steps"] == 1200

    agent = load_checkpoint(round1, device="cpu")  # a round folder loads its best.pt
    frames, stats = play_map(agent.net, agent.env_cfg, bm, agent.dqn_cfg.frame_stack)
    assert len(frames) > 50
    assert stats.judged == len(bm.objects)
    res = judge_replay(bm, frames)
    assert res.n300 + res.n100 + res.n50 + res.n_miss == len(bm.objects)

    # ---- round 2: resume (curriculum). With learning_starts above the round's step budget no gradient step
    # happens, so the saved weights must be exactly the round-1 weights.
    agent = load_checkpoint(resolve_checkpoint(round1, prefer="last.pt"), device="cpu")
    assert agent.opt_state is not None and len(agent.history) == 1
    round1_step = agent.step
    frozen = DQNConfig(**{**cfg.to_dict(), "total_steps": 200, "learning_starts": 10_000, "eps_start": 0.3})
    round2 = tmp_path / "round2"
    r2 = train_dqn([md], [md], round2, env_cfg, frozen, log=lambda *_: None, resume=agent, round_info={"max_stars": 5.0})
    assert r2["round"] == 2 and r2["updates"] == 0
    again = load_checkpoint(round2 / "last.pt", device="cpu")
    assert again.step == round1_step + 200
    # map roles accumulate over rounds (here round 2 trains and validates on the same single test map)
    assert set(again.all_train_maps) >= set(agent.all_train_maps) and set(again.all_val_maps) >= set(agent.all_val_maps)
    assert [h["round"] for h in again.history] == [1, 2] and again.history[1]["max_stars"] == 5.0
    for k, v in agent.net.state_dict().items():
        assert torch.equal(v, again.net.state_dict()[k]), k
    assert (round1 / "best.pt").exists()  # earlier rounds are left untouched

    # ---- round 3: resume with real learning; the network moves away from the loaded weights
    more = DQNConfig(**{**cfg.to_dict(), "total_steps": 600, "learning_starts": 200})
    r3 = train_dqn([md], [md], tmp_path / "round3", env_cfg, more, log=lambda *_: None, resume=again)
    assert r3["round"] == 3 and r3["updates"] > 0
    third = load_checkpoint(tmp_path / "round3" / "last.pt", device="cpu")
    assert any(not torch.equal(v, third.net.state_dict()[k]) for k, v in again.net.state_dict().items())


def test_cli_curriculum_round_folders(tmp_path: Path, capsys, monkeypatch):
    import pytest

    from osuai.cli import main
    from osuai.dqn import latest_round_dir

    monkeypatch.chdir(tmp_path)  # default round folders are created under ./checkpoints
    map_file = tmp_path / "m.osu"
    map_file.write_text(COMPLEX_MAP, encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"maps": [{"path": str(map_file), "star_rating": 3.0}]}), encoding="utf-8")
    common = ["--maps", str(manifest), "--val-ratio", "0", "--envs", "2", "--buffer", "1000", "--batch-size", "8",
              "--learning-starts", "100", "--eval-interval", "200", "--log-interval", "200"]

    assert main(["train", *common, "--max-stars", "4", "--steps", "300"]) == 0
    r1 = tmp_path / "checkpoints" / "round1_0-4star"
    assert (r1 / "best.pt").exists() and (r1 / "last.pt").exists()
    capsys.readouterr()

    # resume from a round folder (its last.pt); the next folder is named after the new round and star range
    assert main(["train", *common, "--max-stars", "5", "--steps", "200", "--resume", str(r1)]) == 0
    out = capsys.readouterr().out
    r2 = tmp_path / "checkpoints" / "round2_0-5star"
    assert "resuming" in out and '"round": 2' in out and (r2 / "round.json").exists()
    r2_ck = load_checkpoint(r2 / "last.pt", device="cpu")
    assert r2_ck.step == load_checkpoint(r1 / "last.pt", device="cpu").step + 200
    assert r2_ck.history[-1]["resumed_from"].endswith("round1_0-4star/last.pt")
    assert latest_round_dir(tmp_path / "checkpoints") == r2

    # an existing round folder is never overwritten by accident
    with pytest.raises(FileExistsError):
        main(["train", *common, "--max-stars", "4", "--steps", "50"])
    assert main(["train", *common, "--max-stars", "4", "--steps", "200", "--force"]) == 0
    assert load_checkpoint(r1, device="cpu").step == 200
    # --out must be a folder, and the network's input size cannot change between rounds
    with pytest.raises(ValueError, match="folder"):
        main(["train", *common, "--steps", "50", "--out", str(tmp_path / "x.pt")])
    with pytest.raises(ValueError, match="--width"):
        main(["train", *common, "--steps", "50", "--resume", str(r1), "--width", "128", "--out", str(tmp_path / "x")])

    # round-1 choices (history length, late presses, aim grid) are stored and kept by later rounds
    custom = tmp_path / "custom"
    assert main(["train", *common, "--steps", "200", "--frame-stack", "6", "--grid", "16x12", "--out", str(custom)]) == 0  # late presses by default
    assert main(["train", *common, "--steps", "200", "--resume", str(custom), "--out", str(tmp_path / "custom2")]) == 0
    ck = load_checkpoint(tmp_path / "custom2", device="cpu")
    assert ck.dqn_cfg.frame_stack == ck.net.in_frames == 6
    assert ck.env_cfg.late_press and (ck.env_cfg.grid_w, ck.env_cfg.grid_h) == (16, 12) and ck.net.n_actions == 16 * 12 * 5
    for bad, match in ((["--frame-stack", "4"], "--frame-stack"), (["--grid", "24x18"], "--grid"), (["--no-late-press"], "--no-late-press")):
        with pytest.raises(ValueError, match=match):
            main(["train", *common, "--steps", "50", "--resume", str(custom), *bad, "--out", str(tmp_path / "x")])
    with pytest.raises(ValueError, match="at least 1"):
        main(["train", *common, "--steps", "50", "--frame-stack", "0", "--out", str(tmp_path / "x")])
    # penalties and costs are positive amounts; a signed value is refused rather than turned into a bonus
    with pytest.raises(ValueError, match="--slider-break-penalty"):
        main(["train", *common, "--steps", "50", "--slider-break-penalty", "-0.3", "--out", str(tmp_path / "x")])
    # --no-late-press keeps 3 key states, and late presses cannot be added back later
    plain = tmp_path / "plain"
    assert main(["train", *common, "--steps", "200", "--no-late-press", "--out", str(plain)]) == 0
    ck = load_checkpoint(plain, device="cpu")
    assert not ck.env_cfg.late_press and ck.net.n_actions == 24 * 18 * 3
    with pytest.raises(ValueError, match="--late-press"):
        main(["train", *common, "--steps", "50", "--resume", str(plain), "--late-press", "--out", str(tmp_path / "x")])
    # one tick per decision has no middle tick: off by default, an error when asked for
    assert main(["train", *common, "--steps", "200", "--action-repeat", "1", "--out", str(tmp_path / "ar1")]) == 0
    assert not load_checkpoint(tmp_path / "ar1", device="cpu").env_cfg.late_press
    with pytest.raises(ValueError, match="--action-repeat"):
        main(["train", *common, "--steps", "50", "--action-repeat", "1", "--late-press", "--out", str(tmp_path / "x")])


BREAK_MAP = """osu file format v14
[General]
Mode: 0
[Difficulty]
CircleSize: 4
OverallDifficulty: 5
ApproachRate: 5
[TimingPoints]
0,500,4,1,0,100,1,0
[HitObjects]
100,100,1000,1,0
400,300,8000,1,0
"""


def test_break_skip_keeps_the_last_hit_before_the_break_in_the_replay():
    """Training fast-forwards through breaks; the tick that triggers the skip must keep its real time in the replay."""
    cfg = EnvConfig(aim_shaping=0.0, skip_breaks=True)
    bm = parse_osu(BREAK_MAP)
    env = OsuEnv(MapData(bm, cfg), cfg, record=True)
    env.reset(mode="full")
    while env.t + cfg.dt_ms < 995.0:
        env.step_raw(100.0, 100.0, False)
    env.step_raw(100.0, 100.0, True)  # hit the circle right before the break -> the break is skipped in this tick
    assert env.stats.n300 == 1 and env.t > 5000.0  # judged as a hit, then fast-forwarded
    press = [f for f in env.frames if f.keys]
    assert press and press[0].time < 1100.0  # the press is recorded at its real time, not after the break
    done = False
    while not done:
        _, _, done, _ = env.step_raw(256.0, 192.0, False)
    res = judge_replay(bm, env.frames)
    assert (res.n300, res.n_miss) == (1, 1)  # the replay still scores the hit before the break


def test_play_map_plays_through_breaks():
    cfg = EnvConfig()
    bm = parse_osu(BREAK_MAP)
    net = QNetwork(4, cfg.obs_shape, cfg.render.field_height, (cfg.grid_h, cfg.grid_w))
    frames, _ = play_map(net, cfg, bm, frame_stack=4)
    gaps = [b.time - a.time for a, b in zip(frames, frames[1:])]
    assert max(gaps) <= cfg.dt_ms + 0.01  # one replay frame every tick (times kept to 3 decimals): no fast-forward
    assert frames[0].time < 1000.0 - 1200.0 and frames[-1].time >= 8000.0  # the whole song, break included


def test_maps_ever_trained_on_are_never_validation(tmp_path: Path):
    """Roles are rebuilt over the whole chain of rounds, and having been trained on always beats validation."""
    from osuai.dqn import LoadedAgent, collect_map_roles

    # an older checkpoint (round 1) that only stores its own round's maps
    r1 = tmp_path / "round1.pt"
    torch.save({"train_maps": ["a", "b"], "val_maps": ["v1", "v2"]}, r1)
    # a round 2 that validated on "a" (trained in round 1) and trained on "v1" (held out in round 1)
    agent = LoadedAgent(
        net=None, env_cfg=EnvConfig(), dqn_cfg=DQNConfig(), train_maps=["b", "c", "v1"], val_maps=["a", "d"], step=0,
        metrics={}, history=[{"round": 1, "resumed_from": None}, {"round": 2, "resumed_from": str(r1)}],
        all_train_maps=["b", "c", "v1"], all_val_maps=["a", "d"],
    )
    seen_train, seen_val = collect_map_roles(agent)
    assert seen_train == {"a", "b", "c", "v1"} and seen_val == {"v2", "d"}
    train, val = split_maps(["a", "b", "c", "d", "v1", "v2"], val_ratio=0.1, forced_train=seen_train, forced_val=seen_val)
    assert set(val) == {"d", "v2"} and set(train) == {"a", "b", "c", "v1"}


def test_late_press_keys_press_at_the_middle_tick():
    cfg = EnvConfig(late_press=True, aim_shaping=0.0, skip_breaks=False)
    assert cfg.n_keys == 5 and cfg.n_actions == cfg.grid_w * cfg.grid_h * 5 and EnvConfig().n_keys == 3
    assert [cfg.key_plan(k) for k in range(5)] == [(0, 0), (KEY_K1, 0), (KEY_K2, 0), (KEY_K1, 1), (KEY_K2, 1)]
    net = QNetwork(4, cfg.obs_shape, cfg.render.field_height, (cfg.grid_h, cfg.grid_w), cfg.n_keys)
    assert net(torch.zeros(1, 4, *cfg.obs_shape, dtype=torch.uint8)).shape == (1, cfg.n_actions)

    # OD 10: 300 window ±19.5 ms. Line the decisions up so that a press on the decision's first tick is a 100
    # and a press on its middle tick (one tick, 16.7 ms, later) is a 300.
    bm_text = _stream_map(2, 1000).replace("OverallDifficulty: 5", "OverallDifficulty: 10")
    table = build_action_table(cfg)
    cell = int(np.argmin(np.hypot(table[:, 0] - 256, table[:, 1] - 192))) // cfg.n_keys
    results = {}
    for key in (1, 3):  # K1, K1 late
        env = OsuEnv(MapData(parse_osu(bm_text), cfg), cfg, record=True)
        env.reset(mode="full")
        hit_at = env.md.bm.objects[0].time
        w300 = 19.5
        while env.t + cfg.dt_ms < hit_at - w300 - cfg.dt_ms:  # single ticks: the next tick lands in [T-36.2, T-19.5)
            env.step_raw(256.0, 192.0, False)
        env.step(cell * cfg.n_keys + key)
        keys = [f.keys for f in env.frames[-2:]]
        assert keys == ([KEY_K1, KEY_K1] if key == 1 else [0, KEY_K1])
        results[key] = (env.stats.n300, env.stats.n100)
    assert results == {1: (0, 1), 3: (1, 0)}

    # a late key re-taps a key that is already held: up for one tick, then down again
    env = OsuEnv(MapData(parse_osu(SIMPLE_MAP), cfg), cfg, record=True)
    env.reset(mode="full")
    env.step(cell * cfg.n_keys + 1)
    presses = env.stats.presses
    env.step(cell * cfg.n_keys + 1)  # still held: no new press
    env.step(cell * cfg.n_keys + 3)
    assert [f.keys for f in env.frames[-2:]] == [0, KEY_K1] and env.stats.presses == presses + 1


def _one_object_map(obj: str) -> str:
    return f"""osu file format v14
[General]
Mode: 0
[Difficulty]
CircleSize: 4
OverallDifficulty: 5
ApproachRate: 9
SliderMultiplier: 1
SliderTickRate: 1
[TimingPoints]
0,500,4,1,0,100,1,0
[HitObjects]
{obj}
"""


def test_spinner_rotation_is_paid_tick_by_tick():
    cfg = EnvConfig(aim_shaping=0.0, skip_breaks=False)
    env = OsuEnv(MapData(parse_osu(_one_object_map("256,192,2000,8,0,5000")), cfg), cfg)
    env.reset(mode="full")
    req = int(env.md.spins_required[0])
    assert req == 7  # OD 5: 2.5 spins/s for 3 s
    step_deg, r = 60.0, 60.0  # one sixth of a rotation per tick, 60 px from the centre
    ang, rewards = 0.0, []
    while env.t + cfg.dt_ms < 2000:
        env.step_raw(256 + r, 192, 0)
    while env.t + cfg.dt_ms < 5000 - cfg.dt_ms:
        ang += step_deg
        x, y = 256 + r * np.cos(np.radians(ang)), 192 + r * np.sin(np.radians(ang))
        _, rew, _, _ = env.step_raw(x, y, KEY_K1)
        rewards.append(rew)
    spins = abs(env.spin_net[0]) / (2 * np.pi)
    assert spins > req + 2
    sixth = 1.0 / 6.0
    # every tick of rotation pays, not just completed turns: r_spin per rotation up to the 300, r_spin_bonus after
    assert np.isclose(rewards[2], cfg.r_spin * sixth)
    assert np.isclose(rewards[-1], cfg.r_spin_bonus * sixth)
    assert np.isclose(sum(rewards[1:]), cfg.r_spin * req + cfg.r_spin_bonus * (spins - req))
    # turning back takes the reward back
    ang -= step_deg
    _, rew, _, _ = env.step_raw(256 + r * np.cos(np.radians(ang)), 192 + r * np.sin(np.radians(ang)), KEY_K1)
    assert np.isclose(rew, -cfg.r_spin_bonus * sixth)


def _slider_run(offset: float = 0.0, release_at: float | None = None) -> OsuEnv:
    """Hit a 1.5 s straight slider's head on time, then follow the ball `offset` px below its centre with K1 held,
    letting go for one tick at `release_at` (ms)."""
    cfg = EnvConfig(aim_shaping=0.0, skip_breaks=False)
    bm = parse_osu(_one_object_map("100,192,2000,2,0,L|400:192,1,300"))
    obj = bm.objects[0]
    env = OsuEnv(MapData(bm, cfg), cfg)
    env.reset(mode="full")
    while env.t + cfg.dt_ms < obj.time - 8:
        env.step_raw(100, 192, 0)
    done = False
    while not done:
        t1 = env.t + cfg.dt_ms
        ball = obj.slider_pos_at_time(min(max(t1, obj.time), obj.end_time))
        key = 0 if release_at is not None and release_at <= t1 < release_at + cfg.dt_ms else KEY_K1
        dy = offset if t1 > obj.time + 50 else 0.0  # the head is hit on its centre
        _, _, done, _ = env.step_raw(ball.x, ball.y + dy, key)
    return env


def test_slider_shaping_costs_offset_and_breaks():
    cfg = EnvConfig()
    centre = _slider_run()
    assert centre.stats.ends_hit == 1 and centre.stats.ticks_missed == 0 and centre.stats.slider_breaks == 0
    # following 30 px off the centre still tracks, but costs slider_offset_cost * time * 30 / follow radius
    off = _slider_run(offset=30.0)
    assert off.stats.ends_hit == 1 and off.stats.ticks_missed == 0 and off.stats.slider_breaks == 0
    expected = cfg.slider_offset_cost * (1.5 - 0.05) * 30.0 / centre.md.follow_radius
    assert abs((centre.stats.reward - off.stats.reward) - expected) < 0.02
    # letting go for one tick breaks the slider at once; on the ball again, it is tracked again
    released = _slider_run(release_at=2700.0)
    assert released.stats.slider_breaks == 1 and released.stats.ends_hit == 1
    assert released.stats.reward < centre.stats.reward - cfg.slider_break_penalty + 0.01
    # off the centre, a release cannot re-enter (tracking restarts only on the ball): ticks and the end are lost
    off_released = _slider_run(offset=50.0, release_at=2700.0)
    assert off_released.stats.slider_breaks == 1 and off_released.stats.ticks_missed > 0 and off_released.stats.ends_hit == 0


def test_spinner_moves_count_under_the_keys_they_start_from():
    """Sliding along a line and holding the key only on the way back spins nothing in lazer: replay playback moves the
    cursor between frames under the keys of the frame it leaves. Env and judge must agree; circling still counts."""
    from osuai.replay import quantize_frames

    cfg = EnvConfig(aim_shaping=0.0, skip_breaks=False)
    bm = parse_osu(_one_object_map("256,192,2000,8,0,5000"))

    def run(path) -> tuple[float, int]:
        env = OsuEnv(MapData(bm, cfg), cfg, record=True)
        env.reset(mode="full")
        i, done = 0, False
        while not done:
            x, y, key = path(i) if env.t + cfg.dt_ms >= 2000 else (75.0, 181.0, 0)
            _, _, done, _ = env.step_raw(x, y, key)
            i += 1
        res = judge_replay(bm, quantize_frames(env.frames))
        return abs(env.spin_net[0]) / (2 * np.pi), res.spinner_spins_hit

    line = [(75.0, 181.0, KEY_K2), (189.0, 250.0, 0), (303.0, 319.0, 0), (189.0, 250.0, KEY_K2)]  # the learned trick
    spins, ticks = run(lambda i: line[i % 4])
    assert spins < 1.0 and ticks == 0

    def circle(i):
        a = np.radians(60.0 * i)
        return (256 + 60 * np.cos(a), 192 + 60 * np.sin(a), KEY_K1)

    spins, ticks = run(circle)
    assert spins > 20 and ticks > 0


def test_eval_maps_spread_over_the_star_range():
    from osuai.dqn import pick_eval_maps

    ids = [f"m{i}" for i in range(10)]
    stars = {m: 10.0 - i for i, m in enumerate(ids)}  # m0 is the hardest
    assert pick_eval_maps(ids, stars, 3) == ["m9", "m5", "m0"]  # easiest, middle, hardest
    assert pick_eval_maps(ids, stars, 20) == sorted(ids, key=lambda m: stars[m])  # fewer maps than asked: all
    assert pick_eval_maps(ids, stars, 1) == ["m4"]
    assert pick_eval_maps(list(reversed(ids)), stars, 3) == pick_eval_maps(ids, stars, 3)  # order does not matter


def test_next_round_replays_the_previous_rounds_eval_maps(tmp_path: Path, capsys, monkeypatch):
    import pytest

    from osuai.cli import main

    monkeypatch.chdir(tmp_path)
    entries = []
    for name, text, stars in (("a", COMPLEX_MAP, 3.0), ("b", SIMPLE_MAP, 2.0)):
        (tmp_path / f"{name}.osu").write_text(text, encoding="utf-8")
        entries.append({"path": str(tmp_path / f"{name}.osu"), "star_rating": stars})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"maps": entries}), encoding="utf-8")
    common = ["--maps", str(manifest), "--val-ratio", "0.5", "--envs", "2", "--buffer", "1000", "--batch-size", "8",
              "--learning-starts", "100", "--eval-interval", "200", "--log-interval", "200"]

    def evals(d: Path) -> list[dict]:
        return [json.loads(line) for line in (d / "log.jsonl").read_text(encoding="utf-8").splitlines() if '"eval"' in line]

    assert main(["train", *common, "--steps", "300", "--out", "r1"]) == 0
    r1 = load_checkpoint(tmp_path / "r1" / "last.pt", device="cpu")
    assert r1.eval_maps and set(r1.eval_maps) <= set(r1.val_maps)
    assert evals(tmp_path / "r1") and not any("eval_prev_accuracy" in e for e in evals(tmp_path / "r1"))

    # round 2 evaluates on its own maps and plays round 1's again, logged as eval_prev_*
    assert main(["train", *common, "--steps", "200", "--resume", "r1", "--out", "r2"]) == 0
    summary = json.loads((tmp_path / "r2" / "round.json").read_text(encoding="utf-8"))
    assert summary["prev_eval_maps"] == r1.eval_maps and summary["eval_maps"]
    assert all("eval_prev_accuracy" in e and "eval_accuracy" in e for e in evals(tmp_path / "r2"))
    with pytest.raises(ValueError, match="--eval-maps"):
        main(["train", *common, "--steps", "50", "--eval-maps", "0", "--out", "x"])

    # hit rewards default to lazer's accuracy weights, can be set per round, and must stay ordered 0 <= 50 <= 100 <= 1
    assert np.isclose(r1.env_cfg.r100, 1 / 3) and np.isclose(r1.env_cfg.r50, 1 / 6) and r1.env_cfg.slider_offset_cost == 1.0
    custom = {"--reward-300": 1.2, "--reward-100": 0.25, "--reward-50": 0.1, "--reward-miss": -0.2, "--reward-tick": 0.05,
              "--reward-tick-miss": -0.01, "--reward-tail": 0.4}
    args = [str(x) for kv in custom.items() for x in kv]
    assert main(["train", *common, "--steps", "200", "--resume", "r2", *args, "--out", "r3"]) == 0
    e = load_checkpoint(tmp_path / "r3" / "last.pt", device="cpu").env_cfg
    assert (e.r300, e.r100, e.r50, e.r_miss, e.r_tick, e.r_tick_miss, e.r_tail) == tuple(custom.values())
    for bad, match in (
        (["--reward-100", "1.5"], "--reward-50 <= --reward-100 <= --reward-300"),  # a 100 worth more than a 300
        (["--reward-100", "0.2", "--reward-50", "0.3"], "--reward-50 <= --reward-100"),
        (["--reward-50", "-0.1"], "--reward-50 is the reward for a hit"),
        (["--reward-miss", "0.1"], "--reward-miss is the reward for a miss"),
        (["--reward-tick-miss", "0.02"], "--reward-tick-miss is the reward for a miss"),
    ):
        with pytest.raises(ValueError, match=match):
            main(["train", *common, "--steps", "50", *bad, "--out", "x"])
    capsys.readouterr()


def test_a_map_in_both_eval_sets_is_played_once_per_evaluation(tmp_path: Path, monkeypatch):
    import osuai.dqn as dqn

    env_cfg = EnvConfig(episode_seconds=4.0)
    a, b = MapData(parse_osu(COMPLEX_MAP), env_cfg), MapData(parse_osu(SIMPLE_MAP), env_cfg)
    played: list[str] = []
    real_play = dqn.play_map

    def counting_play(net, cfg, md, *args, **kwargs):
        played.append(md.bm.md5)
        return real_play(net, cfg, md, *args, **kwargs)

    monkeypatch.setattr(dqn, "play_map", counting_play)
    cfg = DQNConfig(total_steps=400, n_envs=4, buffer_size=1000, batch_size=8, learning_starts=200,
                    eval_interval=200, eval_seconds=2.0, log_interval=200)
    train_dqn([a], [a, b], tmp_path / "r", env_cfg, cfg, eval_maps=[a, b], prev_eval_maps=[b], log=lambda *_: None)
    evals = [json.loads(line) for line in (tmp_path / "r" / "log.jsonl").read_text(encoding="utf-8").splitlines() if '"eval"' in line]
    assert len(evals) == 2
    # b is in both sets but is played once per evaluation; both sets still get their numbers
    assert played.count(b.bm.md5) == played.count(a.bm.md5) == len(evals)
    assert all(e["eval_prev_accuracy"] is not None and "eval_accuracy" in e for e in evals)


def test_holding_a_key_costs_only_when_nothing_needs_it():
    cfg = EnvConfig(aim_shaping=0.0, skip_breaks=False)
    dt = cfg.dt_ms

    def play(obj: str, follow, press_from: float, hold_until_after_end: float = 1000.0) -> OsuEnv:
        """No key until `press_from` (ms), then K1 held until that long after the object ends, cursor at follow(t)."""
        bm = parse_osu(_one_object_map(obj))
        env = OsuEnv(MapData(bm, cfg), cfg)
        env.reset(mode="full")
        o = bm.objects[0]
        done = False
        while not done and env.t + dt < o.end_time + hold_until_after_end:
            t1 = env.t + dt
            _, _, done, _ = env.step_raw(*follow(t1), KEY_K1 if t1 >= press_from else 0)
        return env

    # a slider: its head pressed on time, then held through to its end (free) and 1 s past it (charged)
    slider = "100,192,2000,2,0,L|400:192,1,300\n256,192,6000,1,0"  # a circle later keeps the map going
    obj = parse_osu(_one_object_map(slider)).objects[0]
    ball = lambda t: (lambda p: (p.x, p.y))(obj.slider_pos_at_time(min(max(t, obj.time), obj.end_time)))  # noqa: E731
    env = play(slider, ball, press_from=obj.time - 8)
    assert env.stats.n300 == 1 and env.stats.ends_hit == 1 and env.stats.slider_breaks == 0
    after_end = round((1000 - TAIL_LENIENCY) / dt)  # the end is judged within its last 36 ms; holding after that is idle
    assert abs(env.stats.idle_hold_ticks - after_end) <= 2
    # holding while waiting for a circle-less moment is charged: 1 s too early before the head
    early = play(slider, ball, press_from=obj.time - 1000)
    assert early.stats.idle_hold_ticks > env.stats.idle_hold_ticks + 50
    # a spinner wants the key held from its start to its end
    spin = play("256,192,2000,8,0,5000\n256,192,8000,1,0", lambda t: (256 + 60 * np.cos(t / 50), 192 + 60 * np.sin(t / 50)), press_from=1000)
    assert abs(spin.stats.idle_hold_ticks - round(2000 / dt)) <= 3  # 1 s before it + 1 s after it
    # the cost is hold_cost per second of idle holding
    free = EnvConfig(aim_shaping=0.0, skip_breaks=False, hold_cost=0.0)
    a, b = OsuEnv(MapData(parse_osu(_one_object_map(slider)), cfg), cfg), OsuEnv(MapData(parse_osu(_one_object_map(slider)), free), free)
    for e in (a, b):
        e.reset(mode="full")
        for _ in range(60):  # one second holding K1 long before the slider
            e.step_raw(100, 192, KEY_K1)
    assert np.isclose(b.stats.reward - a.stats.reward, cfg.hold_cost * 60 * dt / 1000.0)


def test_spin_reward_credits_at_most_spin_max_deg_per_tick():
    assert EnvConfig().spin_max_deg == 0.0  # off by default
    cfg = EnvConfig(aim_shaping=0.0, skip_breaks=False, spin_max_deg=60.0)
    env = OsuEnv(MapData(parse_osu(_one_object_map("256,192,2000,8,0,5000")), cfg), cfg)
    env.reset(mode="full")
    while env.t + cfg.dt_ms < 2000:
        env.step_raw(316, 192, 0)
    ang, rewards = 0.0, []
    for _ in range(12):  # 90 degrees per tick, a quarter turn
        ang += 90.0
        _, rew, _, _ = env.step_raw(256 + 60 * np.cos(np.radians(ang)), 192 + 60 * np.sin(np.radians(ang)), KEY_K1)
        rewards.append(rew)
    # each 90-degree tick is credited as 60 degrees (a sixth of a turn), but the spinner counts all of it
    assert np.isclose(rewards[-1], cfg.r_spin / 6.0)
    assert np.isclose(abs(env.spin_net[0]), np.radians(90.0 * 11))  # the first tick only sets the starting angle
    # turning back costs the full 90 degrees
    ang -= 90.0
    _, rew, _, _ = env.step_raw(256 + 60 * np.cos(np.radians(ang)), 192 + 60 * np.sin(np.radians(ang)), KEY_K1)
    assert np.isclose(rew, -cfg.r_spin / 4.0)
    # with no cap, the full 90 degrees is credited
    free = EnvConfig(aim_shaping=0.0, skip_breaks=False, spin_max_deg=0.0)
    env = OsuEnv(MapData(parse_osu(_one_object_map("256,192,2000,8,0,5000")), free), free)
    env.reset(mode="full")
    while env.t + free.dt_ms < 2000:
        env.step_raw(316, 192, 0)
    ang = 0.0
    for _ in range(3):
        ang += 90.0
        _, rew, _, _ = env.step_raw(256 + 60 * np.cos(np.radians(ang)), 192 + 60 * np.sin(np.radians(ang)), KEY_K1)
    assert np.isclose(rew, free.r_spin / 4.0)


def test_early_hits_are_paid_at_the_objects_time():
    two = "256,192,2000,1,0\n256,192,4000,1,0"

    def run(pay_on_time: bool, press_at: float, obj: str = two, cut_after_press: bool = False) -> tuple[list[float], list[float], OsuEnv]:
        cfg = EnvConfig(aim_shaping=0.0, skip_breaks=False, pay_hits_on_time=pay_on_time)
        env = OsuEnv(MapData(parse_osu(_one_object_map(obj)), cfg), cfg)
        env.reset(mode="full")
        times, rewards, done = [], [], False
        while not done and env.t + cfg.dt_ms < 2300:
            t1 = env.t + cfg.dt_ms
            pressing = press_at <= t1 < press_at + cfg.dt_ms
            if pressing and cut_after_press:
                env.max_steps = env.stats.steps + 1  # this decision ends the episode
            _, r, terminal, truncated = env.step_raw(256, 192, KEY_K1 if pressing else 0)
            times.append(t1)
            rewards.append(r)
            done = terminal or truncated
        return times, rewards, env

    cfg = EnvConfig()
    # an early 300 (about 25 ms early, inside OD 5's +-44.5 ms window) is paid on the first tick at or after 2000 ms
    t_on, r_on, env_on = run(True, 1975)
    t_off, r_off, _ = run(False, 1975)
    assert env_on.stats.n300 == 1
    press = next(i for i, t in enumerate(t_off) if 1975 <= t)
    due = next(i for i, t in enumerate(t_on) if t >= 2000)
    assert due > press
    diff = np.array(r_on) - np.array(r_off)
    assert np.isclose(diff[press], -cfg.r300) and np.isclose(diff[due], cfg.r300)
    assert np.isclose(np.delete(diff, [press, due]), 0).all() and np.isclose(sum(r_on), sum(r_off))
    # a late hit is paid on the press either way
    t_on, r_on, _ = run(True, 2010)
    _, r_off, _ = run(False, 2010)
    assert np.allclose(r_on, r_off)
    # the map ending or the episode being cut short pays what is still waiting
    _, r_end, env_end = run(True, 1975, obj="256,192,2000,1,0")
    hold_tick = cfg.hold_cost * cfg.dt_ms / 1000.0  # the key is still down on the hit's tick, with nothing to hold
    assert env_end.stats.n300 == 1 and np.isclose(r_end[-1], cfg.r300 - cfg.press_penalty - hold_tick) and not env_end.pending
    _, r_cut, env_cut = run(True, 1975, cut_after_press=True)
    assert env_cut.stats.n300 == 1 and r_cut[-1] > cfg.r300 - 0.1 and not env_cut.pending


def test_a_press_that_hits_nothing_costs_extra():
    cfg = EnvConfig(aim_shaping=0.0, skip_breaks=False)

    def env_for(objs: str) -> OsuEnv:
        env = OsuEnv(MapData(parse_osu(_one_object_map(objs)), cfg), cfg)
        env.reset(mode="full")
        return env

    def until(env: OsuEnv, t: float, x: float, y: float, key: int = 0) -> None:
        while env.t + cfg.dt_ms < t:
            env.step_raw(x, y, key)

    # a press in empty space: the press penalty plus the empty-press cost (plus one tick of idle holding)
    env = env_for("256,192,4000,1,0")
    until(env, 1000, 50, 50)
    _, r, _, _ = env.step_raw(50, 50, KEY_K1)
    assert env.stats.empty_presses == 1
    assert np.isclose(r, -cfg.press_penalty - cfg.empty_press_cost - cfg.hold_cost * cfg.dt_ms / 1000.0)
    # a press that hits a circle, or one far too early on it (a miss), judged something: no extra cost
    for press_at in (4000 - 8, 4000 - 300):
        env = env_for("256,192,4000,1,0\n256,192,6000,1,0")
        until(env, press_at, 256, 192)
        env.step_raw(256, 192, KEY_K1)
        assert env.stats.empty_presses == 0 and env.stats.n300 + env.stats.n_miss == 1
    # pressing again during a slider (to pick it back up) or a spinner, or just before a spinner, is not wasted
    slider = "100,192,2000,2,0,L|400:192,1,300\n256,192,6000,1,0"
    obj = parse_osu(_one_object_map(slider)).objects[0]
    env = env_for(slider)
    until(env, obj.time - 8, 100, 192)
    t = obj.time - 8
    while env.t + cfg.dt_ms < obj.time + 600:
        t1 = env.t + cfg.dt_ms
        b = obj.slider_pos_at_time(min(max(t1, obj.time), obj.end_time))
        env.step_raw(b.x, b.y, 0 if obj.time + 300 <= t1 < obj.time + 340 else KEY_K1)  # let go for a moment
    assert env.stats.empty_presses == 0 and env.stats.presses == 2
    for press_at in (2000 - 50, 2500):
        env = env_for("256,192,2000,8,0,5000\n256,192,8000,1,0")
        until(env, press_at, 316, 192)
        env.step_raw(316, 192, KEY_K1)
        assert env.stats.empty_presses == 0


def test_spinner_frames_show_its_area_rotation_and_progress():
    cfg = EnvConfig(aim_shaping=0.0, skip_breaks=False)
    env = OsuEnv(MapData(parse_osu(_one_object_map("256,192,2000,8,0,5000\n256,192,8000,1,0")), cfg), cfg)
    env.reset(mode="full")
    while env.t + cfg.dt_ms < 2000:
        env.step_raw(316, 192, 0)
    s = cfg.render.scale

    def px(frame, x, y):
        return int(frame[int(y * s), int(x * s)])

    before = env.step_raw(256, 100, 0)[0]
    # the whole spinner area is filled, so a running spinner never looks like an empty screen
    assert px(before, 256 + 120, 192) > 0 and px(before, 256 - 120, 192 + 40) > 0
    # the line points up while nothing has been spun, and does not move while no key is held
    up = px(before, 256, 192 - 120)
    assert up > px(before, 256, 192 + 120) + 20
    # spinning with a key held turns the line (a quarter turn clockwise on screen: it points right) and brightens the area
    for a in np.linspace(-90, 0, 7):
        frame = env.step_raw(256 + 80 * np.cos(np.radians(a)), 192 + 80 * np.sin(np.radians(a)), KEY_K1)[0]
    assert np.isclose(env.spin_net[0], np.pi / 2, atol=0.05)
    assert px(frame, 256 + 140, 192) > px(frame, 256, 192 - 140) + 20  # the line now points right
    assert px(frame, 256 - 120, 192 + 40) > px(before, 256 - 120, 192 + 40)  # some progress: a brighter area


def test_jumping_across_the_spinner_centre_spins_nothing():
    """Cells come in pairs exactly opposite each other around the playfield centre, so a jump between them is exactly
    180 degrees, which has no direction. Like lazer and the replay judge, the env keeps +180 and -180 as they are, so
    jumping back and forth across the centre cancels out instead of counting as half a turn every jump."""
    from osuai.replay import quantize_frames

    cfg = EnvConfig(aim_shaping=0.0, skip_breaks=False)
    bm = parse_osu(_one_object_map("256,192,2000,8,0,5000\n256,192,8000,1,0"))
    env = OsuEnv(MapData(bm, cfg), cfg, record=True)
    env.reset(mode="full")
    while env.t + cfg.dt_ms < 2000:
        env.step_raw(267, 181, 0)
    path = [(267, 181), (224, 224), (288, 160), (245, 203), (309, 139)]  # all on one line through (256, 192)
    i = 0
    while env.t + cfg.dt_ms < 5000:
        env.step_raw(*path[(i // 2) % len(path)], KEY_K1)
        i += 1
    assert abs(env.spin_net[0]) < np.pi + 1e-6  # at most the half turn of a single unmatched jump
    while not env.step_raw(256, 192, 0)[2]:
        pass
    assert judge_replay(bm, quantize_frames(env.frames)).spinner_spins_hit == 0
