"""Command-line interface: import maps, train the DQN agent, play maps into replays, judge, benchmark and preview."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from osuai.beatmap import Beatmap, parse_osu
from osuai.judge import compute_pp, hit_errors, judge_replay, unstable_rate
from osuai.replay import decode_osr, quantize_frames, write_osr

MAPS_ROOT = Path("maps")
IMPORTED_DIR = MAPS_ROOT / "imported"  # maps extracted from osu!lazer
DEFAULT_MANIFEST = IMPORTED_DIR / "manifest.json"


def _find_map(md5: str, maps_dir: str | Path = MAPS_ROOT) -> Path:
    """Locate <md5>.osu in maps_dir, or in any of its sub-folders."""
    root = Path(maps_dir)
    direct = root / f"{md5}.osu"
    if direct.exists():
        return direct
    found = next(iter(root.rglob(f"{md5}.osu")), None) if root.is_dir() else None
    if found is None:
        raise FileNotFoundError(f"Beatmap {md5}.osu not found under {root}/")
    return found


def _resolve_osr_out_path(map_path: str | Path, model_name: str, out_arg: str | None) -> Path:
    """Organize generated .osr files under out/<map_folder>/<model_name>.osr."""
    map_folder = Path(map_path).stem
    if not out_arg:
        return Path("out") / map_folder / f"{model_name}.osr"
    out_p = Path(out_arg)
    if out_p.suffix.lower() == ".osr":
        return out_p
    return out_p / map_folder / f"{model_name}.osr"


def cmd_import_lazer(args: argparse.Namespace) -> int:
    lazer_dir = Path(args.lazer_dir or os.path.expandvars(r"%APPDATA%\osu"))
    files_dir = lazer_dir / "files"
    if not files_dir.exists():
        raise FileNotFoundError(f"osu!lazer files directory not found: {files_dir}")

    out_maps = Path(args.out_maps)
    out_maps.mkdir(parents=True, exist_ok=True)

    discovered: list[tuple[float, Beatmap]] = []
    seen_md5: set[str] = set()

    for blob in files_dir.rglob("*"):
        if not blob.is_file():
            continue
        try:
            size = blob.stat().st_size
            if size < 64 or size > 5_000_000:
                continue
            with blob.open("rb") as fh:
                head = fh.read(64)
            if b"osu file format v" not in head:
                continue
            bm = parse_osu(blob)
            if len(bm.objects) < 10 or bm.md5 in seen_md5:
                continue
            if any(
                bm.objects[i + 1].time - bm.objects[i].time < 2.0
                and bm.objects[i + 1].pos.dist(bm.objects[i].pos) > 10.0
                for i in range(len(bm.objects) - 1)
            ):
                continue
            stars, _ = compute_pp(bm)
            if stars < args.min_stars or (args.max_stars > 0 and stars > args.max_stars):
                continue
            seen_md5.add(bm.md5)
            discovered.append((stars, bm))
        except Exception:
            continue

    if not discovered:
        raise RuntimeError(f"No matching standard .osu beatmaps found in {files_dir}")

    # Sort descending by star rating so harder maps are included first
    discovered.sort(key=lambda item: item[0], reverse=True)
    selected = discovered[: args.limit] if args.limit > 0 else discovered

    map_entries = []
    for stars, bm in selected:
        dest_map = out_maps / f"{bm.md5}.osu"
        dest_map.write_bytes(bm.raw_bytes)
        map_entries.append(
            {
                "path": str(dest_map).replace("\\", "/"),
                "md5": bm.md5,
                "title": f"{bm.artist} - {bm.title} [{bm.version}]",
                "star_rating": round(stars, 2),
                "mods": ["NM"],
                "notes": f"objects={len(bm.objects)}",
            }
        )

    (out_maps / "manifest.json").write_text(json.dumps({"maps": map_entries}, indent=2), encoding="utf-8")
    star_vals = [m["star_rating"] for m in map_entries]
    print(
        json.dumps(
            {
                "maps_imported": len(map_entries),
                "star_range": [min(star_vals), max(star_vals)],
                "maps_manifest": str(out_maps / "manifest.json"),
            },
            indent=2,
        )
    )
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    import random

    from osuai.synth import generate_map, parse_ratio, parse_star_range

    name = (args.name if args.name is not None else (args.only or "synthetic")).strip()
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError(f"--name must be a plain folder name inside maps/ (got {args.name!r})")
    if Path(name) == Path(IMPORTED_DIR.name):
        raise ValueError(f"maps/{IMPORTED_DIR.name} holds the maps imported from osu!lazer; pick another --name")
    if args.only and args.ratio:
        raise ValueError("Use either --only or --ratio, not both")
    if args.only:
        ratio = {"circles": (1.0, 0.0, 0.0), "sliders": (0.0, 1.0, 0.0), "spinners": (0.0, 0.0, 1.0)}[args.only]
    else:
        ratio = parse_ratio(args.ratio or "60/35/5")
    lo, hi = parse_star_range(args.stars)
    if args.count < 1:
        raise ValueError("--count must be at least 1")
    if ratio[0] == ratio[1] == 0.0:
        print("[generate] note: spinners carry no star rating, so these maps rate about 0* whatever --stars says", flush=True)
    seed = args.seed if args.seed is not None else random.randrange(1 << 30)

    out_dir = MAPS_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    existing = json.loads(manifest_path.read_text(encoding="utf-8")).get("maps", []) if manifest_path.exists() else []
    by_md5 = {e.get("md5"): e for e in existing}

    made = []
    for i in range(args.count):
        target = lo if args.count == 1 else lo + (hi - lo) * i / (args.count - 1)
        res = generate_map(target, ratio, length_s=args.length, seed=seed + i, tolerance=args.tolerance)
        dest = out_dir / f"{res.md5}.osu"
        # write the exact bytes res.md5 was computed from: write_text would turn "\n" into "\r\n" on Windows, and
        # every reader identifies a map by the MD5 of its bytes, so the file name would no longer match
        dest.write_bytes(res.text.encode("utf-8"))
        title = res.text.split("Title:", 1)[1].split("\n", 1)[0]
        entry = {
            "path": str(dest).replace("\\", "/"),
            "md5": res.md5,
            "title": f"osuai - {title} [seed {seed + i}]",
            "star_rating": round(res.star_rating, 2),
            "mods": ["NM"],
            "notes": f"objects={res.n_objects} circles/sliders/spinners={res.n_circles}/{res.n_sliders}/{res.n_spinners} "
                     f"bpm={res.bpm:.0f} target={target:.2f} synthetic",
            "synthetic": True,
        }
        by_md5[res.md5] = entry
        missed = abs(res.star_rating - target) > args.tolerance
        made.append(entry)
        print(f"[generate] {i + 1}/{args.count}: {res.star_rating:5.2f}* (target {target:.2f}) "
              f"{res.n_circles}/{res.n_sliders}/{res.n_spinners} objects, {res.bpm:.0f} BPM -> {dest.name}"
              + ("  (target out of reach for this mix)" if missed else ""), flush=True)

    entries = sorted(by_md5.values(), key=lambda e: -float(e.get("star_rating", 0.0)))
    manifest_path.write_text(json.dumps({"maps": entries}, indent=2), encoding="utf-8")
    stars = [e["star_rating"] for e in made]
    print(json.dumps({
        "maps_generated": len(made),
        "star_range": [min(stars), max(stars)],
        "maps_in_folder": len(entries),
        "maps_manifest": str(manifest_path).replace("\\", "/"),
    }, indent=2))
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    bm = parse_osu(args.map, mods=args.mods)
    stars, max_pp = compute_pp(bm)
    bpm_min, bpm_max = bm.bpm_range()
    n_circles = sum(1 for o in bm.objects if o.kind == "circle")
    n_sliders = sum(1 for o in bm.objects if o.kind == "slider")
    n_spinners = sum(1 for o in bm.objects if o.kind == "spinner")
    info = {
        "title": f"{bm.artist} - {bm.title} [{bm.version}]",
        "md5": bm.md5,
        "mods": list(bm.difficulty.mods),
        "cs": round(bm.difficulty.cs, 2),
        "ar": round(bm.difficulty.ar, 2),
        "od": round(bm.difficulty.od, 2),
        "hp": round(bm.difficulty.hp, 2),
        "bpm_range": [round(bpm_min, 1), round(bpm_max, 1)],
        "objects": {
            "total": len(bm.objects),
            "circles": n_circles,
            "sliders": n_sliders,
            "spinners": n_spinners,
        },
        "max_combo": bm.max_combo,
        "star_rating": round(stars, 2),
        "ss_pp": round(max_pp, 2),
    }
    print(json.dumps(info, indent=2))
    return 0


def _result_row(res, bm: Beatmap) -> dict:
    """Judge result as osu!lazer shows it for one of our replays (see osuai/judge.py)."""
    row = {
        "title": f"{bm.artist} - {bm.title} [{bm.version}]",
        "map_md5": bm.md5,
        "mods": list(bm.difficulty.mods),
        "star_rating": round(res.star_rating, 2),
        "accuracy": res.accuracy,
        "score": res.score,
        "max_combo": f"{res.max_combo}/{res.beatmap_max_combo}",
        "n300": res.n300,
        "n100": res.n100,
        "n50": res.n50,
        "misses": res.n_miss,
        "slider_ticks": f"{res.slider_ticks_hit}/{res.slider_ticks_total}",
        "slider_ends": f"{res.slider_ends_hit}/{res.slider_ends_total}",
        "pp": round(res.pp, 2),
        "perfect": res.perfect,
        "passed": res.passed,
    }
    if res.spinner_spins_total:
        row["spinner_spins"] = f"{res.spinner_spins_hit}/{res.spinner_spins_total}"
        row["spinner_bonus"] = f"{res.spinner_bonus_hit}/{res.spinner_bonus_total}"
    return row


def cmd_judge(args: argparse.Namespace) -> int:
    rep = decode_osr(args.replay)
    mods = args.mods if args.mods else rep.mods
    bm = parse_osu(args.map, mods=mods)
    res = judge_replay(bm, rep.frames)
    print(json.dumps({"player": rep.player_name, **_result_row(res, bm)}, indent=2))
    return 0


def _dqn_play(agent, bm: Beatmap) -> list:
    from osuai.dqn import play_map

    frames, _ = play_map(agent.net, agent.env_cfg, bm, agent.dqn_cfg.frame_stack)
    return frames


def cmd_bench(args: argparse.Namespace) -> int:
    """Print accuracy and combo across map manifests directly to terminal (no files written)."""
    from osuai.dqn import latest_round_dir, load_checkpoint, load_manifest_entries

    maps_list = [] if args.val_only else load_manifest_entries(args.manifest or [DEFAULT_MANIFEST])

    ckpt = args.checkpoint or latest_round_dir()
    agent = load_checkpoint(ckpt)
    if args.val_only:
        # The validation split lives in the checkpoint; find those maps anywhere under maps/
        maps_list = [{"path": str(_find_map(md5, args.maps_dir)), "md5": md5} for md5 in agent.val_maps]
    if args.limit > 0:
        maps_list = maps_list[: args.limit]

    policy_label = f"{Path(str(ckpt)).name} ({agent.step:,} steps)"
    accuracies: list[float] = []
    perfect_count = 0
    pps: list[float] = []
    errors: list[float] = []
    counts = [0, 0, 0, 0]  # 300 / 100 / 50 / miss over all maps

    print(f"=== Benchmark: {policy_label} | Maps: {len(maps_list)} ===")
    for entry in maps_list:
        raw_p = Path(entry["path"])
        mods = args.mods or entry.get("mods", ["NM"])
        bm = parse_osu(raw_p, mods=mods)

        frames = quantize_frames(_dqn_play(agent, bm))
        res = judge_replay(bm, frames)
        accuracies.append(res.accuracy)
        pps.append(res.pp)
        if res.perfect:
            perfect_count += 1
        errs = hit_errors(bm, res)
        errors += errs
        for k, n in enumerate((res.n300, res.n100, res.n50, res.n_miss)):
            counts[k] += n
        title = entry.get("title", f"{bm.artist} - {bm.title} [{bm.version}]")
        print(
            f"  [{res.star_rating:5.2f}*] {res.accuracy:6.2f}% | {res.score:9,d} | "
            f"{res.max_combo:4d}/{res.beatmap_max_combo:<4d} | "
            f"100={res.n100:<3d} 50={res.n50:<3d} miss={res.n_miss:<4d} | UR {unstable_rate(errs):5.1f} | "
            f"{res.pp:6.1f}pp | {title}",
            flush=True,
        )

    mean_acc = sum(accuracies) / max(len(accuracies), 1)
    fc_rate = perfect_count / max(len(accuracies), 1) * 100.0
    mean_pp = sum(pps) / max(len(pps), 1)
    print(f"=== Summary: Mean Accuracy: {mean_acc:.2f}% | FC Rate: {fc_rate:.1f}% | Mean PP: {mean_pp:.2f} ===")
    if errors:
        n = max(sum(counts), 1)
        mean_err = sum(errors) / len(errors)
        early = sum(e < 0 for e in errors) / len(errors) * 100.0
        print(
            f"=== Timing: 300 {counts[0] / n:.1%} | 100 {counts[1] / n:.1%} | 50 {counts[2] / n:.1%} | miss {counts[3] / n:.1%} | "
            f"hit error {mean_err:+.1f} ms ({early:.0f}% early) | UR {unstable_rate(errors):.1f} ==="
        )
    return 0


def _parse_grid(text: str) -> tuple[int, int]:
    """'32x24' -> (32, 24); a single number gives the columns, with rows following the 4:3 playfield."""
    parts = text.lower().replace("×", "x").split("x")
    try:
        if len(parts) == 1:
            cols = int(parts[0])
            rows = max(1, round(cols * 3 / 4))
        elif len(parts) == 2:
            cols, rows = int(parts[0]), int(parts[1])
        else:
            raise ValueError
    except ValueError:
        raise ValueError(f"--grid must look like 32x24 (columns x rows) or 32, got {text!r}") from None
    if cols < 1 or rows < 1:
        raise ValueError(f"--grid needs at least one column and one row, got {text!r}")
    return (cols, rows)


def _check_grid(grid_w: int, grid_h: int, width: int, hud_rows: int) -> None:
    """A grid finer than the network's feature map adds nothing, so refuse it and say which --width would allow it."""
    from osuai.dqn import QNetwork
    from osuai.render import RenderConfig

    rc = RenderConfig(width=width, hud_rows=hud_rows)
    max_h, max_w = QNetwork.max_grid(rc.shape, rc.field_height)
    if grid_w > max_w or grid_h > max_h:
        need = 4 * max(grid_w, -(-grid_h * 4 // 3))
        raise ValueError(
            f"--grid {grid_w}x{grid_h} is finer than the network can resolve at --width {width} "
            f"(at most {max_w}x{max_h}, about width/4); use --width {need} or more, or a coarser grid"
        )


def _stars_label(min_stars: float, max_stars: float) -> str:
    if max_stars > 0.0:
        return f"{min_stars:g}-{max_stars:g}star"
    if min_stars > 0.0:
        return f"{min_stars:g}+star"
    return "allstars"


def _prepare_round_dir(out_arg: str | None, round_no: int, min_stars: float, max_stars: float, force: bool) -> Path:
    """Pick the round folder (default checkpoints/round<N>_<stars>) and refuse to overwrite a trained round."""
    from osuai.dqn import BEST_CKPT, LAST_CKPT, LOG_FILE, ROUND_FILE

    run_dir = Path(out_arg) if out_arg else Path("checkpoints") / f"round{round_no}_{_stars_label(min_stars, max_stars)}"
    if run_dir.suffix == ".pt":
        raise ValueError(f"--out is a folder for this round's files, not a .pt file (got {run_dir})")
    existing = [n for n in (BEST_CKPT, LAST_CKPT, LOG_FILE, ROUND_FILE) if (run_dir / n).exists()]
    if existing:
        if not force:
            raise FileExistsError(
                f"{run_dir} already holds a training round ({', '.join(existing)}). "
                "Pick another --out, or pass --force to overwrite it."
            )
        for n in (LOG_FILE, ROUND_FILE):
            (run_dir / n).unlink(missing_ok=True)  # do not mix the replaced round's log into the new one
    return run_dir


def cmd_train(args: argparse.Namespace) -> int:
    from osuai.dqn import (
        LAST_CKPT,
        DQNConfig,
        collect_map_roles,
        load_checkpoint,
        load_manifest_entries,
        pick_eval_maps,
        resolve_checkpoint,
        split_maps,
        train_dqn,
    )
    from osuai.env import EnvConfig, MapData

    # a round folder resumes from its latest weights (last.pt); pass <folder>/best.pt to start from the best ones
    resume_path = resolve_checkpoint(args.resume, prefer=LAST_CKPT) if args.resume else None
    resume = load_checkpoint(resume_path) if resume_path else None
    if resume is not None:
        # The frame size, frame rate and action set define the network's input/output: they must stay the same
        rc = resume.env_cfg
        for name, val, ck_val in (
            ("--width", args.width, rc.width),
            ("--hz", args.hz, rc.hz),
            ("--action-repeat", args.action_repeat, rc.action_repeat),
        ):
            if val is not None and float(val) != float(ck_val):
                raise ValueError(f"{name} {val} differs from the checkpoint ({ck_val}); it cannot change when resuming")
        if args.grid is not None and _parse_grid(args.grid) != (rc.grid_w, rc.grid_h):
            raise ValueError(f"--grid {args.grid} differs from the checkpoint ({rc.grid_w}x{rc.grid_h}); it cannot change when resuming")
        if args.frame_stack is not None and args.frame_stack != resume.dqn_cfg.frame_stack:
            raise ValueError(
                f"--frame-stack {args.frame_stack} differs from the checkpoint ({resume.dqn_cfg.frame_stack}); it cannot change when resuming"
            )
        if args.late_press is not None and args.late_press != rc.late_press:
            if args.late_press:
                raise ValueError("--late-press cannot be added when resuming a checkpoint trained without it (it adds actions)")
            raise ValueError("--no-late-press cannot be used when resuming a checkpoint trained with late presses (it removes actions)")
        if args.dueling and not resume.dqn_cfg.dueling:
            raise ValueError("--dueling cannot be used when resuming a checkpoint trained without the dueling head")
        for flag, val, ck_val in (("--channels", args.channels, resume.dqn_cfg.channels), ("--depth", args.depth, resume.dqn_cfg.depth)):
            if val is not None and val != ck_val:
                raise ValueError(f"{flag} {val} differs from the checkpoint ({ck_val}); the network's size cannot change when resuming")
        hz, width, action_repeat = rc.hz, rc.width, rc.action_repeat
        hud_rows, grid_w, grid_h, late_press = rc.hud_rows, rc.grid_w, rc.grid_h, rc.late_press
        frame_stack, dueling = resume.dqn_cfg.frame_stack, resume.dqn_cfg.dueling
        channels, depth = resume.dqn_cfg.channels, resume.dqn_cfg.depth
        print(
            f"[train] resuming {resume_path}: round {len(resume.history) + 1}, "
            f"{resume.step:,} steps so far, previous maps: {len(resume.train_maps)}"
        )
    else:
        defaults = EnvConfig()
        hz = args.hz if args.hz is not None else defaults.hz
        width = args.width if args.width is not None else defaults.width
        action_repeat = args.action_repeat if args.action_repeat is not None else defaults.action_repeat
        hud_rows = defaults.hud_rows
        grid_w, grid_h = _parse_grid(args.grid) if args.grid is not None else (defaults.grid_w, defaults.grid_h)
        frame_stack = args.frame_stack if args.frame_stack is not None else DQNConfig().frame_stack
        if frame_stack < 1:
            raise ValueError(f"--frame-stack must be at least 1, got {frame_stack}")
        dueling = args.dueling
        channels = args.channels if args.channels is not None else DQNConfig().channels
        depth = args.depth if args.depth is not None else DQNConfig().depth
        if channels < 2:
            raise ValueError(f"--channels must be at least 2, got {channels}")
        if depth < 0:
            raise ValueError(f"--depth must be 0 or more, got {depth}")
        _check_grid(grid_w, grid_h, width, hud_rows)
        # on by default; a late key needs a middle tick, so 1 tick per decision turns it off unless asked for explicitly
        if args.late_press and action_repeat < 2:
            raise ValueError("--late-press needs --action-repeat 2 or more (a late key goes down at the decision's middle tick)")
        late_press = args.late_press if args.late_press is not None else action_repeat >= 2
        if args.late_press is None and not late_press:
            print("[train] late presses off: --action-repeat 1 has no middle tick to press on")
    eps_start = args.eps_start if args.eps_start is not None else (0.3 if resume is not None else 1.0)
    round_no = len(resume.history) + 1 if resume is not None else 1
    run_dir = _prepare_round_dir(args.out, round_no, args.min_stars, args.max_stars, args.force)
    print(f"[train] round {round_no} -> {run_dir}")

    manifests = args.maps or [DEFAULT_MANIFEST]
    entries = load_manifest_entries(manifests, min_stars=args.min_stars, max_stars=args.max_stars)
    if args.limit_maps > 0:
        entries = entries[: args.limit_maps]
    if not entries:
        raise ValueError(f"No beatmaps in {', '.join(map(str, manifests))} match the star filter")

    for flag, value in (
        ("--press-penalty", args.press_penalty),
        ("--hold-cost", args.hold_cost),
        ("--empty-press-cost", args.empty_press_cost),
        ("--slider-break-penalty", args.slider_break_penalty),
        ("--slider-offset-cost", args.slider_offset_cost),
    ):
        if value < 0:
            raise ValueError(f"{flag} is an amount taken off the reward: pass a positive number (or 0 to turn it off), got {value}")
    # --reward-* are signed: hits earn (0 or more), misses cost (0 or less), and a better grade never earns less
    for flag, value, sign in (
        ("--reward-300", args.reward_300, 1), ("--reward-100", args.reward_100, 1), ("--reward-50", args.reward_50, 1),
        ("--reward-tick", args.reward_tick, 1), ("--reward-tail", args.reward_tail, 1),
        ("--reward-miss", args.reward_miss, -1), ("--reward-tick-miss", args.reward_tick_miss, -1),
    ):
        if value * sign < 0:
            kind = "a hit: pass 0 or more" if sign > 0 else "a miss: pass 0 or less (e.g. -0.1)"
            raise ValueError(f"{flag} is the reward for {kind}, got {value}")
    if args.spin_max_deg < 0:
        raise ValueError(f"--spin-max-deg must be 0 (no cap) or more, got {args.spin_max_deg}")
    if not args.reward_50 <= args.reward_100 <= args.reward_300:
        raise ValueError(
            f"Hit rewards must keep --reward-50 <= --reward-100 <= --reward-300, got "
            f"{args.reward_50} / {args.reward_100} / {args.reward_300}"
        )
    env_cfg = EnvConfig(
        hz=hz,
        action_repeat=action_repeat,
        width=width,
        hud_rows=hud_rows,
        grid_w=grid_w,
        grid_h=grid_h,
        late_press=late_press,
        max_speed=args.max_speed,
        episode_seconds=args.episode_seconds,
        aim_shaping=args.aim_shaping,
        press_penalty=args.press_penalty,
        pay_hits_on_time=args.pay_hits_on_time,
        hold_cost=args.hold_cost,
        empty_press_cost=args.empty_press_cost,
        r_spin=args.spin_reward,
        r_spin_bonus=args.spin_bonus,
        spin_max_deg=args.spin_max_deg,
        slider_break_penalty=args.slider_break_penalty,
        slider_offset_cost=args.slider_offset_cost,
        r300=args.reward_300,
        r100=args.reward_100,
        r50=args.reward_50,
        r_miss=args.reward_miss,
        r_tick=args.reward_tick,
        r_tick_miss=args.reward_tick_miss,
        r_tail=args.reward_tail,
    )
    cfg = DQNConfig(
        total_steps=args.steps,
        n_envs=args.envs,
        buffer_size=args.buffer,
        batch_size=args.batch_size,
        lr=args.lr,
        gamma=args.gamma,
        n_step=args.n_step,
        double=not args.no_double,
        dueling=dueling,
        frame_stack=frame_stack,
        channels=channels,
        depth=depth,
        learning_starts=args.learning_starts,
        target_update=args.target_update,
        eps_decay_steps=args.eps_decay,
        eps_start=eps_start,
        key_explore_prob=args.key_explore,
        eps_end=args.eps_end,
        eval_interval=args.eval_interval,
        eval_maps=args.eval_maps,
        log_interval=args.log_interval,
        seed=args.seed,
    )

    if args.eval_maps < 1:
        raise ValueError(f"--eval-maps must be at least 1, got {args.eval_maps}")
    beatmaps: dict[str, MapData] = {}
    stars: dict[str, float] = {}
    for e in entries:
        try:
            bm = parse_osu(e["path"], mods=e.get("mods", ["NM"]))
            beatmaps[bm.md5] = MapData(bm, env_cfg)
            stars[bm.md5] = float(e.get("star_rating", 0.0))
        except Exception as exc:  # skip unparsable maps but say so
            print(f"[train] skipping {e['path']}: {exc}")
    # A map keeps its role across rounds: anything ever trained on stays training, earlier validation maps stay
    # validation, and new maps get a permanent role from their MD5
    if resume is not None:
        seen_train, seen_val = collect_map_roles(resume)
        resume.all_train_maps, resume.all_val_maps = sorted(seen_train), sorted(seen_val)
    train_ids, val_ids = split_maps(
        list(beatmaps),
        val_ratio=args.val_ratio,
        seed=42,
        forced_train=resume.all_train_maps if resume is not None else (),
        forced_val=resume.all_val_maps if resume is not None else (),
    )
    if not train_ids:
        raise ValueError("No training maps left after the split")
    print(f"[train] {len(train_ids)} training maps, {len(val_ids)} validation maps")
    # this round evaluates on maps spread over its star range; the previous round's eval maps are played too
    eval_ids = pick_eval_maps(val_ids, stars, cfg.eval_maps)
    prev_eval: list[MapData] = []
    for md5 in resume.eval_maps if resume is not None else []:
        if md5 in beatmaps:
            prev_eval.append(beatmaps[md5])
            continue
        try:
            prev_eval.append(MapData(parse_osu(_find_map(md5)), env_cfg))
        except (FileNotFoundError, ValueError) as exc:
            print(f"[train] previous round's eval map {md5} is unavailable, comparing without it: {exc}")
    result = train_dqn(
        train_maps=[beatmaps[k] for k in train_ids],
        val_maps=[beatmaps[k] for k in val_ids],
        out_dir=run_dir,
        env_cfg=env_cfg,
        cfg=cfg,
        train_names=train_ids,
        val_names=val_ids,
        resume=resume,
        eval_maps=[beatmaps[k] for k in eval_ids],
        prev_eval_maps=prev_eval,
        round_info={
            "manifest": [str(m).replace("\\", "/") for m in manifests],
            "min_stars": args.min_stars,
            "max_stars": args.max_stars,
            "resumed_from": str(resume_path).replace("\\", "/") if resume_path else None,
        },
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_rollout(args: argparse.Namespace) -> int:
    from osuai.dqn import latest_round_dir, load_checkpoint

    ckpt = args.checkpoint or latest_round_dir()
    agent = load_checkpoint(ckpt)
    print(f"[rollout] using {ckpt} (round {len(agent.history) or 1}, {agent.step:,} steps)", flush=True)
    if args.val_only:
        if not agent.val_maps:
            raise ValueError(f"Checkpoint {ckpt} has no validation maps")
        map_paths = [_find_map(md5, args.maps_dir) for md5 in agent.val_maps]
    elif args.map:
        map_paths = [Path(args.map)]
    else:
        raise ValueError("Provide a beatmap path or pass --val-only")

    results = []
    for mp in map_paths:
        bm = parse_osu(mp, mods=args.mods)
        frames = quantize_frames(_dqn_play(agent, bm))  # judge exactly what the .osr will contain
        res = judge_replay(bm, frames)
        out_path = _resolve_osr_out_path(mp, "dqn", args.out)
        write_osr(out_path, bm, frames, res, player_name="osuai_dqn")
        results.append({"out": str(out_path).replace("\\", "/"), **_result_row(res, bm)})

    print(json.dumps(results[0] if len(results) == 1 else results, indent=2))
    return 0


def _parse_seconds(value: str) -> float:
    """--seconds accepts a positive number or 'full' (0 also means full)."""
    if value.strip().lower() in ("full", "0", "0.0"):
        return 0.0
    try:
        sec = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--seconds must be a number or 'full', got {value!r}") from exc
    if sec < 0:
        raise argparse.ArgumentTypeError("--seconds must be positive (or 'full')")
    return sec


def cmd_render(args: argparse.Namespace) -> int:
    """Preview the simulated grayscale frames, as a PNG contact sheet or an animated GIF.

    With --checkpoint the trained agent plays and its cursor and keys are shown; without it only the map is shown
    (objects appearing, approaching and disappearing, no cursor). With --gif the animation runs in real map time
    (breaks included) for --seconds, capped at the end of the map ('full' = whole song).
    """
    import numpy as np

    from osuai.env import EnvConfig, MapData, OsuEnv
    from osuai.render import GifWriter, contact_sheet, write_png

    bm = parse_osu(args.map, mods=args.mods)
    agent = None
    if args.checkpoint:
        from osuai.dqn import greedy_actions, load_checkpoint

        agent = load_checkpoint(args.checkpoint)
        env_cfg = agent.env_cfg
    else:
        env_cfg = EnvConfig(width=args.width, hz=args.hz)
    # Previews follow the song in real time, breaks included (only training fast-forwards through breaks)
    env_cfg = EnvConfig.from_dict({**env_cfg.to_dict(), "skip_breaks": False})
    md = MapData(bm, env_cfg)

    # a small clock that advances either the agent's game or just the map
    clock = {"t": float(md.appear[0]) - 500.0, "steps": 0}
    if agent is not None:
        env = OsuEnv(md, env_cfg)
        frame = env.reset(mode="full")
        stack = np.repeat(frame[None], agent.dqn_cfg.frame_stack, axis=0)
        device = next(agent.net.parameters()).device
        clock["t"] = env.t
    else:
        frame = md.renderer.render_map(clock["t"])
    map_end = float(md.last_end) + 500.0

    def advance() -> tuple[np.ndarray, bool]:
        nonlocal stack
        clock["steps"] += 1
        if agent is not None:
            a = int(greedy_actions(agent.net, stack[None], device)[0])
            f, _, term, _ = env.step(a)
            stack = np.concatenate([stack[1:], f[None]], axis=0)
            clock["t"] = env.t
            return (f, term)
        clock["t"] += env_cfg.dt_ms
        return (md.renderer.render_map(clock["t"]), clock["t"] >= map_end)

    driver = "dqn" if agent else "map only"
    map_folder = Path("out") / Path(args.map).stem

    if not args.gif:
        start_ms = args.start_ms if args.start_ms is not None else float(bm.objects[0].time) - 300.0
        shots: list = []
        while len(shots) < args.count:
            f, terminal = advance()
            if clock["t"] >= start_ms and clock["steps"] % max(args.every, 1) == 0:
                shots.append(f)
            if terminal:
                break
        if not shots:
            raise ValueError("No frames captured: --start-ms is past the end of the map")
        out = Path(args.out) if args.out else map_folder / "frames.png"
        upscale = args.upscale or 4
        write_png(out, contact_sheet(shots, cols=args.cols, upscale=upscale))
        summary = {
            "out": str(out).replace("\\", "/"),
            "frames": len(shots),
            "frame_shape": list(env_cfg.obs_shape),
            "driver": driver,
        }
        print(json.dumps(summary, indent=2))
        return 0

    # ---- animated GIF
    fps = float(args.fps)
    if not 0.0 < fps <= 50.0:
        raise ValueError("--fps must be between 0 and 50 (GIF viewers slow down faster frame delays)")
    frame_ms = 1000.0 / fps
    song_end = map_end  # last object plus a short tail
    start_ms = args.start_ms if args.start_ms is not None else clock["t"]
    if start_ms >= song_end:
        raise ValueError(f"--start-ms {start_ms:.0f} is past the end of the map ({song_end:.0f} ms)")
    full = args.seconds == 0.0
    want_end = song_end if full else start_ms + args.seconds * 1000.0
    end_ms = min(want_end, song_end)
    out = Path(args.out) if args.out else map_folder / "frames.gif"
    if out.suffix.lower() != ".gif":
        out = out.with_suffix(".gif")
    upscale = args.upscale or 3

    n = 0
    next_cap = start_ms
    with GifWriter(out, fps=fps, upscale=upscale) as gif:
        terminal = False
        last = frame
        while True:
            # Emit one GIF frame per 1/fps of map time (repeating the latest frame between agent decisions)
            while clock["t"] + 1e-6 >= next_cap and next_cap <= end_ms + 1e-6:
                gif.add(last)
                n += 1
                next_cap += frame_ms
                if n % 1000 == 0:
                    print(f"[render] {n} frames ({(clock['t'] - start_ms) / 1000.0:.1f}s)", flush=True)
            if clock["t"] >= end_ms - 1e-6:
                break
            if terminal:
                # Every object is judged but the requested window is not over: hold the final frame
                clock["t"] += env_cfg.dt_ms
            else:
                last, terminal = advance()
    summary = {
        "out": str(out).replace("\\", "/"),
        "frames": n,
        "fps": round(fps, 2),
        "decision_hz": round(env_cfg.decision_hz, 2) if agent else env_cfg.hz,
        "seconds": round(n / fps, 2),
        "song_seconds_from_start": round((song_end - start_ms) / 1000.0, 2),
        "capped_at_song_end": (not full) and want_end > song_end,
        "size_mb": round(out.stat().st_size / 1e6, 2),
        "frame_shape": [env_cfg.obs_shape[0] * upscale, env_cfg.obs_shape[1] * upscale],
        "driver": driver,
    }
    print(json.dumps(summary, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python osuai/cli.py", description="osu! DQN agent that learns to play from simulated grayscale frames")
    sub = p.add_subparsers(dest="command", required=True)

    p_imp = sub.add_parser("import-lazer", help="Extract real .osu beatmaps from osu!lazer store into maps/imported/")
    p_imp.add_argument("--lazer-dir", default=None, help="Path to osu!lazer data folder (default: %%APPDATA%%\\osu)")
    p_imp.add_argument("--out-maps", default=str(IMPORTED_DIR), help="Destination folder for extracted .osu maps (default: maps/imported)")
    p_imp.add_argument("--limit", type=int, default=40, help="Max number of beatmaps to import (sorted hardest first; 0 = all)")
    p_imp.add_argument("--min-stars", type=float, default=0.0, help="Minimum star rating filter for imported maps")
    p_imp.add_argument("--max-stars", type=float, default=0.0, help="Maximum star rating filter (0 = no cap)")
    p_imp.set_defaults(func=cmd_import_lazer)

    p_gen = sub.add_parser("generate", help="Generate synthetic maps at a target star rating into maps/<name>/")
    p_gen.add_argument("--stars", required=True, help="Target star rating (e.g. 5) or a range spread over the maps (e.g. 3-6)")
    p_gen.add_argument("--count", type=int, default=10, help="Number of maps to generate")
    p_gen.add_argument("--name", default=None, help="Folder inside maps/ for these maps and their manifest.json (default: the --only type, else synthetic)")
    p_gen.add_argument("--only", choices=["circles", "sliders", "spinners"], default=None, help="Use a single object type")
    p_gen.add_argument("--ratio", default=None, help="Share of circles/sliders/spinners, e.g. 60/30/10 (default 60/35/5)")
    p_gen.add_argument("--length", type=float, default=90.0, help="Map length in seconds")
    p_gen.add_argument("--tolerance", type=float, default=0.05, help="Allowed distance from the target star rating")
    p_gen.add_argument("--seed", type=int, default=None, help="Random seed (default: random); map i uses seed + i")
    p_gen.set_defaults(func=cmd_generate)

    p_ins = sub.add_parser("inspect", help="Inspect a .osu beatmap")
    p_ins.add_argument("map", help="Path to .osu beatmap file")
    p_ins.add_argument("--mods", default="NM", help="Comma-separated mods (e.g. NM, HR, DT, HR,DT)")
    p_ins.set_defaults(func=cmd_inspect)

    p_jdg = sub.add_parser("judge", help="Judge a .osr replay against a .osu beatmap (as osu!lazer scores it)")
    p_jdg.add_argument("map", help="Path to .osu beatmap file")
    p_jdg.add_argument("replay", help="Path to .osr replay file")
    p_jdg.add_argument("--mods", default=None, help="Override mods (default: read from .osr header)")
    p_jdg.set_defaults(func=cmd_judge)

    p_bch = sub.add_parser("bench", help="Print accuracy across a map manifest to terminal")
    p_bch.add_argument("manifest", nargs="*", default=None, help="Manifests or map folders (default: maps/imported/manifest.json)")
    p_bch.add_argument("--mods", default=None, help="Override mods for all maps (e.g. NM, HR,DT)")
    p_bch.add_argument("--checkpoint", default=None, help="Round folder (uses best.pt) or .pt file (default: newest round in checkpoints/)")
    p_bch.add_argument("--val-only", action="store_true", help="Benchmark only the checkpoint's held-out validation maps")
    p_bch.add_argument("--maps-dir", default=str(MAPS_ROOT), help="Where --val-only looks for <md5>.osu files (searched recursively)")
    p_bch.add_argument("--limit", type=int, default=0, help="Only benchmark the first N maps (0 = all)")
    p_bch.set_defaults(func=cmd_bench)

    p_tr = sub.add_parser("train", help="Train the DQN agent on simulated grayscale frames")
    p_tr.add_argument("--maps", nargs="+", default=None, help="Manifests or map folders to train on, e.g. maps/imported maps/synthetic (default: maps/imported/manifest.json)")
    p_tr.add_argument("--out", default=None, help="Folder for this round's best.pt / last.pt / log.jsonl / round.json (default: checkpoints/round<N>_<stars>)")
    p_tr.add_argument("--force", action="store_true", help="Overwrite the --out folder if it already holds a training round")
    p_tr.add_argument("--min-stars", type=float, default=0.0, help="Only train on maps at or above this star rating")
    p_tr.add_argument("--max-stars", type=float, default=0.0, help="Only train on maps at or below this star rating (0 = no cap)")
    p_tr.add_argument("--limit-maps", type=int, default=0, help="Use only the first N matching maps (0 = all)")
    p_tr.add_argument("--val-ratio", type=float, default=0.1, help="Fraction of maps held out for validation")
    p_tr.add_argument("--steps", type=int, default=2_000_000, help="Total environment transitions")
    p_tr.add_argument("--envs", type=int, default=8, help="Parallel simulated games")
    p_tr.add_argument("--resume", default=None, help="Continue from a previous round: its folder (uses last.pt) or a .pt file such as <folder>/best.pt")
    p_tr.add_argument("--hz", type=float, default=None, help="Simulation ticks per second of map time (default 60; fixed when resuming)")
    p_tr.add_argument("--action-repeat", type=int, default=None, help="Ticks per agent decision: 2 = 30 decisions/s (default 2; fixed when resuming)")
    p_tr.add_argument("--width", type=int, default=None, help="Rendered playfield width in pixels, height follows 4:3 (default 96; fixed when resuming)")
    p_tr.add_argument(
        "--channels", type=int, default=None,
        help="Network width (default 64, about 145k parameters; 128 is about 4x). Round 1 only, kept when resuming",
    )
    p_tr.add_argument(
        "--depth", type=int, default=None,
        help="Layers after the two downsampling ones, each reaching twice as far (default 3: a cell sees about 64 frame "
        "pixels around it; 4: about 128, the whole playfield width). Round 1 only, kept when resuming",
    )
    p_tr.add_argument("--frame-stack", type=int, default=None, help="Frames of history the network sees: 4 at 30 decisions/s = the last 133 ms (default 4; round 1 only, kept when resuming)")
    p_tr.add_argument(
        "--late-press",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="K1/K2 'late' keys that press at the decision's middle tick, for presses on a 1-tick grid while deciding at 30 Hz "
        "(on by default; --no-late-press keeps 3 key states; round 1 only, kept when resuming)",
    )
    p_tr.add_argument("--grid", default=None, help="Aim grid as COLSxROWS, e.g. 16x12 (coarser) or 32x24 (finer, needs --width 128); one number = columns, rows follow 4:3 (default 24x18; fixed when resuming)")
    p_tr.add_argument("--episode-seconds", type=float, default=30.0, help="Length of each random training window")
    p_tr.add_argument("--aim-shaping", type=float, default=0.5, help="Potential-based aim shaping weight (0 = pure game reward)")
    p_tr.add_argument(
        "--press-penalty", type=float, default=0.03,
        help="Taken off the reward per key press, including switching K1 <-> K2 while holding (a positive number)",
    )
    p_tr.add_argument(
        "--empty-press-cost", type=float, default=0.1,
        help="Taken off the reward, on top of --press-penalty, for a press that hits nothing: no circle or slider head, "
        "no slider in progress, no spinner running or about to start (a positive number; 0 = off)",
    )
    p_tr.add_argument(
        "--hold-cost", type=float, default=0.2,
        help="Taken off the reward per second a key is held while nothing needs one: no slider in progress, no spinner "
        "running (a positive number; 0 = off)",
    )
    # --reward-* are signed rewards (hits 0 or more, misses 0 or less); --*-penalty / --*-cost are positive amounts
    p_tr.add_argument(
        "--pay-hits-on-time",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pay an early hit when the object's time comes instead of on the press, so pressing early gains nothing "
        "from the discount (on by default; --no-pay-hits-on-time pays on the press)",
    )
    p_tr.add_argument("--reward-300", type=float, default=1.0, help="Reward for a 300 (circle, slider head or spinner)")
    p_tr.add_argument(
        "--reward-100", type=float, default=1.0 / 3.0,
        help="Reward for a 100 (default 1/3, lazer's accuracy weight). Lower makes timing matter more",
    )
    p_tr.add_argument("--reward-50", type=float, default=1.0 / 6.0, help="Reward for a 50 (default 1/6, lazer's accuracy weight)")
    p_tr.add_argument(
        "--reward-miss", type=float, default=-0.1,
        help="Reward for a missed circle, slider head or spinner (0 or less; kept small so hits stand out early in training)",
    )
    p_tr.add_argument("--reward-tick", type=float, default=0.1, help="Reward for a slider tick or repeat")
    p_tr.add_argument("--reward-tick-miss", type=float, default=-0.02, help="Reward for a missed slider tick or repeat (0 or less)")
    p_tr.add_argument("--reward-tail", type=float, default=0.5, help="Reward for a slider end")
    p_tr.add_argument(
        "--spin-reward", type=float, default=0.5,
        help="Reward per spinner rotation toward the spins its 300 needs, paid tick by tick as the net rotation grows",
    )
    p_tr.add_argument("--spin-bonus", type=float, default=0.1, help="Reward per spinner rotation beyond the spins its 300 needs")
    p_tr.add_argument(
        "--spin-max-deg", type=float, default=0.0,
        help="Forward rotation the spin reward credits per tick, in degrees (default 0 = no cap; a cap makes random "
        "spinning lose reward, so an untrained agent learns not to spin); spins and grades count it all",
    )
    p_tr.add_argument(
        "--slider-break-penalty", type=float, default=0.3,
        help="Taken off the reward when a tracked slider is lost before its end: key let go or cursor out of the follow circle "
        "(a positive number; 0 = off)",
    )
    p_tr.add_argument(
        "--slider-offset-cost", type=float, default=1.0,
        help="Taken off the reward per second of a slider in progress with the cursor at the follow circle's edge, linear in "
        "distance from the ball (a positive number; 0 = off)",
    )
    p_tr.add_argument("--buffer", type=int, default=300_000, help="Replay buffer size in transitions")
    p_tr.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    p_tr.add_argument("--lr", type=float, default=1e-4, help="Adam learning rate")
    p_tr.add_argument("--gamma", type=float, default=0.9, help="Discount factor per decision (0.9 at 30 decisions/s looks ~1/3 s ahead)")
    p_tr.add_argument("--n-step", type=int, default=3, help="n-step return length")
    p_tr.add_argument("--no-double", action="store_true", help="Use vanilla max-Q targets instead of Double DQN")
    p_tr.add_argument("--dueling", action="store_true", help="Add a dueling value/advantage split to the spatial Q-map")
    p_tr.add_argument("--max-speed", type=float, default=8000.0, help="Cursor speed cap in osu!px per second")
    p_tr.add_argument("--learning-starts", type=int, default=20_000, help="Transitions collected before learning starts")
    p_tr.add_argument("--target-update", type=int, default=2_000, help="Gradient updates between target-network syncs")
    p_tr.add_argument("--key-explore", type=float, default=0.2, help="Probability that a random action holds a key (K1 or K2)")
    p_tr.add_argument("--eps-start", type=float, default=None, help="Initial exploration epsilon (default 1.0, or 0.3 when resuming)")
    p_tr.add_argument("--eps-decay", type=int, default=250_000, help="Transitions over which epsilon decays eps-start -> eps-end")
    p_tr.add_argument("--eps-end", type=float, default=0.05, help="Final exploration epsilon")
    p_tr.add_argument("--eval-interval", type=int, default=100_000, help="Transitions between validation evaluations")
    p_tr.add_argument(
        "--eval-maps", type=int, default=6,
        help="Validation maps per evaluation, spread over the round's star range (the previous round's are played too)",
    )
    p_tr.add_argument("--log-interval", type=int, default=10_000, help="Transitions between progress log lines")
    p_tr.add_argument("--seed", type=int, default=0, help="Random seed")
    p_tr.set_defaults(func=cmd_train)

    p_ro = sub.add_parser("rollout", help="Let a trained DQN play a map (or all its validation maps) into out/<map>/dqn.osr")
    p_ro.add_argument("map", nargs="?", default=None, help="Path to .osu beatmap file (optional if --val-only is set)")
    p_ro.add_argument("--val-only", action="store_true", help="Play every validation map stored in the checkpoint")
    p_ro.add_argument("--maps-dir", default=str(MAPS_ROOT), help="Where --val-only looks for <md5>.osu files (searched recursively)")
    p_ro.add_argument("--checkpoint", default=None, help="DQN round folder (uses best.pt) or .pt file (default: newest round in checkpoints/)")
    p_ro.add_argument("--mods", default="NM", help="Comma-separated mods (e.g. NM, HR, DT)")
    p_ro.add_argument("--out", default=None, help="Base output directory (default: out/<map>/dqn.osr) or explicit .osr path")
    p_ro.set_defaults(func=cmd_rollout)

    p_rn = sub.add_parser("render", help="Preview the simulated frames the agent sees (PNG contact sheet or animated GIF)")
    p_rn.add_argument("map", help="Path to .osu beatmap file")
    p_rn.add_argument("--checkpoint", default=None, help="Let this trained agent (round folder or .pt file) play; without it only the map is shown")
    p_rn.add_argument("--mods", default="NM", help="Comma-separated mods")
    p_rn.add_argument("--start-ms", type=float, default=None, help="Map time of the first captured frame (default: first object - 300ms for PNG, map start for GIF)")
    p_rn.add_argument("--count", type=int, default=16, help="Number of frames to capture (PNG only)")
    p_rn.add_argument("--every", type=int, default=2, help="Capture every N-th simulation step (PNG only)")
    p_rn.add_argument("--cols", type=int, default=4, help="Columns in the contact sheet")
    p_rn.add_argument("--upscale", type=int, default=None, help="Nearest-neighbour upscale factor (default: 4 for PNG, 3 for GIF)")
    p_rn.add_argument("--width", type=int, default=96, help="Playfield width in pixels (ignored with --checkpoint)")
    p_rn.add_argument("--hz", type=float, default=60.0, help="Simulation rate (ignored with --checkpoint)")
    p_rn.add_argument("--gif", action="store_true", help="Write an animated GIF instead of a PNG contact sheet")
    p_rn.add_argument("--seconds", type=_parse_seconds, default=10.0, help="GIF length in seconds from --start-ms, capped at the end of the map; 'full' = whole song (default: 10)")
    p_rn.add_argument("--fps", type=float, default=30.0, help="GIF frame rate (max 50)")
    p_rn.add_argument("--out", default=None, help="Output file (default: out/<map>/frames.png, or frames.gif with --gif)")
    p_rn.set_defaults(func=cmd_render)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
