"""Step-wise osu!standard game simulation used as a reinforcement-learning environment.

The agent observes ONLY the rendered grayscale frame (see osuai/render.py) and acts with a discrete
*spatial* action: a cell of a grid laid over the playfield (where to aim) combined with a key state: no key, K1
or K2. Pressing the other key than the one held is a fresh tap, so alternating K1/K2 (like a human streaming)
allows a new tap every decision.
The cursor glides toward the chosen cell's centre at a capped speed, so movement stays continuous. The game is
simulated at `hz` ticks per second; the agent decides every `action_repeat` ticks (like the frame skip of
DeepMind's Atari DQN). Judgement happens online, tick by tick, with osu!lazer's rules (the replay judge in
osuai/judge.py implements them in full detail; this is a fast version of the same rules):

* lazer's hit windows; circles and slider heads are graded 300 / 100 / 50 by timing;
* note lock: a press is blocked only while the previous circle / head is unjudged and its time has not come;
  hitting an object misses every earlier circle / head still waiting for a press;
* pressing earlier than the 50 window (but inside the 400 ms miss window) is a miss, so spam clicking is costly;
* sliders are tracked while a key is held and the cursor is on the ball (entering within its radius, staying
  within 2.4x it); ticks and repeats are decided at their time, the end within the slider's last 36 ms;
* spinners count the net (signed) rotation of the cursor around the centre while a key is held. A tick's movement
  counts only if a key was already held when it started: lazer moves a replay's cursor smoothly between frames under
  the key state of the frame it leaves, so pressing only on the way back along a line spins nothing.

The reward is mostly positive game feedback (hit grades, slider ticks and ends, spinner rotation) with small costs
for misses and key presses, plus optional potential-based aim shaping computed per decision
(gamma * phi(s') - phi(s), which does not change the optimal policy). Two slider terms do change it on purpose:
losing a tracked slider before its end costs slider_break_penalty at once (the combo break the game reward only shows
ticks later), and every tick of a slider in progress costs a little for the cursor's distance from the ball
(slider_offset_cost), keeping a margin for re-entering after a release. Spinner rotation is paid tick by tick as the net
rotation grows, so the spinner's grade, which only comes at its end, is not the only signal. None of this is ever
part of the observation.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import asdict, dataclass, fields

import numpy as np

from osuai.beatmap import KEY_K1, KEY_K2, KEY_M1, KEY_M2, PLAYFIELD_H, PLAYFIELD_W, Beatmap, Frame
from osuai.judge import TAIL_LENIENCY, HitWindows, difficulty_range
from osuai.render import KIND_CIRCLE, KIND_CODES, KIND_SLIDER, KIND_SPINNER, FrameRenderer, RenderConfig

PLAYFIELD_DIAG = math.hypot(PLAYFIELD_W, PLAYFIELD_H)


@dataclass(frozen=True, slots=True)
class EnvConfig:
    hz: float = 60.0  # simulation ticks per second of map time (judgement + replay frame rate)
    action_repeat: int = 2  # ticks per agent decision (decision rate = hz / action_repeat = 30 Hz)
    width: int = 96  # rendered playfield width in pixels (height follows 4:3, plus HUD rows)
    hud_rows: int = 4
    grid_w: int = 24  # aim grid columns over the playfield (cells are 512/24 = 21.3 osu!px wide)
    grid_h: int = 18  # aim grid rows (384/18 = 21.3 osu!px tall)
    late_press: bool = False  # add "K1 late" / "K2 late": the key goes down at the decision's middle tick, not its first
    max_speed: float = 8000.0  # cursor speed cap in osu!px per second (11* maps jump at up to ~6300 px/s)
    episode_seconds: float = 30.0  # training episodes are random windows of this length
    skip_breaks: bool = True  # fast-forward through empty stretches longer than one second
    miss_window: float = 400.0  # pressing a head earlier than w50 but inside this window is a miss
    gamma: float = 0.9  # per-decision discount used for the potential-based aim shaping term
    aim_shaping: float = 0.5  # weight of the shaping term (0 disables it)
    # r_* are signed rewards; *_penalty and *_cost are positive amounts taken off the reward
    # per key press (a key going down, so switching K1 <-> K2 while holding counts too): large enough that flipping
    # between keys that look equally good is not free, small next to a hit
    press_penalty: float = 0.03
    # per second a key is held while nothing needs one (no slider in progress, no spinner running): holding is otherwise
    # free, and an agent that always holds a key "presses" by switching keys, which hits upcoming circles too early
    hold_cost: float = 0.2
    # extra cost of a press that hits nothing: no circle or slider head judged, no slider in progress, no spinner
    # running or about to start. Pressing again right after a hit is otherwise nearly free, and when the cursor is
    # already on the next object (or it is stacked on the same spot) that extra press hits it a beat early: a miss
    empty_press_cost: float = 0.1
    # 100 and 50 are worth 1/3 and 1/6 of a 300, the weights of lazer's accuracy, so timing matters as much as it
    # does to the score: a press one tick late that drops a 300 to a 100 loses two thirds of the hit
    # an early hit (before the object's time) is paid on the tick the object's time comes, not on the press: with
    # gamma = 0.9 a reward one decision sooner is worth 10% more, so paying on the press makes pressing early pay in
    # itself and pushes presses to the early edge of the 300 window, where some slip out and become 100s
    pay_hits_on_time: bool = True
    r300: float = 1.0
    r100: float = 1.0 / 3.0
    r50: float = 1.0 / 6.0
    r_miss: float = -0.1  # circles and slider heads alike
    r_tick: float = 0.1  # slider tick or repeat (30 points in lazer)
    r_tick_miss: float = -0.02
    r_tail: float = 0.5  # slider end (150 points in lazer); a missed end gives nothing but breaks no combo
    r_spin: float = 0.5  # per rotation toward the spins a spinner's 300 needs, paid tick by tick as net rotation grows
    r_spin_bonus: float = 0.1  # per rotation beyond those
    # forward rotation credited by the spin reward per tick (degrees; 0 = no cap, the default). Only the reward is capped:
    # spins, grades and the judge count the full rotation, and turning back always costs in full. Off by default
    # because it stops spinning from being learned: uncapped, random movement with a key held still earns a little
    # (the net rotation drifts away from zero), which draws an untrained agent into spinning; capped, it loses
    # reward on average, so the agent learns early on never to hold a key during spinners.
    spin_max_deg: float = 0.0
    slider_break_penalty: float = 0.3  # a tracked slider lost before its end (key let go, or cursor out of the follow circle)
    slider_offset_cost: float = 1.0  # per second of a slider in progress at the follow circle's edge, linear in distance

    @property
    def dt_ms(self) -> float:
        """Length of one simulation tick."""
        return 1000.0 / self.hz

    @property
    def decision_hz(self) -> float:
        return self.hz / max(1, self.action_repeat)

    @property
    def n_cells(self) -> int:
        return self.grid_w * self.grid_h

    @property
    def render(self) -> RenderConfig:
        return RenderConfig(width=self.width, hud_rows=self.hud_rows)

    @property
    def obs_shape(self) -> tuple[int, int]:
        return self.render.shape

    @property
    def n_keys(self) -> int:
        return len(KEY_STATES) + (2 if self.late_press else 0)

    @property
    def late_tick(self) -> int:
        """Tick of a decision at which a late key goes down (the middle tick)."""
        return max(1, self.action_repeat) // 2

    def key_plan(self, key: int) -> tuple[int, int]:
        """(key mask, first tick it is held) of a key-state index: 0 none, 1 K1, 2 K2, then 3 K1 late, 4 K2 late.

        A late key is up for the first half of the decision, so it presses half a decision later. It also
        re-taps a key that was already held.
        """
        if key < len(KEY_STATES):
            return (KEY_STATES[key], 0)
        return (KEY_STATES[key - len(KEY_STATES) + 1], self.late_tick)

    @property
    def n_actions(self) -> int:
        return self.n_keys * self.n_cells

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> EnvConfig:
        known = {f.name for f in fields(cls)}
        kw = {k: v for k, v in d.items() if k in known}
        return cls(**kw)


KEY_STATES = (0, KEY_K1, KEY_K2)  # key part of an action: 0 = no key, 1 = K1 held, 2 = K2 held
# with EnvConfig.late_press, key states 3 and 4 are K1 and K2 pressed at the decision's middle tick (EnvConfig.key_plan)


def build_action_table(cfg: EnvConfig) -> np.ndarray:
    """Rows of (target_x, target_y, key_state) with targets in osu!px and key_state a key-state index (EnvConfig.key_plan).

    Action index = cell * n_keys + key_state, with cell = row * grid_w + col (row-major, top-left first).
    """
    cw = PLAYFIELD_W / cfg.grid_w
    ch = PLAYFIELD_H / cfg.grid_h
    table = np.zeros((cfg.n_actions, 3), dtype=np.float32)
    for r in range(cfg.grid_h):
        for c in range(cfg.grid_w):
            cell = r * cfg.grid_w + c
            for key in range(cfg.n_keys):
                table[cell * cfg.n_keys + key] = ((c + 0.5) * cw, (r + 0.5) * ch, key)
    return table


@dataclass(slots=True)
class PlayStats:
    n300: int = 0
    n100: int = 0
    n50: int = 0
    n_miss: int = 0
    ticks_hit: int = 0  # slider ticks and repeats
    ticks_missed: int = 0
    ends_hit: int = 0  # slider ends
    ends_missed: int = 0
    presses: int = 0
    combo: int = 0
    max_combo: int = 0
    slider_breaks: int = 0  # tracked sliders lost before their end
    idle_hold_ticks: int = 0  # ticks with a key held while nothing needed one
    empty_presses: int = 0  # presses that hit nothing (see EnvConfig.empty_press_cost)
    reward: float = 0.0
    steps: int = 0

    @property
    def judged(self) -> int:
        return self.n300 + self.n100 + self.n50 + self.n_miss

    @property
    def accuracy(self) -> float:
        """Accuracy as osu!lazer computes it: ticks and repeats weigh 30, slider ends 150, other objects 300."""
        maximum = 300.0 * self.judged + 30.0 * (self.ticks_hit + self.ticks_missed) + 150.0 * (self.ends_hit + self.ends_missed)
        if maximum == 0:
            return 0.0
        got = 300.0 * self.n300 + 100.0 * self.n100 + 50.0 * self.n50 + 30.0 * self.ticks_hit + 150.0 * self.ends_hit
        return got / maximum * 100.0

    @property
    def hit_rate(self) -> float:
        n = self.judged
        return 0.0 if n == 0 else (n - self.n_miss) / n

    def as_dict(self) -> dict:
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d["accuracy"] = round(self.accuracy, 2)
        d["hit_rate"] = round(self.hit_rate, 4)
        return d


class MapData:
    """Per-beatmap arrays and slider event schedules, shared by every env that plays this map."""

    def __init__(self, beatmap: Beatmap, cfg: EnvConfig) -> None:
        self.bm = beatmap
        objs = beatmap.objects
        if not objs:
            raise ValueError("Beatmap has no hit objects")
        d = beatmap.difficulty
        self.n = len(objs)
        self.preempt = max(d.preempt, 150.0)
        self.radius = d.circle_radius
        self.follow_radius = d.follow_radius
        hw = HitWindows.for_od(d.od)
        self.w300, self.w100, self.w50 = hw.great, hw.ok, hw.meh
        self.times = np.array([o.time for o in objs], dtype=np.float64)
        self.ends = np.array([o.end_time for o in objs], dtype=np.float64)
        self.appear = self.times - self.preempt
        self.kinds = np.array([KIND_CODES[o.kind] for o in objs], dtype=np.int8)
        self.px = np.array([o.pos.x for o in objs], dtype=np.float64)
        self.py = np.array([o.pos.y for o in objs], dtype=np.float64)
        self.times_list = self.times.tolist()
        self.appear_list = self.appear.tolist()
        self.last_end = float(self.ends.max())
        # spinner: full spins needed for a 300 (lazer's Spinner.SpinsRequired)
        min_rps = difficulty_range(d.od, 90, 150, 225) / 60.0
        self.spins_required = np.array(
            [int(min_rps * o.duration / 1000.0 + 0.0001) if o.kind == "spinner" else 0 for o in objs]
        )
        # slider ticks and repeats (times only), and when each slider's end may be judged
        self.ticks: list[list[float]] = []
        for o in objs:
            self.ticks.append(sorted(list(o.tick_times) + list(o.repeat_times)) if o.kind == "slider" else [])
        self.tail_from = self.ends + TAIL_LENIENCY
        self.renderer = FrameRenderer(beatmap, cfg.render)


class OsuEnv:
    """One osu! play session. reset() / step() return single grayscale frames (H, W) uint8."""

    def __init__(
        self,
        maps: list[MapData] | MapData,
        cfg: EnvConfig | None = None,
        seed: int = 0,
        record: bool = False,
    ) -> None:
        self.cfg = cfg or EnvConfig()
        self.maps = maps if isinstance(maps, list) else [maps]
        if not self.maps:
            raise ValueError("OsuEnv needs at least one map")
        self.rng = np.random.default_rng(seed)
        self.actions = build_action_table(self.cfg)
        self.record = record
        self.frames: list[Frame] = []
        self.stats = PlayStats()
        self.md: MapData = self.maps[0]
        self.max_steps = 0
        self.t = 0.0

    # ------------------------------------------------------------------ episode control
    def reset(self, map_index: int | None = None, mode: str = "random") -> np.ndarray:
        """mode='random': random window of the map starting near a random object (training).
        mode='full': play the whole map from the beginning (evaluation / replay generation)."""
        if map_index is None:
            map_index = int(self.rng.integers(len(self.maps)))
        self.md = md = self.maps[map_index]
        n = md.n
        self.head_pending = md.kinds != KIND_SPINNER
        self.head_hit = np.zeros(n, dtype=bool)
        self.done = np.zeros(n, dtype=bool)
        self.tick_ptr = np.zeros(n, dtype=np.int32)
        self.slider_tracking = np.zeros(n, dtype=bool)
        self.spin_net = np.zeros(n, dtype=np.float64)  # signed rotation in radians
        self.spin_prev = np.full(n, np.nan, dtype=np.float64)
        self.pending: list[tuple[float, float]] = []  # (map time it is paid at, reward) of early hits
        self.stats = PlayStats()
        self.key_down = False  # any key held
        self.key_down_before = False  # any key held during the previous tick (the state a movement into this tick starts from)
        self.held_mask = 0  # which keys are held (KEY_K1 / KEY_K2 bits)
        self.last_bit = KEY_K2  # last key pressed, for the alternation of boolean scripted input

        if mode == "full":
            start_idx = 0
            self.t = float(md.appear[0]) - 500.0
            self.x, self.y = PLAYFIELD_W / 2.0, PLAYFIELD_H / 2.0
            self.max_steps = 1 << 62
        elif mode == "random":
            start_idx = int(self.rng.integers(n))
            self.t = float(md.appear[start_idx]) - float(self.rng.uniform(0.0, 300.0))
            self.x = float(self.rng.uniform(0.0, PLAYFIELD_W))
            self.y = float(self.rng.uniform(0.0, PLAYFIELD_H))
            self.max_steps = max(1, int(self.cfg.episode_seconds * self.cfg.decision_hz))
        else:
            raise ValueError(f"Unknown reset mode: {mode}")

        if start_idx > 0:
            self.done[:start_idx] = True
            self.head_pending[:start_idx] = False
        self.lo = start_idx
        self.head_ptr = start_idx
        self._advance_head_ptr()
        self.tracking = -1
        self.phi = self._potential()
        self.frames = [Frame(time=self.t, x=self.x, y=self.y, keys=0)] if self.record else []
        return self._render()

    @property
    def n_actions(self) -> int:
        return len(self.actions)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool]:
        """One agent decision: glide toward the chosen cell for `action_repeat` ticks (at most max_speed),
        with the chosen key state held (a late key from the middle tick on). Returns (frame, reward, terminal, truncated)."""
        tx, ty, key = self.actions[int(action)]
        tx, ty = float(tx), float(ty)
        k = max(1, self.cfg.action_repeat)
        mask, first_tick = self.cfg.key_plan(int(key))
        max_step = self.cfg.max_speed * self.cfg.dt_ms / 1000.0
        reward = 0.0
        terminal = False
        for i in range(k):
            dx, dy = tx - self.x, ty - self.y
            dist = math.hypot(dx, dy)
            if dist > max_step:
                dx, dy = dx * max_step / dist, dy * max_step / dist
            nx = min(max(self.x + dx, 0.0), PLAYFIELD_W)
            ny = min(max(self.y + dy, 0.0), PLAYFIELD_H)
            r, terminal = self._tick(nx, ny, mask if i >= first_tick else 0)
            reward += r
            if terminal:
                break
        return self._finish_decision(reward, terminal)

    def step_raw(self, x: float, y: float, keys: bool | int) -> tuple[np.ndarray, float, bool, bool]:
        """Advance ONE tick with an explicit cursor position and key state (scripted input, e.g. in tests).

        `keys` is either a replay key mask (K1/K2, with M1/M2 treated as K1/K2) or a bool: True keeps the held
        key down, or presses the next key in K1/K2 alternation if none is held. Counts as one decision.
        Returns (frame, reward, terminal, truncated)."""
        if isinstance(keys, (bool, np.bool_)):
            if not keys:
                mask = 0
            elif self.held_mask:
                mask = self.held_mask
            else:
                mask = KEY_K2 if self.last_bit == KEY_K1 else KEY_K1
        else:
            k = int(keys)
            mask = (KEY_K1 if k & (KEY_K1 | KEY_M1) else 0) | (KEY_K2 if k & (KEY_K2 | KEY_M2) else 0)
        r, terminal = self._tick(x, y, mask)
        return self._finish_decision(r, terminal)

    def _finish_decision(self, reward: float, terminal: bool) -> tuple[np.ndarray, float, bool, bool]:
        cfg = self.cfg
        self.stats.steps += 1
        truncated = (not terminal) and self.stats.steps >= self.max_steps
        if truncated:
            reward += self._pay_pending()  # an episode cut short still gets the hits it made
        if cfg.aim_shaping != 0.0:
            phi_next = 0.0 if terminal else self._potential()
            reward += cfg.aim_shaping * (cfg.gamma * phi_next - self.phi)
            self.phi = phi_next
        self.stats.reward += reward
        return (self._render(), reward, terminal, truncated)

    def _tick(self, x: float, y: float, mask: int) -> tuple[float, bool]:
        """Simulate one tick of map time with the keys in `mask` held. Returns (game reward, terminal)."""
        cfg = self.cfg
        md = self.md
        t1 = self.t + cfg.dt_ms
        reward = 0.0

        new_bits = mask & ~self.held_mask  # keys that went down this tick: each such tick is one tap
        pressed = new_bits != 0
        if pressed:
            self.last_bit = KEY_K2 if new_bits & KEY_K2 else KEY_K1
            self.stats.presses += 1
            reward -= cfg.press_penalty
        self.key_down_before = self.key_down
        self.held_mask = mask
        self.key_down = mask != 0
        self.x = min(max(float(x), 0.0), PLAYFIELD_W)
        self.y = min(max(float(y), 0.0), PLAYFIELD_H)
        self.t = t1

        # 1. Heads whose 50 window has fully expired are misses
        j = self.head_ptr
        while j < md.n and md.times[j] + md.w50 < t1:
            if self.head_pending[j]:
                reward += self._judge_head(j, None)
            j += 1
        self._advance_head_ptr()

        # 2. A press hits the earliest waiting circle / head under the cursor, unless note lock blocks it
        press_judged = False
        if pressed:
            hi_visible = bisect.bisect_right(md.appear_list, t1)
            for j in range(self.head_ptr, hi_visible):
                if not self.head_pending[j] or t1 - md.times[j] < -cfg.miss_window:
                    continue
                if math.hypot(self.x - md.px[j], self.y - md.py[j]) > md.radius:
                    continue
                prev = j - 1  # the latest earlier circle / head
                while prev >= 0 and (md.kinds[prev] == KIND_SPINNER or md.times[prev] >= md.times[j]):
                    prev -= 1
                if prev >= 0 and self.head_pending[prev] and t1 < md.times[prev]:
                    continue  # blocked: the press may still reach an object underneath
                for k in range(self.head_ptr, j):  # hitting this one misses earlier ones still waiting
                    if self.head_pending[k] and md.times[k] < md.times[j]:
                        reward += self._judge_head(k, None)
                reward += self._judge_head(j, t1 - md.times[j])
                press_judged = True
                break
            self._advance_head_ptr()

        # 3. Sliders and spinners that have started
        hi_started = bisect.bisect_right(md.times_list, t1)
        self.tracking = -1
        key_needed = False  # a slider in progress (head judged) or a spinner running wants a key held
        for j in range(self.lo, hi_started):
            if self.done[j]:
                continue
            kind = md.kinds[j]
            if kind == KIND_SLIDER:
                if self.head_pending[j]:
                    continue
                key_needed = True
                reward += self._update_slider(j, t1)
            elif kind == KIND_SPINNER:
                key_needed = key_needed or t1 < md.ends[j]
                reward += self._update_spinner(j, t1)
        if self.key_down and not key_needed:
            self.stats.idle_hold_ticks += 1
            reward -= cfg.hold_cost * cfg.dt_ms / 1000.0
        if pressed and not press_judged and not key_needed and not self._spinner_soon(t1):
            self.stats.empty_presses += 1
            reward -= cfg.empty_press_cost
        while self.lo < md.n and self.done[self.lo]:
            self.lo += 1

        terminal = self.lo >= md.n
        reward += self._pay_pending(None if terminal else t1)  # early hits whose time has come (all, at the map's end)

        # Record this tick at its real time, before any fast-forward below moves the clock
        if self.record:
            keys = self.held_mask
            self.frames.append(Frame(time=round(self.t, 3), x=self.x, y=self.y, keys=keys))

        # 4. Fast-forward through breaks (nothing on screen for more than a second)
        if cfg.skip_breaks and not terminal and self.lo < md.n:
            if hi_started <= self.lo and md.appear[self.lo] > self.t + 1000.0:
                self.t = float(md.appear[self.lo]) - 250.0

        return (reward, terminal)

    # ------------------------------------------------------------------ judgement helpers
    def _advance_head_ptr(self) -> None:
        while self.head_ptr < self.md.n and not self.head_pending[self.head_ptr]:
            self.head_ptr += 1

    def _combo_hit(self) -> None:
        self.stats.combo += 1
        self.stats.max_combo = max(self.stats.max_combo, self.stats.combo)

    def _grade_reward(self, grade: int) -> float:
        c = self.cfg
        return {300: c.r300, 100: c.r100, 50: c.r50}.get(grade, c.r_miss)

    def _count_grade(self, grade: int) -> None:
        if grade == 300:
            self.stats.n300 += 1
        elif grade == 100:
            self.stats.n100 += 1
        elif grade == 50:
            self.stats.n50 += 1
        else:
            self.stats.n_miss += 1

    def _judge_head(self, j: int, dt: float | None) -> float:
        """Judge circle / slider head j. dt=None means it expired unpressed."""
        md = self.md
        self.head_pending[j] = False
        if dt is None or dt < -md.w50:
            grade = 0
        else:
            adt = abs(dt)
            grade = 300 if adt <= md.w300 else (100 if adt <= md.w100 else 50)

        # circles and slider heads alike
        if md.kinds[j] == KIND_CIRCLE:
            self.done[j] = True
        self.head_hit[j] = grade > 0
        self._count_grade(grade)
        if grade:
            self._combo_hit()
        else:
            self.stats.combo = 0
        reward = self._grade_reward(grade)
        if grade and dt < 0.0 and self.cfg.pay_hits_on_time:
            self.pending.append((float(md.times[j]), reward))  # paid when the object's time comes
            return 0.0
        return reward

    def _spinner_soon(self, t1: float, within: float = 100.0) -> bool:
        """A spinner starts within `within` ms after t1 (pressing just before it is preparation, not a wasted press)."""
        md = self.md
        lo = bisect.bisect_right(md.times_list, t1)
        hi = bisect.bisect_right(md.times_list, t1 + within)
        return any(md.kinds[k] == KIND_SPINNER for k in range(lo, hi))

    def _pay_pending(self, t1: float | None = None) -> float:
        """Pay the early hits whose object time has come by t1 (all of them when t1 is None)."""
        if not self.pending:
            return 0.0
        due = [r for t, r in self.pending if t1 is None or t <= t1]
        self.pending = [(t, r) for t, r in self.pending if not (t1 is None or t <= t1)]
        return float(sum(due))

    def _update_slider(self, j: int, t1: float) -> float:
        md = self.md
        cfg = self.cfg
        obj = md.bm.objects[j]
        reward = 0.0
        ball = obj.slider_pos_at_time(min(t1, obj.end_time))
        dist = math.hypot(self.x - ball.x, self.y - ball.y)
        was_tracking = bool(self.slider_tracking[j])
        reach = md.follow_radius if was_tracking else md.radius  # enter on the ball, stay in 2.4x it
        tracking = self.key_down and t1 <= obj.end_time and dist <= reach
        self.slider_tracking[j] = tracking
        if t1 <= obj.end_time:
            # shaping: stay near the ball's centre, and losing the slider costs at once, not ticks later
            reward -= cfg.slider_offset_cost * cfg.dt_ms / 1000.0 * min(dist / md.follow_radius, 1.0)
            if was_tracking and not tracking:
                self.stats.slider_breaks += 1
                reward -= cfg.slider_break_penalty
        if tracking:
            self.tracking = j

        ticks = md.ticks[j]
        p = int(self.tick_ptr[j])
        while p < len(ticks) and ticks[p] <= t1:  # ticks and repeats: decided at their time
            if tracking:
                self.stats.ticks_hit += 1
                self._combo_hit()
                reward += cfg.r_tick
            else:
                self.stats.ticks_missed += 1
                self.stats.combo = 0
                reward += cfg.r_tick_miss
            p += 1
        self.tick_ptr[j] = p

        # the end: a hit as soon as the slider is tracked in its last 36 ms, a miss once it is over
        if p >= len(ticks) and t1 >= md.tail_from[j]:
            if tracking:
                self.stats.ends_hit += 1
                self._combo_hit()
                reward += cfg.r_tail
                self.done[j] = True
            elif t1 >= obj.end_time:
                self.stats.ends_missed += 1  # costs accuracy, but breaks no combo
                self.done[j] = True
        return reward

    def _update_spinner(self, j: int, t1: float) -> float:
        md = self.md
        reward = 0.0
        dx = self.x - md.px[j]
        dy = self.y - md.py[j]
        if abs(dx) > 1e-6 or abs(dy) > 1e-6:
            ang = math.atan2(dy, dx)
            prev = self.spin_prev[j]
            # the move from the previous position to this one is made under the previous tick's keys
            if self.key_down_before and not math.isnan(prev) and t1 < md.ends[j]:
                # signed: back-and-forth movement cancels out, as in lazer
                before = abs(float(self.spin_net[j]))
                # wrapped exactly as lazer's SpinnerRotationTracker (and the replay judge) do: only differences beyond
                # +-180 degrees are wrapped, so a jump of exactly 180 (straight through the centre) keeps its sign and
                # jumping back and forth across the centre cancels out. A modulo wrap would turn both +180 and -180
                # into -180, counting every jump across the centre as half a turn the same way.
                delta = ang - prev
                if delta > math.pi:
                    delta -= 2.0 * math.pi
                elif delta < -math.pi:
                    delta += 2.0 * math.pi
                self.spin_net[j] += delta
                gain = abs(float(self.spin_net[j])) - before  # paid as the net rotation grows; turning back costs it again
                if gain > 0.0 and self.cfg.spin_max_deg > 0.0:
                    gain = min(gain, math.radians(self.cfg.spin_max_deg))
                reward += self._spin_value_at(j, before + gain) - self._spin_value_at(j, before)
            self.spin_prev[j] = ang
        if t1 >= md.ends[j]:
            self.done[j] = True
            req = md.spins_required[j]
            progress = 1.0 if req == 0 else abs(self.spin_net[j]) / (2.0 * math.pi) / req
            grade = 300 if progress >= 1.0 else (100 if progress > 0.9 else (50 if progress > 0.75 else 0))
            self._count_grade(grade)
            if grade:
                self._combo_hit()
            else:
                self.stats.combo = 0
            reward += self._grade_reward(grade)
        return reward

    def _spin_value_at(self, j: int, rotation: float) -> float:
        """Spin reward for `rotation` radians on spinner j: r_spin per rotation up to the spins a 300 needs, r_spin_bonus after."""
        spins = rotation / (2.0 * math.pi)
        req = float(self.md.spins_required[j])
        return self.cfg.r_spin * min(spins, req) + self.cfg.r_spin_bonus * max(0.0, spins - req)

    # ------------------------------------------------------------------ shaping & rendering
    def _potential(self) -> float:
        """phi(s) = -distance(cursor, current aim target) / playfield diagonal (0 when nothing to aim at)."""
        md = self.md
        t = self.t
        hi_started = bisect.bisect_right(md.times_list, t)
        for j in range(self.lo, hi_started):
            if not self.done[j] and md.kinds[j] == KIND_SLIDER and not self.head_pending[j]:
                b = md.bm.objects[j].slider_pos_at_time(t)
                return -math.hypot(self.x - b.x, self.y - b.y) / PLAYFIELD_DIAG
        j = self.head_ptr
        if j < md.n and md.appear[j] <= t:
            return -math.hypot(self.x - md.px[j], self.y - md.py[j]) / PLAYFIELD_DIAG
        return 0.0

    def _visible(self) -> list[int]:
        md = self.md
        hi = bisect.bisect_right(md.appear_list, self.t)
        return [j for j in range(self.lo, hi) if not self.done[j]]

    def _render(self) -> np.ndarray:
        k1 = bool(self.held_mask & KEY_K1)
        k2 = bool(self.held_mask & KEY_K2)
        return self.md.renderer.render(
            self.t, self.x, self.y, k1, k2, self._visible(), self.head_pending, self.tracking,
            spin_rotation=self.spin_net, spins_required=self.md.spins_required,
        )
