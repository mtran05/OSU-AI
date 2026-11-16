"""Replay judgement and scoring the way osu!lazer does it (standard rules, no mods that change judging), plus PP.

Ported from lazer's source (ppy/osu): OsuHitWindows; StartTimeOrderedHitPolicy (note lock); SliderInputManager
(slider tracking); slider heads timed like circles, ticks and repeats worth 30, slider ends worth 150 (a missed end
costs accuracy but not combo); spinner rotation and ticks (Spinner / DrawableSpinner); ScoreProcessor (accuracy and
standardised total score).

Results closely match lazer's but not exactly: lazer judges on rendered frames, so even the same replay can score
slightly differently between playbacks. Here sliders are tracked on a ~240 Hz grid plus every replay frame and tick
time, spinners per replay frame, and stacking offsets are not applied. PP comes from rosu-pp's osu!stable formula.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field

import rosu_pp_py as rosu

from osuai.beatmap import KEY_K1, KEY_K2, KEY_M1, KEY_M2, Beatmap, Frame

LEFT = 1
RIGHT = 2
FOLLOW_AREA = 2.4
TAIL_LENIENCY = -36.0
MISS_WINDOW = 400.0
UPDATE_HZ = 240.0
COMBO_EXPONENT = 0.5
SCORE_MULTIPLIER = 1.0  # no mods that change the score multiplier
SPIN_EITHER_DIRECTION = True  # spins counted from |net rotation| (True) or clockwise net rotation only (False)

# HitResult base scores (ScoreProcessor.GetBaseScoreForResult)
GREAT, OK, MEH, MISS = "great", "ok", "meh", "miss"
LARGE_TICK_HIT, LARGE_TICK_MISS = "large_tick_hit", "large_tick_miss"
SLIDER_TAIL_HIT = "slider_tail_hit"
SMALL_BONUS, LARGE_BONUS = "small_bonus", "large_bonus"
IGNORE_HIT, IGNORE_MISS = "ignore_hit", "ignore_miss"
BASE_SCORE = {GREAT: 300, OK: 100, MEH: 50, MISS: 0, LARGE_TICK_HIT: 30, LARGE_TICK_MISS: 0, SLIDER_TAIL_HIT: 150,
              SMALL_BONUS: 10, LARGE_BONUS: 50, IGNORE_HIT: 0, IGNORE_MISS: 0}
IS_HIT = {GREAT, OK, MEH, LARGE_TICK_HIT, SLIDER_TAIL_HIT, SMALL_BONUS, LARGE_BONUS, IGNORE_HIT}
AFFECTS_COMBO = {GREAT, OK, MEH, MISS, LARGE_TICK_HIT, LARGE_TICK_MISS, SLIDER_TAIL_HIT}
IS_BONUS = {SMALL_BONUS, LARGE_BONUS}
IGNORED = {IGNORE_HIT, IGNORE_MISS}  # no score, no accuracy, no combo


def _in_hit_area(dx: float, dy: float, r: float) -> bool:
    return dx * dx + dy * dy <= r * r


def difficulty_range(d: float, lo: float, mid: float, hi: float) -> float:
    if d > 5.0:
        return mid + (hi - mid) * (d - 5.0) / 5.0
    if d < 5.0:
        return mid + (mid - lo) * (d - 5.0) / 5.0
    return mid


@dataclass(slots=True)
class HitWindows:
    great: float
    ok: float
    meh: float
    miss: float = MISS_WINDOW

    @classmethod
    def for_od(cls, od: float) -> HitWindows:
        return cls(
            great=math.floor(difficulty_range(od, 80, 50, 20)) - 0.5,
            ok=math.floor(difficulty_range(od, 140, 100, 60)) - 0.5,
            meh=math.floor(difficulty_range(od, 200, 150, 100)) - 0.5,
        )

    def result_for(self, offset: float) -> str | None:
        t = abs(offset)
        for res, w in ((GREAT, self.great), (OK, self.ok), (MEH, self.meh), (MISS, self.miss)):
            if t <= w:
                return res
        return None


@dataclass(slots=True)
class Judgement:
    time: float
    result: str
    max_result: str
    index: int = -1  # hit object this judgement belongs to


@dataclass(slots=True)
class JudgeResult:
    n300: int  # Great (circles, slider heads and spinners)
    n100: int  # Ok
    n50: int  # Meh
    n_miss: int
    slider_ticks_hit: int  # lazer's "slider tick": slider ticks and repeats
    slider_ticks_total: int
    slider_ends_hit: int  # lazer's "slider end"
    slider_ends_total: int
    spinner_spins_hit: int  # lazer's "spinner spin" ticks
    spinner_spins_total: int
    spinner_bonus_hit: int  # lazer's "spinner bonus" ticks
    spinner_bonus_total: int
    max_combo: int
    beatmap_max_combo: int
    accuracy: float  # percent, floored to 2 decimals as lazer shows it
    accuracy_exact: float  # fraction 0..1
    score: int  # lazer's standardised total score
    perfect: bool  # no misses and full combo
    passed: bool
    star_rating: float = 0.0
    pp: float = 0.0
    statistics: dict[str, int] = field(default_factory=dict)  # lazer's per-result counts, e.g. {"great": 120, ...}
    maximum_statistics: dict[str, int] = field(default_factory=dict)  # the same for a perfect play
    judgements: list[Judgement] = field(default_factory=list)


class _Input:
    """Replay playback: cursor interpolated linearly between frames, buttons switch exactly at frame times."""

    def __init__(self, frames: list[Frame]) -> None:
        self.frames = sorted(frames, key=lambda f: f.time)
        self.times = [f.time for f in self.frames]

    @staticmethod
    def actions(keys: int) -> int:
        a = 0
        if keys & (KEY_M1 | KEY_K1):
            a |= LEFT
        if keys & (KEY_M2 | KEY_K2):
            a |= RIGHT
        return a

    def pressed_at(self, t: float) -> int:
        i = bisect.bisect_right(self.times, t) - 1
        return self.actions(self.frames[i].keys) if i >= 0 else 0

    def pos_at(self, t: float) -> tuple[float, float]:
        fr = self.frames
        i = bisect.bisect_right(self.times, t) - 1
        if i < 0:
            return (fr[0].x, fr[0].y)
        if i >= len(fr) - 1:
            return (fr[-1].x, fr[-1].y)
        a, b = fr[i], fr[i + 1]
        span = b.time - a.time
        u = 0.0 if span <= 0 else (t - a.time) / span
        return (a.x + (b.x - a.x) * u, a.y + (b.y - a.y) * u)

    def presses(self) -> list[tuple[float, int, float, float]]:
        """(time, action, x, y) for every button that goes down."""
        out = []
        prev = 0
        for f in self.frames:
            cur = self.actions(f.keys)
            for act in (LEFT, RIGHT):
                if cur & act and not prev & act:
                    out.append((f.time, act, f.x, f.y))
            prev = cur
        return out


def _slider_nested(o) -> list[tuple[float, float, float, str]]:
    """Ticks and repeats as (time, x, y, kind) sorted by time, then the tail check time."""
    ev = [(t, p.x, p.y, "tick") for t, p in zip(o.tick_times, o.tick_positions)]
    ev += [(t, p.x, p.y, "repeat") for t, p in zip(o.repeat_times, o.repeat_positions)]
    ev.sort(key=lambda e: e[0])
    return ev


def judge_replay(beatmap: Beatmap, frames: list[Frame], score_multiplier: float = SCORE_MULTIPLIER) -> JudgeResult:
    """Judge a replay (frames in map time) against a beatmap the way osu!lazer does."""
    objs = beatmap.objects
    d = beatmap.difficulty
    hw = HitWindows.for_od(d.od)
    radius = d.circle_radius
    preempt = d.preempt
    inp = _Input(frames)
    judgements: list[Judgement] = []

    # ---------------------------------------------------------------- 1. circles and slider heads (note lock)
    head_time: dict[int, float] = {}  # object index -> time its circle/head was judged
    head_hit: dict[int, bool] = {}
    head_result: dict[int, str] = {}  # great / ok / meh / miss of every circle and slider head
    head_action: dict[int, int] = {}
    hittable = [i for i, o in enumerate(objs) if o.kind in ("circle", "slider")]
    pending = set(hittable)

    def expire_until(t: float) -> None:
        """Auto-miss heads whose meh window has passed by time t."""
        for i in sorted(pending):
            o = objs[i]
            if o.time + hw.meh < t:
                pending.discard(i)
                head_time[i] = o.time + hw.meh
                head_hit[i] = False
                head_result[i] = MISS
                if o.kind == "circle":
                    judgements.append(Judgement(head_time[i], MISS, GREAT, i))

    for (t, act, x, y) in inp.presses():
        expire_until(t)
        # the press goes to the objects under the cursor, earliest (topmost) first
        cands = [i for i in sorted(pending) if objs[i].time - preempt <= t and _in_hit_area(x - objs[i].pos.x, y - objs[i].pos.y, radius)]
        for i in cands:
            o = objs[i]
            res = hw.result_for(t - o.time)
            if res is None:
                continue  # shake
            # StartTimeOrderedHitPolicy: blocked only by the latest earlier circle / head, and only before its time
            earlier = [j for j in hittable if objs[j].time < o.time and objs[j].time - preempt <= t]
            if earlier and earlier[-1] in pending and t < objs[earlier[-1]].time:
                continue  # shake: the press may still reach an object underneath
            # hitting an object misses every earlier circle / head still waiting for a press
            for j in sorted(pending):
                if objs[j].time < o.time:
                    pending.discard(j)
                    head_time[j], head_hit[j], head_result[j] = t, False, MISS
                    if objs[j].kind == "circle":
                        judgements.append(Judgement(t, MISS, GREAT, j))
            pending.discard(i)
            head_time[i] = t
            head_hit[i] = res != MISS
            head_result[i] = res
            if head_hit[i]:
                head_action[i] = act
            if o.kind == "circle":
                judgements.append(Judgement(t, res, GREAT, i))
            break
    expire_until(float("inf"))

    # ---------------------------------------------------------------- 2. sliders (tracking, nested objects, body)
    ticks_hit = ticks_total = ends_hit = ends_total = 0
    for i, o in enumerate(objs):
        if o.kind != "slider":
            continue
        th = head_time[i]
        hhit = head_hit[i]
        judgements.append(Judgement(th, head_result[i], GREAT, i))  # the head is timed like a circle
        nested = _slider_nested(o)
        # the tail may be hit at any update from 36 ms before the end; it is only missed once the slider has ended
        tail_from = o.end_time + TAIL_LENIENCY
        r_follow = radius * FOLLOW_AREA

        def ball(t: float) -> tuple[float, float]:
            p = o.slider_pos_at_time(min(max(t, o.time), o.end_time))
            return (p.x, p.y)

        def in_area(t: float, expanded: bool) -> bool:
            cx, cy = inp.pos_at(t)
            bx, by = ball(t)
            r = r_follow if expanded else radius
            return (cx - bx) ** 2 + (cy - by) ** 2 <= r * r

        results: dict[int, bool] = {}  # nested index -> hit
        tracking = False
        h_act = head_action.get(i)
        tta: float | None = None
        last_pressed = inp.pressed_at(th - 1e-6)

        # PostProcessHeadJudgement: a (late) head hit settles every nested object that has already passed
        if hhit and in_area(th, True):
            passed = [k for k, e in enumerate(nested) if e[0] <= th]
            cx, cy = inp.pos_at(th)
            all_in = all((e[1] - cx) ** 2 + (e[2] - cy) ** 2 <= r_follow ** 2 for e in (nested[k] for k in passed))
            for k in passed:
                results[k] = all_in
            tracking = (all_in or in_area(th, False)) and bool(inp.pressed_at(th) & (h_act or (LEFT | RIGHT)))

        # update times: grid + replay frames + nested times, from the head judgement to the slider end
        t0 = min(th, o.time)
        t_end = max(o.end_time, th)
        grid = [t0 + k * (1000.0 / UPDATE_HZ) for k in range(int((t_end - t0) * UPDATE_HZ / 1000.0) + 1)]
        lo = bisect.bisect_left(inp.times, t0)
        hi = bisect.bisect_right(inp.times, t_end)
        times = sorted(set(grid) | set(inp.times[lo:hi]) | {e[0] for e in nested} | {tail_from, t_end, th})
        tail_hit: bool | None = None
        tail_time = t_end
        nk = 0
        for t in times:
            if t < th:
                continue
            pressed = inp.pressed_at(t)
            # key validity (SliderInputManager)
            if h_act is not None and tta is None:
                other = RIGHT if h_act == LEFT else LEFT
                if not last_pressed & other:
                    tta = t
            if h_act is not None and (tta is None or t <= tta):
                valid = bool(pressed & h_act)
            else:
                valid = bool(pressed & (LEFT | RIGHT))
            tracking = valid and in_area(t, tracking) and t <= o.end_time
            last_pressed = pressed
            # nested ticks / repeats: decided at the first update at or after their time
            while nk < len(nested) and nested[nk][0] <= t:
                if nk not in results:
                    results[nk] = tracking
                nk += 1
            # tail: judged only after every tick / repeat; a hit as soon as tracking in its window, a miss at the end
            if t >= tail_from and nk >= len(nested):
                if tracking:
                    tail_hit, tail_time = True, t
                    break
                if t >= o.end_time:
                    tail_hit, tail_time = False, t
                    break
        if tail_hit is None:
            tail_hit = False
        # emit nested judgements in time order
        for k, e in enumerate(nested):
            hit = results.get(k, False)
            judgements.append(Judgement(max(e[0], th), LARGE_TICK_HIT if hit else LARGE_TICK_MISS, LARGE_TICK_HIT, i))
        # 150-point tail (a miss costs accuracy but not combo); the slider itself is not graded
        parts = [hhit] + [results.get(k, False) for k in range(len(nested))] + [tail_hit]
        judgements.append(Judgement(tail_time, SLIDER_TAIL_HIT if tail_hit else IGNORE_MISS, SLIDER_TAIL_HIT, i))
        judgements.append(Judgement(tail_time + 1e-6, IGNORE_HIT if any(parts) else IGNORE_MISS, IGNORE_HIT, i))
        ticks_total += len(nested)
        ticks_hit += sum(results.get(k, False) for k in range(len(nested)))
        ends_total += 1
        ends_hit += int(tail_hit)

    # ---------------------------------------------------------------- 3. spinners
    for i, o in enumerate(objs):
        if o.kind != "spinner":
            continue
        sec = o.duration / 1000.0
        min_rps = difficulty_range(d.od, 90, 150, 225) / 60.0
        max_rps = difficulty_range(d.od, 250, 380, 430) / 60.0
        spins_required = int(min_rps * sec + 0.0001)
        max_bonus = max(0, int(max_rps * sec + 0.0001) - spins_required - 2)
        # SpinnerRotationTracker: per update, the angle change around the centre is wrapped to +-180 degrees and
        # added with its sign while a key is held inside [start, end); the angle is tracked even when not spinning.
        # Replay playback moves the cursor in a straight line from one frame to the next under the keys of the frame
        # it leaves, so each move counts with the previous frame's keys (a straight move sweeps under 180 degrees,
        # so one wrapped difference equals lazer's sum over its updates)
        lo = max(bisect.bisect_left(inp.times, o.time) - 1, 0)
        hi = bisect.bisect_left(inp.times, o.end_time)
        net, prev, prev_keys = 0.0, None, 0
        for f in inp.frames[lo:hi]:
            ang = math.degrees(math.atan2(f.y - o.pos.y, f.x - o.pos.x))
            held = bool(inp.actions(prev_keys))
            prev_keys = f.keys
            if prev is not None and f.time >= o.time and held:
                delta = ang - prev
                if delta > 180.0:
                    delta -= 360.0
                elif delta < -180.0:
                    delta += 360.0
                net += delta
            prev = ang
        rotation = abs(net) if SPIN_EITHER_DIRECTION else max(net, 0.0)
        spins = int(rotation / 360.0)
        progress = 1.0 if spins_required == 0 else min(rotation / 360.0 / spins_required, 1.0)
        for k in range(spins_required + 2 + max_bonus):
            res = (SMALL_BONUS if k < spins_required + 2 else LARGE_BONUS) if k < spins else IGNORE_MISS
            judgements.append(Judgement(o.time + (k + 1) / (spins_required + 2 + max_bonus) * o.duration, res,
                                        SMALL_BONUS if k < spins_required + 2 else LARGE_BONUS, i))
        res = GREAT if progress >= 1.0 else (OK if progress > 0.9 else (MEH if progress > 0.75 else MISS))
        judgements.append(Judgement(o.end_time, res, GREAT, i))

    # ---------------------------------------------------------------- 4. ScoreProcessor
    judgements.sort(key=lambda j: j.time)

    def process(seq: list[tuple[str, str]]) -> tuple[float, float, float, float, int]:
        combo = max_combo = 0
        base = max_base = combo_portion = bonus = 0.0
        for res, mx in seq:
            if res in AFFECTS_COMBO:
                combo = combo + 1 if res in IS_HIT else 0
                max_combo = max(max_combo, combo)
            if mx not in IS_BONUS and mx not in IGNORED:
                max_base += BASE_SCORE[mx]
            if res in IS_BONUS:
                bonus += BASE_SCORE[res]
            elif res not in IGNORED:
                base += BASE_SCORE[res]
                combo_portion += BASE_SCORE[res] * combo ** COMBO_EXPONENT
        return (base, max_base, combo_portion, bonus, max_combo)

    base, max_base, combo_portion, bonus, max_combo = process([(j.result, j.max_result) for j in judgements])
    _, _, max_combo_portion, _, beatmap_max_combo = process([(j.max_result, j.max_result) for j in judgements])
    acc = 1.0 if max_base == 0 else base / max_base
    combo_progress = 1.0 if max_combo_portion == 0 else combo_portion / max_combo_portion
    total = round(round(500000 * acc * combo_progress + 500000 * acc ** 5 + bonus) * score_multiplier)

    counts = {GREAT: 0, OK: 0, MEH: 0, MISS: 0}
    spins = {SMALL_BONUS: [0, 0], LARGE_BONUS: [0, 0]}
    for j in judgements:
        if j.max_result == GREAT:
            counts[j.result] += 1
        elif j.max_result in spins:
            spins[j.max_result][1] += 1
            spins[j.max_result][0] += int(j.result in IS_HIT)

    mods = set(beatmap.difficulty.mods)
    n_miss = counts[MISS]
    perfect = n_miss == 0 and max_combo == beatmap_max_combo
    passed = True
    if ("SD" in mods or "PF" in mods) and not perfect:
        passed = False
    if "PF" in mods and (counts[OK] or counts[MEH]):
        passed = False
    judged = sum(counts.values())
    if "NF" not in mods and judged and acc < 0.3 and n_miss > judged // 2:
        passed = False

    stars, pp = compute_pp(beatmap, n300=counts[GREAT], n100=counts[OK], n50=counts[MEH], n_miss=n_miss, max_combo=max_combo)
    statistics: dict[str, int] = {}
    maximum_statistics: dict[str, int] = {}
    for j in judgements:
        statistics[j.result] = statistics.get(j.result, 0) + 1
        maximum_statistics[j.max_result] = maximum_statistics.get(j.max_result, 0) + 1
    return JudgeResult(
        n300=counts[GREAT],
        n100=counts[OK],
        n50=counts[MEH],
        n_miss=n_miss,
        slider_ticks_hit=ticks_hit,
        slider_ticks_total=ticks_total,
        slider_ends_hit=ends_hit,
        slider_ends_total=ends_total,
        spinner_spins_hit=spins[SMALL_BONUS][0],
        spinner_spins_total=spins[SMALL_BONUS][1],
        spinner_bonus_hit=spins[LARGE_BONUS][0],
        spinner_bonus_total=spins[LARGE_BONUS][1],
        max_combo=max_combo,
        beatmap_max_combo=beatmap_max_combo,
        accuracy=math.floor(acc * 10000) / 100,
        accuracy_exact=acc,
        score=int(total),
        perfect=perfect,
        passed=passed,
        star_rating=stars,
        pp=pp,
        statistics=statistics,
        maximum_statistics=maximum_statistics,
        judgements=judgements,
    )


def hit_errors(beatmap: Beatmap, result: JudgeResult) -> list[float]:
    """Press time minus object time (ms; negative = early) of every circle and slider head that was hit."""
    objs = beatmap.objects
    # on circles and sliders only the circle / head judgement has GREAT as its maximum (ticks and ends have their own)
    return [
        j.time - objs[j.index].time
        for j in result.judgements
        if j.max_result == GREAT and j.result in (GREAT, OK, MEH) and objs[j.index].kind in ("circle", "slider")
    ]


def unstable_rate(errors: list[float]) -> float:
    """osu!'s unstable rate: 10 × the standard deviation of the hit errors."""
    if len(errors) < 2:
        return 0.0
    m = sum(errors) / len(errors)
    return 10.0 * math.sqrt(sum((e - m) ** 2 for e in errors) / len(errors))


def compute_pp(
    beatmap: Beatmap,
    n300: int | None = None,
    n100: int = 0,
    n50: int = 0,
    n_miss: int = 0,
    max_combo: int | None = None,
) -> tuple[float, float]:
    """Compute (star_rating, pp) using rosu-pp-py."""
    if not beatmap.raw_bytes:
        return (0.0, 0.0)
    try:
        rosu_map = rosu.Beatmap(bytes=beatmap.raw_bytes)
        mods_bitmask = beatmap.difficulty.mods_bitmask
        diff_calc = rosu.Difficulty(mods=mods_bitmask, lazer=False)
        diff_attrs = diff_calc.calculate(rosu_map)
        stars = float(diff_attrs.stars)

        perf = rosu.Performance(
            mods=mods_bitmask,
            lazer=False,
            n300=len(beatmap.objects) if n300 is None else n300,
            n100=n100,
            n50=n50,
            misses=n_miss,
            combo=int(diff_attrs.max_combo) if max_combo is None else max_combo,
        )
        perf_attrs = perf.calculate(diff_attrs)
        return (stars, float(perf_attrs.pp))
    except Exception:
        return (0.0, 0.0)
