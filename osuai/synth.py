"""Synthetic beatmap generator: playable osu!standard maps aimed at a target star rating.

A map is a stream of objects (circles, sliders, spinners in a chosen ratio) on a fixed BPM. Its rhythm (how often
1/1, 1/2 and 1/4 gaps appear), BPM, AR / OD and slider velocity are picked from the target star rating; the jump
spacing (cursor speed between objects) is then tuned by bisection until rosu-pp's star rating hits the target.
When spacing alone cannot get there, the BPM is moved and the search repeats.

Spinners carry no difficulty, so a map of only spinners cannot reach a star rating; it is written anyway and gets
whatever rating rosu-pp gives it (about 0).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from osuai.beatmap import Vec2, build_slider_path, parse_osu
from osuai.judge import compute_pp

PLAYFIELD_W, PLAYFIELD_H = 512.0, 384.0
MARGIN = 24.0  # keep objects this far from the playfield edge
OBJECT_TYPES = ("circles", "sliders", "spinners")
FIRST_OBJECT_MS = 2000.0
MAX_JUMP = 420.0


@dataclass
class SynthResult:
    text: str
    md5: str
    star_rating: float
    target_stars: float
    n_circles: int
    n_sliders: int
    n_spinners: int
    bpm: float

    @property
    def n_objects(self) -> int:
        return self.n_circles + self.n_sliders + self.n_spinners


def parse_ratio(text: str) -> tuple[float, float, float]:
    """'60/30/10' (circles/sliders/spinners, any scale) -> fractions summing to 1."""
    parts = [p for p in text.replace(",", "/").replace(":", "/").split("/") if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"--ratio needs three numbers circles/sliders/spinners, e.g. 60/30/10 (got {text!r})")
    vals = [float(p.strip().rstrip("%")) for p in parts]
    if any(v < 0 for v in vals) or sum(vals) <= 0:
        raise ValueError(f"--ratio values must be >= 0 and not all 0 (got {text!r})")
    s = sum(vals)
    return (vals[0] / s, vals[1] / s, vals[2] / s)


def parse_star_range(text: str) -> tuple[float, float]:
    """'5' -> (5, 5); '3-6' -> (3, 6)."""
    t = text.strip()
    if "-" in t[1:]:
        i = t.index("-", 1)
        lo, hi = float(t[:i]), float(t[i + 1:])
    else:
        lo = hi = float(t)
    if lo < 0 or hi < lo:
        raise ValueError(f"--stars must be a rating like 5 or a range like 3-6 (got {text!r})")
    return lo, hi


# ---------------------------------------------------------------------------------------------- layout
class _Layout:
    """Everything about a map except the jump spacing and BPM, drawn once per seed so the search is stable."""

    def __init__(self, stars: float, ratio: tuple[float, float, float], length_s: float, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.stars = stars
        self.ratio = ratio
        self.length_s = length_s
        s = stars
        self.bpm = float(np.clip(90.0 + 18.0 * s, 80.0, 260.0) * self.rng.uniform(0.93, 1.07))
        self.ar = round(float(np.clip(5.0 + 0.75 * s, 5.0, 10.0)), 1)
        self.od = round(float(np.clip(4.0 + 0.8 * s, 4.0, 10.0)), 1)
        self.cs = round(float(self.rng.uniform(3.6, 4.4)), 1)
        self.slider_mult = round(float(np.clip(1.0 + 0.12 * s, 0.8, 2.2)), 2)
        # rhythm: probabilities of a 2 / 1 / 1/2 beat gap, or a 1/4 stream of circles
        if s < 2.0:
            self.gap_p = {2.0: 0.3, 1.0: 0.6, 0.5: 0.1, "stream": 0.0}
        elif s < 3.5:
            self.gap_p = {2.0: 0.1, 1.0: 0.5, 0.5: 0.4, "stream": 0.0}
        elif s < 5.0:
            self.gap_p = {2.0: 0.0, 1.0: 0.3, 0.5: 0.6, "stream": 0.1}
        elif s < 6.5:
            self.gap_p = {2.0: 0.0, 1.0: 0.15, 0.5: 0.65, "stream": 0.2}
        else:
            self.gap_p = {2.0: 0.0, 1.0: 0.05, 0.5: 0.6, "stream": 0.35}
        self.stream_max = int(np.clip(3 + s, 3, 16))
        self.seed = seed


def _inside(p: Vec2) -> bool:
    return MARGIN <= p.x <= PLAYFIELD_W - MARGIN and MARGIN <= p.y <= PLAYFIELD_H - MARGIN


def _bezier_length(pts: list[tuple[float, float]], n: int = 64) -> float:
    a = np.array(pts, dtype=np.float64)
    ts = np.linspace(0.0, 1.0, n)
    k = len(a) - 1
    coef = np.array([math.comb(k, i) for i in range(k + 1)], dtype=np.float64)
    basis = coef[None, :] * (ts[:, None] ** np.arange(k + 1)) * ((1 - ts[:, None]) ** (k - np.arange(k + 1)))
    curve = basis @ a
    return float(np.sum(np.linalg.norm(np.diff(curve, axis=0), axis=1)))


def _render(layout: _Layout, spacing: float, bpm: float) -> tuple[str, tuple[int, int, int]]:
    """Build the .osu text for a given jump spacing (osu!px per ms of gap) and BPM."""
    rng = np.random.default_rng(layout.seed + 1)  # same draws for every spacing / BPM tried
    beat = 60000.0 / bpm
    end_ms = FIRST_OBJECT_MS + layout.length_s * 1000.0
    gap_keys = [k for k, v in layout.gap_p.items() if v > 0]
    gap_w = np.array([layout.gap_p[k] for k in gap_keys], dtype=np.float64)
    gap_w /= gap_w.sum()
    plain_gaps = [g for g in gap_keys if g != "stream"]
    plain_w = np.array([layout.gap_p[g] for g in plain_gaps], dtype=np.float64)
    plain_w /= plain_w.sum()
    lines: list[str] = []
    counts = {"circles": 0, "sliders": 0, "spinners": 0}

    t = FIRST_OBJECT_MS
    pos = Vec2(float(rng.uniform(128, 384)), float(rng.uniform(96, 288)))
    heading = float(rng.uniform(0, 2 * math.pi))
    since_combo = 0
    stream_left = 0
    allowed = np.array([r > 0 for r in layout.ratio])

    def pick_kind() -> str:
        """The type furthest below its share so far (streams add circles, so the ratio holds over the whole map)."""
        n = sum(counts.values()) + 1
        deficit = np.array([layout.ratio[i] * n - counts[OBJECT_TYPES[i]] for i in range(3)])
        deficit = deficit + rng.uniform(0.0, 1.5, 3)  # some randomness in the order
        deficit[~allowed] = -np.inf
        return OBJECT_TYPES[int(np.argmax(deficit))]

    def place_next(gap_ms: float, scale: float = 1.0) -> Vec2:
        nonlocal heading
        dist = min(MAX_JUMP, spacing * gap_ms * scale)
        turn_sd = 1.2 if dist > 60 else 0.5
        for attempt in range(24):
            ang = heading + float(rng.normal(0.0, turn_sd)) + (math.pi * (attempt // 6) / 2 if attempt >= 6 else 0.0)
            cand = Vec2(pos.x + dist * math.cos(ang), pos.y + dist * math.sin(ang))
            if _inside(cand):
                heading = ang
                return cand
        # no room in any direction we tried: head toward the centre
        cx, cy = PLAYFIELD_W / 2 - pos.x, PLAYFIELD_H / 2 - pos.y
        ang = math.atan2(cy, cx)
        heading = ang
        d = min(dist, math.hypot(cx, cy) + 100.0)
        return Vec2(float(np.clip(pos.x + d * math.cos(ang), MARGIN, PLAYFIELD_W - MARGIN)),
                    float(np.clip(pos.y + d * math.sin(ang), MARGIN, PLAYFIELD_H - MARGIN)))

    def combo_flag() -> int:
        nonlocal since_combo
        since_combo += 1
        if since_combo >= int(rng.integers(4, 9)):
            since_combo = 0
            return 4
        return 0

    next_kind = pick_kind()
    while t < end_ms:
        kind = next_kind if stream_left == 0 else "circles"
        if kind == "spinners":
            dur = float(np.clip(rng.uniform(1.0, 2.5) * 1000.0, beat * 2, beat * 8))
            dur = round(dur / beat) * beat
            lines.append(f"256,192,{int(round(t))},12,0,{int(round(t + dur))},0:0:0:0:")
            counts["spinners"] += 1
            since_combo = 0
            next_kind = pick_kind()
            pos = Vec2(256.0, 192.0)
            t += dur + beat
            continue

        if kind == "sliders":
            beats = float(rng.choice([0.5, 1.0, 2.0], p=[0.35, 0.45, 0.2] if layout.stars >= 3 else [0.1, 0.6, 0.3]))
            slides = int(rng.choice([1, 2, 3], p=[0.75, 0.2, 0.05]))
            placed = None
            for _ in range(3):
                length = layout.slider_mult * 100.0 * beats
                for _try in range(16):
                    ang = heading + float(rng.normal(0.0, 1.0))
                    bend = float(rng.normal(0.0, 0.35)) * length
                    if abs(bend) < 0.08 * length:
                        ctrl = [(pos.x, pos.y), (pos.x + math.cos(ang), pos.y + math.sin(ang))]
                        raw = 1.0
                        curve = "L"
                    else:
                        mid = (pos.x + 0.5 * math.cos(ang), pos.y + 0.5 * math.sin(ang))
                        ctrl = [(pos.x, pos.y), (mid[0] - bend / length * math.sin(ang), mid[1] + bend / length * math.cos(ang)),
                                (pos.x + math.cos(ang), pos.y + math.sin(ang))]
                        raw = _bezier_length(ctrl)
                        curve = "B"
                    f = length * 1.02 / raw  # a hair longer than needed; the game cuts the path at the length
                    pts = [(round(pos.x + (x - pos.x) * f), round(pos.y + (y - pos.y) * f)) for x, y in ctrl[1:]]
                    curve_str = curve + "".join(f"|{x}:{y}" for x, y in pts)
                    path, _, got = build_slider_path(pos, curve_str, length)
                    if got >= length * 0.99 and all(_inside(p) for p in path[:: max(1, len(path) // 24)] + [path[-1]]):
                        placed = (curve_str, length, path[-1], ang)
                        break
                if placed:
                    break
                beats /= 2.0  # no room for this one: try a shorter slider
            if not placed:  # last resort: a short straight slider toward the centre always fits
                length = layout.slider_mult * 100.0 * beats
                ang = math.atan2(PLAYFIELD_H / 2 - pos.y, PLAYFIELD_W / 2 - pos.x)
                end = (round(pos.x + length * 1.02 * math.cos(ang)), round(pos.y + length * 1.02 * math.sin(ang)))
                curve_str = f"L|{end[0]}:{end[1]}"
                path, _, got = build_slider_path(pos, curve_str, length)
                if got >= length * 0.99:
                    placed = (curve_str, length, path[-1], ang)
            if placed:
                curve_str, length, tail, ang = placed
                lines.append(f"{int(round(pos.x))},{int(round(pos.y))},{int(round(t))},{2 | combo_flag()},0,{curve_str},{slides},{length:.2f}")
                counts["sliders"] += 1
                t += beats * beat * slides
                if slides % 2 == 1:
                    pos = tail
                    heading = ang
                else:
                    heading = ang + math.pi
                gap = max(float(rng.choice(plain_gaps, p=plain_w)), 0.5)  # no stream straight after a slider
                next_kind = pick_kind()
                t += gap * beat
                pos = place_next(gap * beat)
                continue
            kind = "circles"  # could not fit a slider anywhere: put a circle there instead

        # circle
        lines.append(f"{int(round(pos.x))},{int(round(pos.y))},{int(round(t))},{1 | combo_flag()},0,0:0:0:0:")
        counts["circles"] += 1
        if stream_left > 0:
            stream_left -= 1
            gap_ms = beat / 4.0
            scale = 0.6
        else:
            g = gap_keys[int(rng.choice(len(gap_keys), p=gap_w))]
            if g == "stream" and layout.ratio[0] > 0:
                stream_left = int(rng.integers(2, layout.stream_max))  # circles that follow at 1/4
                gap_ms, scale = beat / 4.0, 0.6
            else:
                g = 0.5 if g == "stream" else g
                gap_ms, scale = float(g) * beat, 1.0
        next_kind = pick_kind() if stream_left == 0 else "circles"
        if next_kind == "spinners":
            gap_ms = max(gap_ms, beat)
        t += gap_ms
        pos = place_next(gap_ms, scale)

    title = f"Synthetic {layout.stars:.2f}* ({round(layout.ratio[0] * 100)}/{round(layout.ratio[1] * 100)}/{round(layout.ratio[2] * 100)})"
    text = "\n".join([
        "osu file format v14",
        "",
        "[General]",
        "AudioFilename: virtual",
        "AudioLeadIn: 0",
        "PreviewTime: -1",
        "Countdown: 0",
        "SampleSet: Normal",
        "StackLeniency: 0.7",
        "Mode: 0",
        "",
        "[Metadata]",
        f"Title:{title}",
        f"TitleUnicode:{title}",
        "Artist:osuai",
        "ArtistUnicode:osuai",
        "Creator:osuai",
        f"Version:seed {layout.seed}",
        "Source:",
        "Tags:synthetic",
        "BeatmapID:0",
        "BeatmapSetID:-1",
        "",
        "[Difficulty]",
        "HPDrainRate:5",
        f"CircleSize:{layout.cs}",
        f"OverallDifficulty:{layout.od}",
        f"ApproachRate:{layout.ar}",
        f"SliderMultiplier:{layout.slider_mult}",
        "SliderTickRate:1",
        "",
        "[Events]",
        "//Background and Video events",
        "",
        "[TimingPoints]",
        f"{int(FIRST_OBJECT_MS)},{beat:.6f},4,1,0,60,1,0",
        "",
        "[HitObjects]",
        *lines,
        "",
    ])
    return text, (counts["circles"], counts["sliders"], counts["spinners"])


def _stars_of(text: str) -> float:
    return compute_pp(parse_osu(text))[0]


def generate_map(
    target_stars: float,
    ratio: tuple[float, float, float] = (0.6, 0.35, 0.05),
    length_s: float = 90.0,
    seed: int = 0,
    tolerance: float = 0.05,
) -> SynthResult:
    """Generate one map whose star rating is within `tolerance` of `target_stars` (when reachable)."""
    layout = _Layout(target_stars, ratio, length_s, seed)
    bpm = layout.bpm
    lo_s, hi_s = 0.0, 4.0  # jump spacing in osu!px per ms of gap
    best: tuple[float, str, tuple[int, int, int], float] | None = None

    def consider(text: str, counts: tuple[int, int, int], b: float) -> float:
        nonlocal best
        st = _stars_of(text)
        if best is None or abs(st - target_stars) < abs(best[0] - target_stars):
            best = (st, text, counts, b)
        return st

    for _bpm_try in range(8):
        text, counts = _render(layout, lo_s, bpm)
        st_lo = consider(text, counts, bpm)
        if st_lo > target_stars + tolerance and bpm > 60.0:
            bpm = max(60.0, bpm / 1.15)  # too hard even stacked: slow the song down
            continue
        text, counts = _render(layout, hi_s, bpm)
        st_hi = consider(text, counts, bpm)
        if st_hi < target_stars - tolerance and bpm < 320.0:
            bpm = min(320.0, bpm * 1.15)  # too easy even at the widest spacing: speed it up
            continue
        # The rating is not perfectly smooth in the spacing (objects shift around as jumps widen), so a search
        # can stall just short of the target; then it is repeated at a slightly different BPM.
        for nudge in (1.0, 1.03, 0.97, 1.06, 0.94, 1.09, 0.91):
            if abs(best[0] - target_stars) <= tolerance:
                break
            a, b = lo_s, hi_s
            for _ in range(18):
                m = 0.5 * (a + b)
                text, counts = _render(layout, m, bpm * nudge)
                st = consider(text, counts, bpm * nudge)
                if abs(best[0] - target_stars) <= tolerance:
                    break
                if st < target_stars:
                    a = m
                else:
                    b = m
        break

    st, text, counts, b = best
    return SynthResult(
        text=text,
        md5=hashlib.md5(text.encode("utf-8")).hexdigest(),
        star_rating=st,
        target_stars=target_stars,
        n_circles=counts[0],
        n_sliders=counts[1],
        n_spinners=counts[2],
        bpm=b,
    )
