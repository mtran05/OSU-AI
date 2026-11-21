"""Shared test maps and a scripted perfect player used to check that judging and scoring reward flawless input."""

from __future__ import annotations

import bisect
import math

from osuai.beatmap import KEY_K1, KEY_K2, Beatmap, Frame

COMPLEX_MAP = """osu file format v14
[General]
Mode: 0
StackLeniency: 0.7
[Metadata]
Title: Judge Test
Artist: osuai
Creator: osuai
Version: Insane
[Difficulty]
HPDrainRate: 6
CircleSize: 4.2
OverallDifficulty: 8.5
ApproachRate: 9.2
SliderMultiplier: 1.6
SliderTickRate: 2
[TimingPoints]
0,400,4,1,0,100,1,0
2000,-75,4,1,0,100,0,0
[HitObjects]
120,120,1000,1,0,0:0:0:0:
160,120,1100,1,0,0:0:0:0:
200,120,1200,1,0,0:0:0:0:
240,120,1300,1,0,0:0:0:0:
100,200,1600,2,0,B|200:250|300:200,2,220
350,250,3000,2,0,P|380:200|350:150,1,120
256,192,4200,8,0,5400
400,100,5800,1,0,0:0:0:0:
"""

TAIL_TEST_MAP = """osu file format v14
[General]
Mode: 0
[Difficulty]
CircleSize: 4
OverallDifficulty: 5
SliderMultiplier: 1.0
SliderTickRate: 1
[TimingPoints]
0,500,4,1,0,100,1,0
[HitObjects]
100,100,1000,2,0,L|400:100,1,300
"""

SPIN_RADIUS = 100.0
SPIN_RPS = 8.0


def follow_slider(bm: Beatmap, key_at) -> list[Frame]:
    """Cursor on the ball of the map's first slider every 10 ms; key_at(t) gives the key mask held at time t."""
    s = bm.objects[0]
    frames = []
    t = s.time - 1.0
    while t <= s.end_time + 50.0:
        p = s.slider_pos_at_time(t)
        frames.append(Frame(time=t, x=p.x, y=p.y, keys=key_at(t)))
        t += 10.0
    return frames


def perfect_inputs(bm: Beatmap, rate: float = 240.0) -> list[Frame]:
    """Flawless scripted play: press every object exactly on time with alternating keys, stay on the slider ball,
    spin spinners clockwise, and glide in a straight line between objects."""
    objs = bm.objects
    holds: list[tuple[float, float, int]] = []  # (press, release, key)
    for i, o in enumerate(objs):
        key = KEY_K1 if i % 2 == 0 else KEY_K2
        release = (o.end_time + 10.0) if o.kind != "circle" else (o.time + 20.0)
        if i + 1 < len(objs):
            release = min(release, objs[i + 1].time - 1.0)
        holds.append((o.time, max(release, o.time + 0.5), key))
    starts = [o.time for o in objs]

    def pos(t: float) -> tuple[float, float]:
        i = bisect.bisect_right(starts, t) - 1
        if i < 0:
            return (objs[0].pos.x, objs[0].pos.y)
        o = objs[i]
        if t <= o.end_time:
            if o.kind == "slider":
                p = o.slider_pos_at_time(t)
                return (p.x, p.y)
            if o.kind == "spinner":
                a = 2.0 * math.pi * SPIN_RPS * (t - o.time) / 1000.0
                return (o.pos.x + SPIN_RADIUS * math.cos(a), o.pos.y + SPIN_RADIUS * math.sin(a))
            return (o.pos.x, o.pos.y)
        end = pos(o.end_time) if o.kind != "circle" else (o.pos.x, o.pos.y)
        if i + 1 >= len(objs):
            return end
        nxt = objs[i + 1]
        arrive = max(o.end_time, nxt.time - min(60.0, (nxt.time - o.end_time) / 2.0))
        if t >= arrive or arrive <= o.end_time:
            return (nxt.pos.x, nxt.pos.y)
        u = (t - o.end_time) / (arrive - o.end_time)
        return (end[0] + (nxt.pos.x - end[0]) * u, end[1] + (nxt.pos.y - end[1]) * u)

    def keys(t: float) -> int:
        mask = 0
        for press, release, key in holds:
            if press <= t < release:
                mask |= key
        return mask

    t0, t1 = objs[0].time - 500.0, objs[-1].end_time + 100.0
    times = {t0 + k * 1000.0 / rate for k in range(int((t1 - t0) * rate / 1000.0) + 1)}
    for press, release, _ in holds:
        times |= {press, release}
    frames = []
    for t in sorted(times):
        x, y = pos(t)
        frames.append(Frame(time=t, x=x, y=y, keys=keys(t)))
    return frames
