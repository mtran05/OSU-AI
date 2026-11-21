"""Tests for the replay judge: hit windows, note lock, sliders, spinners, perfect play, PP and speed."""

from __future__ import annotations

import math
import time

from osuai.beatmap import KEY_K1, KEY_K2, Frame, parse_osu
from osuai.judge import judge_replay
from tests.helpers import COMPLEX_MAP, TAIL_TEST_MAP, follow_slider, perfect_inputs

def test_circle_hit_windows_and_note_lock():
    bm = parse_osu(
        """osu file format v14
[General]
Mode: 0
[Difficulty]
CircleSize: 4
OverallDifficulty: 5
ApproachRate: 5
[TimingPoints]
0,500,4,1,0,100,1,0
[HitObjects]
256,192,1000,1,0
256,192,2000,1,0
256,192,3000,1,0
256,192,4000,1,0
"""
    )
    # At OD5 lazer's windows are 300: +-49.5 ms, 100: +-99.5 ms, 50: +-149.5 ms
    frames = [
        Frame(time=0.0, x=256.0, y=192.0, keys=0),
        # Object 1 at 1000ms: hit at 1020ms -> 300
        Frame(time=1020.0, x=256.0, y=192.0, keys=KEY_K1),
        Frame(time=1050.0, x=256.0, y=192.0, keys=0),
        # Object 2 at 2000ms: hit at 2075ms -> 100
        Frame(time=2075.0, x=256.0, y=192.0, keys=KEY_K2),
        Frame(time=2100.0, x=256.0, y=192.0, keys=0),
        # Object 3 at 3000ms: hit at 3130ms -> 50
        Frame(time=3130.0, x=256.0, y=192.0, keys=KEY_K1),
        Frame(time=3160.0, x=256.0, y=192.0, keys=0),
        # Object 4 at 4000ms: press outside radius -> Miss
        Frame(time=4000.0, x=10.0, y=10.0, keys=KEY_K2),
        Frame(time=4030.0, x=10.0, y=10.0, keys=0),
    ]
    res = judge_replay(bm, frames)
    assert (res.n300, res.n100, res.n50, res.n_miss) == (1, 1, 1, 1)
    assert res.max_combo == 3
    assert not res.perfect


def test_perfect_play_scores_100_percent_full_combo_with_mods():
    for mod in ("NM", "HR", "DT", "HR,DT", "EZ", "HT"):
        bm = parse_osu(COMPLEX_MAP, mods=mod)
        frames = perfect_inputs(bm)
        assert [f.time for f in frames] == sorted(f.time for f in frames)

        res = judge_replay(bm, frames)
        assert res.accuracy == 100.0, f"Failed 100% on {mod}: {res}"
        assert res.n_miss == 0
        assert res.slider_ticks_hit == res.slider_ticks_total and res.slider_ends_hit == res.slider_ends_total
        assert res.max_combo == bm.max_combo == res.beatmap_max_combo
        assert res.perfect
        assert res.pp > 0.0
        assert res.star_rating > 0.0


def test_slider_early_release_drops_tail_and_ticks():
    bm = parse_osu(
        """osu file format v14
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
    )
    # Release key at 1100ms (well before slider end at 2500ms)
    frames = [
        Frame(time=999.0, x=100.0, y=100.0, keys=KEY_K1),
        Frame(time=1100.0, x=120.0, y=100.0, keys=0),
        Frame(time=2500.0, x=400.0, y=100.0, keys=0),
    ]
    res = judge_replay(bm, frames)
    assert res.slider_ticks_hit < res.slider_ticks_total and res.slider_ends_hit == 0
    assert not res.perfect


def test_slider_tail_needs_any_key_not_the_head_press():
    """Lazer tracks a slider while a valid key is held: switching keys or re-pressing must not drop the tail."""
    bm = parse_osu(TAIL_TEST_MAP)
    end = bm.objects[0].end_time

    switched = judge_replay(bm, follow_slider(bm, lambda t: KEY_K1 if t < 1700.0 else KEY_K2))
    assert switched.slider_ends_hit == 1 and switched.n300 == 1 and switched.perfect

    repressed = judge_replay(
        bm, follow_slider(bm, lambda t: 0 if 1520.0 <= t < 1580.0 else (KEY_K1 if t < 1520.0 else KEY_K2))
    )
    assert repressed.slider_ends_hit == 1  # released between ticks, pressed again before the tail

    released = judge_replay(bm, follow_slider(bm, lambda t: KEY_K1 if t < end - 300.0 else 0))
    # nothing held at the end: the end is missed, but the head's 300 stands and the combo is not broken
    assert released.slider_ends_hit == 0 and released.n300 == 1 and released.max_combo == 3


def test_performance_5_minute_map_under_budget():
    # Generate a 5-minute (300,000 ms) map with 600 objects (§15 Performance tests)
    obj_lines = [f"{100 + (i * 37) % 300},{80 + (i * 29) % 220},{1000 + i * 500},1,0" for i in range(600)]
    big_osu = (
        "osu file format v14\n[General]\nMode: 0\n[Difficulty]\nCircleSize:4\nOverallDifficulty:7\n"
        "[TimingPoints]\n0,500,4,1,0,100,1,0\n[HitObjects]\n" + "\n".join(obj_lines)
    )
    t0 = time.perf_counter()
    bm = parse_osu(big_osu)
    t_parse = time.perf_counter() - t0

    t1 = time.perf_counter()
    frames = perfect_inputs(bm)
    t_gen = time.perf_counter() - t1

    t2 = time.perf_counter()
    res = judge_replay(bm, frames)
    t_judge = time.perf_counter() - t2

    assert t_parse < 1.0
    assert t_gen < 2.0
    assert t_judge < 1.0
    assert math.isclose(res.accuracy, 100.0)
