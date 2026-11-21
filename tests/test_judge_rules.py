"""Tests for the judge's osu!lazer rules: hit windows, SS scoring, note lock, slider scoring and tracking."""

from __future__ import annotations

from osuai.beatmap import KEY_K1, KEY_K2, parse_osu
from osuai.judge import HitWindows, judge_replay
from osuai.replay import quantize_frames
from tests.helpers import COMPLEX_MAP, TAIL_TEST_MAP, follow_slider, perfect_inputs


def test_hit_windows_use_lazers_rounding():
    w = HitWindows.for_od(2.0)
    assert (w.great, w.ok, w.meh, w.miss) == (67.5, 123.5, 179.5, 400.0)
    assert w.result_for(67.0) == "great" and w.result_for(68.0) == "ok"
    assert w.result_for(-250.0) == "miss" and w.result_for(401.0) is None  # early press inside 400 ms is a miss
    w8 = HitWindows.for_od(8.0)
    assert (w8.great, w8.ok, w8.meh) == (31.5, 75.5, 119.5)


def test_perfect_play_scores_like_a_lazer_ss():
    bm = parse_osu(COMPLEX_MAP)
    r = judge_replay(bm, quantize_frames(perfect_inputs(bm)))
    assert (r.n100, r.n50, r.n_miss) == (0, 0, 0) and r.n300 == len(bm.objects)
    assert r.slider_ticks_hit == r.slider_ticks_total and r.slider_ends_hit == r.slider_ends_total
    assert r.spinner_spins_hit == r.spinner_spins_total > 0
    assert r.max_combo == r.beatmap_max_combo and r.accuracy == 100.0 and r.perfect
    assert r.score >= 1_000_000  # 1M plus spinner bonus


def test_slider_tracking():
    bm = parse_osu(TAIL_TEST_MAP)  # one slider 1000-2500 ms, ticks at 1500 and 2000
    # holding K1 throughout, then switching to K2: lazer accepts the other key once it was not held at the head
    clean = judge_replay(bm, follow_slider(bm, lambda t: KEY_K1 if t < 1700.0 else KEY_K2))
    assert (clean.n300, clean.slider_ticks_hit, clean.slider_ends_hit) == (1, 2, 1) and clean.accuracy_exact == 1.0

    # releasing across the first tick misses it (breaking combo); pressing again resumes tracking (cursor on the ball)
    dropped = judge_replay(bm, follow_slider(bm, lambda t: 0 if 1450.0 <= t < 1550.0 else KEY_K1))
    assert (dropped.n300, dropped.slider_ticks_hit, dropped.slider_ends_hit) == (1, 1, 1)
    assert abs(dropped.accuracy_exact - (300 + 30 + 150) / (300 + 60 + 150)) < 1e-9
    assert dropped.max_combo == 2  # head | tick 2, end


TWO_CIRCLES = """osu file format v14
[General]
Mode: 0
[Difficulty]
CircleSize: 4
OverallDifficulty: 5
ApproachRate: 9
[TimingPoints]
0,500,4,1,0,100,1,0
[HitObjects]
100,100,1000,1,0
400,300,1100,1,0
"""


def test_note_lock():
    from osuai.beatmap import Frame

    bm = parse_osu(TWO_CIRCLES)
    # skip the first circle and press only on the second one, on time: the hit counts, the earlier circle is missed
    frames = [Frame(0.0, 400.0, 300.0, 0), Frame(1100.0, 400.0, 300.0, KEY_K1), Frame(1130.0, 400.0, 300.0, 0)]
    r = judge_replay(bm, frames)
    assert (r.n300, r.n_miss) == (1, 1)
    # pressing the second circle before the first one's time is blocked (the press does nothing)
    frames = [Frame(0.0, 400.0, 300.0, 0), Frame(960.0, 400.0, 300.0, KEY_K1), Frame(980.0, 400.0, 300.0, 0)]
    r = judge_replay(bm, frames)
    assert (r.n300, r.n100, r.n50, r.n_miss) == (0, 0, 0, 2)


def test_slider_scoring():
    bm = parse_osu(TAIL_TEST_MAP)  # one slider 1000-2500 ms, ticks at 1500 and 2000
    end = bm.objects[0].end_time
    clean = judge_replay(bm, follow_slider(bm, lambda t: KEY_K1))
    assert (clean.n300, clean.slider_ticks_hit, clean.slider_ticks_total, clean.slider_ends_hit) == (1, 2, 2, 1)
    assert clean.accuracy_exact == 1.0 and clean.max_combo == clean.beatmap_max_combo == 4
    # release before the end: the tail is lost (150 off the accuracy) but the combo is not broken
    no_tail = judge_replay(bm, follow_slider(bm, lambda t: KEY_K1 if t < end - 300.0 else 0))
    assert no_tail.slider_ends_hit == 0 and no_tail.n300 == 1  # the head's timing is what counts as the 300
    assert abs(no_tail.accuracy_exact - (300 + 60) / (300 + 60 + 150)) < 1e-9
    assert no_tail.max_combo == 3 and no_tail.statistics.get("ignore_miss") == 1
