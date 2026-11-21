"""Unit and property tests for beatmap parser, timing/SV, L/P/B/C slider geometry, stacking, and mods."""

from __future__ import annotations

import math
import pytest

from osuai.beatmap import Vec2, build_slider_path, parse_mods, parse_osu

MINIMAL_OSU = """\ufeffosu file format v14
// comment line
[General]
AudioFilename: test.mp3
Mode: 0
StackLeniency: 0.7

[Metadata]
Title: Test Map
Artist: Tester
Creator: Mapper
Version: Hard

[Difficulty]
HPDrainRate: 6
CircleSize: 4
OverallDifficulty: 8
SliderMultiplier: 1.5
SliderTickRate: 2

[TimingPoints]
-100,500,4,1,0,100,1,0
500,-50
500,-200
1500,250,4,1,0,100,1,0

[HitObjects]
256,192,1000,1,0,0:0:0:0:
100,100,2000,2,0,L|400:100,2,300
256,192,4000,8,0,5500
"""


def test_parse_osu_bom_crlf_missing_ar_and_two_field_timing():
    raw_crlf = MINIMAL_OSU.replace("\n", "\r\n").encode("utf-8")
    bm = parse_osu(raw_crlf)

    assert bm.title == "Test Map"
    # Missing ApproachRate falls back to OD (§8.1)
    assert bm.difficulty.od == 8.0
    assert bm.difficulty.ar == 8.0
    assert math.isclose(bm.difficulty.circle_radius, 54.4 - 4.48 * 4.0)

    # Duplicate time at 500ms: later line (-200 -> sv=0.5) wins (§8.2)
    bl0, sv0 = bm.timing_at(100.0)
    assert bl0 == 500.0 and sv0 == 1.0
    bl1, sv1 = bm.timing_at(600.0)
    assert bl1 == 500.0 and math.isclose(sv1, 0.5)
    # Red line at 1500ms changes BPM to 250ms/beat and resets SV to 1.0
    bl2, sv2 = bm.timing_at(1600.0)
    assert bl2 == 250.0 and math.isclose(sv2, 1.0)


def test_reject_non_standard_mode():
    taiko = MINIMAL_OSU.replace("Mode: 0", "Mode: 1")
    with pytest.raises(ValueError, match="Mode=1"):
        parse_osu(taiko)


def test_slider_curves_linear_bezier_circle_catmull_and_length_parity():
    start = Vec2(100.0, 100.0)
    # Linear
    pts_l, cum_l, len_l = build_slider_path(start, "L|100:100|400:100", 300.0)
    assert math.isclose(len_l, 300.0)
    assert pts_l[-1].dist(Vec2(400.0, 100.0)) < 1e-3

    # Perfect circle arc (quarter circle radius 100 -> arc length ~157.08)
    s_arc = Vec2(200.0, 100.0)
    mid_arc = Vec2(100.0 + 100.0 * math.cos(math.pi / 4), 100.0 + 100.0 * math.sin(math.pi / 4))
    end_arc = Vec2(100.0, 200.0)
    curve_p = f"P|{mid_arc.x:.3f}:{mid_arc.y:.3f}|{end_arc.x:.3f}:{end_arc.y:.3f}"
    pts_p, cum_p, len_p = build_slider_path(s_arc, curve_p, 157.08)
    assert abs(cum_p[-1] - 157.08) / 157.08 < 0.01
    # Midpoint should be at distance ~100 from center (100,100)
    mid_sampled = pts_p[len(pts_p) // 2]
    assert abs(mid_sampled.dist(Vec2(100.0, 100.0)) - 100.0) < 1.5

    # Collinear perfect circle falls back to line without crash (§8.7)
    pts_col, _, len_col = build_slider_path(start, "P|200:100|300:100", 200.0)
    assert math.isclose(len_col, 200.0)
    assert pts_col[-1].dist(Vec2(300.0, 100.0)) < 1e-3

    # Bezier with duplicate control point (red anchor split)
    pts_b, cum_b, len_b = build_slider_path(start, "B|200:100|200:100|200:200", 200.0)
    assert math.isclose(len_b, 200.0)
    assert pts_b[-1].dist(Vec2(200.0, 200.0)) < 1.0

    # Catmull-Rom
    pts_c, cum_c, len_c = build_slider_path(start, "C|180:140|260:100|340:140", 240.0)
    assert math.isclose(len_c, 240.0)
    assert len(pts_c) > 10


def test_repeat_slider_end_parity_and_monotonic_ticks():
    bm = parse_osu(MINIMAL_OSU)
    slider = bm.objects[1]
    assert slider.kind == "slider"
    assert slider.slides == 2
    # slides=2 (even) returns to start position (§8.4)
    assert slider.end_pos.dist(Vec2(100.0, 100.0)) < 1e-3
    assert len(slider.repeat_times) == 1
    assert slider.repeat_positions[0].dist(Vec2(400.0, 100.0)) < 1e-3
    # Ticks monotonic and strictly inside slider duration
    assert slider.tick_times == sorted(slider.tick_times)
    assert all(slider.time < tt < slider.end_time for tt in slider.tick_times)


def test_mods_hr_ez_dt_ht_and_bitmask():
    bm = parse_osu(MINIMAL_OSU)
    hr_bm = bm.with_mods("HR,DT")
    assert math.isclose(hr_bm.difficulty.cs, min(4.0 * 1.4, 10.0))
    assert math.isclose(hr_bm.difficulty.od, 10.0)  # 8 * 1.4 capped at 10
    assert math.isclose(hr_bm.difficulty.clock_rate, 1.5)
    assert hr_bm.difficulty.mods_bitmask == (16 | 64)

    ez_bm = bm.with_mods(["EZ", "HT"])
    assert math.isclose(ez_bm.difficulty.cs, 2.0)
    assert math.isclose(ez_bm.difficulty.od, 4.0)
    assert math.isclose(ez_bm.difficulty.clock_rate, 0.75)

    with pytest.raises(ValueError):
        parse_mods("HR,EZ")
    with pytest.raises(ValueError):
        parse_mods("RX")
