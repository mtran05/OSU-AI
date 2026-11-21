"""Tests for the synthetic map generator and the `generate` command."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import pytest

from osuai.beatmap import parse_osu
from osuai.judge import HitWindows, hit_errors, judge_replay, unstable_rate
from osuai.replay import quantize_frames
from osuai.synth import generate_map, parse_ratio, parse_star_range
from tests.helpers import perfect_inputs


def test_parse_ratio_and_stars():
    assert parse_ratio("60/30/10") == pytest.approx((0.6, 0.3, 0.1))
    assert parse_ratio("3/1/0") == pytest.approx((0.75, 0.25, 0.0))
    assert parse_star_range("5") == (5.0, 5.0) and parse_star_range("3-6.5") == (3.0, 6.5)
    for bad in ("60/40", "0/0/0", "-1/2/3"):
        with pytest.raises(ValueError):
            parse_ratio(bad)
    with pytest.raises(ValueError):
        parse_star_range("6-3")


@pytest.mark.parametrize("stars", [2.0, 5.0, 7.0])
def test_hits_the_target_star_rating_with_the_ratio(stars: float):
    r = generate_map(stars, parse_ratio("60/30/10"), length_s=60, seed=3)
    assert abs(r.star_rating - stars) <= 0.05
    n = r.n_objects
    assert abs(r.n_circles / n - 0.6) < 0.03 and abs(r.n_sliders / n - 0.3) < 0.03 and abs(r.n_spinners / n - 0.1) < 0.03


def test_single_object_type_maps_are_pure_and_inside_the_playfield():
    sliders = generate_map(4.0, (0.0, 1.0, 0.0), length_s=60, seed=1)
    assert sliders.n_circles == sliders.n_spinners == 0 and sliders.n_sliders > 20
    bm = parse_osu(sliders.text)
    for o in bm.objects:
        for t in (o.time, (o.time + o.end_time) / 2, o.end_time):
            p = o.slider_pos_at_time(t)
            assert 0 <= p.x <= 512 and 0 <= p.y <= 384
    circles = generate_map(5.0, (1.0, 0.0, 0.0), length_s=60, seed=2)
    assert circles.n_sliders == circles.n_spinners == 0 and abs(circles.star_rating - 5.0) <= 0.05
    spinners = generate_map(3.0, (0.0, 0.0, 1.0), length_s=60, seed=4)
    assert spinners.n_circles == spinners.n_sliders == 0 and spinners.n_spinners > 5


def test_generated_maps_are_playable_to_an_ss():
    r = generate_map(4.0, parse_ratio("60/30/10"), length_s=40, seed=5)
    bm = parse_osu(r.text)
    res = judge_replay(bm, quantize_frames(perfect_inputs(bm)))
    assert res.n_miss == 0 and res.accuracy == 100.0 and res.max_combo == res.beatmap_max_combo
    # every circle and slider head has one hit error, all inside the 300 window
    errs = hit_errors(bm, res)
    assert len(errs) == sum(o.kind in ("circle", "slider") for o in bm.objects)
    assert max(abs(e) for e in errs) <= HitWindows.for_od(bm.difficulty.od).great and unstable_rate(errs) < 100


def test_generate_command_writes_a_named_folder_and_merges_its_manifest(tmp_path: Path, monkeypatch):
    from osuai.cli import main

    monkeypatch.chdir(tmp_path)

    def run(*argv: str) -> dict:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(["generate", *argv])
        return json.loads(buf.getvalue()[buf.getvalue().index("{"):])

    out = run("--stars", "2-4", "--count", "3", "--name", "drills", "--only", "circles", "--length", "30", "--seed", "1")
    assert out["maps_generated"] == 3 and out["maps_manifest"] == "maps/drills/manifest.json"
    out = run("--stars", "3", "--count", "2", "--name", "drills", "--ratio", "50/50/0", "--length", "30", "--seed", "10")
    assert out["maps_in_folder"] == 5
    entries = json.loads(Path("maps/drills/manifest.json").read_text(encoding="utf-8"))["maps"]
    assert all(Path(e["path"]).exists() and e["path"].startswith("maps/drills/") for e in entries)
    assert [e["star_rating"] for e in entries] == sorted((e["star_rating"] for e in entries), reverse=True)
    # --only picks the folder when --name is not given; mixed maps default to maps/synthetic
    out = run("--stars", "3", "--count", "1", "--only", "sliders", "--length", "20", "--seed", "20")
    assert out["maps_manifest"] == "maps/sliders/manifest.json"
    out = run("--stars", "3", "--count", "1", "--length", "20", "--seed", "30")
    assert out["maps_manifest"] == "maps/synthetic/manifest.json"
    for bad in (["--name", "imported"], ["--name", "a/b"], ["--only", "circles", "--ratio", "1/1/1"]):
        with pytest.raises(ValueError):
            main(["generate", "--stars", "3", "--count", "1", *bad])


def test_training_reads_several_manifests_and_folders(tmp_path: Path):
    from osuai.dqn import load_manifest_entries

    for name, stars in (("a", 2.0), ("b", 5.0)):
        r = generate_map(stars, (1.0, 0.0, 0.0), length_s=20, seed=int(stars))
        d = tmp_path / name
        d.mkdir()
        (d / f"{r.md5}.osu").write_text(r.text, encoding="utf-8")
        entry = {"path": f"{r.md5}.osu", "md5": r.md5, "star_rating": round(r.star_rating, 2)}
        (d / "manifest.json").write_text(json.dumps({"maps": [entry]}), encoding="utf-8")
    both = load_manifest_entries([tmp_path / "a", tmp_path / "b" / "manifest.json", tmp_path / "a"])
    assert len(both) == 2  # folders or manifest files; a map listed twice is kept once
    assert len(load_manifest_entries([tmp_path / "a", tmp_path / "b"], max_stars=3.0)) == 1
