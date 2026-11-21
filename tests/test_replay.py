"""Unit tests for .osr binary writer/reader and cross-verification against osrparse."""

from __future__ import annotations

from pathlib import Path
import osrparse

from osuai.beatmap import parse_osu
from osuai.judge import judge_replay
from osuai.replay import decode_osr, decode_osu_string, encode_osu_string, write_osr
from tests.helpers import COMPLEX_MAP, perfect_inputs


def test_osu_string_roundtrip():
    for s in ("", "osuai", "日本語プレイヤー_123", "a" * 300):
        enc = encode_osu_string(s)
        dec, off = decode_osu_string(enc, 0)
        assert dec == s
        assert off == len(enc)


def test_osr_roundtrip_and_osrparse_compatibility(tmp_path: Path):
    bm = parse_osu(COMPLEX_MAP, mods="HR,DT")
    frames = perfect_inputs(bm)
    res = judge_replay(bm, frames)

    osr_file = tmp_path / "test_hrdt.osr"
    written = write_osr(osr_file, bm, frames, res, player_name="osuai_bot")

    decoded = decode_osr(osr_file)
    assert decoded.beatmap_md5 == bm.md5
    assert decoded.player_name == "osuai_bot"
    assert decoded.n300 == res.n300
    assert decoded.max_combo == res.max_combo
    assert decoded.perfect is True
    assert decoded.mods == bm.difficulty.mods_bitmask
    assert len(decoded.frames) == len(frames)

    # Re-judge decoded frames (integer-rounded in .osr) and verify 100% FC holds
    res_after = judge_replay(bm, decoded.frames)
    assert res_after.accuracy == 100.0
    assert res_after.max_combo == bm.max_combo

    # Verify third-party osrparse parses the written .osr cleanly (§11.3)
    parsed_ext = osrparse.Replay.from_path(osr_file)
    assert parsed_ext.beatmap_hash == bm.md5
    assert parsed_ext.username == "osuai_bot"
    assert parsed_ext.count_300 == res.n300
    assert parsed_ext.max_combo == res.max_combo
    assert len(parsed_ext.replay_data) == len(frames)
    assert written.replay_md5 == parsed_ext.replay_hash


def test_quantized_frames_are_exactly_what_the_osr_stores(tmp_path: Path):
    """Judging quantize_frames(frames) must give the same result as re-judging the written .osr."""
    from osuai.beatmap import Frame
    from osuai.replay import quantize_frames

    bm = parse_osu(COMPLEX_MAP)
    # frames shifted by fractional milliseconds / sub-pixel offsets, like the agent's 60 Hz frames
    frames = [Frame(time=f.time + 0.37, x=f.x + 0.123456, y=f.y - 0.0004, keys=f.keys) for f in perfect_inputs(bm)]
    q = quantize_frames(frames)
    res = judge_replay(bm, q)
    osr_file = tmp_path / "q.osr"
    write_osr(osr_file, bm, q, res, player_name="osuai_bot")
    dec = decode_osr(osr_file)
    assert [(f.time, f.x, f.y, f.keys) for f in dec.frames] == [(f.time, f.x, f.y, f.keys) for f in q]
    again = judge_replay(bm, dec.frames)
    assert (again.n300, again.n100, again.n50, again.n_miss) == (dec.n300, dec.n100, dec.n50, dec.n_miss)


def test_osr_is_lazer_format_with_full_statistics(tmp_path: Path):
    """Replays carry lazer's score-info block: statistics incl. slider ticks / ends, maximums and the mod list."""
    import json as _json

    from osuai.cli import main
    from osuai.replay import LAZER_REPLAY_VERSION
    from osuai.replay import quantize_frames

    bm = parse_osu(COMPLEX_MAP)
    frames = quantize_frames(perfect_inputs(bm))
    res = judge_replay(bm, frames)
    out = tmp_path / "standard.osr"
    write_osr(out, bm, frames, res, player_name="osuai_dqn")
    rep = decode_osr(out)
    assert rep.game_version == LAZER_REPLAY_VERSION and rep.score_info is not None
    stats, maxima = rep.score_info["statistics"], rep.score_info["maximum_statistics"]
    assert stats == maxima  # perfect play
    assert (rep.n300, rep.n_miss, rep.score) == (res.n300, res.n_miss, res.score)
    assert rep.mod_acronyms == [] and stats["slider_tail_hit"] == 2 and stats["large_tick_hit"] and stats["small_bonus"]
    # the judge command reads it back to the same result
    map_file = tmp_path / "m.osu"
    map_file.write_text(COMPLEX_MAP, encoding="utf-8")
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main(["judge", str(map_file), str(out)])
    row = _json.loads(buf.getvalue())
    assert (row["accuracy"], row["max_combo"]) == (res.accuracy, f"{res.max_combo}/{res.beatmap_max_combo}")
