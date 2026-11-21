"""Tests for the animated GIF preview: encoder round trip (independent decoder) and the render --gif CLI."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from osuai.cli import main
from osuai.render import GifWriter
from tests.helpers import COMPLEX_MAP


def _lzw_decode(data: bytes, min_code_size: int) -> bytes:
    """Reference GIF LZW decoder (standard rules, written independently of the encoder)."""
    clear = 1 << min_code_size
    eoi = clear + 1
    bitpos = 0
    total_bits = len(data) * 8
    as_int = int.from_bytes(data, "little")
    out = bytearray()
    table: list[bytes] = []
    cs = min_code_size + 1
    prev: bytes | None = None
    while bitpos + cs <= total_bits:
        code = (as_int >> bitpos) & ((1 << cs) - 1)
        bitpos += cs
        if code == clear:
            table = [bytes((i,)) for i in range(clear)] + [b"", b""]
            cs = min_code_size + 1
            prev = None
            continue
        if code == eoi:
            break
        if prev is None:
            entry = table[code]
        else:
            entry = table[code] if code < len(table) else prev + prev[:1]
            table.append(prev + entry[:1])
        out += entry
        prev = entry
        if len(table) == (1 << cs) and cs < 12:
            cs += 1
    return bytes(out)


def _decode_gif(path: Path) -> tuple[list[np.ndarray], list[int], int]:
    """Return composited frames, per-frame delays (cs) and the NETSCAPE loop count."""
    b = path.read_bytes()
    assert b[:6] == b"GIF89a"
    w, h, flags = struct.unpack("<HHB", b[6:11])
    pos = 13 + (3 * (2 ** ((flags & 7) + 1)) if flags & 0x80 else 0)
    canvas = np.zeros((h, w), dtype=np.uint8)
    frames, delays, loop, delay = [], [], None, 0

    def sub_blocks(p: int) -> tuple[bytes, int]:
        buf = bytearray()
        while b[p]:
            buf += b[p + 1 : p + 1 + b[p]]
            p += 1 + b[p]
        return (bytes(buf), p + 1)

    while True:
        tag = b[pos]
        if tag == 0x3B:
            break
        if tag == 0x21:
            label = b[pos + 1]
            payload, pos = sub_blocks(pos + 2)
            if label == 0xF9:
                delay = struct.unpack("<H", payload[1:3])[0]
            elif label == 0xFF and payload.startswith(b"NETSCAPE2.0"):
                loop = struct.unpack("<H", payload[12:14])[0]
        elif tag == 0x2C:
            x, y, fw, fh, _ = struct.unpack("<HHHHB", b[pos + 1 : pos + 10])
            mcs = b[pos + 10]
            data, pos = sub_blocks(pos + 11)
            pix = np.frombuffer(_lzw_decode(data, mcs), dtype=np.uint8)
            assert pix.size == fw * fh
            canvas = canvas.copy()
            canvas[y : y + fh, x : x + fw] = pix.reshape(fh, fw)
            frames.append(canvas)
            delays.append(delay)
        else:
            raise AssertionError(f"unexpected GIF block 0x{tag:02x}")
    return (frames, delays, loop)


def test_gif_writer_round_trip(tmp_path: Path):
    rng = np.random.default_rng(1)
    frames = [rng.integers(0, 256, size=(30, 40), dtype=np.uint8)]  # incompressible: 12-bit codes + table resets
    for k in range(1, 9):
        f = frames[-1].copy()
        if k != 4:  # frame 4 repeats frame 3 and must be merged into a longer delay
            y, x = rng.integers(0, 25), rng.integers(0, 35)
            f[y : y + 5, x : x + 5] = rng.integers(0, 256)
        frames.append(f)
    path = tmp_path / "a.gif"
    with GifWriter(path, fps=30.0, upscale=2) as gif:
        for f in frames:
            gif.add(f)
    assert gif.n_frames == len(frames)

    decoded, delays, loop = _decode_gif(path)
    unique = [f for i, f in enumerate(frames) if i == 0 or not np.array_equal(f, frames[i - 1])]
    assert len(decoded) == len(unique)
    for got, want in zip(decoded, unique):
        assert np.array_equal(got, np.repeat(np.repeat(want, 2, axis=0), 2, axis=1))
    assert loop == 0
    assert sum(delays) in (29, 30, 31)  # 9 frames at 30 fps ~= 30 cs in total (merged frame holds longer)
    assert max(delays) >= 6


def _render(tmp_path: Path, *extra: str, capsys) -> dict:
    map_file = tmp_path / "m.osu"
    map_file.write_text(COMPLEX_MAP, encoding="utf-8")
    assert main(["render", str(map_file), "--gif", "--upscale", "1", *extra]) == 0
    return json.loads(capsys.readouterr().out)


def test_render_gif_length_is_capped_at_song_end(tmp_path: Path, capsys):
    out = tmp_path / "x.gif"
    info = _render(tmp_path, "--seconds", "2", "--fps", "20", "--out", str(out), capsys=capsys)
    assert info["frames"] == 41 and info["fps"] == 20.0 and not info["capped_at_song_end"]
    frames, _, _ = _decode_gif(out)
    assert frames[0].shape == (76, 96)

    full = _render(tmp_path, "--seconds", "full", "--out", str(tmp_path / "full.gif"), capsys=capsys)
    capped = _render(tmp_path, "--seconds", "999", "--out", str(tmp_path / "cap.gif"), capsys=capsys)
    assert capped["capped_at_song_end"] and not full["capped_at_song_end"]
    assert capped["frames"] == full["frames"]
    assert abs(full["seconds"] - full["song_seconds_from_start"]) < 0.1


def test_render_gif_rejects_bad_seconds(tmp_path: Path):
    map_file = tmp_path / "m.osu"
    map_file.write_text(COMPLEX_MAP, encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["render", str(map_file), "--gif", "--seconds", "soon"])
