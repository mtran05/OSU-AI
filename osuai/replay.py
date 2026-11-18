"""Binary .osr replay reader and writer in osu!lazer's format.

Besides the stable-style header (300/100/50/miss counts, score, combo, mods) and the LZMA-compressed cursor frames, a
lazer replay (version >= 30000001) carries a compressed JSON "score info" block with the full statistics: slider
ticks and ends, spinner ticks, the maximum possible counts and the mod list. osu!lazer reads that block
on import, so its results screen shows every statistic right away.
"""

from __future__ import annotations

import hashlib
import json
import lzma
import struct
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from osuai.beatmap import Beatmap, Frame
from osuai.judge import JudgeResult

UNIX_EPOCH_DOTNET_TICKS = 621355968000000000
LAZER_REPLAY_VERSION = 30000019  # osu!lazer's LegacyScoreEncoder.LATEST_VERSION
FIRST_SCORE_INFO_VERSION = 30000001  # replays from this version on carry the JSON score-info block


def current_dotnet_ticks() -> int:
    """Return current UTC time as .NET DateTime ticks (100-ns intervals since 0001-01-01)."""
    return UNIX_EPOCH_DOTNET_TICKS + int(datetime.now(timezone.utc).timestamp() * 10_000_000)


@dataclass(slots=True)
class ReplayData:
    game_mode: int
    game_version: int
    beatmap_md5: str
    player_name: str
    replay_md5: str
    n300: int
    n100: int
    n50: int
    n_geki: int
    n_katu: int
    n_miss: int
    score: int
    max_combo: int
    perfect: bool
    mods: int
    life_bar: str
    timestamp: int
    frames: list[Frame] = field(default_factory=list)
    online_score_id: int = 0
    rng_seed: int = 32767
    score_info: dict | None = None  # lazer's JSON score-info block (statistics, mods, ...)

    @property
    def mod_acronyms(self) -> list[str]:
        """Mods from the score-info block (e.g. ["HR"]), or [] when there is none."""
        return [str(m.get("acronym", "")).upper() for m in (self.score_info or {}).get("mods", [])]


def encode_uleb128(value: int) -> bytes:
    if value < 0:
        raise ValueError("ULEB128 value must be non-negative")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value != 0:
            byte |= 0x80
        out.append(byte)
        if value == 0:
            break
    return bytes(out)


def decode_uleb128(data: bytes, offset: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        if offset >= len(data):
            raise ValueError("Unexpected EOF while reading ULEB128")
        byte = data[offset]
        offset += 1
        result |= (byte & 0x7F) << shift
        if (byte & 0x80) == 0:
            break
        shift += 7
    return (result, offset)


def encode_osu_string(text: str) -> bytes:
    if not text:
        return b"\x00"
    raw = text.encode("utf-8")
    return b"\x0b" + encode_uleb128(len(raw)) + raw


def decode_osu_string(data: bytes, offset: int) -> tuple[str, int]:
    if offset >= len(data):
        raise ValueError("Unexpected EOF while reading osu!string tag")
    tag = data[offset]
    offset += 1
    if tag == 0x00:
        return ("", offset)
    if tag != 0x0B:
        raise ValueError(f"Invalid osu!string tag: 0x{tag:02x}")
    length, offset = decode_uleb128(data, offset)
    end = offset + length
    if end > len(data):
        raise ValueError("Unexpected EOF while reading osu!string payload")
    return (data[offset:end].decode("utf-8", errors="replace"), end)


def _fmt_coord(v: float) -> str:
    s = f"{v:.3f}".rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def quantize_frames(frames: list[Frame]) -> list[Frame]:
    """Return the frames exactly as they will be stored in (and decoded from) an .osr file.

    The .osr format stores time as whole milliseconds; cursor positions are kept to 3 decimals. Judge this list,
    not the raw frames, so the score written to the .osr header matches what any client re-judging it sees.
    """
    out: list[Frame] = []
    for f in sorted(frames, key=lambda fr: fr.time):
        out.append(Frame(time=float(int(round(f.time))), x=float(_fmt_coord(f.x)), y=float(_fmt_coord(f.y)), keys=int(f.keys)))
    return out


def _lzma_compress(raw: bytes) -> bytes:
    return lzma.compress(
        raw,
        format=lzma.FORMAT_ALONE,
        filters=[{"id": lzma.FILTER_LZMA1, "lc": 3, "lp": 0, "pb": 2, "dict_size": 1 << 21}],
    )


def compress_replay_frames(frames: list[Frame], rng_seed: int = 32767) -> bytes:
    """Delta-encode frames and compress with LZMA_ALONE."""
    sorted_frames = sorted(frames, key=lambda f: f.time)
    parts: list[str] = []
    prev_ms = 0
    for f in sorted_frames:
        t_ms = int(round(f.time))
        delta = t_ms - prev_ms
        prev_ms = t_ms
        parts.append(f"{delta}|{_fmt_coord(f.x)}|{_fmt_coord(f.y)}|{int(f.keys)}")
    parts.append(f"-12345|0|0|{int(rng_seed)},")
    return _lzma_compress(",".join(parts).encode("ascii"))


def decompress_replay_frames(compressed_data: bytes) -> tuple[list[Frame], int]:
    """Decompress LZMA replay payload into sorted Frame list and rng_seed."""
    if not compressed_data:
        return ([], 0)
    raw = lzma.decompress(compressed_data, format=lzma.FORMAT_ALONE).decode("ascii", errors="replace")
    frames: list[Frame] = []
    cur_ms = 0.0
    seed = 0
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        fields = token.split("|")
        if len(fields) < 4:
            continue
        dt = float(fields[0])
        x = float(fields[1])
        y = float(fields[2])
        keys = int(float(fields[3]))
        if int(round(dt)) == -12345:
            seed = keys
            continue
        cur_ms += dt
        frames.append(Frame(time=cur_ms, x=x, y=y, keys=keys))
    return (frames, seed)


def compute_replay_md5(
    max_combo: int,
    perfect: bool,
    player_name: str,
    beatmap_md5: str,
    n300: int,
    n100: int,
    n50: int,
    n_geki: int,
    n_katu: int,
    n_miss: int,
    score: int,
    mods: int,
) -> str:
    payload = (
        f"{max_combo}osu{player_name}{beatmap_md5}{n300}{n100}{n50}{n_geki}{n_katu}{n_miss}"
        f"{score}{'True' if perfect else 'False'}{mods}"
    )
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def encode_osr(replay: ReplayData) -> bytes:
    """Serialize ReplayData to .osr bytes."""
    compressed = compress_replay_frames(replay.frames, rng_seed=replay.rng_seed)
    replay_md5 = replay.replay_md5 or compute_replay_md5(
        max_combo=replay.max_combo,
        perfect=replay.perfect,
        player_name=replay.player_name,
        beatmap_md5=replay.beatmap_md5,
        n300=replay.n300,
        n100=replay.n100,
        n50=replay.n50,
        n_geki=replay.n_geki,
        n_katu=replay.n_katu,
        n_miss=replay.n_miss,
        score=replay.score,
        mods=replay.mods,
    )

    buf = bytearray()
    buf.extend(struct.pack("<Bi", replay.game_mode, replay.game_version))
    buf.extend(encode_osu_string(replay.beatmap_md5))
    buf.extend(encode_osu_string(replay.player_name))
    buf.extend(encode_osu_string(replay_md5))
    buf.extend(
        struct.pack(
            "<HHHHHHiHBi",
            min(max(replay.n300, 0), 65535),
            min(max(replay.n100, 0), 65535),
            min(max(replay.n50, 0), 65535),
            min(max(replay.n_geki, 0), 65535),
            min(max(replay.n_katu, 0), 65535),
            min(max(replay.n_miss, 0), 65535),
            int(replay.score),
            min(max(replay.max_combo, 0), 65535),
            1 if replay.perfect else 0,
            int(replay.mods),
        )
    )
    buf.extend(encode_osu_string(replay.life_bar))
    buf.extend(struct.pack("<qi", int(replay.timestamp), len(compressed)))
    buf.extend(compressed)
    buf.extend(struct.pack("<q", int(replay.online_score_id)))
    if replay.game_version >= FIRST_SCORE_INFO_VERSION and replay.score_info is not None:
        info = _lzma_compress(json.dumps(replay.score_info, separators=(",", ":")).encode("utf-8"))
        buf.extend(struct.pack("<i", len(info)))
        buf.extend(info)
    return bytes(buf)


def decode_osr(source: bytes | str | Path) -> ReplayData:
    """Deserialize .osr file or bytes into ReplayData."""
    data = Path(source).read_bytes() if isinstance(source, (str, Path)) else source
    offset = 0
    game_mode, game_version = struct.unpack_from("<Bi", data, offset)
    offset += 5
    beatmap_md5, offset = decode_osu_string(data, offset)
    player_name, offset = decode_osu_string(data, offset)
    replay_md5, offset = decode_osu_string(data, offset)

    n300, n100, n50, n_geki, n_katu, n_miss, score, max_combo, perfect_byte, mods = struct.unpack_from(
        "<HHHHHHiHBi", data, offset
    )
    offset += 23
    life_bar, offset = decode_osu_string(data, offset)
    timestamp, replay_len = struct.unpack_from("<qi", data, offset)
    offset += 12

    if replay_len < 0 or offset + replay_len > len(data):
        raise ValueError(f"Invalid compressed replay length: {replay_len}")
    compressed = data[offset : offset + replay_len]
    offset += replay_len

    online_score_id = 0
    if offset + 8 <= len(data):
        (online_score_id,) = struct.unpack_from("<q", data, offset)
        offset += 8
    score_info = None
    if game_version >= FIRST_SCORE_INFO_VERSION and offset + 4 <= len(data):
        (info_len,) = struct.unpack_from("<i", data, offset)
        offset += 4
        if 0 < info_len <= len(data) - offset:
            raw = lzma.decompress(data[offset : offset + info_len], format=lzma.FORMAT_ALONE)
            score_info = json.loads(raw.decode("utf-8"))

    frames, seed = decompress_replay_frames(compressed)
    return ReplayData(
        game_mode=game_mode,
        game_version=game_version,
        beatmap_md5=beatmap_md5,
        player_name=player_name,
        replay_md5=replay_md5,
        n300=n300,
        n100=n100,
        n50=n50,
        n_geki=n_geki,
        n_katu=n_katu,
        n_miss=n_miss,
        score=score,
        max_combo=max_combo,
        perfect=bool(perfect_byte),
        mods=mods,
        life_bar=life_bar,
        timestamp=timestamp,
        frames=frames,
        online_score_id=online_score_id,
        rng_seed=seed,
        score_info=score_info,
    )


def write_osr(
    path: str | Path,
    beatmap: Beatmap,
    frames: list[Frame],
    judge_result: JudgeResult,
    player_name: str = "osuai",
) -> ReplayData:
    """Write a lazer-format .osr: the stable-style header plus a score-info block with the judge's full statistics."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mods_bitmask = beatmap.difficulty.mods_bitmask
    score_val = int(judge_result.score)
    acronyms = [m for m in beatmap.difficulty.mods if m != "NM"]
    score_info = {
        "online_id": -1,
        "mods": [{"acronym": a} for a in acronyms],
        "statistics": {k: v for k, v in judge_result.statistics.items() if v},
        "maximum_statistics": {k: v for k, v in judge_result.maximum_statistics.items() if v},
        "client_version": "",
        "user_id": -1,
        "total_score_without_mods": score_val,
        "pauses": [],
    }
    replay_md5 = hashlib.md5(f"lazer-{player_name}-{current_dotnet_ticks()}".encode("utf-8")).hexdigest()
    replay = ReplayData(
        game_mode=0,
        game_version=LAZER_REPLAY_VERSION,
        beatmap_md5=beatmap.md5,
        player_name=player_name,
        replay_md5=replay_md5,
        n300=judge_result.n300,
        n100=judge_result.n100,
        n50=judge_result.n50,
        n_geki=0,
        n_katu=0,
        n_miss=judge_result.n_miss,
        score=score_val,
        max_combo=judge_result.max_combo,
        perfect=judge_result.perfect,
        mods=mods_bitmask,
        life_bar="",
        timestamp=current_dotnet_ticks(),
        frames=frames,
        score_info=score_info,
    )
    out_path.write_bytes(encode_osr(replay))
    return replay
