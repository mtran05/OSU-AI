"""Grayscale playfield simulator.

Renders what the osu! client would show on screen (approach circles, hit circles, slider bodies, slider ball,
follow circle, spinners, cursor, key overlay) as a small grayscale image. This is the ONLY observation the DQN
agent receives: no object coordinates, timings or other game-state numbers are ever fed to the network.

Intensity conventions (0..1, stored as uint8 0..255). Every playfield element is drawn at <= 0.85 so the cursor
(1.0) is always the brightest thing on screen:
    spinner area 0.12 (rising to 0.32 as the required spins are reached), slider body 0.20, hit circle fill 0.35,
    follow circle 0.50, slider border 0.55, spinner ring / centre / rotation line 0.60, hit circle border 0.70,
    slider ball 0.75, approach circle 0.85, cursor 1.00
A spinner fills its whole area (so a running spinner never looks like an empty screen), shows a ring shrinking
over its duration (time left), and a line from its centre that turns with the rotation that counts toward it, as
lazer's spinner disc turns: it only moves while a key is held, so the frame shows whether spinning registers.
A HUD strip under the playfield shows the K1 / K2 key overlay (lit while the key is held).
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

import numpy as np

from osuai.beatmap import PLAYFIELD_H, PLAYFIELD_W, Beatmap, HitObject

KIND_CIRCLE = 0
KIND_SLIDER = 1
KIND_SPINNER = 2
KIND_CODES = {"circle": KIND_CIRCLE, "slider": KIND_SLIDER, "spinner": KIND_SPINNER}

I_SLIDER_BODY = 0.20
I_CIRCLE_FILL = 0.35
I_FOLLOW = 0.50
I_SLIDER_BORDER = 0.55
I_SPINNER = 0.60
I_SPINNER_FILL = 0.12  # the spinner's area at the start
I_SPINNER_FILL_DONE = 0.32  # ... once the spins a 300 needs are reached
I_CIRCLE_BORDER = 0.70
I_BALL = 0.75
I_APPROACH = 0.85
I_CURSOR = 1.00

SPINNER_MAX_R = 180.0
SPINNER_MIN_R = 16.0


@dataclass(frozen=True, slots=True)
class RenderConfig:
    width: int = 96  # playfield width in pixels; height follows the 4:3 playfield
    hud_rows: int = 4

    @property
    def scale(self) -> float:
        return self.width / PLAYFIELD_W

    @property
    def field_height(self) -> int:
        return int(round(PLAYFIELD_H * self.scale))

    @property
    def height(self) -> int:
        return self.field_height + self.hud_rows

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)


def fade_in_ms(preempt: float) -> float:
    """Lazer hit object fade-in duration."""
    return 400.0 * min(1.0, preempt / 450.0)


class FrameRenderer:
    """Rasterizes one beatmap at a given map time into a grayscale uint8 frame."""

    def __init__(self, beatmap: Beatmap, cfg: RenderConfig | None = None) -> None:
        self.bm = beatmap
        self.cfg = cfg or RenderConfig()
        d = beatmap.difficulty
        self.preempt = max(d.preempt, 150.0)
        self.fade_in = fade_in_ms(self.preempt)
        self.w50 = d.w50
        self.s = self.cfg.scale
        self.r_px = d.circle_radius * self.s
        self.follow_px = d.follow_radius * self.s
        self.fw = self.cfg.width
        self.fh = self.cfg.field_height
        self.cursor_px = max(1.3, 7.0 * self.s)
        self.ring_w = 1.0  # ring stroke width in pixels
        self._slider_cache: dict[int, tuple[int, int, int, int, np.ndarray]] = {}
        self._appear = [o.time - self.preempt for o in beatmap.objects]  # for render_map
        self._head_pending = np.zeros(len(beatmap.objects), dtype=bool)

    # ------------------------------------------------------------------ primitives
    def _to_px(self, x: float, y: float) -> tuple[float, float]:
        return (x * self.s - 0.5, y * self.s - 0.5)

    def _bbox(self, u: float, v: float, reach: float) -> tuple[int, int, int, int] | None:
        x0 = max(int(math.floor(u - reach)), 0)
        x1 = min(int(math.ceil(u + reach)) + 1, self.fw)
        y0 = max(int(math.floor(v - reach)), 0)
        y1 = min(int(math.ceil(v + reach)) + 1, self.fh)
        if x0 >= x1 or y0 >= y1:
            return None
        return (y0, y1, x0, x1)

    def _dist(self, bb: tuple[int, int, int, int], u: float, v: float) -> np.ndarray:
        y0, y1, x0, x1 = bb
        dy = np.arange(y0, y1, dtype=np.float32)[:, None] - v
        dx = np.arange(x0, x1, dtype=np.float32)[None, :] - u
        return np.sqrt(dx * dx + dy * dy)

    def _blend(self, canvas: np.ndarray, bb: tuple[int, int, int, int], val: np.ndarray) -> None:
        y0, y1, x0, x1 = bb
        np.maximum(canvas[y0:y1, x0:x1], val, out=canvas[y0:y1, x0:x1])

    def _disc(self, canvas: np.ndarray, x: float, y: float, r_px: float, inten: float) -> None:
        u, v = self._to_px(x, y)
        bb = self._bbox(u, v, r_px + 1.0)
        if bb is None or inten <= 0.0:
            return
        d = self._dist(bb, u, v)
        self._blend(canvas, bb, np.clip(r_px - d + 0.5, 0.0, 1.0) * inten)

    def _ring(self, canvas: np.ndarray, x: float, y: float, r_px: float, inten: float) -> None:
        u, v = self._to_px(x, y)
        bb = self._bbox(u, v, r_px + self.ring_w + 1.0)
        if bb is None or inten <= 0.0:
            return
        d = self._dist(bb, u, v)
        self._blend(canvas, bb, np.clip(self.ring_w * 0.5 + 0.5 - np.abs(d - r_px), 0.0, 1.0) * inten)

    def _segment(self, canvas: np.ndarray, x0: float, y0: float, x1: float, y1: float, inten: float) -> None:
        """An anti-aliased line of ring width from (x0, y0) to (x1, y1), in osu!px."""
        (u0, v0), (u1, v1) = self._to_px(x0, y0), self._to_px(x1, y1)
        reach = self.ring_w + 1.0
        bx0 = max(int(math.floor(min(u0, u1) - reach)), 0)
        bx1 = min(int(math.ceil(max(u0, u1) + reach)) + 1, self.fw)
        by0 = max(int(math.floor(min(v0, v1) - reach)), 0)
        by1 = min(int(math.ceil(max(v0, v1) + reach)) + 1, self.fh)
        if bx0 >= bx1 or by0 >= by1 or inten <= 0.0:
            return
        gy, gx = np.mgrid[by0:by1, bx0:bx1].astype(np.float32)
        du, dv = u1 - u0, v1 - v0
        t = np.clip(((gx - u0) * du + (gy - v0) * dv) / max(du * du + dv * dv, 1e-9), 0.0, 1.0)
        d = np.hypot(gx - (u0 + t * du), gy - (v0 + t * dv))
        self._blend(canvas, (by0, by1, bx0, bx1), np.clip(self.ring_w * 0.5 + 0.5 - d, 0.0, 1.0) * inten)

    def _hit_circle(self, canvas: np.ndarray, x: float, y: float, inten: float) -> None:
        u, v = self._to_px(x, y)
        bb = self._bbox(u, v, self.r_px + self.ring_w + 1.0)
        if bb is None or inten <= 0.0:
            return
        d = self._dist(bb, u, v)
        fill = np.clip(self.r_px - d + 0.5, 0.0, 1.0) * I_CIRCLE_FILL
        border = np.clip(self.ring_w * 0.5 + 0.5 - np.abs(d - self.r_px), 0.0, 1.0) * I_CIRCLE_BORDER
        self._blend(canvas, bb, np.maximum(fill, border) * inten)

    def _slider_mask(self, idx: int, obj: HitObject) -> tuple[int, int, int, int, np.ndarray] | None:
        if idx in self._slider_cache:
            return self._slider_cache[idx]
        pts = obj.path if obj.path else [obj.pos]
        # Resample the polyline at ~1 pixel spacing to bound the distance computation
        step = 1.0 / self.s
        keep = [pts[0]]
        acc = 0.0
        for a, b in zip(pts, pts[1:]):
            acc += a.dist(b)
            if acc >= step:
                keep.append(b)
                acc = 0.0
        if keep[-1] is not pts[-1]:
            keep.append(pts[-1])
        P = np.array([self._to_px(p.x, p.y) for p in keep], dtype=np.float32)  # (S+1, 2) as (u, v)
        reach = self.r_px + self.ring_w + 1.0
        x0 = max(int(math.floor(P[:, 0].min() - reach)), 0)
        x1 = min(int(math.ceil(P[:, 0].max() + reach)) + 1, self.fw)
        y0 = max(int(math.floor(P[:, 1].min() - reach)), 0)
        y1 = min(int(math.ceil(P[:, 1].max() + reach)) + 1, self.fh)
        if x0 >= x1 or y0 >= y1:
            self._slider_cache[idx] = None  # type: ignore[assignment]
            return None
        gy, gx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        pix = np.stack([gx.ravel(), gy.ravel()], axis=1)  # (K, 2)
        if len(P) == 1:
            dist = np.hypot(pix[:, 0] - P[0, 0], pix[:, 1] - P[0, 1])
        else:
            A = P[:-1]
            B = P[1:]
            AB = B - A
            denom = np.maximum((AB * AB).sum(axis=1), 1e-9)
            dist = np.full(len(pix), np.inf, dtype=np.float32)
            chunk = max(1, 200_000 // max(len(A), 1))
            for s0 in range(0, len(pix), chunk):
                q = pix[s0 : s0 + chunk, None, :]  # (k, 1, 2)
                t = np.clip(((q - A[None]) * AB[None]).sum(axis=2) / denom[None], 0.0, 1.0)
                proj = A[None] + t[..., None] * AB[None]
                dd = np.sqrt(((q - proj) ** 2).sum(axis=2)).min(axis=1)
                dist[s0 : s0 + chunk] = dd
        dist = dist.reshape(gy.shape)
        body = np.clip(self.r_px - dist + 0.5, 0.0, 1.0) * I_SLIDER_BODY
        border = np.clip(self.ring_w * 0.5 + 0.5 - np.abs(dist - self.r_px), 0.0, 1.0) * I_SLIDER_BORDER
        mask = np.maximum(body, border).astype(np.float32)
        entry = (y0, y1, x0, x1, mask)
        self._slider_cache[idx] = entry
        return entry

    # ------------------------------------------------------------------ frame
    def render(
        self,
        t: float,
        cursor_x: float | None,
        cursor_y: float | None,
        k1: bool,
        k2: bool,
        visible: list[int],
        head_pending: np.ndarray,
        tracking_slider: int = -1,
        spin_rotation: np.ndarray | None = None,
        spins_required: np.ndarray | None = None,
    ) -> np.ndarray:
        """Render the frame at map time t.

        cursor_x / cursor_y: cursor position, or None to draw no cursor.
        visible: indices of objects that are on screen (not yet finished) at time t.
        head_pending: per-object flag, True while a circle / slider head has not been judged yet.
        tracking_slider: index of the slider whose follow circle is shown (being tracked), or -1.
        spin_rotation / spins_required: per-object counted spinner rotation (signed radians) and the full spins a
            300 needs; they turn the spinner's line and brighten its area (no rotation when not given).
        """
        canvas = np.zeros((self.cfg.height, self.fw), dtype=np.float32)
        field = canvas[: self.fh]
        objs = self.bm.objects

        for i in visible:
            o = objs[i]
            alpha = min(max((t - (o.time - self.preempt)) / self.fade_in, 0.0), 1.0)
            if alpha <= 0.0:
                continue
            if o.kind == "spinner":
                if t < o.time:
                    frac = 0.0
                else:
                    frac = min(max((t - o.time) / max(o.duration, 1.0), 0.0), 1.0)
                rot = float(spin_rotation[i]) if spin_rotation is not None else 0.0
                req = float(spins_required[i]) if spins_required is not None else 0.0
                progress = 1.0 if req <= 0.0 and rot != 0.0 else min(abs(rot) / (2.0 * math.pi) / max(req, 1e-9), 1.0)
                fill = I_SPINNER_FILL + (I_SPINNER_FILL_DONE - I_SPINNER_FILL) * progress
                self._disc(field, o.pos.x, o.pos.y, SPINNER_MAX_R * self.s, fill * alpha)  # the whole spinner area
                r = (SPINNER_MAX_R - (SPINNER_MAX_R - SPINNER_MIN_R) * frac) * self.s
                self._ring(field, o.pos.x, o.pos.y, r, I_SPINNER * alpha)  # time left
                self._disc(field, o.pos.x, o.pos.y, max(1.0, 4.0 * self.s), I_SPINNER * alpha)
                ang = rot - math.pi / 2.0  # starts pointing up, turns with the counted rotation (osu! y points down)
                self._segment(field, o.pos.x, o.pos.y, o.pos.x + SPINNER_MAX_R * math.cos(ang),
                              o.pos.y + SPINNER_MAX_R * math.sin(ang), I_SPINNER * alpha)
                continue

            if o.kind == "slider":
                m = self._slider_mask(i, o)
                if m is not None:
                    y0, y1, x0, x1, mask = m
                    self._blend(field, (y0, y1, x0, x1), mask * alpha)
                if o.time <= t <= o.end_time:
                    b = o.slider_pos_at_time(t)
                    self._disc(field, b.x, b.y, self.r_px * 0.8, I_BALL)
                    if tracking_slider == i:
                        self._ring(field, b.x, b.y, self.follow_px, I_FOLLOW)

            if head_pending[i]:
                self._hit_circle(field, o.pos.x, o.pos.y, alpha)
                if t < o.time:
                    prog = min(max((t - (o.time - self.preempt)) / self.preempt, 0.0), 1.0)
                    self._ring(field, o.pos.x, o.pos.y, self.r_px * (1.0 + 3.0 * (1.0 - prog)), I_APPROACH * alpha)

        if cursor_x is not None and cursor_y is not None:
            self._disc(field, cursor_x, cursor_y, self.cursor_px, I_CURSOR)

        if self.cfg.hud_rows > 0:
            hud = canvas[self.fh :]
            kw = max(2, self.fw // 24)
            if k1:
                hud[:, self.fw - 2 * kw - 2 : self.fw - kw - 2] = 1.0
            if k2:
                hud[:, self.fw - kw - 1 : self.fw - 1] = 1.0
            hud[0, :] = np.maximum(hud[0, :], 0.15)  # faint separator line between playfield and HUD

        return (canvas * 255.0 + 0.5).astype(np.uint8)


    def render_map(self, t: float) -> np.ndarray:
        """The map alone at time t, with no player: every object is shown from when it appears until its own time
        (circles) or end time (sliders and spinners); no cursor, no follow circle, key overlay off."""
        objs = self.bm.objects
        hi = bisect.bisect_right(self._appear, t)
        visible = [i for i in range(hi) if t <= (objs[i].time if objs[i].kind == "circle" else objs[i].end_time)]
        for i in visible:
            self._head_pending[i] = objs[i].kind != "spinner" and t <= objs[i].time
        return self.render(t, None, None, False, False, visible, self._head_pending)


def write_png(path, image: np.ndarray) -> None:
    """Write a 2-D uint8 grayscale array as a PNG (no third-party imaging dependency)."""
    import struct
    import zlib
    from pathlib import Path

    img = np.ascontiguousarray(image, dtype=np.uint8)
    h, w = img.shape
    raw = b"".join(b"\x00" + img[row].tobytes() for row in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(png)


def contact_sheet(frames: list[np.ndarray], cols: int = 4, upscale: int = 3, pad: int = 2) -> np.ndarray:
    """Tile frames into one image (nearest-neighbour upscaled) for previewing what the agent sees."""
    if not frames:
        raise ValueError("No frames to tile")
    h, w = frames[0].shape
    rows = int(math.ceil(len(frames) / cols))
    H = rows * (h * upscale + pad) + pad
    W = cols * (w * upscale + pad) + pad
    sheet = np.full((H, W), 40, dtype=np.uint8)
    for k, f in enumerate(frames):
        r, c = divmod(k, cols)
        big = np.repeat(np.repeat(f, upscale, axis=0), upscale, axis=1)
        y = pad + r * (h * upscale + pad)
        x = pad + c * (w * upscale + pad)
        sheet[y : y + h * upscale, x : x + w * upscale] = big
    return sheet


def _lzw_encode(data: bytes, min_code_size: int = 8) -> bytes:
    """GIF-flavoured variable-width LZW (code-size growth and table reset follow giflib's encoder)."""
    clear = 1 << min_code_size
    eoi = clear + 1
    out = bytearray()
    acc = 0
    nbits = 0
    cs = min_code_size + 1
    next_code = eoi + 1
    table: dict[int, int] = {}

    def emit(code: int) -> None:
        nonlocal acc, nbits, cs
        acc |= code << nbits
        nbits += cs
        while nbits >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            nbits -= 8
        if next_code >= (1 << cs) and cs < 12:
            cs += 1

    emit(clear)
    if not data:
        emit(eoi)
        if nbits:
            out.append(acc & 0xFF)
        return bytes(out)
    prefix = data[0]
    for c in data[1:]:
        key = (prefix << 8) | c
        code = table.get(key)
        if code is not None:
            prefix = code
            continue
        emit(prefix)
        if next_code >= 4095:
            emit(clear)
            table.clear()
            next_code = eoi + 1
            cs = min_code_size + 1
        else:
            table[key] = next_code
            next_code += 1
        prefix = c
    emit(prefix)
    emit(eoi)
    if nbits:
        out.append(acc & 0xFF)
    return bytes(out)


class GifWriter:
    """Streaming grayscale animated-GIF writer (no third-party imaging dependency).

    Frames are nearest-neighbour upscaled and only the rectangle that changed since the previous frame is
    re-encoded, so long (full-song) animations stay small and fast to write. Usage:

        with GifWriter("out.gif", fps=30, upscale=3) as gif:
            for frame in frames:
                gif.add(frame)
    """

    def __init__(self, path, fps: float = 30.0, upscale: int = 3, loop: int = 0) -> None:
        from pathlib import Path

        if fps <= 0:
            raise ValueError("fps must be positive")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.upscale = max(1, int(upscale))
        self.loop = loop
        self.frame_cs = 100.0 / fps  # GIF delays are whole centiseconds; keep the remainder so the average is exact
        self._carry = 0.0
        self._fh = None
        self._prev: np.ndarray | None = None
        self._pending: tuple[np.ndarray, tuple[int, int, int, int], int] | None = None
        self.n_frames = 0

    def __enter__(self) -> GifWriter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _open(self, h: int, w: int) -> None:
        import struct

        if w > 0xFFFF or h > 0xFFFF:
            raise ValueError("Frame too large for GIF")
        self._fh = self.path.open("wb")
        palette = np.repeat(np.arange(256, dtype=np.uint8), 3).tobytes()  # 256-level grayscale
        self._fh.write(b"GIF89a" + struct.pack("<HHBBB", w, h, 0xF7, 0, 0) + palette)
        # NETSCAPE2.0 application extension: loop count (0 = forever)
        self._fh.write(b"\x21\xff\x0bNETSCAPE2.0\x03\x01" + struct.pack("<H", self.loop) + b"\x00")

    def _next_delay(self) -> int:
        self._carry += self.frame_cs
        d = max(2, int(self._carry))  # browsers treat delays below 2 cs as 10 cs
        self._carry -= d
        return d

    def _write_frame(self, img: np.ndarray, box: tuple[int, int, int, int], delay: int) -> None:
        import struct

        y0, y1, x0, x1 = box
        fh = self._fh
        # Graphic control extension: disposal 1 (leave in place), no transparency
        fh.write(b"\x21\xf9\x04" + struct.pack("<BHB", 0x04, min(int(delay), 0xFFFF), 0) + b"\x00")
        fh.write(b"\x2c" + struct.pack("<HHHHB", x0, y0, x1 - x0, y1 - y0, 0))
        data = _lzw_encode(np.ascontiguousarray(img[y0:y1, x0:x1]).tobytes(), 8)
        fh.write(b"\x08")
        for i in range(0, len(data), 255):
            block = data[i : i + 255]
            fh.write(bytes((len(block),)) + block)
        fh.write(b"\x00")

    def add(self, frame: np.ndarray) -> None:
        img = np.asarray(frame, dtype=np.uint8)
        if img.ndim != 2:
            raise ValueError("GifWriter expects 2-D grayscale frames")
        if self.upscale > 1:
            img = np.repeat(np.repeat(img, self.upscale, axis=0), self.upscale, axis=1)
        h, w = img.shape
        if self._fh is None:
            self._open(h, w)
            box = (0, h, 0, w)
        else:
            if self._prev is None or self._prev.shape != img.shape:
                raise ValueError("All GIF frames must have the same shape")
            diff = img != self._prev
            rows = np.flatnonzero(diff.any(axis=1))
            if len(rows) == 0:
                # Unchanged frame: just lengthen the previous frame's display time
                if self._pending is not None:
                    pimg, pbox, pdelay = self._pending
                    self._pending = (pimg, pbox, pdelay + self._next_delay())
                self.n_frames += 1
                return
            cols = np.flatnonzero(diff.any(axis=0))
            box = (int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1)
        if self._pending is not None:
            self._write_frame(*self._pending)
        self._pending = (img, box, self._next_delay())
        self._prev = img
        self.n_frames += 1

    def close(self) -> None:
        if self._fh is None:
            return
        if self._pending is not None:
            self._write_frame(*self._pending)
            self._pending = None
        self._fh.write(b"\x3b")
        self._fh.close()
        self._fh = None
