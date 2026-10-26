"""Synthetic frame content, generated ONCE per (format, width, height) and cached.

Frames reuse the cached bytes (copied into each pool buffer once at start_stream), so the
mock costs no per-frame CPU for image content.
"""
from __future__ import annotations

import math
import threading

import numpy as np

from .enums import BITS_PER_PIXEL, PixelFormat

_cache: dict[tuple, bytes] = {}
_cache_lock = threading.Lock()

# Helios2+ field of view 67 x 51 deg (LUCID product page) -> pinhole intrinsics.
_HELIOS_HFOV_DEG = 67.0
_HELIOS_VFOV_DEG = 51.0
# Scan3d coordinate conversion used by the mock (value_mm = raw * scale + offset).
# NOT CONFIRMED on real hardware - see README_MOCK.md.
SCAN3D_SCALE_MM = 0.25
SCAN3D_OFFSET_MM = {"CoordinateA": -8192.0, "CoordinateB": -8192.0, "CoordinateC": 0.0}
_PLANE_DISTANCE_MM = 3000.0


def frame_bytes(fmt: PixelFormat, width: int, height: int) -> int:
    return width * height * BITS_PER_PIXEL[fmt] // 8


def _helios_xyz(width: int, height: int):
    fx = (width / 2) / math.tan(math.radians(_HELIOS_HFOV_DEG / 2))
    fy = (height / 2) / math.tan(math.radians(_HELIOS_VFOV_DEG / 2))
    v, u = np.mgrid[0:height, 0:width].astype(np.float64)
    # A plane ~3 m away, tilted a little so Z is not constant (like a floor/wall view).
    z = _PLANE_DISTANCE_MM + 400.0 * (v / height - 0.5)
    x = (u - width / 2 + 0.5) / fx * z
    y = (v - height / 2 + 0.5) / fy * z
    # Intensity: radial fall-off like an active illuminator; corners very dark = invalid.
    r = np.hypot((u - width / 2) / (width / 2), (v - height / 2) / (height / 2))
    inten = np.clip(3000.0 * (1.0 - 0.6 * r * r), 0, 65535)
    invalid = r > 1.25
    return x, y, z, inten, invalid


def _to_raw(mm: np.ndarray, coord: str) -> np.ndarray:
    raw = np.round((mm - SCAN3D_OFFSET_MM[coord]) / SCAN3D_SCALE_MM)
    return np.clip(raw, 0, 65535).astype(np.uint16)


def _make(fmt: PixelFormat, width: int, height: int) -> bytes:
    if fmt in (PixelFormat.Coord3D_ABCY16, PixelFormat.Coord3D_ABC16,
               PixelFormat.Coord3D_C16, PixelFormat.Mono16):
        x, y, z, inten, invalid = _helios_xyz(width, height)
        a, b, c = _to_raw(x, "CoordinateA"), _to_raw(y, "CoordinateB"), _to_raw(z, "CoordinateC")
        iy = inten.astype(np.uint16)
        c[invalid] = 0
        iy[invalid] = 0
        if fmt == PixelFormat.Coord3D_ABCY16:
            arr = np.stack([a, b, c, iy], axis=-1)
        elif fmt == PixelFormat.Coord3D_ABC16:
            arr = np.stack([a, b, c], axis=-1)
        elif fmt == PixelFormat.Coord3D_C16:
            arr = c
        else:
            arr = iy
        return np.ascontiguousarray(arr, dtype="<u2").tobytes()
    # Colour / mono 8-bit: smooth gradient with a Bayer-like channel modulation.
    u = (np.arange(width, dtype=np.uint16) * 200 // max(width, 1))[None, :]
    v = (np.arange(height, dtype=np.uint16) * 55 // max(height, 1))[:, None]
    base = u + v                                     # 0..255, broadcast (h, w) uint16
    if fmt == PixelFormat.BayerRG8:
        out = base.astype(np.uint8)
        out[0::2, 0::2] = np.minimum(base[0::2, 0::2] * 6 // 5, 255)   # R brighter
        out[1::2, 1::2] = base[1::2, 1::2] * 7 // 10                    # B darker
        return out.tobytes()
    if fmt == PixelFormat.BayerRG16:
        return (np.clip(base, 0, 255).astype(np.uint16) << 4).astype("<u2").tobytes()
    if fmt == PixelFormat.RGB8:
        g = np.clip(base, 0, 255).astype(np.uint8)
        return np.stack([g, g // 2 + 60, 255 - g], axis=-1).tobytes()
    return np.clip(base, 0, 255).astype(np.uint8).tobytes()  # Mono8


def template(fmt: PixelFormat, width: int, height: int) -> bytes:
    key = (fmt, width, height)
    with _cache_lock:
        data = _cache.get(key)
        if data is None:
            data = _make(fmt, width, height)
            assert len(data) == frame_bytes(fmt, width, height)
            _cache[key] = data
        return data
