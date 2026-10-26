"""Enumerations mirroring ``arena_api.enums`` (only the members the mock needs).

Integer values follow the GenICam PFNC codes where known; the real SDK may use different
Python enum values, so scripts must compare by member/name, never by integer.
"""
from __future__ import annotations

import enum


class PixelFormat(enum.IntEnum):
    Mono8 = 0x01080001
    Mono16 = 0x01100007
    BayerRG8 = 0x01080009
    BayerRG16 = 0x0110002F           # Triton support NOT CONFIRMED (mock only)
    RGB8 = 0x02180014                # Triton support NOT CONFIRMED (mock only)
    Coord3D_C16 = 0x011000B8         # PFNC code; HTP-only format per LUCID datasheet
    Coord3D_ABC16 = 0x023000B9       # PFNC code
    Coord3D_ABCY16 = 0x024000B9      # LUCID-specific; integer value NOT CONFIRMED


# Bits per pixel per format (what buffer.bits_per_pixel reports).
BITS_PER_PIXEL = {
    PixelFormat.Mono8: 8,
    PixelFormat.Mono16: 16,
    PixelFormat.BayerRG8: 8,
    PixelFormat.BayerRG16: 16,
    PixelFormat.RGB8: 24,
    PixelFormat.Coord3D_C16: 16,
    PixelFormat.Coord3D_ABC16: 48,
    PixelFormat.Coord3D_ABCY16: 64,
}


class InterfaceType(enum.IntEnum):
    """GenICam interface types (values follow GenApi EInterfaceType)."""
    VALUE = 0
    BASE = 1
    INTEGER = 2
    BOOLEAN = 3
    COMMAND = 4
    FLOAT = 5
    STRING = 6
    REGISTER = 7
    CATEGORY = 8
    ENUMERATION = 9
    ENUMENTRY = 10
    PORT = 11


class AccessMode(enum.IntEnum):
    """GenICam access modes (values follow GenApi EAccessMode)."""
    NI = 0   # not implemented
    NA = 1   # not available
    WO = 2
    RO = 3
    RW = 4
