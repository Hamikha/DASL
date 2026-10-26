"""Buffers handed out by Device.get_buffer(), and BufferFactory (copy / destroy).

Use-after-requeue detection: every get_buffer() returns a NEW Buffer view on a pool slot.
requeue_buffer() invalidates that view (any attribute access then raises RuntimeError),
because on real hardware the memory is refilled by the next frame. Copy first
(BufferFactory.copy or np.ctypeslib.as_array(...).copy()) if you need data later.
"""
from __future__ import annotations

import ctypes
from typing import Any, Iterable

from ._mock_errors import MockStreamError
from .enums import BITS_PER_PIXEL, InterfaceType, PixelFormat


class _Slot:
    """One pool buffer: real memory of the full payload size, filled once at allocation."""

    __slots__ = ("mem", "nbytes", "generation", "meta", "owner", "outstanding")

    def __init__(self, nbytes: int, content: bytes, owner: Any) -> None:
        # One memcpy (bytearray) instead of zero-fill + memmove; the ctypes array keeps the
        # bytearray alive through its _objects reference.
        self.mem = (ctypes.c_ubyte * nbytes).from_buffer(bytearray(content[:nbytes]))
        self.nbytes = nbytes
        self.generation = 0
        self.meta: dict = {}
        self.owner = owner          # the StreamEngine
        self.outstanding = False    # handed to the user, not yet requeued


class ChunkNode:
    """Minimal node-like object returned by Buffer.get_chunk()."""

    interface_type = InterfaceType.FLOAT

    def __init__(self, name: str, value: Any) -> None:
        self.name = name
        self._value = value
        self.is_readable = True
        self.is_writable = False
        if isinstance(value, int) and not isinstance(value, bool):
            self.interface_type = InterfaceType.INTEGER

    @property
    def value(self) -> Any:
        return self._value

    def __repr__(self) -> str:
        return f"<ChunkNode {self.name}={self._value!r}>"


class Buffer:
    """Image buffer (mirrors the arena_api Buffer attributes used by capture code)."""

    def __init__(self, slot: _Slot, *, standalone: bool = False) -> None:
        self._slot = slot
        self._gen = slot.generation
        self._standalone = standalone
        self._valid = True

    # ------------------------------------------------------------------ validity
    def _check(self) -> _Slot:
        if not self._valid or self._slot.generation != self._gen:
            raise MockStreamError(
                "buffer was requeued/destroyed; its memory now belongs to the SDK "
                "(copy the data before requeue_buffer)")
        return self._slot

    def _m(self, key: str) -> Any:
        return self._check().meta[key]

    # ------------------------------------------------------------------ data
    @property
    def pdata(self) -> "ctypes._Pointer":
        s = self._check()
        return ctypes.cast(s.mem, ctypes.POINTER(ctypes.c_ubyte))

    @property
    def data(self) -> memoryview:
        # The real SDK returns a Python list (slow for MB frames); a read-only memoryview is
        # equally indexable/sliceable/len()-able and free.
        s = self._check()
        return memoryview(s.mem).cast("B").toreadonly()

    # ------------------------------------------------------------------ metadata
    width = property(lambda self: self._m("width"))
    height = property(lambda self: self._m("height"))
    pixel_format = property(lambda self: self._m("pixel_format"))
    bits_per_pixel = property(lambda self: BITS_PER_PIXEL[self._m("pixel_format")])
    size_filled = property(lambda self: self._m("size_filled"))
    payload_size = property(lambda self: self._check().nbytes)
    buffer_size = property(lambda self: self._check().nbytes)
    timestamp_ns = property(lambda self: self._m("timestamp_ns"))
    timestamp = property(lambda self: self._m("timestamp_ns"))
    frame_id = property(lambda self: self._m("frame_id"))
    is_incomplete = property(lambda self: self._m("is_incomplete"))
    has_imagedata = property(lambda self: True)
    has_chunkdata = property(lambda self: bool(self._m("chunks")))

    def get_chunk(self, names: str | Iterable[str]) -> Any:
        """str -> chunk node (KeyError if absent); list -> {name: node or None}."""
        chunks: dict = self._m("chunks")
        if isinstance(names, str):
            if names not in chunks:
                raise KeyError(f"chunk {names!r} not present (ChunkModeActive/ChunkEnable?)")
            return ChunkNode(names, chunks[names])
        return {n: (ChunkNode(n, chunks[n]) if n in chunks else None) for n in names}

    # Mock-only diagnostics (NOT in the real SDK): model times of this frame.
    @property
    def mock_info(self) -> dict:
        return dict(self._m("mock"))

    def __repr__(self) -> str:
        try:
            return (f"<Buffer id={self.frame_id} ts={self.timestamp_ns} "
                    f"{self.width}x{self.height} {self.pixel_format.name}"
                    f"{' INCOMPLETE' if self.is_incomplete else ''}>")
        except MockStreamError:
            return "<Buffer (requeued)>"


class BufferFactory:
    """Mirror of arena_api.buffer.BufferFactory (copy / destroy)."""

    @staticmethod
    def copy(buffer: Buffer) -> Buffer:
        src = buffer._check()
        slot = _Slot(src.nbytes, bytes(src.mem), owner=None)
        slot.meta = dict(src.meta)
        return Buffer(slot, standalone=True)

    @staticmethod
    def destroy(buffer: Buffer) -> None:
        if not buffer._standalone:
            raise MockStreamError("BufferFactory.destroy() only accepts buffers created by "
                                  "BufferFactory.copy(); use device.requeue_buffer()")
        buffer._check()
        buffer._valid = False
        buffer._slot.generation += 1

    @staticmethod
    def convert(buffer: Buffer, pixel_format: PixelFormat) -> Buffer:  # pragma: no cover
        raise NotImplementedError("BufferFactory.convert is not modelled by the mock")
