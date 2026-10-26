"""NodeKeeper: safe access to GenICam nodes with a written record of every problem.

The user's original capture.py had a ``NodeKeeper`` that set nodes, logged failures and
restored the original values at the end. This module keeps that idea and makes it strict
enough for a thesis:

* A node is "missing" when ``nodemap.get_node(name)`` returns None OR raises. The real SDK's
  behaviour for unknown names is not documented, so we accept both.
* Every missing node and every failed write is appended to a shared `NodeReport`, with camera
  name, node map, node, value and error text. That report goes into run.json.
* We never invent nodes or values: an enum entry is written only if it appears in
  ``enumentry_names``; a number outside [min, max] is refused unless the caller explicitly
  asks for ``clamp=True`` (and the clamp is logged).
* ALLOW-LIST: only the node names in ALLOWED_WRITES / ALLOWED_COMMANDS may be written or
  executed. Everything else (IP configuration, user sets, firmware, resets, ...) is refused
  and logged. A short explicit list is easy to check for the thesis.
* A write of a value the node already holds is skipped (logged at DEBUG); fewer writes means
  fewer side effects.
* Every successful write stores the old value, so `restore_writes()` can put the cameras
  back exactly as they were found (RESTORE_ON_EXIT).
* Each keeper owns a lock. Every node access through the keeper holds it, so the PTP
  monitor, the action scheduler and the main thread never interleave a latch+read pair on
  the same camera. The acquisition threads never take it: they only call get_buffer /
  requeue_buffer. (Whether the real SDK tolerates node access on a device while another
  thread waits in get_buffer is UNVERIFIED - A open question / review item 20.7.)
"""
from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from typing import Any, Iterable

log = logging.getLogger(__name__)

#: Every node the sync code is permitted to WRITE (exact names). Anything else is refused.
ALLOWED_WRITES = frozenset({
    # image format / Helios ToF mode / Triton ROI (as in the user's capture.py)
    "PixelFormat", "Scan3dOperatingMode", "ExposureTimeSelector", "ConversionGain",
    "Scan3dCoordinateSelector", "AcquisitionMode",
    "BinningHorizontal", "BinningVertical", "OffsetX", "OffsetY", "Width", "Height",
    # Triton exposure / colour
    "ExposureAuto", "ExposureTime", "ExposureAutoLimitAuto", "ExposureAutoLowerLimit",
    "ExposureAutoUpperLimit", "BalanceWhiteAuto",
    "ChunkModeActive", "ChunkSelector", "ChunkEnable",
    # bandwidth / host stream
    "DeviceLinkThroughputLimitMode", "DeviceLinkThroughputLimit",
    "StreamAutoNegotiatePacketSize", "StreamPacketResendEnable", "StreamBufferHandlingMode",
    # acquisition / trigger / PTPSync / action (device side)
    "TriggerSelector", "TriggerSource", "TriggerMode", "TriggerDelay",
    "AcquisitionStartMode", "AcquisitionFrameRateEnable", "AcquisitionFrameRate",
    "PTPSyncFrameRate", "PTPSyncOffset",
    "ActionUnconditionalMode", "ActionSelector", "ActionDeviceKey", "ActionGroupKey",
    "ActionGroupMask",
    "PtpEnable", "PtpSlaveOnly",
    # host TL system node map (action commands; not a camera setting)
    "ActionCommandDeviceKey", "ActionCommandGroupKey", "ActionCommandGroupMask",
    "ActionCommandTargetIP", "ActionCommandExecuteTime",
})
#: Every command node the sync code is permitted to EXECUTE.
ALLOWED_COMMANDS = frozenset({"PtpDataSetLatch", "ActionCommandFireCommand", "TriggerSoftware"})


class NodeError(Exception):
    """A required node operation failed (used to trigger the sync-method fallback)."""


class MissingNodeError(NodeError):
    pass


class NodeWriteError(NodeError):
    pass


def entry_name(v: Any) -> Any:
    """Enum members (e.g. arena_api.enums.PixelFormat.BayerRG8) -> their name; else v."""
    if not isinstance(v, str) and isinstance(getattr(v, "name", None), str):
        return v.name
    return v


def _jsonable(v: Any) -> Any:
    v = entry_name(v)
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    return repr(v)


@dataclass
class NodeReport:
    """Shared log of node activity for one run (one report for all node maps/cameras)."""

    missing: list[dict] = field(default_factory=list)
    failed: list[dict] = field(default_factory=list)
    clamped: list[dict] = field(default_factory=list)
    writes: list[dict] = field(default_factory=list)
    keepers: dict = field(default_factory=dict, repr=False)   # (camera, map) -> keeper
    restore_result: dict | None = None        # set once restore_writes() has run
    _missing_seen: set = field(default_factory=set, repr=False)

    def add_missing(self, camera: str, nodemap: str, node: str, action: str,
                    value: Any = None, error: str = "") -> None:
        key = (camera, nodemap, node)
        if key in self._missing_seen:          # log each missing node once
            return
        self._missing_seen.add(key)
        self.missing.append(dict(camera=camera, nodemap=nodemap, node=node, action=action,
                                 value=_jsonable(value), error=error))
        log.warning("[%s/%s] node %s is MISSING (%s%s)", camera, nodemap, node, action,
                    f": {error}" if error else "")

    def add_failed(self, camera: str, nodemap: str, node: str, action: str,
                   value: Any, error: str) -> None:
        self.failed.append(dict(camera=camera, nodemap=nodemap, node=node, action=action,
                                value=_jsonable(value), error=error))
        log.warning("[%s/%s] %s %s = %r FAILED: %s", camera, nodemap, action, node,
                    _jsonable(value), error)

    def to_dict(self) -> dict:
        return dict(missing_nodes=list(self.missing), failed_node_writes=list(self.failed),
                    clamped_node_writes=list(self.clamped),
                    node_writes=[{k: v for k, v in w.items() if k != "raw_old"}
                                 for w in self.writes])


def node_kind(node: Any) -> str:
    """'enum' | 'int' | 'float' | 'bool' | 'str' | 'command' | 'unknown'.

    Uses ``node.interface_type`` (arena_api / GenApi interface enum), compared by NAME
    because the integer values are not documented for the real SDK.
    """
    it = getattr(node, "interface_type", None)
    name = str(getattr(it, "name", it) or "").upper()
    for key, kind in (("ENUMERATION", "enum"), ("INTEGER", "int"), ("FLOAT", "float"),
                      ("BOOLEAN", "bool"), ("STRING", "str"), ("COMMAND", "command")):
        if key in name:
            return kind
    if hasattr(node, "enumentry_names"):
        return "enum"
    if hasattr(node, "execute"):
        return "command"
    return "unknown"


def _equal(a: Any, b: Any) -> bool:
    a, b = entry_name(a), entry_name(b)
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9)
        except (TypeError, ValueError):
            return False
    return a == b


class NodeKeeper:
    """Wrapper around ONE node map of ONE camera.

    Parameters
    ----------
    nodemap: the arena_api node map (device.nodemap, device.tl_stream_nodemap, ...).
    camera:  short camera name for the logs ("helios", "triton", "host").
    nodemap_name: which map ("device", "stream", "tl_device", "tl_system").
    report:  shared NodeReport (create one per run and pass it to every keeper).
    allow:   override of the write allow-list (tests only).
    """

    def __init__(self, nodemap: Any, camera: str, nodemap_name: str = "device",
                 report: NodeReport | None = None, allow: Iterable[str] | None = None) -> None:
        self.nodemap = nodemap
        self.camera = camera
        self.nodemap_name = nodemap_name
        self.report = report if report is not None else NodeReport()
        self.allow = frozenset(allow) if allow is not None else ALLOWED_WRITES
        self.lock = threading.RLock()
        self.report.keepers[(camera, nodemap_name)] = self

    # ------------------------------------------------------------------ convenience
    @property
    def missing(self) -> list[dict]:
        return self.report.missing

    @property
    def failed(self) -> list[dict]:
        return self.report.failed

    def __repr__(self) -> str:
        return f"<NodeKeeper {self.camera}/{self.nodemap_name}>"

    # ------------------------------------------------------------------ lookup
    def node(self, name: str) -> Any | None:
        """The node object, or None if missing (never raises, never logs)."""
        if self.nodemap is None:
            return None
        try:
            with self.lock:
                return self.nodemap.get_node(name)
        except Exception:     # real SDK may raise for unknown names (undocumented)
            return None

    def has(self, name: str) -> bool:
        """True if the node exists. Does NOT record anything (used for capability checks)."""
        return self.node(name) is not None

    def entries(self, name: str) -> list[str]:
        """Enum entry names of a node ([] if missing or not an enum)."""
        n = self.node(name)
        if n is None:
            return []
        try:
            with self.lock:
                return [str(e) for e in (n.enumentry_names or [])]
        except Exception:
            return []

    def limits(self, name: str) -> tuple[Any, Any, Any]:
        """(min, max, inc) of a numeric node; None for anything unavailable."""
        n = self.node(name)
        out = []
        for attr in ("min", "max", "inc"):
            try:
                with self.lock:
                    out.append(getattr(n, attr) if n is not None else None)
            except Exception:
                out.append(None)
        return out[0], out[1], out[2]

    def _missing(self, name: str, action: str, value: Any = None, required: bool = False,
                 error: str = "") -> None:
        self.report.add_missing(self.camera, self.nodemap_name, name, action, value, error)
        if required:
            raise MissingNodeError(f"{self.camera}: required node {name} is missing")

    # ------------------------------------------------------------------ read
    def get(self, name: str, default: Any = None, required: bool = False) -> Any:
        """Read a node value; ``default`` if missing / unreadable (and record why).
        Enum values are returned as entry names (str)."""
        n = self.node(name)
        if n is None:
            self._missing(name, "get", required=required)
            return default
        try:
            with self.lock:
                if not n.is_readable:
                    raise RuntimeError("node is not readable")
                return entry_name(n.value)
        except Exception as exc:
            self.report.add_failed(self.camera, self.nodemap_name, name, "get", None, str(exc))
            if required:
                raise NodeError(f"{self.camera}: cannot read {name}: {exc}") from exc
            return default

    # ------------------------------------------------------------------ write
    def set(self, name: str, value: Any, required: bool = False, clamp: bool = False,
            record: bool = True) -> bool:
        """Write ``value`` to node ``name``. Returns True on success (or already equal).

        required=True raises MissingNodeError / NodeWriteError instead of returning False.
        clamp=True clamps numbers into [min, max] (and onto the increment); logged.
        record=False: do not remember the old value (used by restore itself).
        """
        def fail(msg: str) -> bool:
            self.report.add_failed(self.camera, self.nodemap_name, name, "set", value, msg)
            if required:
                raise NodeWriteError(f"{self.camera}: {name} = {_jsonable(value)!r} "
                                     f"failed: {msg}")
            return False

        if name not in self.allow:
            return fail("refused: node is not on the sync code's write allow-list")
        n = self.node(name)
        if n is None:
            self._missing(name, "set", value, required=required)
            return False
        with self.lock:
            kind = node_kind(n)
            if kind == "enum":
                ok_entries = self.entries(name)
                if str(entry_name(value)) not in ok_entries:
                    return fail(f"entry {entry_name(value)!r} not available; "
                                f"entries = {ok_entries}")
            if kind in ("int", "float") and not isinstance(value, bool):
                value, msg = self._range_check(n, name, value, kind, clamp)
                if msg:
                    return fail(msg)
            old = None
            try:
                old = n.value if n.is_readable else None
                if old is not None and _equal(old, value):
                    log.debug("[%s] %s already = %r", self.camera, name, _jsonable(value))
                    return True
            except Exception:
                pass                      # unreadable / write-only node: just write it
            try:
                if not n.is_writable:
                    return fail("node is not writable now (read-only, locked while "
                                "streaming, or its enable/selector node is not set)")
                n.value = value
            except Exception as exc:
                return fail(f"{type(exc).__name__}: {exc}")
        if record:
            self.report.writes.append(dict(camera=self.camera, nodemap=self.nodemap_name,
                                           node=name, old=_jsonable(old),
                                           new=_jsonable(value), raw_old=old))
        log.info("[%s] %s = %r", self.camera, name, _jsonable(value))
        return True

    def _range_check(self, n: Any, name: str, value: Any, kind: str,
                     clamp: bool) -> tuple[Any, str]:
        """Return (value_to_write, error_message_or_empty)."""
        try:
            lo, hi = n.min, n.max
        except Exception:
            lo = hi = None
        v = int(value) if kind == "int" else float(value)
        if (lo is not None and v < lo) or (hi is not None and v > hi):
            if not clamp:
                return v, f"{v} outside [{lo}, {hi}]"
            new = min(max(v, lo if lo is not None else v), hi if hi is not None else v)
            if kind == "int":
                new = self._snap_int(n, int(new))
            self.report.clamped.append(dict(camera=self.camera, node=name, requested=v,
                                            written=new, min=lo, max=hi))
            log.warning("[%s] %s: requested %r clamped to %r (range [%s, %s])",
                        self.camera, name, v, new, lo, hi)
            v = new
        elif clamp and kind == "int":
            v = self._snap_int(n, v)
        return v, ""

    @staticmethod
    def _snap_int(n: Any, v: int) -> int:
        """Snap an integer down onto min + k*inc (GenICam increment rule)."""
        try:
            inc, lo = int(n.inc or 1), int(n.min or 0)
        except Exception:
            return v
        if inc > 1:
            v = lo + ((v - lo) // inc) * inc
        return v

    def execute(self, name: str, required: bool = False) -> bool:
        """Execute a command node (allow-listed commands only). Returns True on success."""
        if name not in ALLOWED_COMMANDS:
            self.report.add_failed(self.camera, self.nodemap_name, name, "execute", None,
                                   "refused: command is not on the allow-list")
            if required:
                raise NodeWriteError(f"{self.camera}: refused to execute {name}")
            return False
        n = self.node(name)
        if n is None:
            self._missing(name, "execute", required=required)
            return False
        try:
            with self.lock:
                n.execute()
            return True
        except Exception as exc:
            self.report.add_failed(self.camera, self.nodemap_name, name, "execute", None,
                                   f"{type(exc).__name__}: {exc}")
            if required:
                raise NodeWriteError(f"{self.camera}: execute {name} failed: {exc}") from exc
            return False

    def try_set_first(self, name: str, candidates: Iterable[Any],
                      required: bool = False) -> Any | None:
        """Write the first candidate the node accepts; return it (None if none worked).

        For enums only candidates listed in enumentry_names are tried, so nothing is
        invented and no spurious failure is logged for entries that simply do not exist.
        """
        candidates = list(candidates)
        n = self.node(name)
        if n is None:
            self._missing(name, "set", candidates, required=required)
            return None
        if node_kind(n) == "enum":
            available = self.entries(name)
            usable = [c for c in candidates if str(entry_name(c)) in available]
            if not usable:
                self.report.add_failed(self.camera, self.nodemap_name, name, "set",
                                       candidates, f"none of the candidates in {available}")
                if required:
                    raise NodeWriteError(f"{self.camera}: {name}: no usable entry "
                                         f"among {candidates}")
                return None
            candidates = usable
        for c in candidates:
            if self.set(name, c):
                return c
        if required:
            raise NodeWriteError(f"{self.camera}: {name}: no candidate could be written")
        return None


def restore_writes(report: NodeReport) -> dict:
    """Put every node we wrote back to its value before our first write (best effort).

    Writes are undone in REVERSE order, which also handles dependencies: e.g. the
    throughput limit is restored while its "Mode On" is still set, and TriggerMode is
    restored while TriggerSelector still points at the entry we used. Must run after
    stop_stream (some nodes are locked while streaming). Returns
    {"restored": [...], "failed": [...], "skipped": [...]} for run.json.
    """
    if report.restore_result is not None:      # idempotent: restore only once
        return report.restore_result
    out: dict = dict(restored=[], failed=[], skipped=[])
    for w in reversed(report.writes):
        label = f"{w['camera']}/{w['nodemap']}/{w['node']}"
        keeper = report.keepers.get((w["camera"], w["nodemap"]))
        old = w.get("raw_old")
        if keeper is None or old is None:
            out["skipped"].append(dict(node=label, reason="old value unknown"))
            continue
        n_failed = len(report.failed)
        ok = keeper.set(w["node"], old, record=False)
        if ok:
            out["restored"].append(dict(node=label, value=_jsonable(old)))
        else:
            err = report.failed[-1]["error"] if len(report.failed) > n_failed else "missing"
            out["failed"].append(dict(node=label, value=_jsonable(old), error=err))
    log.info("restore: %d nodes restored, %d failed, %d skipped", len(out["restored"]),
             len(out["failed"]), len(out["skipped"]))
    report.restore_result = out
    return out
