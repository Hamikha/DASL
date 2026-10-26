"""IEEE 1588 PTP between the two cameras: enable + lock, settle gate, in-run monitor.

Source tags: [HT] LUCID "Helios2 and Triton Sync" app note + C++ sample, [BW] "Bandwidth
Sharing" app note, [TRM-T] Triton manual, [C §1] research/C_constraints.md section 1.

Thread safety: every latch+read pair runs under the camera keeper's lock, so the monitor,
the action scheduler and the main thread never interleave on one camera. The acquisition
threads never touch node maps. Whether the real SDK allows node access on a device while
another thread waits in get_buffer() on it is UNVERIFIED (review item 20.7).
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

from .nodes import NodeError, NodeKeeper

log = logging.getLogger(__name__)

HELIOS, TRITON = "helios", "triton"
LOCKED = ["Master", "Slave"]


class SyncSetupError(Exception):
    """Base class: a sync method cannot be used (the fallback chain catches these)."""


class MethodUnavailable(SyncSetupError):
    """A node / enum entry the method needs does not exist, or a read-back failed."""


class PtpNotLocked(SyncSetupError):
    """PTP did not reach exactly one Master and one Slave within the timeout."""

    def __init__(self, message: str, report: "PtpReport | None" = None) -> None:
        super().__init__(message)
        self.report = report


@dataclass
class PtpReport:
    helios_status: str | None = None
    triton_status: str | None = None
    master: str | None = None
    lock_seconds: float | None = None
    settled: bool | None = None           # None = could not be measured (fixed wait)
    settle_method: str | None = None      # "offset_threshold" | "fixed_wait"
    settle_seconds: float | None = None
    threshold_ns: int | None = None
    offset_from_master_ns_samples: list = field(default_factory=list)
    servo_status: str | None = None
    status_changes: list = field(default_factory=list)   # [t_s, helios, triton]
    roles_reset: list = field(default_factory=list)      # cameras whose PTP was toggled
    monitor: dict | None = None                          # filled by the engine

    def to_dict(self) -> dict:
        return asdict(self)


# =========================================================================== latching
def ptp_now_ns(master: NodeKeeper) -> int:
    """Current PTP time of a camera: execute PtpDataSetLatch, read PtpDataSetLatchValue
    (ns) [SAC][HT]. Raises NodeError if that is impossible."""
    with master.lock:
        master.execute("PtpDataSetLatch", required=True)
        v = master.get("PtpDataSetLatchValue", required=True)
    if v is None:
        raise NodeError(f"{master.camera}: PtpDataSetLatchValue unreadable")
    return int(v)


def latch_offset_ns(slave: NodeKeeper) -> int | None:
    """Latched PtpOffsetFromMaster of the slave in ns (-1 = sync not started) [TRM-T]."""
    with slave.lock:
        if not slave.execute("PtpDataSetLatch"):
            return None
        v = slave.get("PtpOffsetFromMaster")
    return None if v is None else int(v)


def can_measure_offset(slave: NodeKeeper) -> bool:
    return slave.has("PtpDataSetLatch") and slave.has("PtpOffsetFromMaster")


# =========================================================================== enable
def _set_role(k: NodeKeeper, slave_only: bool, rep: PtpReport) -> None:
    """PtpSlaveOnly = slave_only, then PtpEnable = True.

    If PTP is already on with the OTHER role (left by an earlier session), we switch PTP
    off, change the role and switch it on again: whether a role change takes effect while
    PTP runs is undocumented, and toggling PtpEnable restarts the best-master negotiation
    with the new role (review item 15).
    """
    enabled = k.get("PtpEnable")
    role = k.get("PtpSlaveOnly") if k.has("PtpSlaveOnly") else None
    if enabled and role is not None and bool(role) != slave_only:
        k.set("PtpEnable", False)
        rep.roles_reset.append(k.camera)
        log.info("%s: PTP was on with PtpSlaveOnly=%s -> toggled off/on to change role",
                 k.camera, role)
    k.set("PtpSlaveOnly", slave_only)
    k.set("PtpEnable", True, required=True)


def enable_ptp(helios: NodeKeeper, triton: NodeKeeper, timeout_s: float = 60.0,
               poll_s: float = 0.5, threshold_ns: int = 10_000, settle_samples: int = 5,
               settle_max_s: float = 30.0, min_settle_s: float = 10.0,
               sleep: Callable[[float], None] = time.sleep, **_ignored) -> PtpReport:
    """Enable PTP with the Helios as master, wait for the lock, then for the servo to settle.

    1. Helios PtpSlaveOnly=False, Triton PtpSlaveOnly=True, PtpEnable=True on both [HT].
    2. Poll PtpStatus until exactly one "Master" and one "Slave" exist [BW]; LUCID: this
       "can take about 40 seconds" [BW]. Timeout -> PtpNotLocked (fallback).
    3. Settle gate (review item 2): "Master/Slave" is only a status string. Research C §1:
       "Let the servo settle for about 10-60 s before recording". We wait until
       |PtpOffsetFromMaster| < threshold_ns for settle_samples consecutive samples (at most
       settle_max_s). If it never settles we continue (the clock is still shared) but record
       settled=False and warn loudly. Without PtpOffsetFromMaster we wait min_settle_s.
    """
    for k in (helios, triton):
        for name in ("PtpEnable", "PtpStatus"):
            if not k.has(name):
                k.get(name)                                   # records it as missing
                raise MethodUnavailable(f"{k.camera}: node {name} missing (no PTP)")
    rep = PtpReport(threshold_ns=int(threshold_ns))
    _set_role(helios, False, rep)       # Helios = master, as in LUCID's sample [HT]
    _set_role(triton, True, rep)

    t0 = time.monotonic()
    last = None
    while True:
        st = (helios.get("PtpStatus"), triton.get("PtpStatus"))
        elapsed = time.monotonic() - t0
        if st != last:
            log.info("PTP status after %.1f s: helios=%s triton=%s", elapsed, *st)
            rep.status_changes.append([round(elapsed, 2), st[0], st[1]])
            last = st
        rep.helios_status, rep.triton_status = st
        if sorted(str(x) for x in st) == LOCKED:
            rep.lock_seconds = round(elapsed, 3)
            rep.master = HELIOS if st[0] == "Master" else TRITON
            break
        if elapsed >= timeout_s:
            raise PtpNotLocked(f"PTP not locked after {timeout_s:.0f} s "
                               f"(helios={st[0]}, triton={st[1]})", rep)
        sleep(poll_s)

    slave = triton if rep.master == HELIOS else helios
    _settle(helios, triton, slave, rep, poll_s, threshold_ns, settle_samples, settle_max_s,
            min_settle_s, sleep)
    rep.servo_status = slave.get("PtpServoStatus") if slave.has("PtpServoStatus") else None
    log.info("PTP locked in %.1f s (master=%s), settled=%s after %.1f s", rep.lock_seconds,
             rep.master, rep.settled, rep.settle_seconds or 0.0)
    return rep


def _settle(helios: NodeKeeper, triton: NodeKeeper, slave: NodeKeeper, rep: PtpReport,
            poll_s: float, threshold_ns: int, n_needed: int, max_s: float, min_s: float,
            sleep: Callable[[float], None]) -> None:
    t0 = time.monotonic()
    if not can_measure_offset(slave):
        rep.settle_method = "fixed_wait"
        log.warning("%s: PtpOffsetFromMaster unavailable -> waiting a fixed %.0f s for the "
                    "PTP servo to settle (quality cannot be measured)", slave.camera, min_s)
        while time.monotonic() - t0 < min_s:
            sleep(min(poll_s, max(0.0, min_s - (time.monotonic() - t0))))
        rep.settled = None
        rep.settle_seconds = round(time.monotonic() - t0, 3)
        return
    rep.settle_method = "offset_threshold"
    good = 0
    while True:
        off = latch_offset_ns(slave)
        st = sorted(str(x) for x in (helios.get("PtpStatus"), triton.get("PtpStatus")))
        if off is not None and len(rep.offset_from_master_ns_samples) < 200:
            rep.offset_from_master_ns_samples.append(off)
        ok = st == LOCKED and off is not None and off != -1 and abs(off) < threshold_ns
        good = good + 1 if ok else 0
        elapsed = time.monotonic() - t0
        if good >= n_needed:
            rep.settled = True
            break
        if elapsed >= max_s:
            rep.settled = False
            log.error("PTP DID NOT SETTLE: |PtpOffsetFromMaster| stayed >= %d ns for %.0f s "
                      "(last %s ns). Recording anyway on the shared clock; check run.json "
                      "ptp.monitor and the network.", threshold_ns, max_s, off)
            break
        sleep(poll_s)
    rep.settle_seconds = round(time.monotonic() - t0, 3)


def disable_ptp(*keepers: NodeKeeper) -> None:
    """PtpEnable=False where it is currently on (legacy freerun = no PTP at all)."""
    for k in keepers:
        if k.has("PtpEnable") and k.get("PtpEnable"):
            k.set("PtpEnable", False)


# =========================================================================== monitor
class PtpMonitor:
    """Polls PTP health during the capture (review item 1).

    Every period_s: PtpStatus of both cameras, the master's PTP time (latch) and the slave's
    PtpOffsetFromMaster (latch). A sample is BAD if the statuses are not exactly one
    Master + one Slave, or |offset| > threshold_ns. The PTP time between two consecutive
    samples is a bad interval if either sample is bad; pairs whose master-camera timestamp
    falls in a bad interval are marked suspect by the engine and not written.

    The timeline stored in run.json is decimated: at most one sample per second plus every
    status change.
    """

    def __init__(self, helios: NodeKeeper, triton: NodeKeeper, master: str, period_s: float,
                 threshold_ns: int) -> None:
        self.h, self.t = helios, triton
        self.master = helios if master == HELIOS else triton
        self.slave = triton if master == HELIOS else helios
        self.period_s = float(period_s)
        self.threshold_ns = int(threshold_ns)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._prev: dict | None = None
        self.bad_intervals: list[list] = []     # [ptp_lo, ptp_hi, reason]
        self.timeline: list[dict] = []
        self.samples = 0
        self.bad_samples = 0
        self.covered_ptp: int | None = None     # master PTP time of the latest sample
        self.t0 = time.perf_counter_ns()
        self._measure_offset = can_measure_offset(self.slave)

    # ------------------------------------------------------------------ thread
    def start(self) -> None:
        self.sample()
        self._thread = threading.Thread(target=self._run, name="ptp-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(5.0)

    def _run(self) -> None:
        while not self._stop.wait(self.period_s):
            try:
                self.sample()
            except Exception as exc:              # never let the monitor die silently
                log.warning("PTP monitor sample failed: %s", exc)

    # ------------------------------------------------------------------ sampling
    def sample(self) -> dict:
        host = time.perf_counter_ns()
        hs, ts = self.h.get("PtpStatus"), self.t.get("PtpStatus")
        try:
            ptp = ptp_now_ns(self.master)
        except Exception:
            ptp = None
        off = latch_offset_ns(self.slave) if self._measure_offset else None
        reason = ""
        if sorted(str(x) for x in (hs, ts)) != LOCKED:
            reason = f"status helios={hs} triton={ts}"
        elif off is not None and (off == -1 or abs(off) > self.threshold_ns):
            reason = f"offset {off} ns"
        s = dict(t_s=round((host - self.t0) / 1e9, 3), helios=hs, triton=ts, offset_ns=off,
                 ptp_ns=ptp, ok=not reason)
        with self._lock:
            self.samples += 1
            prev = self._prev
            if reason:
                self.bad_samples += 1
                log.warning("PTP monitor: %s", reason)
            if ptp is not None:
                if prev is not None and prev["ptp_ns"] is not None and (reason or not prev["ok"]):
                    lo, hi = sorted((prev["ptp_ns"], ptp))
                    self._add_bad(lo, hi, reason or prev["reason"])
                elif prev is None and reason:
                    self._add_bad(ptp - int(self.period_s * 1e9), ptp, reason)
                self.covered_ptp = ptp
            s["reason"] = reason
            last = self.timeline[-1] if self.timeline else None
            if (last is None or (last["helios"], last["triton"], last["ok"]) != (hs, ts, s["ok"])
                    or s["t_s"] - last["t_s"] >= 1.0):
                self.timeline.append({k: v for k, v in s.items() if k != "reason"}
                                     | ({"reason": reason} if reason else {}))
            self._prev = s
        return s

    def _add_bad(self, lo: int, hi: int, reason: str) -> None:
        if self.bad_intervals and lo <= self.bad_intervals[-1][1]:
            self.bad_intervals[-1][1] = max(self.bad_intervals[-1][1], hi)
        else:
            self.bad_intervals.append([lo, hi, reason])

    # ------------------------------------------------------------------ queries
    def covered_until(self) -> int | None:
        with self._lock:
            return self.covered_ptp

    def bad_reason(self, ptp_ns: int) -> str | None:
        """Why the PTP time ptp_ns lies in a bad interval, or None."""
        with self._lock:
            for lo, hi, why in self.bad_intervals:
                if lo <= ptp_ns <= hi:
                    return f"ptp_monitor: {why}"
        return None

    def to_dict(self) -> dict:
        with self._lock:
            return dict(period_s=self.period_s, threshold_ns=self.threshold_ns,
                        offset_measured=self._measure_offset, samples=self.samples,
                        bad_samples=self.bad_samples,
                        bad_intervals_ptp_ns=[list(b) for b in self.bad_intervals[:200]],
                        timeline=list(self.timeline[:5000]))
