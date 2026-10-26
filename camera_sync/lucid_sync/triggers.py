"""Host-side trigger threads: scheduled action commands and paced software triggers.

Both record every trigger index k they attempted (fired / skipped / error), so frame drops
can be counted exactly against what was actually requested. Both expose ``failure`` (a
reason string once they gave up); the engine watches it and stops the run loudly.

Timing on Windows: Event.wait / lock timeouts have ~15.6 ms granularity there, so
`wait_until_ns` waits coarsely with Event.wait until ~BUSY_WAIT_NS before the target and
then spins on perf_counter with time.sleep(0) (which only yields the GIL). No OS timer
setting is changed (no timeBeginPeriod) - the user forbids OS changes.
"""
from __future__ import annotations

import bisect
import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from fractions import Fraction

from .nodes import NodeKeeper
from .ptp import ptp_now_ns

log = logging.getLogger(__name__)

BUSY_WAIT_NS = 2_000_000          # spin for the last 2 ms
COARSE_SLACK_NS = 20_000_000      # Event.wait stops this early (covers 15.6 ms ticks)


def wait_until_ns(target_ns: int, stop: threading.Event) -> bool:
    """Sleep until perf_counter_ns() >= target_ns. False if `stop` was set meanwhile."""
    while True:
        left = target_ns - time.perf_counter_ns()
        if left <= 0:
            return not stop.is_set()
        if left > COARSE_SLACK_NS:
            if stop.wait(min((left - COARSE_SLACK_NS) / 1e9, 0.05)):
                return False
        elif left > BUSY_WAIT_NS:
            time.sleep(min((left - BUSY_WAIT_NS) / 1e9, 0.001))   # high-res on 3.11+ Windows
        else:
            time.sleep(0)                                      # spin, yield the GIL
            if stop.is_set():
                return False


class _ClockMap:
    """Linear map host perf_counter_ns <-> PTP ns, refreshed by latching the PTP clock.

    Host and PTP clocks do not run at the same rate (oscillators differ by tens of ppm), so
    the rate is fitted (least squares) over the recent samples instead of assumed to be 1.
    Each sample pairs the PTP latch with the MIDPOINT of the host round trip; the latch
    happens somewhere inside the round trip (probably early, on the write), so the mapping
    may be biased by up to half the round trip (~0.05 ms here) - irrelevant against a
    50 ms lead.
    """

    def __init__(self, max_samples: int = 30) -> None:
        self.samples: deque[tuple[int, int]] = deque(maxlen=max_samples)
        self.rate = 1.0                 # d(ptp) / d(host)
        self.max_roundtrip_ns = 0

    def add(self, host_ns: int, ptp_ns: int, roundtrip_ns: int = 0) -> None:
        self.samples.append((int(host_ns), int(ptp_ns)))
        self.max_roundtrip_ns = max(self.max_roundtrip_ns, int(roundtrip_ns))
        if len(self.samples) >= 3 and self.samples[-1][0] - self.samples[0][0] > 500_000_000:
            h0, p0 = self.samples[0]
            xs = [h - h0 for h, _ in self.samples]
            ys = [p - p0 for _, p in self.samples]
            mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
            sxx = sum((x - mx) ** 2 for x in xs)
            if sxx > 0:
                self.rate = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx

    def reset(self) -> None:
        self.samples.clear()
        self.rate = 1.0

    def ptp_at(self, host_ns: int) -> int:
        h, p = self.samples[-1]
        return p + int(round((host_ns - h) * self.rate))

    def host_at(self, ptp_ns: int) -> int:
        h, p = self.samples[-1]
        return h + int(round((ptp_ns - p) / self.rate))


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    v = sorted(values)
    return round(v[min(len(v) - 1, int(q / 100 * len(v)))], 3)


# =========================================================================== action
@dataclass(slots=True)
class ScheduledAction:
    k: int
    execute_time_ns: int
    fire_host_ns: int
    margin_ns: int          # estimated PTP time left until execution when sent
    status: str             # "fired" | "skipped_late" | "error"
    error: str = ""


class ActionScheduler:
    """Thread that fires one scheduled action command per frame on a fixed PTP grid.

    Grid: T_k = T0 + k * period (PTP ns). Command k is sent ACTION_LEAD_MS before T_k with
    ActionCommandExecuteTime = T_k [SAC][HT]. A command whose time is (almost) past is
    SKIPPED, not sent: a late command may execute immediately (undocumented, A open question
    7) and would expose off the grid. A PTP time step (the latched PTP time disagrees with
    the clock map by more than max(1 ms, period/4)) stops the scheduler with `failure` set:
    the grid T0 would no longer mean anything.
    """

    def __init__(self, tl: NodeKeeper, master: NodeKeeper, period_ns: float, lead_ms: float,
                 resync_s: float = 1.0, min_margin_ms: float | None = None,
                 max_consecutive_errors: int = 20) -> None:
        self.tl, self.master = tl, master
        self.period = Fraction(period_ns).limit_denominator(1000)
        self.lead_ns = int(lead_ms * 1e6)
        self.resync_ns = int(resync_s * 1e9)
        # Do not send if less than this is left: covers network latency (~0.1 ms) plus
        # host jitter with a wide margin.
        self.min_margin_ns = int((min_margin_ms if min_margin_ms is not None
                                  else max(2.0, lead_ms / 4)) * 1e6)
        self.max_consecutive_errors = max_consecutive_errors
        self.step_limit_ns = max(1_000_000, int(float(self.period) / 4))
        self.clock = _ClockMap()
        self.records: list[ScheduledAction] = []
        self.t0: int | None = None
        self.resyncs = 0
        self.failure: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._exec_node = tl.node("ActionCommandExecuteTime")
        self._fire_node = tl.node("ActionCommandFireCommand")

    # ------------------------------------------------------------------ clock
    def resync(self) -> None:
        """Latch the master's PTP clock once and add the (host, PTP) sample."""
        h1 = time.perf_counter_ns()
        p = ptp_now_ns(self.master)
        h2 = time.perf_counter_ns()
        mid = (h1 + h2) // 2
        if self.clock.samples:
            jump = p - self.clock.ptp_at(mid)
            if abs(jump) > self.step_limit_ns:
                raise RuntimeError(f"PTP time step of {jump} ns detected on the master "
                                   f"clock; action grid invalid")
        self.clock.add(mid, p, h2 - h1)
        self.resyncs += 1

    def grid_time(self, k: int) -> int:
        return self.t0 + int(round(k * self.period))

    def prepare(self) -> int:
        """Measure the clock and choose T0 (a period multiple > lead + 1 period ahead)."""
        self.resync()
        now = self.clock.ptp_at(time.perf_counter_ns())
        P = self.period
        self.t0 = int(math.ceil(Fraction(now + self.lead_ns) / P + 1) * P)
        log.info("action scheduler: T0 = %d ns PTP, period %.3f ms, lead %.1f ms",
                 self.t0, float(P) / 1e6, self.lead_ns / 1e6)
        return self.t0

    # ------------------------------------------------------------------ thread
    def start(self) -> None:
        if self.t0 is None:
            self.prepare()
        self._thread = threading.Thread(target=self._run, name="action-scheduler", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        try:
            self._loop()
        except Exception as exc:                       # never die silently
            self.failure = f"{type(exc).__name__}: {exc}"
            log.error("action scheduler stopped: %s", self.failure)

    def _loop(self) -> None:
        k, errors = 0, 0
        next_resync = time.perf_counter_ns() + self.resync_ns
        while not self._stop.is_set():
            T = self.grid_time(k)
            if not wait_until_ns(self.clock.host_at(T - self.lead_ns), self._stop):
                break
            fire_host = time.perf_counter_ns()
            margin = T - self.clock.ptp_at(fire_host)
            if margin < self.min_margin_ns:
                self.records.append(ScheduledAction(k, T, fire_host, margin, "skipped_late"))
                if margin < -float(self.period):      # host stalled: jump to the next
                    behind = int((-margin) // float(self.period))   # feasible grid point
                    for j in range(1, min(behind, 10_000) + 1):
                        self.records.append(ScheduledAction(k + j, self.grid_time(k + j),
                                                            fire_host, margin, "skipped_late"))
                    k += min(behind, 10_000)
            else:
                try:
                    self._exec_node.value = T
                    self._fire_node.execute()
                    self.records.append(ScheduledAction(k, T, fire_host, margin, "fired"))
                    errors = 0
                except Exception as exc:
                    errors += 1
                    self.records.append(ScheduledAction(k, T, fire_host, margin, "error",
                                                        f"{type(exc).__name__}: {exc}"))
                    if errors >= self.max_consecutive_errors:
                        raise RuntimeError(f"{errors} consecutive action command errors: {exc}")
            k += 1
            if time.perf_counter_ns() >= next_resync:
                self.resync()                          # raises on a PTP time step
                next_resync += self.resync_ns

    # ------------------------------------------------------------------ results
    def fired_k(self) -> list[int]:
        return [r.k for r in self.records if r.status == "fired"]

    def fired_in(self, ptp_lo: int | None, ptp_hi: int | None) -> list[int]:
        """Fired k whose execute time lies in [ptp_lo, ptp_hi] (None = open)."""
        return [r.k for r in self.records if r.status == "fired"
                and (ptp_lo is None or r.execute_time_ns >= ptp_lo)
                and (ptp_hi is None or r.execute_time_ns <= ptp_hi)]

    def summary(self) -> dict:
        n = {st: sum(1 for r in self.records if r.status == st)
             for st in ("fired", "skipped_late", "error")}
        margins = [r.margin_ns / 1e6 for r in self.records if r.status == "fired"]
        return dict(t0_ns=self.t0, period_ns=float(self.period), lead_ms=self.lead_ns / 1e6,
                    scheduled=len(self.records), **n,
                    margin_ms=dict(min=_pct(margins, 0), p1=_pct(margins, 1),
                                   p50=_pct(margins, 50)),
                    resyncs=self.resyncs,
                    host_to_ptp_rate_ppm=round((self.clock.rate - 1.0) * 1e6, 3),
                    max_latch_roundtrip_us=round(self.clock.max_roundtrip_ns / 1e3, 1),
                    failure=self.failure,
                    errors_text=sorted({r.error for r in self.records if r.error})[:5])


# =========================================================================== software
@dataclass(slots=True)
class SoftwareFire:
    k: int
    host_ns: int            # perf_counter_ns just before TriggerSoftware was executed
    status: str             # "fired" | "not_armed" | "error"
    error: str = ""


class SoftwareTrigger:
    """Thread that paces software triggers at FRAME_RATE, as the user's capture.py did.

    Each slot k: wait until due (no bursts after falling behind - the schedule is resynced,
    as capture.py's _pace did), wait for TriggerArmed on the Triton and the Helios (if the
    node exists), then execute TriggerSoftware on the Triton and then the Helios (the
    original order). The skew between the two cameras is the GigE command latency (~ms);
    with PTP on it is MEASURED from the timestamps.
    """

    def __init__(self, helios: NodeKeeper, triton: NodeKeeper, period_ns: float,
                 armed_timeout_s: float = 1.0, max_consecutive_failures: int = 10) -> None:
        self.h, self.t = helios, triton
        self.period_ns = int(round(period_ns))
        self.armed_timeout_s = armed_timeout_s
        self.max_failures = max_consecutive_failures
        self.records: list[SoftwareFire] = []
        self._fired_host: list[int] = []        # host times of fired k (sorted)
        self._fired_k: list[int] = []
        self._lock = threading.Lock()
        self.failure: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._has_armed = {k.camera: k.has("TriggerArmed") for k in (helios, triton)}

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="software-trigger", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _armed(self, k: NodeKeeper) -> bool:
        if not self._has_armed[k.camera]:
            return True
        end = time.perf_counter() + self.armed_timeout_s
        while time.perf_counter() < end and not self._stop.is_set():
            if k.get("TriggerArmed", False):
                return True
            time.sleep(0.001)
        return False

    def _run(self) -> None:
        try:
            self._loop()
        except Exception as exc:
            self.failure = f"{type(exc).__name__}: {exc}"
            log.error("software trigger stopped: %s", self.failure)

    def _loop(self) -> None:
        k, failures = 0, 0
        due = time.perf_counter_ns()
        while not self._stop.is_set():
            now = time.perf_counter_ns()
            if now - due > self.period_ns:
                due = now                                  # fell behind: resync, no burst
            if not wait_until_ns(due, self._stop):
                break
            due += self.period_ns
            if not (self._armed(self.t) and self._armed(self.h)):
                self.records.append(SoftwareFire(k, time.perf_counter_ns(), "not_armed"))
                failures += 1
            else:
                host = time.perf_counter_ns()
                rec = SoftwareFire(k, host, "fired")
                with self._lock:                           # visible before the frame exists
                    self.records.append(rec)
                    self._fired_host.append(host)
                    self._fired_k.append(k)
                ok = self.t.execute("TriggerSoftware") and self.h.execute("TriggerSoftware")
                if ok:
                    failures = 0
                else:
                    rec.status, rec.error = "error", "TriggerSoftware execute failed"
                    failures += 1
            if failures >= self.max_failures:
                raise RuntimeError(f"cameras stopped answering software triggers "
                                   f"({failures} consecutive failures)")
            k += 1

    # ------------------------------------------------------------------ lookups (pairer)
    def first_fired(self) -> tuple[int, int] | None:
        with self._lock:
            return (self._fired_k[0], self._fired_host[0]) if self._fired_k else None

    def nearest_fired(self, host_ns: int) -> tuple[int, int] | None:
        """(k, fire host time) of the fired trigger nearest to host_ns."""
        with self._lock:
            if not self._fired_host:
                return None
            i = bisect.bisect_left(self._fired_host, host_ns)
            cands = [j for j in (i - 1, i) if 0 <= j < len(self._fired_host)]
            j = min(cands, key=lambda j: abs(self._fired_host[j] - host_ns))
            return self._fired_k[j], self._fired_host[j]

    def fired_in(self, host_lo: int | None, host_hi: int | None) -> list[int]:
        return [r.k for r in self.records if r.status == "fired"
                and (host_lo is None or r.host_ns >= host_lo)
                and (host_hi is None or r.host_ns <= host_hi)]

    def summary(self) -> dict:
        n = {st: sum(1 for r in self.records if r.status == st)
             for st in ("fired", "not_armed", "error")}
        gaps = [(b - a) / 1e6 for a, b in zip(self._fired_host, self._fired_host[1:])]
        return dict(period_ns=self.period_ns, attempted=len(self.records), **n,
                    interval_ms=dict(p50=_pct(gaps, 50), max=_pct(gaps, 100)),
                    failure=self.failure,
                    armed_node={k: v for k, v in self._has_armed.items()})

