"""CaptureEngine: open both cameras, configure a sync method, stream, pair, hand pairs to a writer.

Threads (C §3 "recommended writer design"):

    helios acquisition thread --+                                   +--> writer thread
                                +--> pairer --> guard --> hold -----+     writer(pair)
    triton acquisition thread --+    (one lock)  (PTP-   (until the  (bounded queue; full ->
                                                 free     PTP monitor pair DROPPED + counted,
                                                 checks)  covered it) acquisition never blocks)
    trigger thread (action scheduler or software trigger), PTP monitor thread

Each acquisition thread: get_buffer(timeout) -> copy the pixels out with numpy -> read the
metadata -> requeue_buffer immediately -> give (record, pixels) to the pairer. The camera's
buffer goes back to the SDK within ~1 ms, so slow disks can never starve the stream. The
acquisition threads never touch node maps.

Suspect pairs (PairGuard: host-time / frame-counter inconsistency; PTP monitor: lock lost or
offset above threshold around the pair's timestamp) are recorded but NOT given to the writer.

writer=None is the "no_save" mode used by sync tests: no pixels are copied at all.
"""
from __future__ import annotations

import csv
import logging
import queue
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from . import stats as st
from .methods import (HELIOS, TRITON, ActionScheduler, Cam, MethodResult, SoftwareTrigger,
                      configure_common, configure_with_fallback, open_cameras,
                      read_helios_scale_offsets, read_state, system_tl_keeper,
                      verify_ptpsync_rate)
from .nodes import NodeKeeper, NodeReport, restore_writes
from .pairing import (FRAME_COLUMNS, PAIR_COLUMNS, FrameRecord, GridPairer, NearestPairer,
                      Pair, PairGuard, TriggerPairer)
from .ptp import PtpMonitor, ptp_now_ns
from .settings import SyncSettings

log = logging.getLogger(__name__)

Writer = Callable[[Pair], None]
LOG_EVERY_S = 10.0                 # rate limit for repeated grab warnings
WRITER_JOIN_S = 60.0               # max wait for the writer to finish the queue


# =========================================================================== helpers
def pixel_format_name(pf: Any) -> str:
    return str(getattr(pf, "name", pf))


def buffer_to_array(buf: Any) -> np.ndarray:
    """Copy a buffer's pixels into a new numpy array (safe to keep after requeue).

    8-bit formats -> uint8 (H, W) or (H, W, C); 16-bit formats (Mono16, Coord3D_*16, ...)
    -> little-endian uint16, e.g. Coord3D_ABCY16 -> (H, W, 4).
    """
    w, h, bpp = int(buf.width), int(buf.height), int(buf.bits_per_pixel)
    nbytes = w * h * bpp // 8
    raw = np.ctypeslib.as_array(buf.pdata, shape=(nbytes,)).copy()   # the one real copy
    if "16" in pixel_format_name(buf.pixel_format):
        arr, ch = raw.view("<u2"), bpp // 16
    else:
        arr, ch = raw, bpp // 8
    return arr.reshape(h, w, ch) if ch > 1 else arr.reshape(h, w)


def write_csv(path: str | Path, columns: Iterable[str], rows: Iterable[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})


@dataclass
class _CamCounters:
    received: int = 0
    incomplete: int = 0
    warmup_frames: int = 0
    timeouts: int = 0
    timeouts_before_first: int = 0
    errors: int = 0
    pairer_errors: int = 0
    last_error: str = ""
    _last_log: dict = field(default_factory=dict)
    _suppressed: dict = field(default_factory=dict)


@dataclass
class CaptureResult:
    """Everything a run produced, without pixels (those went to the writer)."""

    pairs: list[Pair]
    frames: list[FrameRecord]
    stats: dict
    run_info: dict                 # the "sync" object of run.json
    interrupted: bool = False

    def pair_rows(self, only_written: bool = False) -> list[dict]:
        return [p.csv_row() for p in self.pairs if not only_written or p.status == "written"]

    def frame_rows(self) -> list[dict]:
        return [f.csv_row() for f in self.frames]

    def write_sync_csvs(self, pairs_csv: str | Path, frames_csv: str | Path,
                        only_written: bool = False) -> None:
        """Write the sync columns of frames.csv and the full sync_frames.csv."""
        write_csv(pairs_csv, PAIR_COLUMNS, self.pair_rows(only_written))
        write_csv(frames_csv, FRAME_COLUMNS, self.frame_rows())


# =========================================================================== engine
class CaptureEngine:
    """Synchronised Helios2+ / Triton capture.

    Usage::

        with CaptureEngine(SyncSettings(SYNC_METHOD="ptpsync"), writer=my_writer) as eng:
            eng.configure()
            result = eng.run(seconds=10)
        json.dump({"sync": result.run_info}, ...)
    """

    def __init__(self, settings: SyncSettings, writer: Writer | None = None,
                 system: Any = None, sdk_info: dict | None = None, helios_serial: str = "",
                 triton_serial: str = "", open_retries: int = 1) -> None:
        settings.validate()
        self.s = settings
        self.writer = writer
        self.system = system
        self.sdk_info = sdk_info or {}
        self.serials = (helios_serial, triton_serial)
        self.open_retries = open_retries
        self.report = NodeReport()
        self.helios: Cam | None = None
        self.triton: Cam | None = None
        self.tl: NodeKeeper | None = None
        self.exposure: dict = {}
        self.scan3d: tuple[float | None, list] = (None, [])
        self.image_info: dict = {}
        self.method: MethodResult | None = None
        self.restore_result: dict | None = None
        self._streaming: list[Cam] = []
        self._trigger: ActionScheduler | SoftwareTrigger | None = None
        self._monitor: PtpMonitor | None = None
        self._guard: PairGuard | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._recording = threading.Event()
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, int(settings.WRITER_QUEUE_PAIRS)))
        self._wstats = dict(written=0, dropped_queue_full=0, errors=0, max_queue_depth=0)
        self._counters = {HELIOS: _CamCounters(), TRITON: _CamCounters()}
        self._pairer: GridPairer | NearestPairer | None = None
        self._hold: list[Pair] = []
        self._save_counter = 0
        self._rejected = 0
        self._window = dict(ptp_start=None, ptp_end=None, host_start=None, host_end=None)

    @property
    def trigger(self) -> ActionScheduler | SoftwareTrigger | None:
        """The trigger thread object of the last run (action / software), else None."""
        return self._trigger

    # ------------------------------------------------------------------ lifecycle
    def __enter__(self) -> "CaptureEngine":
        self.open()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def open(self) -> None:
        """Load the SDK (unless a system object was given) and open both cameras."""
        if self.system is None:
            from .sdk import load_system   # noqa: PLC0415
            self.system, self.sdk_info = load_system()
        self.helios, self.triton = open_cameras(self.system, self.report, *self.serials,
                                                retries=self.open_retries)
        self.tl = system_tl_keeper(self.system, self.report)

    def configure(self) -> MethodResult:
        """Common settings + the requested sync method (with automatic fallback)."""
        if self.helios is None:
            self.open()
        self.exposure = configure_common(self.helios, self.triton, self.s)
        self.scan3d = read_helios_scale_offsets(self.helios.nm)
        for cam in (self.helios, self.triton):
            self.image_info[cam.name] = dict(width=cam.nm.get("Width"),
                                             height=cam.nm.get("Height"),
                                             pixel_format=cam.nm.get("PixelFormat"))
        self.method = configure_with_fallback(self.s.SYNC_METHOD, self.helios, self.triton,
                                              self.tl, self.s, self.exposure)
        return self.method

    def restore(self) -> dict | None:
        """Put back every node we wrote (RESTORE_ON_EXIT); streams must be stopped."""
        if not self.s.RESTORE_ON_EXIT:
            return None
        self._stop_streams()
        self.restore_result = restore_writes(self.report)
        return self.restore_result

    def close(self) -> None:
        """Stop everything, restore nodes, release the cameras. Safe to call twice."""
        for th in (self._trigger, self._monitor):
            if th is not None:
                th.stop()
        self._stop.set()
        self._stop_streams()
        if self.helios is not None:
            self.restore()
        for cam in (self.helios, self.triton):
            if cam is not None and self.system is not None:
                try:
                    self.system.destroy_device(cam.device)
                except Exception as exc:
                    log.warning("destroy_device(%s) failed: %s", cam.name, exc)
        self.helios = self.triton = None

    def _stop_streams(self) -> None:
        for cam in list(self._streaming):
            try:
                cam.device.stop_stream()
            except Exception as exc:
                log.warning("%s: stop_stream failed: %s", cam.name, exc)
            self._streaming.remove(cam)

    # ------------------------------------------------------------------ run
    def _master(self) -> tuple[str, NodeKeeper] | None:
        """(name, keeper) of the PTP master if PTP is locked and its time can be latched."""
        m = self.method
        if m is None or m.plan.clock_domain != "ptp" or m.ptp is None or m.ptp.master is None:
            return None
        cam = self.helios if m.ptp.master == HELIOS else self.triton
        if not (cam.nm.has("PtpDataSetLatch") and cam.nm.has("PtpDataSetLatchValue")):
            return None
        return cam.name, cam.nm

    def _latch_master(self) -> int | None:
        mk = self._master()
        if mk is None:
            return None
        try:
            return ptp_now_ns(mk[1])
        except Exception as exc:
            log.warning("could not latch PTP time: %s", exc)
            return None

    def _make_pairer(self) -> GridPairer | NearestPairer:
        plan, s = self.method.plan, self.s
        common = dict(clock_domain=plan.clock_domain, helios_center_us=s.helios_center_us(),
                      planned_lag_ns=plan.planned_lag_ns)
        if plan.pairing == "grid":
            return GridPairer(plan.period_ns, lags_ns=plan.lags_ns, anchor_ns=plan.anchor_ns,
                              horizon=s.PAIR_HORIZON_FRAMES, **common)
        if plan.pairing == "trigger":
            return TriggerPairer(plan.period_ns, fires=self._trigger,
                                 horizon=s.PAIR_HORIZON_FRAMES, **common)
        key = "ts_ns" if plan.pairing == "nearest_ts" else "host_ns"
        return NearestPairer(plan.period_ns, key=key, max_pending=s.PAIR_HORIZON_FRAMES,
                             **common)

    def _status(self, t_rec: float) -> dict:
        return dict(elapsed_s=time.monotonic() - t_rec, pairs=self._save_counter,
                    written=self._wstats["written"],
                    writer_dropped=self._wstats["dropped_queue_full"],
                    queue_depth=self._queue.qsize(), rejected_suspect=self._rejected,
                    failed_grabs=sum(c.timeouts + c.errors for c in self._counters.values()))

    def run(self, seconds: float | None = None,
            should_stop: Callable[[dict], str | None] | None = None,
            warmup_s: float = 0.0) -> CaptureResult:
        """Stream for `seconds` (default SECONDS; 0/None = until stopped) or until
        should_stop(status) returns a reason, Ctrl+C, or a trigger thread dies.
        Frames during the first `warmup_s` (auto exposure settling) are discarded.
        Always returns a result with statistics."""
        if self.method is None:
            self.configure()
        seconds = float(self.s.SECONDS if seconds is None else seconds)
        plan, s = self.method.plan, self.s
        self._stop.clear()
        self._recording.clear()
        threads: list[threading.Thread] = []
        writer_thread: threading.Thread | None = None
        info: dict = dict(rate_check=None, state_after_start=None, stop_reason=None,
                          aborted_reason=None, warnings=[])
        interrupted = False
        t_rec = time.monotonic()
        try:
            for cam in (self.helios, self.triton):
                cam.device.start_stream(int(s.BUFFERS))
                self._streaming.append(cam)
            info["state_after_start"] = {c.name: read_state(c) for c in (self.helios, self.triton)}
            if plan.method == "ptpsync":
                info["rate_check"] = verify_ptpsync_rate(self.helios, self.triton, s)
                if not info["rate_check"]["ok"]:
                    info["warnings"].append(f"AcquisitionFrameRate after start reads "
                                            f"{info['rate_check']}, expected {s.FRAME_RATE}")
            if plan.method == "action":
                self._trigger = ActionScheduler(self.tl, self.helios.nm, plan.period_ns,
                                                plan.action["lead_ms"], resync_s=s.ACTION_RESYNC_S)
                plan.anchor_ns = self._trigger.prepare()   # grid k = 0 at T0
            elif plan.method == "software":
                self._trigger = SoftwareTrigger(self.helios.nm, self.triton.nm, plan.period_ns,
                                                s.SOFTWARE_ARMED_TIMEOUT_S)
            self._pairer = self._make_pairer()
            if plan.clock_domain == "ptp":
                self._guard = PairGuard(plan.period_ns)
                mk = self._master()
                if mk is not None and s.PTP_MONITOR_S > 0:
                    self._monitor = PtpMonitor(self.helios.nm, self.triton.nm, mk[0],
                                               s.PTP_MONITOR_S, s.PTP_LOCK_THRESHOLD_NS)
                    self._monitor.start()
            if self.writer is not None:
                writer_thread = threading.Thread(target=self._writer_loop, name="pair-writer",
                                                 daemon=True)
                writer_thread.start()
            for cam in (self.helios, self.triton):
                th = threading.Thread(target=self._acquire_loop, args=(cam,),
                                      name=f"acquire-{cam.name}", daemon=True)
                th.start()
                threads.append(th)
            if self._trigger is not None:
                self._trigger.start()
            if warmup_s > 0:
                log.info("warming up %.1f s (frames discarded)", warmup_s)
                t_w = time.monotonic()
                while time.monotonic() - t_w < warmup_s and not self._trigger_dead(info):
                    time.sleep(0.05)
            self._window["ptp_start"] = self._latch_master()
            self._window["host_start"] = time.perf_counter_ns()
            self._recording.set()
            t_rec = time.monotonic()
            log.info("capturing for %s with method %s ...",
                     f"{seconds:.1f} s" if seconds else "an open-ended time", plan.method)
            while info["aborted_reason"] is None:
                if seconds and time.monotonic() - t_rec >= seconds:
                    info["stop_reason"] = "time limit"
                    break
                if should_stop is not None:
                    reason = should_stop(self._status(t_rec))
                    if reason:
                        info["stop_reason"] = reason
                        break
                if self._trigger_dead(info):
                    break
                with self._lock:
                    self._release()
                time.sleep(0.05)
        except KeyboardInterrupt:
            interrupted = True
            info["stop_reason"] = "Ctrl+C"
            log.warning("Ctrl+C: stopping capture, statistics will still be written")
        finally:
            if info["stop_reason"] is None:
                info["stop_reason"] = info["aborted_reason"] or "error"
            elapsed = time.monotonic() - t_rec
            summary = self._shutdown(threads, writer_thread)
        while True:                     # a second Ctrl+C must not lose the statistics
            try:
                return self._result(seconds, elapsed, summary, info, interrupted)
            except KeyboardInterrupt:
                log.warning("Ctrl+C again - still writing the statistics")

    def _trigger_dead(self, info: dict) -> bool:
        trig = self._trigger
        if trig is None or info["aborted_reason"] is not None:
            return info["aborted_reason"] is not None
        if trig.failure or not trig.is_alive():
            info["aborted_reason"] = (f"{type(trig).__name__} died: "
                                      f"{trig.failure or 'thread ended'}")
            log.error("RUN ABORTED: %s", info["aborted_reason"])
            return True
        return False

    # ------------------------------------------------------------------ shutdown
    def _step(self, name: str, fn: Callable, *args: Any) -> Any:
        """Run one shutdown step; a (second) Ctrl+C is logged and the shutdown goes on."""
        try:
            return fn(*args)
        except KeyboardInterrupt:
            log.warning("Ctrl+C during shutdown (%s) - finishing shutdown anyway", name)
        except Exception as exc:
            log.exception("shutdown step %s failed: %s", name, exc)
        return None

    def _shutdown(self, threads: list[threading.Thread],
                  writer_thread: threading.Thread | None) -> dict:
        """Stop in a safe order: latch end, trigger thread, drain, acquisition, monitor,
        pairer, writer, streams, restore."""
        s = self.s
        self._window["ptp_end"] = self._step("latch end", self._latch_master)
        self._window["host_end"] = time.perf_counter_ns()
        lead_s = 0.0
        if self._trigger is not None:
            self._step("stop trigger", self._trigger.stop)
            lead_s = getattr(self._trigger, "lead_ns", 0) / 1e9
        # Frames exposed before the end are still in transit (readout + transfer ~0.1 s):
        # keep acquiring for a moment so they are counted.
        if self._recording.is_set() and self.method.plan.pairing in ("grid", "trigger"):
            self._step("drain", time.sleep, lead_s + 2 * s.period_ns / 1e9 + 0.3)
        self._stop.set()
        for th in threads:
            self._step("join acquisition", th.join, s.get_buffer_timeout_ms() / 1000.0 + 2.0)
        if self._monitor is not None:
            self._step("final PTP sample", self._monitor.sample)
            self._step("stop monitor", self._monitor.stop)
        summary = self._step("finalize pairer", self._finalize) or {}
        if writer_thread is not None:
            self._step("finish writer", self._finish_writer, writer_thread)
        self._step("stop streams", self._stop_streams)
        if self.s.RESTORE_ON_EXIT:
            self._step("restore nodes", self.restore)
        return summary

    def _finalize(self) -> dict:
        pairer, plan, w = self._pairer, self.method.plan, self._window
        if pairer is None:
            return {}
        P = int(plan.period_ns)
        expected, k_window = None, None
        if plan.method == "action" and isinstance(self._trigger, ActionScheduler):
            lo = None if w["ptp_start"] is None else w["ptp_start"] + P
            expected = self._trigger.fired_in(lo, w["ptp_end"])
        elif plan.method == "software" and isinstance(self._trigger, SoftwareTrigger):
            expected = self._trigger.fired_in(w["host_start"] + P, w["host_end"])
        elif isinstance(pairer, GridPairer) and w["ptp_start"] is not None:
            k_window = pairer.k_window(w["ptp_start"] + P, w["ptp_end"])
        with self._lock:
            new_pairs, summary = pairer.finalize(expected_k=expected, k_window=k_window)
            self._on_pairs(new_pairs)
            self._release(final=True)
        return summary

    def _finish_writer(self, writer_thread: threading.Thread) -> None:
        deadline = time.monotonic() + WRITER_JOIN_S
        while writer_thread.is_alive() and time.monotonic() < deadline:
            try:
                self._queue.put(None, timeout=0.5)       # sentinel
                break
            except queue.Full:
                continue
        writer_thread.join(max(0.0, deadline - time.monotonic()))
        if writer_thread.is_alive():
            log.error("writer did not finish within %.0f s (disk hung?) - %d pairs unsaved",
                      WRITER_JOIN_S, self._queue.qsize())

    # ------------------------------------------------------------------ acquisition
    def _ratelog(self, c: _CamCounters, cam: str, kind: str, msg: str) -> None:
        """Log the first occurrence, then at most every LOG_EVERY_S with a count."""
        now = time.monotonic()
        last = c._last_log.get(kind)
        if last is None or now - last >= LOG_EVERY_S:
            extra = c._suppressed.get(kind, 0)
            log.warning("%s: %s%s", cam, msg, f" (+{extra} more since last message)"
                        if extra else "")
            c._last_log[kind], c._suppressed[kind] = now, 0
        else:
            c._suppressed[kind] = c._suppressed.get(kind, 0) + 1

    def _acquire_loop(self, cam: Cam) -> None:
        c = self._counters[cam.name]
        timeout_ms = self.s.get_buffer_timeout_ms()
        while not self._stop.is_set():
            t_call = time.perf_counter_ns()
            try:
                buf = cam.device.get_buffer(timeout=timeout_ms)
            except Exception as exc:
                if self._stop.is_set():
                    break
                if isinstance(exc, TimeoutError) or "timeout" in str(exc).lower():
                    c.timeouts += 1
                    if c.received == 0:
                        c.timeouts_before_first += 1
                    self._ratelog(c, cam.name, "timeout",
                                  f"no frame for {timeout_ms} ms (get_buffer timeout)")
                else:
                    c.errors += 1
                    c.last_error = f"{type(exc).__name__}: {exc}"
                    self._ratelog(c, cam.name, "error", f"get_buffer failed: {c.last_error}")
                    time.sleep(0.01)                     # never spin on a broken stream
                continue
            host_ns, wall_ns = time.perf_counter_ns(), time.time_ns()
            if not self._recording.is_set():             # warm-up: discard
                c.warmup_frames += 1
                self._requeue(cam, buf)
                continue
            rec = image = None
            try:
                rec, image = self._read_buffer(cam, buf, host_ns, wall_ns)
                rec.grab_ms = (host_ns - t_call) / 1e6
            except Exception as exc:
                c.errors += 1
                c.last_error = f"read buffer: {type(exc).__name__}: {exc}"
                self._ratelog(c, cam.name, "read", c.last_error)
            finally:
                self._requeue(cam, buf)                  # give the memory back at once
            if rec is None:
                continue
            c.received += 1
            if rec.incomplete:
                c.incomplete += 1
            try:
                with self._lock:
                    self._on_pairs(self._pairer.add(rec, image))
            except Exception:
                c.pairer_errors += 1
                if c.pairer_errors == 1:
                    log.exception("%s: pairing failed (logged once, counted)", cam.name)

    def _requeue(self, cam: Cam, buf: Any) -> None:
        try:
            cam.device.requeue_buffer(buf)
        except Exception as exc:
            self._ratelog(self._counters[cam.name], cam.name, "requeue",
                          f"requeue_buffer failed: {exc}")

    def _read_buffer(self, cam: Cam, buf: Any, host_ns: int,
                     wall_ns: int) -> tuple[FrameRecord, np.ndarray | None]:
        """Copy everything we need out of the buffer (it is requeued right after)."""
        rec = FrameRecord(camera=cam.name, frame_id=int(buf.frame_id),
                          ts_ns=int(buf.timestamp_ns), host_ns=host_ns, host_wall_ns=wall_ns,
                          incomplete=bool(buf.is_incomplete), width=int(buf.width),
                          height=int(buf.height),
                          pixel_format=pixel_format_name(buf.pixel_format),
                          size_filled=int(buf.size_filled))
        if cam.name == TRITON:
            rec.exposure_us, rec.exposure_source = self._triton_exposure(buf)
            if rec.exposure_source == "cap_upper_bound":
                rec.add_note("exposure_is_cap_upper_bound")   # not the real per-frame value
            elif rec.exposure_source == "fixed_setting":
                rec.add_note("exposure_from_setting")
        image = None
        if self.writer is not None and not rec.incomplete:
            t0 = time.perf_counter_ns()
            image = buffer_to_array(buf)
            rec.copy_ms = (time.perf_counter_ns() - t0) / 1e6
        return rec, image

    def _triton_exposure(self, buf: Any) -> tuple[float | None, str]:
        """Per-frame exposure: chunk value if enabled, else the fixed setting, else the cap
        (an UPPER BOUND under auto exposure - marked as such)."""
        if self.exposure.get("chunk_exposure"):
            try:
                if buf.has_chunkdata:
                    node = buf.get_chunk(["ChunkExposureTime"]).get("ChunkExposureTime")
                    if node is not None:
                        return float(node.value), "chunk"
            except Exception:
                pass
        if self.exposure.get("fixed_us"):
            return float(self.exposure["fixed_us"]), "fixed_setting"
        cap = self.exposure.get("cap_us")
        return (float(cap), "cap_upper_bound") if cap else (None, "")

    # ------------------------------------------------------------------ pairs
    def _on_pairs(self, pairs: list[Pair]) -> None:
        """Guard check, then hold until the PTP monitor has covered the pair (under lock)."""
        end = self._window["ptp_end"]
        for p in pairs:
            if end is not None and self._master_ts(p) > end:
                # exposed after the recording stopped (collected during the drain):
                # outside the capture window -> not saved, not in the statistics
                p.status, p.images = "after_stop", None
                continue
            if self._guard is not None:
                why = self._guard.check(p)
                if why:
                    p.suspect, p.suspect_reason = True, why
            self._hold.append(p)
        self._release()

    def _master_ts(self, p: Pair) -> int:
        m = self.method.ptp.master if self.method.ptp else HELIOS
        return p.triton.ts_ns if m == TRITON else p.helios.ts_ns

    def _release(self, final: bool = False) -> None:
        """Dispatch held pairs once the PTP monitor has a sample after their timestamp
        (or they waited too long). Called under self._lock."""
        mon = self._monitor
        max_hold_ns = int((3 * self.s.PTP_MONITOR_S + 1.0) * 1e9)
        now = time.perf_counter_ns()
        while self._hold:
            p = self._hold[0]
            if mon is not None:
                ts = self._master_ts(p)
                cov = mon.covered_until()
                if not final and (cov is None or cov < ts) and now - p.helios.host_ns < max_hold_ns:
                    break
                bad = mon.bad_reason(ts)
                if bad:
                    p.suspect = True
                    p.suspect_reason = f"{p.suspect_reason}; {bad}" if p.suspect_reason else bad
            self._hold.pop(0)
            self._dispatch(p)

    def _dispatch(self, p: Pair) -> None:
        """Hand a pair to the writer queue; never blocks."""
        if p.suspect:
            p.status, p.images = "rejected_suspect", None
            self._rejected += 1
            for f in (p.helios, p.triton):
                f.add_note("suspect_pair")
            log.warning("pair %d rejected (suspect): %s", p.pair_index, p.suspect_reason)
            return
        p.save_index = self._save_counter
        self._save_counter += 1
        if self.writer is None:
            p.status, p.images = "not_saved", None
            return
        try:
            self._queue.put_nowait(p)
            p.queued = True
        except queue.Full:
            p.status, p.images = "writer_dropped", None
            self._wstats["dropped_queue_full"] += 1
            for f in (p.helios, p.triton):
                f.add_note("writer_dropped")
            return
        p.queue_depth = self._queue.qsize()
        self._wstats["max_queue_depth"] = max(self._wstats["max_queue_depth"], p.queue_depth)

    def _writer_loop(self) -> None:
        while True:
            p = self._queue.get()
            if p is None:
                break
            try:
                self.writer(p)
                p.status = "written"
                self._wstats["written"] += 1
            except Exception:
                p.status = "writer_error"
                self._wstats["errors"] += 1
                for f in (p.helios, p.triton):
                    f.add_note("writer_error")
                log.exception("writer failed for pair %d", p.pair_index)
            finally:
                p.images = None                          # free the pixels

    # ------------------------------------------------------------------ results
    def _result(self, seconds: float, elapsed: float, summary: dict, info: dict,
                interrupted: bool) -> CaptureResult:
        pairer = self._pairer
        pairs = list(pairer.pairs) if pairer else []
        frames = list(pairer.frames) if pairer else []
        stats = self._stats(pairs, frames, summary)
        run_info = self._run_info(seconds, elapsed, stats, info, interrupted)
        log.info("capture done: %d pairs (%d rejected), %.2f fps, residual median %s us, "
                 "drops h/t %s/%s", stats["pairs"], stats["rejected_suspect"],
                 stats["fps"] or 0.0, stats["residual_us"]["median"],
                 stats["dropped_helios"], stats["dropped_triton"])
        return CaptureResult(pairs, frames, stats, run_info, interrupted)

    def _freerun_drops(self, frames: list[FrameRecord]) -> dict:
        """Free-run drop ESTIMATE per camera: frame_id gaps + frames missing at the start or
        end relative to the other camera's host arrival times (a camera that starts late
        or dies early shows up)."""
        P = self.method.plan.period_ns
        good = {c: [f for f in frames if f.camera == c and f.valid_ts] for c in (HELIOS, TRITON)}
        hosts = [f.host_ns for fs in good.values() for f in fs]
        out = {}
        for c, fs in good.items():
            gaps = st.frame_id_gaps(f.frame_id for f in fs)
            if not fs:
                n = int(round((max(hosts) - min(hosts)) / P)) + 1 if hosts else 0
                out[c] = dict(dropped=n, gaps=0, head=n, tail=0, received=0)
                continue
            head = max(0, int(round((min(f.host_ns for f in fs) - min(hosts)) / P)))
            tail = max(0, int(round((max(hosts) - max(f.host_ns for f in fs)) / P)))
            out[c] = dict(dropped=gaps + head + tail, gaps=gaps, head=head, tail=tail,
                          received=len(fs))
        return out

    def _stats(self, pairs: list[Pair], frames: list[FrameRecord], summary: dict) -> dict:
        plan, s = self.method.plan, self.s
        ptp = plan.clock_domain == "ptp"
        ok = [p for p in pairs if not p.suspect and p.status != "after_stop"]
        pair_t = [p.helios.ts_ns if ptp else p.helios.host_ns for p in ok]
        grid = plan.pairing in ("grid", "trigger")
        free = None if grid else self._freerun_drops(frames)
        per_cam = {}
        for cam in (HELIOS, TRITON):
            fr = [f for f in frames if f.camera == cam]
            c = self._counters[cam]
            entry = dict(received=c.received, incomplete=c.incomplete,
                         warmup_frames_discarded=c.warmup_frames,
                         failed_grabs=c.timeouts + c.errors, timeouts=c.timeouts,
                         timeouts_before_first_frame=c.timeouts_before_first,
                         grab_errors=c.errors, pairer_errors=c.pairer_errors,
                         last_error=c.last_error,
                         frame_id_gaps=st.frame_id_gaps(f.frame_id for f in fr),
                         invalid_ts=summary.get("invalid_ts", {}).get(cam, 0),
                         unpaired=summary.get("unpaired", {}).get(cam),
                         edge_frames=summary.get("edge_frames", {}).get(cam),
                         fps_received=_round(st.fps_from_timestamps(
                             [f.ts_ns for f in fr if not f.incomplete and f.valid_ts])))
            if grid:
                entry["dropped"] = summary.get("dropped", {}).get(cam)
                entry["frames_expected"] = summary.get("frames_expected", {}).get(cam)
                entry["frames_received"] = summary.get("frames_received", {}).get(cam)
                entry["drop_definition"] = ("expected grid/trigger index with no valid buffer "
                                            f"({summary.get('window_source')})")
                if plan.pairing == "grid":
                    entry["grid_residual_us"] = st.summarize(
                        f.grid_residual_ns / 1000.0 for f in fr
                        if f.grid_residual_ns is not None and not f.incomplete)
            else:
                fd = free[cam]
                entry["dropped"] = fd["dropped"]
                entry["frames_received"] = fd["received"]
                entry["frames_expected"] = fd["received"] + fd["dropped"]
                entry["drop_definition"] = ("ESTIMATE: frame_id gaps + missing head/tail vs the "
                                            f"other camera ({fd})")
            per_cam[cam] = entry
        rec_s = None
        if self._window["host_start"] is not None and self._window["host_end"] is not None:
            rec_s = (self._window["host_end"] - self._window["host_start"]) / 1e9
        trig = self._trigger.summary() if self._trigger is not None else {}
        expected = summary.get("_expected")
        if expected is not None:      # grid/trigger: pairs whose k was expected / expected k
            in_win = sum(1 for p in ok if p.grid_k in expected[HELIOS])
            pair_yield = in_win / len(expected[HELIOS]) if expected[HELIOS] else None
        else:                         # free-run: pairs / (recording time x frame rate)
            pair_yield = len(ok) / (rec_s * s.FRAME_RATE) if rec_s else None
        return dict(
            pairs=len(ok),
            pairs_total=len(pairs),
            pairs_after_stop=sum(1 for p in pairs if p.status == "after_stop"),
            rejected_suspect=sum(1 for p in pairs if p.suspect),
            suspect_reasons=dict(Counter(_reason_kind(p.suspect_reason)
                                         for p in pairs if p.suspect).most_common(10)),
            recording_s=_round(rec_s),
            duration_s=_round(st.duration_s(pair_t)),
            fps=_round(st.fps_from_timestamps(pair_t)),
            pair_yield=_round(pair_yield),
            dropped_helios=per_cam[HELIOS]["dropped"],
            dropped_triton=per_cam[TRITON]["dropped"],
            frames_expected={c: per_cam[c]["frames_expected"] for c in per_cam},
            frames_received={c: per_cam[c]["frames_received"] for c in per_cam},
            failed_grabs=sum(per_cam[c]["failed_grabs"] for c in per_cam),
            incomplete=sum(per_cam[c]["incomplete"] for c in per_cam),
            invalid_ts=sum(per_cam[c]["invalid_ts"] for c in per_cam),
            offset_start_us=st.summarize(p.offset_start_us for p in ok),
            residual_us=st.summarize(p.residual_us for p in ok),
            offset_center_model_us=st.summarize(p.offset_center_model_us for p in ok),
            host_delta_ms=st.summarize(p.host_delta_ms for p in ok),
            action_skipped_late=trig.get("skipped_late"),
            software_not_armed=trig.get("not_armed"),
            triton_exposure_source=dict(Counter(f.exposure_source or "unknown" for f in frames
                                                if f.camera == TRITON)),
            per_camera=per_cam,
            pairing={k: v for k, v in summary.items()
                     if k not in ("unpaired", "edge_frames", "dropped", "frames_expected",
                                  "frames_received", "invalid_ts", "_expected")},
            writer=dict(self._wstats, enabled=self.writer is not None),
        )

    def _run_info(self, seconds: float, elapsed: float, stats: dict, info: dict,
                  interrupted: bool) -> dict:
        m, plan, s = self.method, self.method.plan, self.s
        rep = self.report.to_dict()
        ptp = m.ptp.to_dict() if m.ptp is not None else None
        if ptp is not None and self._monitor is not None:
            ptp["monitor"] = self._monitor.to_dict()
        src = stats["triton_exposure_source"]
        if src.get("cap_upper_bound"):
            info["warnings"].append("Triton exposure chunk unavailable: the exposure CAP was "
                                    "used as a stand-in, offset_center_model_us is an upper-"
                                    "bound model only")
        if ptp is not None and ptp.get("settled") is False:
            info["warnings"].append("PTP never settled below the lock threshold")
        return dict(
            method_requested=m.method_requested,
            method_used=m.method_used,
            fallback_chain=m.fallback_chain,
            clock_domain=plan.clock_domain,
            frame_rate=s.FRAME_RATE,
            seconds=seconds,
            seconds_actual=round(elapsed, 3),
            stop_reason=info["stop_reason"],
            aborted_reason=info["aborted_reason"],
            interrupted=interrupted,
            exposure_cap_us=s.EXPOSURE_CAP_US,
            helios_center_us=s.helios_center_us(),
            helios_window=s.helios_window(),
            alignment_requested=plan.alignment_requested,
            alignment_used=plan.alignment_used,
            alignment_note=plan.alignment_note,
            planned_lag_us=plan.planned_lag_ns / 1000.0,
            triton_lag_us=plan.planned_lag_ns / 1000.0,
            triton_trigger_delay_us=plan.trigger_delay_us.get(TRITON),
            ptpsync_offset_ns=plan.ptpsync_offset_ns,
            ptpsync_rate_check=info["rate_check"],
            state_after_configure=plan.state,
            state_after_start=info["state_after_start"],
            capture_window=dict(self._window),
            ptp=ptp,
            cameras={c.name: c.identity for c in (self.helios, self.triton) if c is not None},
            exposure={k: v for k, v in self.exposure.items()},
            triton_exposure_source=src,
            trigger=self._trigger.summary() if self._trigger is not None else None,
            action=(self._trigger.summary() if isinstance(self._trigger, ActionScheduler)
                    else None),
            plan=plan.to_dict(),
            warnings=info["warnings"],
            missing_nodes=rep["missing_nodes"],
            failed_node_writes=rep["failed_node_writes"],
            clamped_node_writes=rep["clamped_node_writes"],
            node_writes=rep["node_writes"],
            restore=self.restore_result,
            stats=stats,
            sdk=dict(arena_api_version=self.sdk_info.get("arena_api_version"),
                     mock=self.sdk_info.get("mock")),
            settings=s.as_dict(),
        )


def _reason_kind(reason: str) -> str:
    """'host_delta -99.0 ms vs ...; ptp_monitor: offset 5 ns' -> 'host_delta+ptp_monitor'."""
    return "+".join(sorted({part.split()[0].rstrip(":") for part in reason.split("; ")
                            if part.strip()}))


def _round(v: float | None, nd: int = 4) -> float | None:
    return None if v is None else round(v, nd)
