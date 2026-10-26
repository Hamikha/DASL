"""
capture_sync.py - synchronised raw paired capture from the Helios2+ and the Triton.

The successor of capture.py. Same output folder layout, so process.py works unchanged, but
the two cameras are now synchronised in hardware time:

    "ptpsync"  (default) both cameras share one clock (IEEE 1588 PTP) and their firmware
               starts every exposure on the same time grid. Falls back automatically to
               "action" -> "software" -> "freerun_ptp" -> "freerun" and says why.

    python capture_sync.py

Stopping, whichever is most convenient:
    - wait for SECONDS to run out
    - press q or Esc in this console window
    - press Ctrl+C
    - create a file named STOP inside the run folder

Each run gets its own folder:
    run_YYYYMMDD_HHMMSS/
        raw/helios_000000.npy   Helios Coord3D_ABCY16, uint16 (H, W, 4)
        raw/triton_000000.npy   Triton pixels, uint8 (H, W) Bayer or (H, W, 3) RGB
        frames.csv              one row per captured pair: capture.py's 11 columns, then
                                the sync columns (timestamps, offsets, suspect flag, ...)
        sync_frames.csv         one row per received frame of either camera
        run.json                every setting, camera and result (capture.py keys) + "sync"
        orientation.yml         copy of the calibration in use

Do not click inside the console while recording. Windows "QuickEdit" freezes
any program whose console is being selected, which looks like a hang.

LUCID_MOCK=1 runs it on the simulated cameras (lucid_sync/../mock), no hardware needed.
"""

import csv
import datetime
import json
import logging
import os
import shutil
import signal
import sys
import time

import numpy as np

try:
    import msvcrt                     # Windows console keys, optional
except ImportError:
    msvcrt = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # lucid_sync/ next to us

from lucid_sync.engine import CaptureEngine, write_csv  # noqa: E402
from lucid_sync.methods import CameraNotFound  # noqa: E402
from lucid_sync.pairing import FRAME_COLUMNS, PAIR_COLUMNS  # noqa: E402
from lucid_sync.ptp import SyncSetupError  # noqa: E402
from lucid_sync.sdk import load_system  # noqa: E402
from lucid_sync.settings import SyncSettings  # noqa: E402


# ============================================================================
# SETTINGS
# ============================================================================

# Keep recordings out of OneDrive. A synced folder uploads every frame while
# you record, competing for disk and CPU. This resolves to C:\Users\<you>\captures.
OUTPUT_ROOT = os.path.join(os.path.expanduser("~"), "captures")
CALIB_FILE = "orientation.yml"         # copied into each run for process.py

SECONDS = 60                           # 0 = no time limit
MAX_FRAMES = 0                         # 0 = no frame limit
FRAME_RATE = 10.0                      # pairs per second (15 is the 1 GbE maximum)

# "ptpsync":     PTP clock + camera-scheduled exposures on one grid. Best.
# "action":      PTP clock + host-scheduled action commands.
# "software":    software trigger as in capture.py (PTP timestamps if PTP works).
# "freerun_ptp": cameras free-run on a shared PTP clock (offset measured, not controlled).
# "freerun":     cameras free-run on their own clocks (the old behaviour).
# A method that cannot work on these cameras falls back to the next one, logged.
SYNC_METHOD = "ptpsync"

# Triton auto exposure stays on but never exceeds this. 8 ms keeps hand-held blur at
# about 3 px at 10 deg/s. 20000 also keeps 10 fps but blurs ~2.5x more when hand-held.
EXPOSURE_CAP_US = 8000
TRITON_FIXED_EXPOSURE_US = None        # a number = fixed exposure instead (us)

# "center": the Triton exposure is centred on the middle of the Helios depth frame
#           (the Helios collects light for ~12 ms in this mode). "start": both start together.
ALIGNMENT = "center"
HELIOS_CENTER_US = None                # None = timing model (~6405 us here); set after a scope check

# BayerRG8 sends 1 byte per pixel instead of 3, so the Triton fits three
# times more frames through its share of the link. process.py converts it to
# colour. Same sensor resolution, so the calibration stays valid.
TRITON_PIXEL_FORMAT = "BayerRG8"       # "BayerRG8" or "RGB8"

# user choice 2026-10-26: multi-frequency for fewer range-wrap errors;
# capture.py used Distance6000mmSingleFreq
HLT_OPERATING_MODE = "Distance5000mmMultiFreq"
HLT_EXPOSURE = "Exp250Us"              # Exp1000Us / Exp250Us / Exp62_5Us
HLT_CONVERSION_GAIN = "High"           # Low / High

# Both cameras share one gigabit link to the laptop. Each gets a fixed share
# in bytes per second so they never overflow it together. Keep the sum under
# 110_000_000.
HLT_THROUGHPUT = 40_000_000
TRI_THROUGHPUT = 60_000_000

# Leave empty to pick the cameras automatically by model.
HELIOS_SERIAL = ""
TRITON_SERIAL = ""

WARMUP_SECONDS = 3.0                   # lets Triton auto-exposure settle
PTP_TIMEOUT_S = 60                     # PTP negotiation can take ~40 s (LUCID)
GRAB_TIMEOUT_MS = 1000                 # a camera silent this long counts a failed grab
STREAM_BUFFERS = 10
WRITE_QUEUE_SIZE = 100                 # pairs held in RAM waiting for the disk
MIN_FREE_GB_START = 10
MIN_FREE_GB_STOP = 3

# Rarely needed: any other lucid_sync SyncSettings field, e.g. {"PTP_MONITOR_S": 2.0}.
ADVANCED = {}


# ============================================================================
# SMALL HELPERS
# ============================================================================

TAB = "  "
ORIGINAL_COLUMNS = ["frame", "host_time", "hlt_frame_id", "tri_frame_id",
                    "hlt_ts_ns", "tri_ts_ns", "arrival_gap_ms", "grab_ms",
                    "copy_ms", "queue_depth", "queued"]


def log(message):
    print(f"{TAB}{message}", flush=True)


def plain(value):
    """Convert node values (enums, numpy scalars) into JSON-safe types."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [plain(v) for v in value]
    name = getattr(value, "name", None)
    return name if isinstance(name, str) else str(value)


def free_gb(path):
    return shutil.disk_usage(path).free / 1e9


def write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(plain(data), f, indent=2)
    os.replace(tmp, path)


def sync_settings():
    s = dict(SYNC_METHOD=SYNC_METHOD, FRAME_RATE=float(FRAME_RATE), SECONDS=float(SECONDS),
             EXPOSURE_CAP_US=float(EXPOSURE_CAP_US),
             TRITON_FIXED_EXPOSURE_US=TRITON_FIXED_EXPOSURE_US, ALIGNMENT=ALIGNMENT,
             HELIOS_CENTER_US=HELIOS_CENTER_US, TRITON_PIXEL_FORMAT=TRITON_PIXEL_FORMAT,
             HELIOS_OPERATING_MODE=HLT_OPERATING_MODE, HELIOS_EXPOSURE_SELECTOR=HLT_EXPOSURE,
             HELIOS_CONVERSION_GAIN=HLT_CONVERSION_GAIN,
             THROUGHPUT_LIMIT_HELIOS=int(HLT_THROUGHPUT),
             THROUGHPUT_LIMIT_TRITON=int(TRI_THROUGHPUT), PTP_TIMEOUT_S=float(PTP_TIMEOUT_S),
             GET_BUFFER_TIMEOUT_MS=int(GRAB_TIMEOUT_MS), BUFFERS=int(STREAM_BUFFERS),
             WRITER_QUEUE_PAIRS=int(WRITE_QUEUE_SIZE), RESTORE_ON_EXIT=True)
    s.update(ADVANCED)
    return SyncSettings(**s)


# ============================================================================
# DISK WRITER (runs on the engine's writer thread)
# ============================================================================

class RawWriter:
    """np.save of both frames per pair, same file names as capture.py. A disk error stops
    the run (as capture.py did); pairs that do not fit in the queue are dropped and
    counted by the engine rather than stalling the cameras."""

    def __init__(self, raw_dir):
        self.raw_dir = raw_dir
        self.error = None
        self.saved = 0
        self.write_seconds = 0.0

    def __call__(self, pair):
        if self.error is not None:
            raise RuntimeError(f"disk write failed earlier: {self.error}")
        index = pair.save_index
        try:
            t0 = time.perf_counter()
            np.save(os.path.join(self.raw_dir, f"helios_{index:06d}.npy"), pair.images["helios"])
            np.save(os.path.join(self.raw_dir, f"triton_{index:06d}.npy"), pair.images["triton"])
            self.write_seconds += time.perf_counter() - t0
            self.saved += 1
        except Exception as exc:
            self.error = exc
            raise


# ============================================================================
# STOPPING
# ============================================================================

class StopControl:
    """Collects every way of stopping. A user request, once seen, is latched
    so it is never lost between checks."""

    def __init__(self):
        self.stop_file = None
        self._interrupted = False
        self._latched = None
        self._previous = None

    def install(self):
        self._previous = signal.signal(signal.SIGINT, self._on_sigint)

    def uninstall(self):
        if self._previous is not None:
            signal.signal(signal.SIGINT, self._previous)

    def _on_sigint(self, *_):
        self._interrupted = True

    def user_request(self):
        if self._latched:
            return self._latched
        if self._interrupted:
            self._latched = "Ctrl+C"
        elif msvcrt is not None:
            while msvcrt.kbhit():
                if msvcrt.getch() in (b"q", b"Q", b"\x1b"):
                    self._latched = "stop key"
                    break
        if not self._latched and self.stop_file and os.path.exists(self.stop_file):
            self._latched = "STOP file"
        return self._latched


class Monitor:
    """The engine calls this every 50 ms: stop requests, limits, disk, status line."""

    def __init__(self, stop, writer, run_dir):
        self.stop, self.writer, self.run_dir = stop, writer, run_dir
        self.last_status = time.perf_counter()
        self.last_disk = 0.0
        self.t0 = time.perf_counter()

    def __call__(self, status):
        request = self.stop.user_request()
        if request:
            return request
        if MAX_FRAMES and status["pairs"] >= MAX_FRAMES:
            return "frame limit"
        if self.writer.error is not None:
            return f"disk write failed: {self.writer.error}"
        now = time.perf_counter()
        if now - self.last_disk >= 2.5:
            self.last_disk = now
            if free_gb(self.run_dir) < MIN_FREE_GB_STOP:
                return "disk nearly full"
        if now - self.last_status >= 1.0:
            fps = status["pairs"] / max(status["elapsed_s"], 1e-6)
            print(f"\r{TAB}{status['pairs']:6d} pairs  {fps:5.1f} fps  "
                  f"queue {status['queue_depth']:3d}  dropped {status['writer_dropped']}  "
                  f"failed {status['failed_grabs']}  suspect {status['rejected_suspect']}   ",
                  end="", flush=True)
            self.last_status = now
        return None


# ============================================================================
# CHECKS
# ============================================================================

def check_calibration(calib_path, width, height):
    """Warn early if the calibration clearly belongs to another resolution."""
    try:
        import cv2
        fs = cv2.FileStorage(calib_path, cv2.FileStorage_READ)
        k = fs.getNode("cameraMatrix").mat()
        fs.release()
        cx, cy = float(k[0, 2]), float(k[1, 2])
        if not (0.25 * width < cx < 0.75 * width and 0.25 * height < cy < 0.75 * height):
            log(f"WARNING: calibration centre ({cx:.0f}, {cy:.0f}) does not fit a "
                f"{width} x {height} Triton image. Recalibrate.")
    except Exception as exc:
        log(f"Could not check calibration ({exc})")


# ============================================================================
# OUTPUT FILES
# ============================================================================

def camera_info(engine):
    """run.json "helios" / "triton" blocks, with the same keys as capture.py."""
    h, t = engine.helios.nm, engine.triton.nm
    scale, offsets = engine.scan3d
    hi, ti = engine.image_info["helios"], engine.image_info["triton"]
    helios = {
        "serial": plain(engine.helios.identity.get("serial")),
        "model": plain(engine.helios.identity.get("model")),
        "width": plain(hi["width"]), "height": plain(hi["height"]),
        "pixel_format": "Coord3D_ABCY16",
        "scale": scale, "offsets": offsets,
        "operating_mode": plain(h.get("Scan3dOperatingMode")),
        "exposure": plain(h.get("ExposureTimeSelector")),
        "conversion_gain": plain(h.get("ConversionGain")),
        "throughput": plain(h.get("DeviceLinkThroughputLimit")),
    }
    triton = {
        "serial": plain(engine.triton.identity.get("serial")),
        "model": plain(engine.triton.identity.get("model")),
        "width": plain(ti["width"]), "height": plain(ti["height"]),
        "pixel_format": TRITON_PIXEL_FORMAT,
        "throughput": plain(t.get("DeviceLinkThroughputLimit")),
    }
    return helios, triton


def grab_timer(sync, trigger):
    """Function pair -> grab_ms, the closest equivalent of capture.py's grab_ms ("trigger
    to both buffers received"):
      PTP clock: exposure start (earlier timestamp, mapped to host time through the PTP/host
                 times latched at the start and end of recording) -> later frame arrival;
      software trigger without PTP: TriggerSoftware execution -> later frame arrival;
      free-run without PTP: time the later camera's get_buffer() was blocked (no trigger
                 exists; best effort)."""
    w = (sync or {}).get("capture_window") or {}
    if (sync or {}).get("clock_domain") == "ptp" and w.get("ptp_start") and w.get("ptp_end") \
            and w["ptp_end"] != w["ptp_start"]:
        rate = (w["host_end"] - w["host_start"]) / (w["ptp_end"] - w["ptp_start"])

        def from_ptp(p):
            start = w["host_start"] + (min(p.helios.ts_ns, p.triton.ts_ns) - w["ptp_start"]) * rate
            return (max(p.helios.host_ns, p.triton.host_ns) - start) / 1e6
        return from_ptp
    records = getattr(trigger, "records", None)
    if (sync or {}).get("method_used") == "software" and records:
        def from_fire(p):
            k = p.grid_k
            if k is None or not 0 <= k < len(records):
                return max(p.helios.grab_ms, p.triton.grab_ms)
            return (max(p.helios.host_ns, p.triton.host_ns) - records[k].host_ns) / 1e6
        return from_fire
    return lambda p: max(p.helios.grab_ms, p.triton.grab_ms)


def pair_row(p, grab_ms):
    """capture.py's 11 columns (same meaning) followed by the sync columns."""
    gap = abs(p.triton.host_ns - p.helios.host_ns) / 1e6
    row = [p.save_index, f"{p.helios.host_wall_ns / 1e9:.6f}", p.helios.frame_id,
           p.triton.frame_id, p.helios.ts_ns, p.triton.ts_ns, f"{gap:.1f}",
           f"{grab_ms(p):.1f}",
           f"{p.helios.copy_ms + p.triton.copy_ms:.1f}", p.queue_depth, int(p.queued)]
    sync = p.csv_row()
    return row + ["" if sync[c] is None else sync[c] for c in PAIR_COLUMNS]


def write_frames_csv(path, pairs, grab_ms):
    """One row per captured (non-suspect) pair, in file-index order - like capture.py."""
    rows = sorted((p for p in pairs if p.save_index is not None), key=lambda p: p.save_index)
    with open(path, "w", newline="") as f:
        table = csv.writer(f)
        table.writerow(ORIGINAL_COLUMNS + list(PAIR_COLUMNS))
        for p in rows:
            table.writerow(pair_row(p, grab_ms))
    return rows


def result_block(sync, rows, writer, stop_reason, error_text, grab_ms):
    """run.json "result" with capture.py's keys (same meaning where possible)."""
    st = sync.get("stats", {}) if sync else {}
    duration = float(sync.get("seconds_actual") or 0.0) if sync else 0.0
    captured = len(rows)
    gaps = [abs(p.triton.host_ns - p.helios.host_ns) / 1e6 for p in rows]
    grabs = [grab_ms(p) for p in rows]
    failed = {}
    for cam, pc in (st.get("per_camera") or {}).items():
        for key, label in (("timeouts", "timeout"), ("grab_errors", "error"),
                           ("incomplete", "incomplete"), ("invalid_ts", "invalid timestamp")):
            if pc.get(key):
                failed[f"{cam} {label}"] = pc[key]
    return {
        "stop_reason": stop_reason,
        "error": error_text,
        "duration_s": round(duration, 2),
        "pairs_captured": captured,
        "pairs_saved": writer.saved,
        "pairs_dropped_disk_slow": (st.get("writer") or {}).get("dropped_queue_full", 0),
        "failed_grabs": failed,
        "mean_fps": round(captured / duration, 2) if duration > 0 else 0,
        "mean_grab_ms": round(float(np.mean(grabs)), 1) if grabs else None,
        "mean_arrival_gap_ms": round(float(np.mean(gaps)), 1) if gaps else None,
        "max_arrival_gap_ms": round(float(np.max(gaps)), 1) if gaps else None,
        "disk_write_error": str(writer.error) if writer.error else None,
        "pairs_rejected_suspect": st.get("rejected_suspect", 0),
    }


# ============================================================================
# SUMMARY
# ============================================================================

def _us(d, key):
    v = d.get(key)
    return "-" if v is None else f"{v:.1f}"


def print_summary(info, run_dir):
    r, sync = info.get("result", {}), info.get("sync") or {}
    st = sync.get("stats", {})
    log(f"Stopped: {r.get('stop_reason')}")
    if sync:
        log(f"Sync method: {sync['method_used']} (requested {sync['method_requested']}), "
            f"clock {sync['clock_domain']}")
        for c in sync.get("fallback_chain", []):
            log(f"  fallback from {c['method']}: {c['reason']}")
        ptp = sync.get("ptp")
        if ptp:
            mon = ptp.get("monitor") or {}
            log(f"PTP: helios={ptp.get('helios_status')} triton={ptp.get('triton_status')}, "
                f"locked in {ptp.get('lock_seconds')} s, settled={ptp.get('settled')} "
                f"({ptp.get('settle_method')}), monitor {mon.get('bad_samples', '-')} bad of "
                f"{mon.get('samples', '-')} samples")
        if sync.get("aborted_reason"):
            log(f"RUN ABORTED: {sync['aborted_reason']}")
    log(f"Saved {r.get('pairs_saved')} pairs in {r.get('duration_s', 0):.1f} s "
        f"({r.get('mean_fps')} fps)")
    if st:
        pc = st.get("per_camera", {})
        log(f"Dropped frames: Helios {st.get('dropped_helios')}, Triton {st.get('dropped_triton')}"
            f" (expected {st.get('frames_expected')}; {pc.get('helios', {}).get('drop_definition')})")
    if r.get("pairs_dropped_disk_slow"):
        log(f"{r['pairs_dropped_disk_slow']} pairs dropped because the disk fell behind")
    if r.get("failed_grabs"):
        log("Failed grabs: " + ", ".join(f"{k} x{v}" for k, v in r["failed_grabs"].items()))
    if st.get("rejected_suspect"):
        log(f"{st['rejected_suspect']} suspect pairs rejected (not saved): "
            f"{st.get('suspect_reasons')}")
    if st and sync.get("clock_domain") == "ptp":
        o, res, c = st["offset_start_us"], st["residual_us"], st["offset_center_model_us"]
        log(f"Timestamp offset Triton-Helios (PTP clock, measured): mean {_us(o, 'mean')} us, "
            f"median {_us(o, 'median')}, p95|.| {_us(o, 'p95_abs')}, max|.| {_us(o, 'max_abs')}")
        log(f"Residual vs plan (planned lag {sync['planned_lag_us']:.1f} us, alignment "
            f"{sync['alignment_used']}): median {_us(res, 'median')} us, "
            f"p95|.| {_us(res, 'p95_abs')} us")
        log(f"Exposure-centre offset (MODELLED, Helios centre {sync['helios_center_us']:.0f} us, "
            f"Triton exposure from {sync.get('triton_exposure_source')}): median "
            f"{_us(c, 'median')} us, p95|.| {_us(c, 'p95_abs')} us")
    for w in sync.get("warnings", []) if sync else []:
        log(f"WARNING: {w}")
    if r.get("mean_arrival_gap_ms") is not None:
        synced = sync.get("method_used") in ("ptpsync", "action", "software")
        note = ("transfer time only, both shot together" if synced
                else "includes capture offset")
        log(f"Triton arrived after Helios by {r['mean_arrival_gap_ms']} ms "
            f"on average ({note})")
    log(f"Run folder: {run_dir}")
    if r.get("pairs_saved"):
        log(f"Next: python process.py \"{run_dir}\"")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\ncapture_sync.py\n")
    logging.basicConfig(level=logging.WARNING, format=f"{TAB}%(levelname)s %(name)s: %(message)s")
    stop = StopControl()

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    if free_gb(OUTPUT_ROOT) < MIN_FREE_GB_START:
        log(f"Only {free_gb(OUTPUT_ROOT):.1f} GB free at {OUTPUT_ROOT}. "
            f"Need {MIN_FREE_GB_START} GB. Free some space or change OUTPUT_ROOT.")
        return 1

    run_name = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_ROOT, run_name)
    raw_dir = os.path.join(run_dir, "raw")
    os.makedirs(raw_dir)
    stop.stop_file = os.path.join(run_dir, "STOP")
    json_path = os.path.join(run_dir, "run.json")

    calib_copied = False
    if os.path.exists(CALIB_FILE):
        shutil.copy2(CALIB_FILE, os.path.join(run_dir, "orientation.yml"))
        calib_copied = True
    else:
        log(f"WARNING: {CALIB_FILE} not found here. Recording anyway; give "
            f"process.py the calibration with --calib.")

    system, sdk_info = load_system()
    writer = RawWriter(raw_dir)
    info = {"created": datetime.datetime.now().isoformat(timespec="seconds"), "run": run_name}
    result = None
    stop_reason, error_text = "error", None
    try:
        engine = CaptureEngine(sync_settings(), writer=writer, system=system, sdk_info=sdk_info,
                               helios_serial=HELIOS_SERIAL, triton_serial=TRITON_SERIAL,
                               open_retries=6)
    except ValueError as exc:
        log(f"ERROR in settings: {exc}")
        shutil.rmtree(run_dir, ignore_errors=True)
        return 1
    try:
        try:
            engine.open()
        except CameraNotFound as exc:
            log(f"ERROR: {exc}")
            shutil.rmtree(run_dir, ignore_errors=True)
            return 1
        log(f"Helios {engine.helios.identity['model']}  S/N {engine.helios.identity['serial']}")
        log(f"Triton {engine.triton.identity['model']}  S/N {engine.triton.identity['serial']}")
        log("Configuring (PTP negotiation can take up to a minute)...")
        method = engine.configure()
        helios, triton = camera_info(engine)
        if calib_copied:
            check_calibration(CALIB_FILE, triton["width"], triton["height"])
        log(f"Sync method {method.method_used}" + (
            f" (fallback from {SYNC_METHOD})" if method.fallback_chain else ""))
        info.update({
            "sync_mode": method.method_used,
            "frame_rate_target": FRAME_RATE,
            "helios": helios,
            "triton": triton,
            "calibration": "orientation.yml" if calib_copied else None,
            "files": {"helios": "raw/helios_{:06d}.npy", "triton": "raw/triton_{:06d}.npy"},
        })
        write_json(json_path, info)

        print()
        log(f"RECORDING to {run_dir} (after {WARMUP_SECONDS:.0f} s warm-up)")
        log("Stop: q / Esc here, Ctrl+C, or create a file named STOP in that folder.")
        log("Do not click inside this window while recording.\n")
        stop.install()
        try:
            result = engine.run(seconds=float(SECONDS), warmup_s=float(WARMUP_SECONDS),
                                should_stop=Monitor(stop, writer, run_dir))
        finally:
            stop.uninstall()
        stop_reason = result.run_info.get("stop_reason") or "error"
    except (SyncSetupError, RuntimeError) as exc:
        error_text = str(exc)
        print()
        log(f"ERROR: {exc}")
    except KeyboardInterrupt:
        stop_reason = "Ctrl+C"
    finally:
        print("\n")
        engine.close()                                  # stops streams, restores nodes
        rows = []
        grab_ms = grab_timer(result.run_info if result else None, engine.trigger)
        if result is not None:
            rows = write_frames_csv(os.path.join(run_dir, "frames.csv"), result.pairs, grab_ms)
            write_csv(os.path.join(run_dir, "sync_frames.csv"), FRAME_COLUMNS,
                      result.frame_rows())
            if result.run_info.get("restore") is None:
                result.run_info["restore"] = engine.restore_result
        info["result"] = result_block(result.run_info if result else None, rows, writer,
                                      stop_reason, error_text, grab_ms)
        info["sync"] = result.run_info if result else None
        try:
            write_json(json_path, info)
        except Exception as exc:
            log(f"Could not write run.json ({exc})")
        print_summary(info, run_dir)
    return 0 if result is not None else 1


if __name__ == "__main__":
    sys.exit(main())
