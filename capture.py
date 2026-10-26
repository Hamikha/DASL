"""
capture.py - raw paired capture from the Helios2+ and the Triton.

Records every frame pair exactly as the cameras send it, as fast as the link
allows. Nothing is projected or coloured here; that happens afterwards in
process.py, which can be re-run at any time without re-recording.

    python capture.py

Stopping, whichever is most convenient:
    - wait for SECONDS to run out
    - press q or Esc in this console window
    - press Ctrl+C
    - create a file named STOP inside the run folder

Each run gets its own folder:
    run_YYYYMMDD_HHMMSS/
        raw/helios_000000.npy   Helios Coord3D_ABCY16, uint16 (H, W, 4)
        raw/triton_000000.npy   Triton pixels, uint8 (H, W) Bayer or (H, W, 3) RGB
        frames.csv              one row per captured pair: timing and health
        run.json                every setting, camera and result, for process.py
        orientation.yml         copy of the calibration in use

Do not click inside the console while recording. Windows "QuickEdit" freezes
any program whose console is being selected, which looks like a hang.
"""

import csv
import ctypes
import datetime
import json
import os
import queue
import shutil
import signal
import threading
import time

import numpy as np

try:
    import msvcrt                     # Windows console keys, optional
except ImportError:
    msvcrt = None

from arena_api.system import system
from arena_api.enums import PixelFormat


# ============================================================================
# SETTINGS
# ============================================================================

# Keep recordings out of OneDrive. A synced folder uploads every frame while
# you record, competing for disk and CPU. This resolves to C:\Users\<you>\captures.
OUTPUT_ROOT = os.path.join(os.path.expanduser("~"), "captures")
CALIB_FILE = "orientation.yml"         # copied into each run for process.py

SECONDS = 60                           # 0 = no time limit
MAX_FRAMES = 0                         # 0 = no frame limit
FRAME_RATE = 10.0                      # pairs per second to aim for

# "trigger": both cameras fired by one software trigger, frames taken within
#            milliseconds of each other. Best pairing. Falls back to freerun
#            automatically if a camera does not respond to triggers.
# "freerun": each camera streams on its own clock. Pairs can be up to one
#            frame period apart.
SYNC_MODE = "trigger"

# BayerRG8 sends 1 byte per pixel instead of 3, so the Triton fits three
# times more frames through its share of the link. process.py converts it to
# colour. Same sensor resolution, so the calibration stays valid.
TRITON_PIXEL_FORMAT = "BayerRG8"       # "BayerRG8" or "RGB8"

HLT_OPERATING_MODE = "Distance6000mmSingleFreq"
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
GRAB_TIMEOUT_MS = 3000
MAX_CONSECUTIVE_FAILURES = 10          # give up after this many bad pairs in a row
STREAM_BUFFERS = 10
WRITE_QUEUE_SIZE = 100                 # pairs held in RAM waiting for the disk
MIN_FREE_GB_START = 10
MIN_FREE_GB_STOP = 3


# ============================================================================
# SMALL HELPERS
# ============================================================================

TAB = "  "


def log(message):
    print(f"{TAB}{message}", flush=True)


def read_node(nodemap, name, default=None):
    try:
        return nodemap.get_node(name).value
    except Exception:
        return default


def plain(value):
    """Convert node values (enums, numpy scalars) into JSON-safe types."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    name = getattr(value, "name", None)
    return name if isinstance(name, str) else str(value)


def free_gb(path):
    return shutil.disk_usage(path).free / 1e9


def write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


class NodeKeeper:
    """Writes camera nodes and remembers the original values, so the cameras
    are left exactly as they were found (e.g. for the next ArenaView session)."""

    def __init__(self):
        self._saved = []

    def set(self, nodemap, name, value, label, required=False):
        try:
            node = nodemap.get_node(name)
            if node is None:
                raise KeyError(name)
            try:
                original = node.value
            except Exception:
                original = None
            node.value = value
            if original is not None:
                self._saved.append((nodemap, name, original))
            return True
        except Exception as exc:
            if required:
                raise RuntimeError(f"{label}: cannot set {name} = {value} ({exc})") from exc
            log(f"{label}: {name} not set ({exc})")
            return False

    def restore(self):
        for nodemap, name, original in reversed(self._saved):
            try:
                nodemap.get_node(name).value = original
            except Exception:
                pass
        self._saved.clear()


# ============================================================================
# CAMERAS
# ============================================================================

class Camera:
    """Thin wrapper holding one device and the facts needed to read it."""

    def __init__(self, device, label):
        self.device = device
        self.label = label
        self.nodemap = device.nodemap
        self.serial = str(read_node(self.nodemap, "DeviceSerialNumber", ""))
        self.model = str(read_node(self.nodemap, "DeviceModelName", ""))
        self.width = 0
        self.height = 0
        self.channels = 1
        self.dtype = np.uint8
        self.pixel_format = ""
        self._timeout_supported = True

    def describe_frame(self, pixel_format, channels, dtype):
        self.pixel_format = pixel_format
        self.channels = channels
        self.dtype = dtype
        self.width = int(read_node(self.nodemap, "Width", 0))
        self.height = int(read_node(self.nodemap, "Height", 0))
        if not self.width or not self.height:
            raise RuntimeError(f"{self.label}: could not read image size")

    @property
    def frame_bytes(self):
        return self.width * self.height * self.channels * np.dtype(self.dtype).itemsize

    def get_buffer(self, timeout_ms):
        if self._timeout_supported:
            try:
                return self.device.get_buffer(timeout=timeout_ms)
            except TypeError:
                self._timeout_supported = False
        return self.device.get_buffer()

    def requeue(self, buffer):
        try:
            self.device.requeue_buffer(buffer)
        except Exception:
            pass

    def check_buffer(self, buffer):
        """Return None if the buffer is safe to read, else a short reason."""
        if getattr(buffer, "is_incomplete", False):
            return "incomplete"
        if int(buffer.width) != self.width or int(buffer.height) != self.height:
            return "size mismatch"
        fmt = getattr(getattr(buffer, "pixel_format", None), "name", None)
        if fmt is not None and fmt != self.pixel_format:
            return "format mismatch"
        filled = getattr(buffer, "size_filled", None)
        if filled is not None and int(filled) < self.frame_bytes:
            return "short buffer"
        return None

    def copy_pixels(self, buffer):
        """Copy the frame straight out of the driver's memory. No Python lists."""
        ctype = np.ctypeslib.as_ctypes_type(self.dtype)
        ptr = ctypes.cast(buffer.pdata, ctypes.POINTER(ctype))
        count = self.width * self.height * self.channels
        flat = np.ctypeslib.as_array(ptr, shape=(count,))
        shape = (self.height, self.width) if self.channels == 1 \
            else (self.height, self.width, self.channels)
        return flat.reshape(shape).copy()


def find_cameras():
    devices = []
    for attempt in range(6):
        devices = system.create_device()
        if devices:
            break
        log(f"No cameras yet, retrying in 5 s ({attempt + 1}/6)")
        time.sleep(5)
    if not devices:
        raise RuntimeError("No cameras found. Check cables and switch power.")

    helios = triton = None
    for device in devices:
        model = str(read_node(device.nodemap, "DeviceModelName", ""))
        serial = str(read_node(device.nodemap, "DeviceSerialNumber", ""))
        is_helios = ("HT" in model or "HLT" in model) and not model.startswith("TRI")
        is_triton = model.startswith("TRI") and "-C" in model
        if is_helios and helios is None and (not HELIOS_SERIAL or serial == HELIOS_SERIAL):
            helios = Camera(device, "Helios")
        elif is_triton and triton is None and (not TRITON_SERIAL or serial == TRITON_SERIAL):
            triton = Camera(device, "Triton")

    if helios is None or triton is None:
        found = ", ".join(str(read_node(d.nodemap, "DeviceModelName", "?")) for d in devices)
        raise RuntimeError(f"Need one Helios and one colour Triton, found: {found}")

    log(f"Helios {helios.model}  S/N {helios.serial}")
    log(f"Triton {triton.model}  S/N {triton.serial}")
    return helios, triton


# ============================================================================
# CONFIGURATION
# ============================================================================

def configure_helios(cam, keeper):
    nm = cam.nodemap
    keeper.set(nm, "AcquisitionMode", "Continuous", "Helios", required=True)
    keeper.set(nm, "PixelFormat", PixelFormat.Coord3D_ABCY16, "Helios", required=True)
    keeper.set(nm, "Scan3dOperatingMode", HLT_OPERATING_MODE, "Helios")
    keeper.set(nm, "ExposureTimeSelector", HLT_EXPOSURE, "Helios")
    keeper.set(nm, "ConversionGain", HLT_CONVERSION_GAIN, "Helios")

    scale = float(read_node(nm, "Scan3dCoordinateScale"))
    offsets = []
    for axis in ("CoordinateA", "CoordinateB", "CoordinateC"):
        nm.get_node("Scan3dCoordinateSelector").value = axis
        offsets.append(float(read_node(nm, "Scan3dCoordinateOffset")))

    cam.describe_frame("Coord3D_ABCY16", channels=4, dtype=np.uint16)
    log(f"Helios {cam.width} x {cam.height}, scale {scale}, offsets {offsets}")
    return scale, offsets


def configure_triton(cam, keeper):
    nm = cam.nodemap
    keeper.set(nm, "AcquisitionMode", "Continuous", "Triton", required=True)

    # Full sensor, no binning. The calibration was made at full resolution; any
    # ROI or binning left over from an ArenaView session would silently break it.
    for name in ("BinningHorizontal", "BinningVertical"):
        if read_node(nm, name) is not None:
            keeper.set(nm, name, 1, "Triton")
    for name in ("OffsetX", "OffsetY"):
        keeper.set(nm, name, 0, "Triton")
    for name in ("Width", "Height"):
        try:
            keeper.set(nm, name, int(nm.get_node(name).max), "Triton")
        except Exception as exc:
            log(f"Triton: {name} not maximised ({exc})")

    fmt = TRITON_PIXEL_FORMAT
    if fmt not in ("BayerRG8", "RGB8"):
        raise RuntimeError(f"TRITON_PIXEL_FORMAT must be BayerRG8 or RGB8, not {fmt}")
    keeper.set(nm, "PixelFormat", getattr(PixelFormat, fmt), "Triton", required=True)
    keeper.set(nm, "ExposureAuto", "Continuous", "Triton")
    keeper.set(nm, "BalanceWhiteAuto", "Continuous", "Triton")

    cam.describe_frame(fmt, channels=1 if fmt == "BayerRG8" else 3, dtype=np.uint8)
    log(f"Triton {cam.width} x {cam.height} {fmt}")


def configure_host_stream(cam):
    stream = cam.device.tl_stream_nodemap
    stream["StreamAutoNegotiatePacketSize"].value = True
    stream["StreamPacketResendEnable"].value = True
    # NewestOnly never lets a stale frame build up in the queue, so a pair can
    # never be made of one fresh and one old image.
    stream["StreamBufferHandlingMode"].value = "NewestOnly"


def configure_link(cam, keeper, throughput):
    nm = cam.nodemap
    if not keeper.set(nm, "DeviceLinkThroughputLimitMode", "On", cam.label):
        return
    try:
        node = nm.get_node("DeviceLinkThroughputLimit")
        value = int(min(max(throughput, node.min), node.max))
        keeper.set(nm, "DeviceLinkThroughputLimit", value, cam.label)
        log(f"{cam.label} link share {read_node(nm, 'DeviceLinkThroughputLimit') / 1e6:.0f} MB/s")
    except Exception as exc:
        log(f"{cam.label}: link share not set ({exc})")


def configure_freerun(cam, keeper, fps):
    nm = cam.nodemap
    keeper.set(nm, "TriggerMode", "Off", cam.label)
    keeper.set(nm, "AcquisitionFrameRateEnable", True, cam.label)
    try:
        node = nm.get_node("AcquisitionFrameRate")
        keeper.set(nm, "AcquisitionFrameRate", float(min(fps, node.max)), cam.label)
        rate = float(node.value)
        note = "" if rate >= fps - 0.01 else "  (capped by link share)"
        log(f"{cam.label} free-running at {rate:.2f} fps{note}")
    except Exception as exc:
        log(f"{cam.label}: frame rate not set ({exc})")


def configure_trigger(cam, keeper):
    nm = cam.nodemap
    ok = keeper.set(nm, "AcquisitionFrameRateEnable", False, cam.label)
    ok = keeper.set(nm, "TriggerSelector", "FrameStart", cam.label) and ok
    ok = keeper.set(nm, "TriggerSource", "Software", cam.label) and ok
    ok = keeper.set(nm, "TriggerMode", "On", cam.label) and ok
    return ok


# ============================================================================
# GRABBING PAIRS
# ============================================================================

class Pair:
    __slots__ = ("ok", "reason", "helios", "triton", "meta")

    def __init__(self, ok, reason="", helios=None, triton=None, meta=None):
        self.ok = ok
        self.reason = reason
        self.helios = helios
        self.triton = triton
        self.meta = meta or {}


class Grabber:
    def __init__(self, helios, triton, mode, fps):
        self.helios = helios
        self.triton = triton
        self.mode = mode
        self.period = 1.0 / fps
        self._next_due = None

    def _wait_armed(self, cam, timeout_s):
        end = time.perf_counter() + timeout_s
        while time.perf_counter() < end:
            if read_node(cam.nodemap, "TriggerArmed", False):
                return True
            time.sleep(0.001)
        return False

    def _pace(self):
        now = time.perf_counter()
        if self._next_due is None or now - self._next_due > self.period:
            self._next_due = now          # fell behind: resync, don't burst
        elif self._next_due > now:
            time.sleep(self._next_due - now)
        self._next_due += self.period

    def grab(self, keep_pixels=True):
        hb = tb = None
        meta = {}
        try:
            if self.mode == "trigger":
                self._pace()
                if not (self._wait_armed(self.triton, 1.0) and self._wait_armed(self.helios, 1.0)):
                    return Pair(False, "trigger not armed")
                meta["trigger_host"] = time.time()
                self.triton.nodemap.get_node("TriggerSoftware").execute()
                self.helios.nodemap.get_node("TriggerSoftware").execute()

            t0 = time.perf_counter()
            hb = self.helios.get_buffer(GRAB_TIMEOUT_MS)
            meta["hlt_rx_host"] = time.time()
            tb = self.triton.get_buffer(GRAB_TIMEOUT_MS)
            meta["tri_rx_host"] = time.time()
            meta["grab_ms"] = (time.perf_counter() - t0) * 1000.0

            for cam, buf in ((self.helios, hb), (self.triton, tb)):
                problem = cam.check_buffer(buf)
                if problem:
                    return Pair(False, f"{cam.label} {problem}")

            meta["hlt_ts_ns"] = int(getattr(hb, "timestamp_ns", 0))
            meta["tri_ts_ns"] = int(getattr(tb, "timestamp_ns", 0))
            meta["hlt_frame_id"] = int(getattr(hb, "frame_id", -1))
            meta["tri_frame_id"] = int(getattr(tb, "frame_id", -1))

            if not keep_pixels:
                return Pair(True, meta=meta)

            t1 = time.perf_counter()
            h_pixels = self.helios.copy_pixels(hb)
            t_pixels = self.triton.copy_pixels(tb)
            meta["copy_ms"] = (time.perf_counter() - t1) * 1000.0
            return Pair(True, helios=h_pixels, triton=t_pixels, meta=meta)

        except Exception as exc:
            text = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
            return Pair(False, text[:80])
        finally:
            if hb is not None:
                self.helios.requeue(hb)
            if tb is not None:
                self.triton.requeue(tb)


# ============================================================================
# DISK WRITER
# ============================================================================

class FrameWriter(threading.Thread):
    """Saves frames on a background thread so the capture loop never waits
    for the disk. If the disk falls behind and the queue fills, whole pairs
    are dropped and counted rather than stalling the cameras."""

    def __init__(self, raw_dir, capacity):
        super().__init__(daemon=True)
        self.raw_dir = raw_dir
        self.queue = queue.Queue(maxsize=capacity)
        self.error = None
        self.saved = 0
        self.write_seconds = 0.0

    def submit(self, index, helios, triton):
        try:
            self.queue.put_nowait((index, helios, triton))
            return True
        except queue.Full:
            return False

    def run(self):
        while True:
            item = self.queue.get()
            if item is None:
                return
            if self.error is not None:
                continue
            index, helios, triton = item
            try:
                t0 = time.perf_counter()
                np.save(os.path.join(self.raw_dir, f"helios_{index:06d}.npy"), helios)
                np.save(os.path.join(self.raw_dir, f"triton_{index:06d}.npy"), triton)
                self.write_seconds += time.perf_counter() - t0
                self.saved += 1
            except Exception as exc:
                self.error = exc

    def finish(self):
        while self.is_alive():
            try:
                self.queue.put(None, timeout=0.5)
                break
            except queue.Full:
                continue
        self.join()


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
        signal.signal(signal.SIGINT, self._on_sigint)

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

    def reason(self, elapsed, frames, honour_limits=True):
        request = self.user_request()
        if request:
            return request
        if honour_limits:
            if SECONDS and elapsed >= SECONDS:
                return "time limit"
            if MAX_FRAMES and frames >= MAX_FRAMES:
                return "frame limit"
        return None


# ============================================================================
# MAIN
# ============================================================================

def start_streams(helios, triton):
    helios.device.start_stream(STREAM_BUFFERS)
    triton.device.start_stream(STREAM_BUFFERS)


def stop_streams(helios, triton):
    for cam in (helios, triton):
        try:
            cam.device.stop_stream()
        except Exception:
            pass


def warm_up(grabber, stop, seconds):
    good = bad = 0
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        if stop.reason(0, 0, honour_limits=False):
            break
        if grabber.grab(keep_pixels=False).ok:
            good += 1
        else:
            bad += 1
    return good, bad


def check_calibration(calib_path, triton):
    """Warn early if the calibration clearly belongs to another resolution."""
    try:
        import cv2
        fs = cv2.FileStorage(calib_path, cv2.FileStorage_READ)
        k = fs.getNode("cameraMatrix").mat()
        fs.release()
        cx, cy = float(k[0, 2]), float(k[1, 2])
        if not (0.25 * triton.width < cx < 0.75 * triton.width and
                0.25 * triton.height < cy < 0.75 * triton.height):
            log(f"WARNING: calibration centre ({cx:.0f}, {cy:.0f}) does not fit a "
                f"{triton.width} x {triton.height} Triton image. Recalibrate.")
    except Exception as exc:
        log(f"Could not check calibration ({exc})")


def main():
    print("\ncapture.py\n")

    if SYNC_MODE not in ("trigger", "freerun"):
        log(f"SYNC_MODE must be 'trigger' or 'freerun', not '{SYNC_MODE}'")
        return
    stop = StopControl()

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    if free_gb(OUTPUT_ROOT) < MIN_FREE_GB_START:
        log(f"Only {free_gb(OUTPUT_ROOT):.1f} GB free at {OUTPUT_ROOT}. "
            f"Need {MIN_FREE_GB_START} GB. Free some space or change OUTPUT_ROOT.")
        return

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

    try:
        helios, triton = find_cameras()
    except RuntimeError as exc:
        log(f"ERROR: {exc}")
        try:
            system.destroy_device()
        except Exception:
            pass
        shutil.rmtree(run_dir, ignore_errors=True)
        return
    keeper = NodeKeeper()

    streaming = False
    writer = None
    csv_file = None
    counters = {"captured": 0, "dropped": 0}
    failures = {}
    grab_ms = []
    skew_ms = []
    stop_reason = "error"
    error_text = None
    rec_start = None
    info = {"created": datetime.datetime.now().isoformat(timespec="seconds"),
            "run": run_name}

    try:
        scale, offsets = configure_helios(helios, keeper)
        configure_triton(triton, keeper)
        if calib_copied:
            check_calibration(CALIB_FILE, triton)
        for cam, share in ((helios, HLT_THROUGHPUT), (triton, TRI_THROUGHPUT)):
            configure_host_stream(cam)
            configure_link(cam, keeper, share)

        mode = SYNC_MODE
        if mode == "trigger":
            if not (configure_trigger(helios, keeper) and configure_trigger(triton, keeper)):
                log("Trigger setup failed, using freerun")
                mode = "freerun"
        if mode == "freerun":
            for cam in (helios, triton):
                configure_freerun(cam, keeper, FRAME_RATE)

        grabber = Grabber(helios, triton, mode, FRAME_RATE)
        start_streams(helios, triton)
        streaming = True

        log(f"Warming up {WARMUP_SECONDS:.0f} s ({mode})")
        good, bad = warm_up(grabber, stop, WARMUP_SECONDS)
        if stop.user_request():
            raise KeyboardInterrupt
        if mode == "trigger" and good == 0:
            log("Cameras did not answer the software trigger. Switching to freerun.")
            stop_streams(helios, triton)
            for cam in (helios, triton):
                configure_freerun(cam, keeper, FRAME_RATE)
            grabber = Grabber(helios, triton, "freerun", FRAME_RATE)
            mode = "freerun"
            start_streams(helios, triton)
            good, bad = warm_up(grabber, stop, WARMUP_SECONDS)
            if stop.user_request():
                raise KeyboardInterrupt
        if good == 0:
            raise RuntimeError("No complete frame pairs during warm-up. "
                               "Check both cameras stream in ArenaView.")
        log(f"Warm-up: {good} good pairs, {bad} failed")

        info.update({
            "sync_mode": mode,
            "frame_rate_target": FRAME_RATE,
            "helios": {
                "serial": helios.serial, "model": helios.model,
                "width": helios.width, "height": helios.height,
                "pixel_format": helios.pixel_format,
                "scale": scale, "offsets": offsets,
                "operating_mode": plain(read_node(helios.nodemap, "Scan3dOperatingMode")),
                "exposure": plain(read_node(helios.nodemap, "ExposureTimeSelector")),
                "conversion_gain": plain(read_node(helios.nodemap, "ConversionGain")),
                "throughput": plain(read_node(helios.nodemap, "DeviceLinkThroughputLimit")),
            },
            "triton": {
                "serial": triton.serial, "model": triton.model,
                "width": triton.width, "height": triton.height,
                "pixel_format": triton.pixel_format,
                "throughput": plain(read_node(triton.nodemap, "DeviceLinkThroughputLimit")),
            },
            "calibration": "orientation.yml" if calib_copied else None,
            "files": {"helios": "raw/helios_{:06d}.npy", "triton": "raw/triton_{:06d}.npy"},
        })
        write_json(json_path, info)

        writer = FrameWriter(raw_dir, WRITE_QUEUE_SIZE)
        writer.start()
        csv_file = open(os.path.join(run_dir, "frames.csv"), "w", newline="")
        table = csv.writer(csv_file)
        table.writerow(["frame", "host_time", "hlt_frame_id", "tri_frame_id",
                        "hlt_ts_ns", "tri_ts_ns", "arrival_gap_ms", "grab_ms",
                        "copy_ms", "queue_depth", "queued"])

        print()
        log(f"RECORDING to {run_dir}")
        log("Stop: q / Esc here, Ctrl+C, or create a file named STOP in that folder.")
        log("Do not click inside this window while recording.\n")

        rec_start = time.perf_counter()
        last_status = rec_start
        consecutive = 0

        while True:
            elapsed = time.perf_counter() - rec_start
            stop_reason = stop.reason(elapsed, counters["captured"])
            if stop_reason:
                break
            if writer.error is not None:
                stop_reason = f"disk write failed: {writer.error}"
                break

            pair = grabber.grab()
            if not pair.ok:
                failures[pair.reason] = failures.get(pair.reason, 0) + 1
                consecutive += 1
                if consecutive >= MAX_CONSECUTIVE_FAILURES:
                    stop_reason = f"cameras stopped delivering ({pair.reason})"
                    break
                continue
            consecutive = 0

            index = counters["captured"]
            counters["captured"] += 1
            saved = writer.submit(index, pair.helios, pair.triton)
            if not saved:
                counters["dropped"] += 1

            m = pair.meta
            skew = abs(m["tri_rx_host"] - m["hlt_rx_host"]) * 1000.0
            grab_ms.append(m["grab_ms"])
            skew_ms.append(skew)
            table.writerow([index, f"{m['hlt_rx_host']:.6f}", m["hlt_frame_id"],
                            m["tri_frame_id"], m["hlt_ts_ns"], m["tri_ts_ns"],
                            f"{skew:.1f}", f"{m['grab_ms']:.1f}", f"{m['copy_ms']:.1f}",
                            writer.queue.qsize(), int(saved)])
            if index % 20 == 0:
                csv_file.flush()

            if index % 25 == 0 and free_gb(run_dir) < MIN_FREE_GB_STOP:
                stop_reason = "disk nearly full"
                break

            now = time.perf_counter()
            if now - last_status >= 1.0:
                fps = counters["captured"] / max(now - rec_start, 1e-6)
                print(f"\r{TAB}{counters['captured']:6d} pairs  {fps:5.1f} fps  "
                      f"queue {writer.queue.qsize():3d}  dropped {counters['dropped']}  "
                      f"failed {sum(failures.values())}   ", end="", flush=True)
                last_status = now

    except KeyboardInterrupt:
        stop_reason = stop.user_request() or "Ctrl+C"
    except Exception as exc:
        error_text = str(exc)
        stop_reason = "error"
        print()
        log(f"ERROR: {exc}")

    finally:
        duration = (time.perf_counter() - rec_start) if rec_start else 0.0
        print("\n")
        if streaming:
            stop_streams(helios, triton)
        keeper.restore()
        if writer is not None:
            pending = writer.queue.qsize()
            if pending:
                log(f"Writing the last {pending} pairs to disk...")
            writer.finish()
        if csv_file is not None:
            csv_file.flush()
            os.fsync(csv_file.fileno())
            csv_file.close()
        try:
            system.destroy_device()
        except Exception:
            pass

        saved = writer.saved if writer is not None else 0
        result = {
            "stop_reason": stop_reason,
            "error": error_text,
            "duration_s": round(duration, 2),
            "pairs_captured": counters["captured"],
            "pairs_saved": saved,
            "pairs_dropped_disk_slow": counters["dropped"],
            "failed_grabs": failures,
            "mean_fps": round(counters["captured"] / duration, 2) if duration > 0 else 0,
            "mean_grab_ms": round(float(np.mean(grab_ms)), 1) if grab_ms else None,
            "mean_arrival_gap_ms": round(float(np.mean(skew_ms)), 1) if skew_ms else None,
            "max_arrival_gap_ms": round(float(np.max(skew_ms)), 1) if skew_ms else None,
            "disk_write_error": str(writer.error) if writer is not None and writer.error else None,
        }
        info["result"] = result
        try:
            write_json(json_path, info)
        except Exception:
            pass

        log(f"Stopped: {stop_reason}")
        log(f"Saved {saved} pairs in {duration:.1f} s ({result['mean_fps']} fps)")
        if counters["dropped"]:
            log(f"{counters['dropped']} pairs dropped because the disk fell behind")
        if failures:
            log("Failed grabs: " + ", ".join(f"{k} x{v}" for k, v in failures.items()))
        if skew_ms:
            note = ("transfer time only, both shot together"
                    if info.get("sync_mode") == "trigger" else "includes capture offset")
            log(f"Triton arrived after Helios by {result['mean_arrival_gap_ms']} ms "
                f"on average ({note})")
        log(f"Run folder: {run_dir}")
        if saved:
            log(f"Next: python process.py \"{run_dir}\"")


if __name__ == "__main__":
    main()