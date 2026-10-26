"""Offline analysis of a Helios2+ / Triton capture: how well synchronised were the pairs?

This module is the ANALYSIS half of ``sync_test.py``. It never talks to a camera; it reads the
CSV/JSON files a capture left behind and turns them into numbers, plots and a report.

Rule 1 (the one a thesis reviewer will check): device timestamps of two cameras may only be
subtracted when both cameras were on ONE clock (PTP, clock_domain == "ptp"). Free-running cameras
count time on their own oscillators from their own power-on instants, so ``triton_ts - helios_ts``
is meaningless there.

Rule 2: even on one PTP clock, the timestamp difference is NOT an exposure measurement. LUCID does
not document which instant ``timestamp_ns`` marks (exposure start, end or readout), and under
PTPSync / action commands both timestamps come from the firmware's own schedule (LUCID measured a
~18 us bookkeeping difference between Helios and Triton). The number therefore proves clock lock and
scheduling precision, not that the exposures overlapped. Only a scope / LED / rotation test can.

So every number carries a *basis* label:

    MEASURED       device timestamps on the shared PTP clock
    MODELLED       MEASURED + a model of where each exposure's centre lies (needs a scope to verify)
    HOST-BASED     PC receive times; include network transfer time and OS jitter
    PER-CAMERA     one camera's own device clock (valid for its own intervals, fps and drops)

and the report headline shows three numbers side by side:
    (a) MEASURED residual vs the planned lag  = scheduling precision (ptpsync / action)
    (b) MODELLED exposure-centre offset       = what matters for motion smear
    (c) the measured (or, without PTP, host-estimated) timestamp offset for free-running runs

Folders understood by ``load_capture``:
    a. new captures (capture_sync.py / sync_test live): frames.csv with the DESIGN.md sync columns,
       sync_frames.csv, run.json with a "sync" object
    b. old captures of the original capture.py (frames.csv: frame, host_time, hlt_frame_id,
       tri_frame_id, hlt_ts_ns, tri_ts_ns, arrival_gap_ms, ...; run.json sync_mode / result) -
       recognised natively; other legacy layouts via name guessing or --map
    c. folders that only contain sync_frames.csv

Dependencies: numpy (required), matplotlib (plots only; imported lazily with the Agg backend).
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

# ----------------------------------------------------------------------------------------------
# Settings (change here, not deep in the code)
# ----------------------------------------------------------------------------------------------
# Helios light-collection model (TODO: switch to the shared lucid_sync helper once Coder A adds it).
# A ToF frame = n sub-frames of (0.98 x exposure integration + 1.55 ms readout); single-frequency
# modes use 4 sub-frames, multi-frequency 8. Centroid = (n-1)/2 (int + readout) + int/2.
HELIOS_READOUT_US = 1550.0
HELIOS_DEFAULT_MODE = ("Distance5000mmMultiFreq", "Exp250Us")    # new capture default (~6405 us)

# Typical exposure -> host-receive latency per camera, used ONLY to turn legacy host receive deltas
# into an *estimate* of the camera-to-camera offset (1 GbE transfer of one frame). ASSUMPTIONS.
LEGACY_LATENCY_MS = {"helios": 20.0, "triton": 25.0}

# Drop accounting - ONE place to adjust when the engine's definition is final.
#   DROP_WINDOW "union": a grid slot k counts as expected from the first to the last frame of
#   EITHER camera (so a camera that dies early or starts late is charged for it);
#   "intersection": only where both cameras delivered (old engine behaviour).
DROP_WINDOW = "union"
INCOMPLETE_IS_DROP = False      # an incomplete buffer still proves the camera produced that frame
DROP_GAP_FACTOR = 1.5           # without a grid: a gap > 1.5 x median interval = dropped frame(s)
MAX_K_SPAN_FACTOR = 20          # refuse grid spans > 20 x frames received (timestamp step?)

# Bad timestamps: <= 0, NaN, or further than this from the median are excluded before any windowing
# (one ts=0 on an epoch-scale PTP clock would otherwise mean a 50-year time axis).
TS_PLAUSIBLE_RANGE_S = 3600.0
MAX_WINDOWS = 200_000           # hard cap on fps windows (memory safety)

# A camera whose frames start/stop this long before the other's (or the run's) gets a loud warning.
DEAD_CAMERA_GAP_S = 2.0

# Drift is only called "drift" if the run is long enough and the slope is clearly non-zero.
DRIFT_MIN_DURATION_S = 120.0
DRIFT_MIN_SIGMA = 3.0
DRIFT_EDGE_WINDOW_S = 60.0

# Analysis-side consistency check (independent of the PTP clock, see REVIEW_1 item 1): on a PTP
# run, a pair whose host receive delta is more than this many frame periods from the run's median
# is flagged suspect (a PTP step can pair frames n periods apart with perfect timestamps).
HOST_DELTA_BAND_PERIODS = 0.5

# capture.py fetched the Triton buffer only after the Helios one. An arrival gap below this means
# the Triton frame was already waiting, so its host time is NOT its arrival time.
CAPTURE_PY_WAITING_GAP_MS = 1.0
MIN_PROMPT_FETCHES = 10          # fewer promptly fetched Triton frames -> the estimate is only a bound

FPS_WINDOWS_S = (1.0, 10.0)

# "Why it matters" conversion: misregistration in pixels for a rotating rig (research B section 2).
MOTION_DEG_PER_S = 60.0
F_TRITON_PX = 1739.0            # 6 mm lens on IMX265 (3.45 um pixels)
F_HELIOS_PX = 483.0             # 640 px over 67 deg HFOV
MOTION_CRITERION_MS = 1.0       # ~1 Helios px at 90 deg/s (research B section 3)

# Colour-blind-safe categorical colours: each camera ALWAYS gets the same colour in every plot;
# pair quantities (which belong to neither camera) are neutral ink.
COLOR_HELIOS = "#2a78d6"   # blue
COLOR_TRITON = "#eb6834"   # orange
COLOR_PAIR = "#52514e"     # neutral dark grey
COLOR_INK = "#0b0b0b"
COLOR_SUSPECT = "#e34948"  # red, always with an 'x' marker + label (never colour alone)
COLOR_BEFORE = "#8a8984"   # grey   - baseline in before/after plots
COLOR_AFTER = "#6250d6"    # violet - synchronised run in before/after plots
COLOR_GRID = "#e4e3df"

CAMERAS = ("helios", "triton")

# Column names of the new schema; taken from pairing.py (the writer) so reader and writer cannot
# disagree. The local copy is only a fallback.
try:
    from .pairing import FRAME_COLUMNS, PAIR_COLUMNS
except Exception:  # pragma: no cover
    PAIR_COLUMNS = (
        "pair_index", "grid_k", "helios_frame_id", "triton_frame_id", "helios_ts_ns", "triton_ts_ns",
        "offset_start_us", "triton_exposure_us", "offset_center_model_us", "helios_host_ns",
        "triton_host_ns", "host_delta_ms", "clock_domain",
    )
    FRAME_COLUMNS = (
        "camera", "frame_id", "ts_ns", "host_ns", "grid_k", "incomplete", "paired", "pair_index",
        "exposure_us", "note",
    )
# The original (pre-round-1) DESIGN.md columns, pinned for old-file compatibility tests.
DESIGN_V0_PAIR_COLUMNS = (
    "pair_index", "grid_k", "helios_frame_id", "triton_frame_id", "helios_ts_ns", "triton_ts_ns",
    "offset_start_us", "triton_exposure_us", "offset_center_model_us", "helios_host_ns",
    "triton_host_ns", "host_delta_ms", "clock_domain",
)
DESIGN_V0_FRAME_COLUMNS = ("camera", "frame_id", "ts_ns", "host_ns", "grid_k", "incomplete", "paired",
                           "pair_index", "exposure_us", "note")
# Optional newer columns (round-1 engine contract); written by write_capture when present.
PAIR_COLUMNS_OPTIONAL = ("planned_lag_us", "residual_us", "suspect", "suspect_reason", "segment_id")
FRAME_COLUMNS_OPTIONAL = ("segment_id",)

try:
    from .stats import summarize as _summarize
except Exception:  # pragma: no cover
    _summarize = None

SCHEDULED_METHODS = {"ptpsync", "action"}               # a planned lag exists -> residual
PTP_METHODS = {"ptpsync", "action", "freerun_ptp"}       # always on the PTP clock
UNIT_TO_NS = {"ns": 1, "us": 1_000, "ms": 1_000_000, "s": 1_000_000_000}

BASIS_MEASURED = "MEASURED (device timestamps, shared PTP clock)"
BASIS_MODELLED = "MODELLED (measured timestamp offset + exposure-centre model)"
BASIS_HOST = "HOST-BASED (PC receive time; includes transfer + OS jitter)"
BASIS_PER_CAMERA = "PER-CAMERA (each camera's own device clock)"
BASIS_ESTIMATE = "HOST-BASED ESTIMATE (independent clocks; see legacy_estimate.method)"
BASIS_BOUND = "HOST-BASED WORST-CASE BOUND (independent clocks; Triton frames were already waiting)"

OFFSET_NAME = "timestamp offset (Triton − Helios)"
TIMESTAMP_CAVEAT = (
    "What each camera's timestamp marks (exposure start, end or readout) is NOT CONFIRMED by LUCID. "
    "Under PTPSync / action commands both timestamps come from the firmware's own schedule on the "
    "shared clock (LUCID measured a ≈18 µs bookkeeping difference between Helios and Triton), so "
    "this number proves clock lock and scheduling precision, NOT that the exposures overlapped.")


class ClockDomainError(ValueError):
    """Raised when someone tries to subtract device timestamps of cameras on different clocks."""


# ----------------------------------------------------------------------------------------------
# Small statistics helpers
# ----------------------------------------------------------------------------------------------
def describe(values) -> dict:
    """Summary statistics of a 1-D sample, ignoring NaN.

    Core keys (n, mean, median, std, min, max, p95_abs, max_abs) are those of run.json "stats"
    (lucid_sync.stats.summarize, population SD); extra percentiles are added for the report.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    ax = np.abs(x)
    if _summarize is not None:
        out = _summarize(x.tolist(), ndigits=9)       # no rounding that would hide µs in ms values
    else:
        out = {"n": int(x.size), "mean": float(np.mean(x)), "median": float(np.median(x)),
               "std": float(np.std(x)), "p95_abs": float(np.percentile(ax, 95)),
               "max_abs": float(np.max(ax)), "min": float(np.min(x)), "max": float(np.max(x))}
    out.update({"p5": float(np.percentile(x, 5)), "p95": float(np.percentile(x, 95)),
                "median_abs": float(np.median(ax)), "p99_abs": float(np.percentile(ax, 99))})
    return out


def linear_fit(t, y) -> dict | None:
    """Least-squares line y = a + b t with the standard error of the slope (NaN-safe).

    The SE assumes independent residuals; PTP servo noise is somewhat correlated, so treat it as a
    lower bound (hence the conservative 3-SE significance rule in `drift_summary`)."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(t) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(t[ok]) == 0:
        return None
    tt, yy = t[ok], y[ok]
    b, a = np.polyfit(tt, yy, 1)
    resid = yy - (a + b * tt)
    dof = max(tt.size - 2, 1)
    s2 = float(np.sum(resid ** 2)) / dof
    sxx = float(np.sum((tt - tt.mean()) ** 2))
    se = math.sqrt(s2 / sxx) if sxx > 0 else math.inf
    return {"slope": float(b), "intercept": float(a), "residual_std": math.sqrt(s2), "slope_se": se,
            "n": int(tt.size), "span": float(np.ptp(tt))}


def drift_summary(t_s, y, unit: str) -> dict | None:
    """Slope per minute with SE and 95 % CI; 'detected' only for runs >= 2 min with |slope| > 3 SE."""
    fit = linear_fit(np.asarray(t_s, dtype=float) / 60.0, y)
    if fit is None:
        return None
    dur_s = fit["span"] * 60.0
    significant = abs(fit["slope"]) > DRIFT_MIN_SIGMA * fit["slope_se"]
    long_enough = dur_s >= DRIFT_MIN_DURATION_S
    if not long_enough:
        verdict = f"no drift detectable (run too short: {dur_s:.0f} s < {DRIFT_MIN_DURATION_S:.0f} s)"
    elif not significant:
        verdict = f"no drift detectable (within noise: |slope| < {DRIFT_MIN_SIGMA:g} SE)"
    else:
        verdict = "drift detected"
    out = {"slope_per_min": fit["slope"], "slope_se_per_min": fit["slope_se"],
           "ci95_per_min": [fit["slope"] - 1.96 * fit["slope_se"], fit["slope"] + 1.96 * fit["slope_se"]],
           "residual_std": fit["residual_std"], "window_s": dur_s, "n": fit["n"], "unit": unit,
           "detected": bool(significant and long_enough), "verdict": verdict}
    edge = _edge_change(np.asarray(t_s, dtype=float), np.asarray(y, dtype=float), DRIFT_EDGE_WINDOW_S)
    if edge:
        out.update({"edge_window_s": edge["window_s"], "first_window_median": edge["first_median"],
                    "last_window_median": edge["last_median"], "change_first_to_last": edge["change"]})
    return out


def _edge_change(t_s, y, window_s) -> dict | None:
    ok = np.isfinite(t_s) & np.isfinite(y)
    t, v = t_s[ok], y[ok]
    if t.size < 4:
        return None
    order = np.argsort(t)
    t, v = t[order], v[order]
    dur = t[-1] - t[0]
    w = window_s if dur >= 2 * window_s else dur / 4     # short runs: first vs last quarter
    first, last = v[t <= t[0] + w], v[t >= t[-1] - w]
    return {"window_s": float(w), "first_median": float(np.median(first)),
            "last_median": float(np.median(last)), "change": float(np.median(last) - np.median(first))}


def device_offset_us(helios_ts_ns, triton_ts_ns, clock_domain: str):
    """Triton - Helios device-timestamp difference in microseconds.

    WHY this function exists: it is the only place where two cameras' device timestamps are
    subtracted, and it refuses unless they are on the same (PTP) clock - comparing two free-running
    camera clocks is the classic mistake (different power-on instants, tens of ppm drift).
    Integer inputs (raw ns, ~1.7e18) are subtracted as int64 BEFORE converting to float, because a
    float64 near 1.7e18 can only represent multiples of 256 ns. Float inputs must already be
    relative values (as the loader produces); NaN marks a missing timestamp.
    """
    if clock_domain != "ptp":
        raise ClockDomainError(
            f"clock_domain is {clock_domain!r}: the cameras' device timestamps are on independent "
            "clocks and cannot be subtracted. Only host-receive-based estimates are possible.")
    h, t = np.asarray(helios_ts_ns), np.asarray(triton_ts_ns)
    if h.dtype.kind in "iu" and t.dtype.kind in "iu":
        return (t.astype(np.int64) - h.astype(np.int64)).astype(float) / 1e3
    if h.dtype.kind == "O" or t.dtype.kind == "O":            # Python ints (exact)
        return np.array([math.nan if a is None or b is None else (int(b) - int(a)) / 1e3
                         for a, b in zip(h.tolist(), t.tolist())], dtype=float)
    return (t.astype(float) - h.astype(float)) / 1e3


def helios_center_us_model(operating_mode: str | None, exposure: str | float | None) -> float | None:
    """Estimated centroid of the Helios light collection after its frame start, in µs.

    TODO: replace by the shared lucid_sync helper (Coder A) once it exists - same formula.
    Exp250Us SingleFreq ~2815 us; Exp250Us MultiFreq ~6405 us; Exp1000Us MultiFreq ~9345 us.
    This is a MODEL from data-sheet timings; verify with a scope.
    """
    if not operating_mode or exposure in (None, ""):
        return None
    if isinstance(exposure, (int, float)):
        exp_us = float(exposure)
    else:
        m = re.search(r"(\d+(?:[._]\d+)?)\s*us", str(exposure), re.I)
        if not m:
            return None
        exp_us = float(m.group(1).replace("_", "."))
    n_sub = 8 if "multi" in str(operating_mode).lower() else 4
    integ = 0.98 * exp_us
    return (n_sub - 1) / 2.0 * (integ + HELIOS_READOUT_US) + integ / 2.0


def pixels_at_motion(dt_ms: float | None) -> dict | None:
    """Misregistration in pixels for MOTION_DEG_PER_S rotation: px = omega * dt * f."""
    if dt_ms is None or not math.isfinite(dt_ms):
        return None
    w = math.radians(MOTION_DEG_PER_S)
    return {"triton_px": w * abs(dt_ms) / 1e3 * F_TRITON_PX,
            "helios_px": w * abs(dt_ms) / 1e3 * F_HELIOS_PX}


# ----------------------------------------------------------------------------------------------
# run.json -> normalised info. The ONE place that knows where the engine / capture.py put things.
# ----------------------------------------------------------------------------------------------
def _get(d: Any, path: str):
    for part in path.split("."):
        if isinstance(d, dict) and part in d:
            d = d[part]
        else:
            return None
    return d


def _first(run: dict, *paths, default=None):
    for p in paths:
        v = _get(run, p)
        if v is not None:
            return v
    return default


def _as_float(v) -> float | None:
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _normalise_monitor(mon) -> dict | None:
    """PTP monitor timeline -> {"t_s", "helios_status", "triton_status", "helios_offset_ns",
    "triton_offset_ns"} lists. Accepts a list of dicts (keys guessed) or of lists
    [t, helios_status, triton_status, (triton_offset_ns)] - the engine format is not final."""
    if not isinstance(mon, list) or not mon:
        return None
    out = {k: [] for k in ("t_s", "helios_status", "triton_status", "helios_offset_ns", "triton_offset_ns")}

    def pick(d, *keys):
        for k in keys:
            if k in d and d[k] is not None:
                return d[k]
        return None

    for row in mon:
        if isinstance(row, dict):
            t = pick(row, "t_s", "t", "time_s", "elapsed_s", "time")
            hs = pick(row, "helios_status", "helios")
            ts = pick(row, "triton_status", "triton")
            ho = pick(row, "helios_offset_ns", "helios_offset_from_master_ns")
            to = pick(row, "triton_offset_ns", "triton_offset_from_master_ns", "offset_from_master_ns",
                      "offset_ns", "slave_offset_ns")
        elif isinstance(row, (list, tuple)) and len(row) >= 3:
            t, hs, ts = row[0], row[1], row[2]
            ho, to = None, (row[3] if len(row) > 3 else None)
        else:
            continue
        out["t_s"].append(_as_float(t))
        out["helios_status"].append(None if hs is None else str(hs))
        out["triton_status"].append(None if ts is None else str(ts))
        out["helios_offset_ns"].append(_as_float(ho))
        out["triton_offset_ns"].append(_as_float(to))
    return out if out["t_s"] else None


def run_info(run: dict) -> dict:
    """Everything the analysis needs from run.json, from whichever layout wrote it.

    Sources: the "sync" object (capture_sync.py / sync_test live; engine contract, all keys
    optional) and the original capture.py keys (sync_mode, frame_rate_target, helios, triton,
    result). If the engine moves a key, adjust the candidate paths HERE.
    """
    sync = run.get("sync") if isinstance(run.get("sync"), dict) else {}
    legacy_mode = run.get("sync_mode")
    method_used = sync.get("method_used")
    if not method_used and legacy_mode:
        method_used = {"trigger": "legacy software trigger (capture.py sync_mode=trigger)",
                       "freerun": "legacy freerun (capture.py sync_mode=freerun)"}.get(
            legacy_mode, f"legacy {legacy_mode}")
    helios_mode = _first(run, "helios.operating_mode", "sync.helios.operating_mode")
    helios_exposure = _first(run, "helios.exposure", "sync.helios.exposure")
    centre = _as_float(sync.get("helios_center_us"))
    centre_src = "run.json sync.helios_center_us" if centre is not None else None
    if centre is None:
        centre = helios_center_us_model(helios_mode, helios_exposure)
        if centre is not None:
            centre_src = f"model from run.json helios.operating_mode={helios_mode}, exposure={helios_exposure}"
    if centre is None:
        centre = helios_center_us_model(*HELIOS_DEFAULT_MODE)
        centre_src = f"model, ASSUMED default mode {HELIOS_DEFAULT_MODE[0]} + {HELIOS_DEFAULT_MODE[1]}"

    failed_reasons = _first(run, "result.failed_grabs")
    failed = _first(run, "sync.stats.failed_grabs")
    if failed is None and isinstance(failed_reasons, dict):
        failed = int(sum(int(v) for v in failed_reasons.values()))

    per_cam = {}
    for c in CAMERAS:
        base = f"sync.stats.per_camera.{c}"
        per_cam[c] = {
            "received": _first(run, f"{base}.frames_received", f"{base}.received",
                               f"sync.frames_received.{c}"),
            "expected": _first(run, f"{base}.frames_expected", f"sync.frames_expected.{c}"),
            "dropped": _first(run, f"{base}.dropped", f"sync.stats.dropped_{c}"),
            "drop_definition": _first(run, f"{base}.drop_definition"),
            "incomplete": _first(run, f"{base}.incomplete"),
            "edge_frames": _first(run, f"{base}.edge_frames"),
        }
    cams = sync.get("cameras")
    if not cams and (run.get("helios") or run.get("triton")):
        cams = {c: {k: run[c].get(k) for k in ("model", "serial", "pixel_format", "operating_mode", "exposure")
                    if isinstance(run.get(c), dict) and run[c].get(k) is not None}
                for c in CAMERAS if isinstance(run.get(c), dict)}
    ptp = sync.get("ptp") if isinstance(sync.get("ptp"), dict) else None
    return {
        "method_requested": sync.get("method_requested") or (f"capture.py SYNC_MODE ({legacy_mode})" if legacy_mode else None),
        "method_used": method_used,
        "legacy_sync_mode": legacy_mode if not sync.get("method_used") else None,
        "clock_domain": sync.get("clock_domain"),
        "fallback_chain": sync.get("fallback_chain"),
        "frame_rate": _as_float(_first(run, "sync.frame_rate", "frame_rate_target", "frame_rate", "fps")),
        "seconds": _as_float(sync.get("seconds")),
        "seconds_actual": _as_float(_first(run, "sync.seconds_actual", "result.duration_s")),
        "interrupted": sync.get("interrupted"),
        "stop_reason": _first(run, "result.stop_reason"),
        "alignment_requested": _first(run, "sync.alignment_requested", "sync.alignment"),
        "alignment_used": _first(run, "sync.alignment_used", "sync.alignment"),
        "planned_lag_us": _as_float(_first(run, "sync.planned_lag_us", "sync.triton_lag_us",
                                           "sync.plan.triton_lag_us")),
        "helios_center_us": centre,
        "helios_center_source": centre_src,
        "helios_mode": helios_mode,
        "helios_exposure": helios_exposure,
        "exposure_cap_us": _as_float(sync.get("exposure_cap_us")),
        "triton_trigger_delay_us": _as_float(sync.get("triton_trigger_delay_us")),
        "triton_exposure_source": _first(run, "sync.triton_exposure_source", "sync.exposure.source",
                                         "sync.stats.triton_exposure_source"),
        "ptp": ptp,
        "ptp_settled": _first(run, "sync.ptp.settled", "sync.ptp_settled"),
        "ptp_monitor": _normalise_monitor(_first(run, "sync.ptp.monitor", "sync.ptp_monitor")),
        "ptp_status_changes": _first(run, "sync.ptp.status_changes"),
        "action": sync.get("action") if isinstance(sync.get("action"), dict) else None,
        "action_skipped_late": _first(run, "sync.action_skipped_late", "sync.stats.action_skipped_late",
                                      "sync.action.skipped_late"),
        "rejected_suspect": _first(run, "sync.stats.rejected_suspect"),
        "per_camera": per_cam,
        "failed_grabs": failed,
        "failed_grab_reasons": failed_reasons if isinstance(failed_reasons, dict) else None,
        "incomplete": _first(run, "sync.stats.incomplete"),
        "pairs_engine": _first(run, "sync.stats.pairs", "result.pairs_captured"),
        "pairs_saved": _first(run, "result.pairs_saved"),
        "pairs_dropped_disk_slow": _first(run, "result.pairs_dropped_disk_slow"),
        "cameras": cams,
        "sdk": sync.get("sdk"),
        "missing_nodes": sync.get("missing_nodes"),
        "failed_node_writes": sync.get("failed_node_writes"),
        "ptpsync_rate_check": sync.get("ptpsync_rate_check"),
        "synthetic": bool(run.get("synthetic") or sync.get("synthetic")),
    }


# ----------------------------------------------------------------------------------------------
# Data containers
# ----------------------------------------------------------------------------------------------
@dataclass
class CameraFrames:
    """All frames ONE camera delivered. Times are float ns relative to Capture.ts_ref_ns /
    host_ref_ns (see Capture for why relative). Invalid timestamps are NaN."""
    ts_ns: np.ndarray
    host_ns: np.ndarray
    grid_k: np.ndarray
    frame_id: np.ndarray
    incomplete: np.ndarray       # bool
    source: str                  # "sync_frames.csv" or "frames.csv (paired frames only)"
    notes: list[str] = field(default_factory=list)


@dataclass
class Capture:
    """Everything loaded from one capture folder.

    WHY relative times: PTP timestamps are ~1.7e18 ns. A float64 has a 53-bit mantissa, so at that
    magnitude it only represents multiples of 256 ns - enough to corrupt a ~20 µs offset. We parse
    timestamps as exact Python ints, subtract one common reference (ts_ref_ns, host_ref_ns), and
    only then convert to float64; relative values over hours stay exact.
    """
    folder: str
    source_kind: str                       # "new", "capture_py", "legacy", "sync_frames_only"
    clock_domain: str                      # "ptp", "independent", "unknown"
    clock_domain_source: str
    pairs: dict[str, Any]
    frames: dict[str, CameraFrames]
    run: dict
    ts_ref_ns: int = 0
    host_ref_ns: int = 0
    column_map: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    excluded_ts: dict[str, int] = field(default_factory=dict)   # "<source>": n excluded timestamps
    info: dict = field(default_factory=dict)                     # run_info(run)
    fps_setting: float | None = None
    synthetic: bool = False

    @property
    def n_pairs(self) -> int:
        return int(len(self.pairs.get("pair_index", [])))


# ----------------------------------------------------------------------------------------------
# Parsing helpers
# ----------------------------------------------------------------------------------------------
def _read_csv(path: str) -> tuple[list[str], list[dict]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader]
        return [c.strip() for c in (reader.fieldnames or [])], rows


def _cell(row: dict, col: str | None) -> str:
    if not col:
        return ""
    v = row.get(col)
    return "" if v is None else str(v).strip()


def _parse_time_ns(text: str, unit: str = "ns") -> int | None:
    """Parse a timestamp cell into exact integer ns. Accepts ints, floats and ISO date strings."""
    if text == "" or text.lower() in ("nan", "none", "null"):
        return None
    factor = UNIT_TO_NS.get(unit, 1)              # unit "datetime": ISO strings, handled below
    try:
        return int(text) * factor                 # exact for integer cells
    except ValueError:
        pass
    try:
        v = float(text)
        return int(round(v * factor)) if math.isfinite(v) else None
    except ValueError:
        pass
    try:
        return int(round(datetime.fromisoformat(text).timestamp() * 1e9))
    except ValueError:
        return None


def _parse_float(text: str) -> float:
    if text == "":
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def _clean_times(values: list[int | None], label: str, excluded: dict[str, int],
                 span_hint_s: float | None = None) -> list[int | None]:
    """Drop timestamps that cannot be real: <= 0, or further than TS_PLAUSIBLE_RANGE_S (or twice
    the run length) from the median. WHY: an incomplete buffer may carry ts = 0, a PTP step may
    jump by 1e15 ns; either would stretch the time axis to years and blow up any windowing."""
    good = [v for v in values if v is not None and v > 0]
    n_bad = sum(1 for v in values if v is not None and v <= 0)
    out: list[int | None] = [v if (v is not None and v > 0) else None for v in values]
    if good:
        med = int(np.median(np.array(good, dtype=np.float64)))
        lim = int(max(TS_PLAUSIBLE_RANGE_S, 2 * (span_hint_s or 0)) * 1e9)
        for i, v in enumerate(out):
            if v is not None and abs(v - med) > lim:
                out[i] = None
                n_bad += 1
    if n_bad:
        excluded[label] = excluded.get(label, 0) + n_bad
    return out


def _relative(values: list[int | None], ref: int) -> np.ndarray:
    return np.array([math.nan if v is None else float(v - ref) for v in values], dtype=float)


def _min_ref(*cols: list[int | None]) -> int:
    vals = [v for c in cols for v in c if v is not None]
    return min(vals) if vals else 0


def _load_run_json(folder: str) -> dict:
    path = os.path.join(folder, "run.json")
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            return {"_error": f"run.json unreadable: {e}"}


def _flag(text: str) -> bool:
    return text in ("1", "True", "true", "yes")


# ----------------------------------------------------------------------------------------------
# Legacy column detection
# ----------------------------------------------------------------------------------------------
# The user's original capture.py writes exactly these columns (Grabber.grab / main):
#   frame, host_time, hlt_frame_id, tri_frame_id, hlt_ts_ns, tri_ts_ns, arrival_gap_ms,
#   grab_ms, copy_ms, queue_depth, queued
CAPTURE_PY_COLUMNS = ("frame", "host_time", "hlt_frame_id", "tri_frame_id", "hlt_ts_ns", "tri_ts_ns",
                      "arrival_gap_ms")

_CAM_TOKENS = {
    "helios": ("helios", "hlt", "depth", "tof", "xyz", "range", "h2", "pointcloud", "pcd"),
    "triton": ("triton", "tri", "color", "colour", "rgb", "bgr", "bayer"),
}
_TIME_TOKENS = ("ts", "time", "timestamp", "stamp", "tick", "ticks", "t", "ns", "us", "ms")
_HOST_TOKENS = ("host", "recv", "receive", "received", "rx", "arrival", "arrive", "pc", "sys",
                "system", "wall", "perf", "grab", "now", "local", "cpu", "python", "py", "epoch")
_ARRIVAL_TOKENS = ("recv", "receive", "received", "rx", "arrival", "arrive", "arrived")
_DEVICE_TOKENS = ("device", "dev", "cam", "camera", "hw", "hardware", "sensor", "buffer", "buf",
                  "chunk", "ptp")
_ID_TOKENS = ("frame", "id", "fid", "index", "idx", "number", "nr", "count", "counter", "seq")

MAP_KEYS = ("helios_ts", "triton_ts", "helios_host", "triton_host", "pair_host",
            "helios_frame_id", "triton_frame_id", "units", "ts_units", "host_units")


def _tokens(name: str) -> list[str]:
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)       # camelCase -> camel Case
    return [t for t in re.split(r"[^a-z0-9]+", name.lower()) if t]


def _has(tokens: list[str], vocab) -> bool:
    joined = "".join(tokens)
    return any(t in vocab for t in tokens) or any(len(v) > 3 and v in joined for v in vocab)


def guess_legacy_columns(columns: list[str]) -> dict[str, str]:
    """Guess which columns of an unknown old frames.csv hold which timestamp (name-based).

    Only used for layouts other than the original capture.py (which is recognised exactly).
    A column must mention a camera and a time / receive word; with a host word it is a HOST time,
    otherwise the DEVICE timestamp. The map is printed in the report; --map overrides it.
    """
    found: dict[str, str] = {}
    for col in columns:
        tk = _tokens(col)
        if not tk:
            continue
        is_time = ((_has(tk, _TIME_TOKENS) or _has(tk, _ARRIVAL_TOKENS))
                   and not (_has(tk, _ID_TOKENS) and not _has(tk, ("time", "timestamp", "stamp", "ts"))))
        is_id = _has(tk, _ID_TOKENS) and not is_time
        cams = [c for c in CAMERAS if _has(tk, _CAM_TOKENS[c])]
        if len(cams) != 1:
            if not cams and is_time and _has(tk, _HOST_TOKENS):
                found.setdefault("pair_host", col)
            continue
        cam = cams[0]
        if is_time:
            key = f"{cam}_host" if _has(tk, _HOST_TOKENS) and not _has(tk, _DEVICE_TOKENS) else f"{cam}_ts"
            found.setdefault(key, col)
        elif is_id:
            found.setdefault(f"{cam}_frame_id", col)
    return found


def _guess_unit(raw_values: list[str], fps_hint: float | None, column: str = "") -> str:
    """Unit of a timestamp column: from a _ns/_us/_ms/_s name suffix if present, else from the
    median frame interval (closest to the expected period 1/fps)."""
    m = re.search(r"[_\s](ns|us|ms|s)$", column.lower())
    if m:
        return m.group(1)
    nums = []
    for t in raw_values:
        try:
            nums.append(float(t))
        except ValueError:
            continue
    if len(nums) < 3:
        sample = next((t for t in raw_values if t), "")
        try:
            datetime.fromisoformat(sample)
            return "datetime"
        except ValueError:
            return "ns"
    d = np.diff(np.asarray(nums))
    d = d[d > 0]
    if d.size == 0:
        return "ns"
    med = float(np.median(d))
    target = 1.0 / fps_hint if fps_hint else 0.1
    best, best_err = "ns", math.inf
    for unit, f in UNIT_TO_NS.items():
        period_s = med * f / 1e9
        if not (5e-4 <= period_s <= 5.0):
            continue
        err = abs(math.log(period_s / target))
        if err < best_err:
            best, best_err = unit, err
    return best


def parse_map(spec: str | list[str] | None) -> dict[str, str]:
    """Parse '--map helios_ts=COL,triton_ts=COL,units=ms' (may be given several times)."""
    if not spec:
        return {}
    specs = [spec] if isinstance(spec, str) else list(spec)
    out: dict[str, str] = {}
    for s in specs:
        for part in s.split(","):
            if not part.strip():
                continue
            if "=" not in part:
                raise ValueError(f"--map entry {part!r} must look like key=COLUMN")
            k, v = (p.strip() for p in part.split("=", 1))
            if k not in MAP_KEYS:
                raise ValueError(f"--map key {k!r} unknown; use one of {', '.join(MAP_KEYS)}")
            if k.endswith("units") and v not in UNIT_TO_NS:
                raise ValueError(f"--map {k}={v!r}: units must be one of ns, us, ms, s")
            out[k] = v
    return out


# ----------------------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------------------
def load_capture(folder: str, mapping: dict[str, str] | str | None = None,
                 clock_domain: str | None = None, fps: float | None = None) -> Capture:
    """Load a capture folder (or a path to its frames.csv).

    mapping       optional legacy column map (dict or the --map string)
    clock_domain  optional override ("ptp" / "independent"); use only if you KNOW the clocks
    fps           optional nominal frame rate (else run.json, else estimated)
    """
    if isinstance(mapping, (str, list)):
        mapping = parse_map(mapping)
    mapping = dict(mapping or {})
    frames_csv = None
    if os.path.isfile(folder):
        frames_csv = folder if os.path.basename(folder) != "sync_frames.csv" else None
        folder = os.path.dirname(os.path.abspath(folder))
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"capture folder not found: {folder}")
    frames_csv = frames_csv or os.path.join(folder, "frames.csv")
    sync_csv = os.path.join(folder, "sync_frames.csv")
    has_frames, has_sync = os.path.isfile(frames_csv), os.path.isfile(sync_csv)
    if not has_frames and not has_sync:
        raise FileNotFoundError(f"{folder} has neither frames.csv nor sync_frames.csv")

    run = _load_run_json(folder)
    info = run_info(run)
    notes: list[str] = [run["_error"]] if "_error" in run else []
    excluded: dict[str, int] = {}
    fps_setting = fps or info["frame_rate"]
    span_hint = info["seconds_actual"] or info["seconds"]

    cols, rows = _read_csv(frames_csv) if has_frames else ([], [])
    is_new = has_frames and {"helios_ts_ns", "triton_ts_ns"} <= set(cols) and not mapping
    if is_new:
        cap = _load_new(folder, cols, rows, run, notes, excluded, span_hint)
    elif has_frames and rows:
        cap = _load_legacy(folder, cols, rows, run, notes, mapping, fps_setting, excluded, span_hint)
    else:
        cap = None
    sync_frames = _load_sync_frames(sync_csv) if has_sync else None
    if cap is None and sync_frames is None:
        raise ValueError(f"{frames_csv} has no data rows and there is no sync_frames.csv")
    if cap is None:
        cap = _pairs_from_sync_frames(folder, sync_frames, run, notes, fps_setting, excluded, span_hint)
    elif sync_frames is not None:
        _attach_sync_frames(cap, sync_frames, excluded, span_hint)

    cap.info = info
    cap.excluded_ts = excluded
    cap.fps_setting = fps_setting
    cap.synthetic = info["synthetic"]
    _decide_clock_domain(cap, override=clock_domain)
    if cap.fps_setting is None:
        cap.fps_setting = _estimate_fps(cap)
        if cap.fps_setting:
            cap.notes.append(f"nominal fps not in run.json; estimated {cap.fps_setting:.3g} fps "
                             "from the median frame interval")
    return cap


def _ints(rows, col, unit="ns"):
    return [_parse_time_ns(_cell(r, col), unit) for r in rows]


def _floats(rows, col):
    return np.array([_parse_float(_cell(r, col)) for r in rows], dtype=float)


def _load_new(folder, cols, rows, run, notes, excluded, span_hint) -> Capture:
    """Case a: DESIGN.md schema (+ optional round-1 columns). Old-format columns that capture_sync
    keeps in front (frame, host_time, hlt_ts_ns, ...) are simply ignored here."""
    h_raw, t_raw = _ints(rows, "helios_ts_ns"), _ints(rows, "triton_ts_ns")
    h_ts = _clean_times(h_raw, "frames.csv helios_ts_ns", excluded, span_hint)
    t_ts = _clean_times(t_raw, "frames.csv triton_ts_ns", excluded, span_hint)
    # A pair whose timestamp was present but implausible must not fall back to the CSV's offset
    # columns either: the engine computed those from the same bad timestamp.
    ts_excluded = np.array([(a is not None and b is None) or (c is not None and d is None)
                            for a, b, c, d in zip(h_raw, h_ts, t_raw, t_ts)], dtype=bool)
    h_host = _clean_times(_ints(rows, "helios_host_ns"), "frames.csv helios_host_ns", excluded, span_hint)
    t_host = _clean_times(_ints(rows, "triton_host_ns"), "frames.csv triton_host_ns", excluded, span_hint)
    ts_ref, host_ref = _min_ref(h_ts, t_ts), _min_ref(h_host, t_host)
    n = len(rows)
    pairs = {
        "pair_index": _floats(rows, "pair_index") if "pair_index" in cols else np.arange(n, dtype=float),
        "grid_k": _floats(rows, "grid_k"),
        "helios_frame_id": _floats(rows, "helios_frame_id"),
        "triton_frame_id": _floats(rows, "triton_frame_id"),
        "helios_ts_ns": _relative(h_ts, ts_ref),
        "triton_ts_ns": _relative(t_ts, ts_ref),
        "offset_start_us_file": _floats(rows, "offset_start_us"),
        "triton_exposure_us": _floats(rows, "triton_exposure_us"),
        "offset_center_model_us_file": _floats(rows, "offset_center_model_us"),
        "helios_host_ns": _relative(h_host, host_ref),
        "triton_host_ns": _relative(t_host, host_ref),
        "host_delta_ms_file": _floats(rows, "host_delta_ms"),
        # optional round-1 columns (NaN / False / "" when absent)
        "planned_lag_us": _floats(rows, "planned_lag_us"),
        "residual_us_file": _floats(rows, "residual_us"),
        "suspect": np.array([_flag(_cell(r, "suspect")) for r in rows], dtype=bool),
        "suspect_reason": [_cell(r, "suspect_reason") for r in rows],
        "segment_id": _floats(rows, "segment_id"),
        "ts_excluded": ts_excluded,
    }
    domains = [_cell(r, "clock_domain").lower() for r in rows if _cell(r, "clock_domain")]
    cap = Capture(folder, "new", "unknown", "", pairs, {}, run, ts_ref, host_ref, notes=notes)
    if domains:
        vals, counts = np.unique(domains, return_counts=True)
        cap.column_map["clock_domain_column"] = dict(zip(vals.tolist(), counts.tolist()))
        if len(vals) > 1:
            notes.append(f"frames.csv mixes clock domains {cap.column_map['clock_domain_column']}; "
                         "the majority is used")
    _frames_from_pairs(cap)
    return cap


def _empty_optional(n: int) -> dict:
    return {"planned_lag_us": np.full(n, math.nan), "residual_us_file": np.full(n, math.nan),
            "suspect": np.zeros(n, dtype=bool), "suspect_reason": [""] * n,
            "segment_id": np.full(n, math.nan), "ts_excluded": np.zeros(n, dtype=bool)}


def _load_legacy(folder, cols, rows, run, notes, mapping, fps_hint, excluded, span_hint) -> Capture:
    """Case b: old frames.csv. The original capture.py layout is recognised exactly; anything else
    via name guessing and/or --map."""
    n = len(rows)
    capture_py = set(CAPTURE_PY_COLUMNS) <= set(cols) and not mapping
    if capture_py:
        # host_time = time.time() when the HELIOS get_buffer returned; arrival_gap_ms =
        # |tri_rx_host - hlt_rx_host| (sign lost). The Triton buffer is fetched sequentially AFTER
        # the Helios one, so tri_rx_host >= hlt_rx_host and Triton host time = host_time + gap.
        colmap = {"helios_ts": "hlt_ts_ns", "triton_ts": "tri_ts_ns", "helios_host": "host_time",
                  "triton_host": "host_time + arrival_gap_ms/1000",
                  "helios_frame_id": "hlt_frame_id", "triton_frame_id": "tri_frame_id",
                  "units": {"helios_ts": "ns", "triton_ts": "ns", "helios_host": "s (epoch)",
                            "triton_host": "s (epoch)"},
                  "source": "original capture.py layout (recognised exactly)"}
        h_ts = _ints(rows, "hlt_ts_ns")
        t_ts = _ints(rows, "tri_ts_ns")
        h_host = _ints(rows, "host_time", "s")
        gaps = _floats(rows, "arrival_gap_ms")
        t_host = [None if (h is None or not math.isfinite(g)) else h + int(round(g * 1e6))
                  for h, g in zip(h_host, gaps)]
        p_host = [None] * n
        ids = {"helios_frame_id": _floats(rows, "hlt_frame_id"),
               "triton_frame_id": _floats(rows, "tri_frame_id")}
        notes.append("original capture.py frames.csv: Triton host time reconstructed as host_time + "
                     "arrival_gap_ms (the gap is stored as an absolute value; capture.py fetches the "
                     "Triton buffer after the Helios one, so the Triton host time is never earlier).")
        kind = "capture_py"
    else:
        guessed = guess_legacy_columns(cols)
        colmap = dict(guessed)
        colmap.update({k: v for k, v in mapping.items() if not k.endswith("units")})
        missing = [v for v in colmap.values() if v not in cols]
        if missing:
            raise ValueError(f"--map refers to column(s) {missing} not in frames.csv; columns are {cols}")
        if not any(k in colmap for k in ("helios_ts", "triton_ts", "helios_host", "triton_host")):
            raise ValueError(
                "could not find any timestamp columns in frames.csv. Columns are:\n  " + ", ".join(cols)
                + "\nTell the analyser which column is which, e.g.\n  --map helios_ts=COL,triton_ts=COL,"
                  "helios_host=COL,triton_host=COL,units=ns")
        units = {}
        for key in ("helios_ts", "triton_ts", "helios_host", "triton_host", "pair_host"):
            if key in colmap:
                explicit = mapping.get("host_units" if "host" in key else "ts_units") or mapping.get("units")
                units[key] = explicit or _guess_unit([_cell(r, colmap[key]) for r in rows], fps_hint, colmap[key])
        colmap["units"] = units
        colmap["source"] = "--map + auto-detect" if mapping else "auto-detected from column names"

        def col_ints(key):
            return _ints(rows, colmap[key], units[key]) if key in colmap else [None] * n

        h_ts, t_ts = col_ints("helios_ts"), col_ints("triton_ts")
        h_host, t_host, p_host = col_ints("helios_host"), col_ints("triton_host"), col_ints("pair_host")
        ids = {k: _floats(rows, colmap.get(k)) for k in ("helios_frame_id", "triton_frame_id")}
        notes.append("legacy frames.csv: columns mapped as " + json.dumps(colmap)
                     + " - verify this, override with --map if wrong")
        kind = "legacy"

    h_ts = _clean_times(h_ts, "frames.csv helios device ts", excluded, span_hint)
    t_ts = _clean_times(t_ts, "frames.csv triton device ts", excluded, span_hint)
    h_host = _clean_times(h_host, "frames.csv helios host time", excluded, span_hint)
    t_host = _clean_times(t_host, "frames.csv triton host time", excluded, span_hint)
    p_host = _clean_times(p_host, "frames.csv pair host time", excluded, span_hint)
    host_ref = _min_ref(h_host, t_host, p_host)
    nan = np.full(n, math.nan)
    pairs = {
        "pair_index": _floats(rows, "frame") if capture_py else np.arange(n, dtype=float),
        "grid_k": nan.copy(),
        **ids,
        # Independent clocks: each camera's device time relative to ITS OWN first frame. Valid for
        # per-camera intervals only, never for cross-camera offsets.
        "helios_ts_ns": _relative(h_ts, _min_ref(h_ts)),
        "triton_ts_ns": _relative(t_ts, _min_ref(t_ts)),
        "offset_start_us_file": nan.copy(),
        "triton_exposure_us": nan.copy(),
        "offset_center_model_us_file": nan.copy(),
        "helios_host_ns": _relative(h_host, host_ref),
        "triton_host_ns": _relative(t_host, host_ref),
        "pair_host_ns": _relative(p_host, host_ref),
        "host_delta_ms_file": nan.copy(),
        **_empty_optional(n),
    }
    if capture_py and "queued" in cols:
        pairs["saved"] = np.array([_flag(_cell(r, "queued")) for r in rows], dtype=bool)
    cap = Capture(folder, kind, "unknown", "", pairs, {}, run, 0, host_ref,
                  column_map=colmap, notes=notes)
    _frames_from_pairs(cap)
    return cap


def _frames_from_pairs(cap: Capture) -> None:
    """Per-camera frame lists derived from frames.csv (only frames that ended up in a pair)."""
    p = cap.pairs
    for cam in CAMERAS:
        cap.frames[cam] = CameraFrames(
            ts_ns=p[f"{cam}_ts_ns"], host_ns=p[f"{cam}_host_ns"], grid_k=p["grid_k"],
            frame_id=p[f"{cam}_frame_id"], incomplete=np.zeros(cap.n_pairs, dtype=bool),
            source="frames.csv (paired frames only)")


def _load_sync_frames(path: str) -> dict:
    _, rows = _read_csv(path)
    out = {}
    for cam in CAMERAS:
        rr = [r for r in rows if _cell(r, "camera").lower() == cam]
        out[cam] = {
            "ts": _ints(rr, "ts_ns"),
            "host": _ints(rr, "host_ns"),
            "grid_k": _floats(rr, "grid_k"),
            "frame_id": _floats(rr, "frame_id"),
            "incomplete": np.array([_flag(_cell(r, "incomplete")) for r in rr], dtype=bool),
            "paired": np.array([_flag(_cell(r, "paired")) for r in rr], dtype=bool),
            "pair_index": _floats(rr, "pair_index"),
            "exposure_us": _floats(rr, "exposure_us"),
            "segment_id": _floats(rr, "segment_id"),
            "note": [_cell(r, "note") for r in rr],
        }
    return out


def _attach_sync_frames(cap: Capture, sf: dict, excluded: dict, span_hint) -> None:
    """Replace the pair-derived per-camera lists by the complete per-frame log."""
    for cam in CAMERAS:
        d = sf[cam]
        if not d["ts"] and not d["host"]:
            continue
        ts = _clean_times(d["ts"], f"sync_frames.csv {cam} ts_ns", excluded, span_hint)
        host = _clean_times(d["host"], f"sync_frames.csv {cam} host_ns", excluded, span_hint)
        k = d["grid_k"].copy()
        k[[v is None for v in ts]] = math.nan               # a bad timestamp has no trustworthy k
        notes = [n for n in d["note"] if n]
        cap.frames[cam] = CameraFrames(
            ts_ns=_relative(ts, cap.ts_ref_ns), host_ns=_relative(host, cap.host_ref_ns),
            grid_k=k, frame_id=d["frame_id"], incomplete=d["incomplete"], source="sync_frames.csv",
            notes=notes)


def _pairs_from_sync_frames(folder, sf, run, notes, fps_hint, excluded, span_hint) -> Capture:
    """Case c: rebuild pairs from the per-frame log.

    Preference (use the pairing the capture itself made if it was recorded):
      1. pair_index column   2. same PTP grid index k   3. nearest timestamp within half a period
      (device ts if on PTP, else host time = estimate)
    """
    h, t = sf["helios"], sf["triton"]
    h_ts = _clean_times(h["ts"], "sync_frames.csv helios ts_ns", excluded, span_hint)
    t_ts = _clean_times(t["ts"], "sync_frames.csv triton ts_ns", excluded, span_hint)
    h_host = _clean_times(h["host"], "sync_frames.csv helios host_ns", excluded, span_hint)
    t_host = _clean_times(t["host"], "sync_frames.csv triton host_ns", excluded, span_hint)
    ts_ref, host_ref = _min_ref(h_ts, t_ts), _min_ref(h_host, t_host)
    hi_ts, ti_ts = _relative(h_ts, ts_ref), _relative(t_ts, ts_ref)
    hi_host, ti_host = _relative(h_host, host_ref), _relative(t_host, host_ref)
    ok_h, ok_t = ~h["incomplete"], ~t["incomplete"]

    pairs_idx: list[tuple[int, int]] = []
    if np.isfinite(h["pair_index"]).any() and np.isfinite(t["pair_index"]).any():
        how = "pair_index column"
        tmap = {int(pi): j for j, pi in enumerate(t["pair_index"]) if np.isfinite(pi)}
        pairs_idx = [(i, tmap[int(pi)]) for i, pi in enumerate(h["pair_index"])
                     if np.isfinite(pi) and int(pi) in tmap]
    elif np.isfinite(h["grid_k"]).any() and np.isfinite(t["grid_k"]).any():
        how = "PTP grid index k"
        tmap = {int(k): j for j, k in enumerate(t["grid_k"]) if np.isfinite(k) and ok_t[j]}
        pairs_idx = [(i, tmap[int(k)]) for i, k in enumerate(h["grid_k"])
                     if np.isfinite(k) and ok_h[i] and int(k) in tmap]
    else:
        use_dev = np.isfinite(hi_ts).any() and (run.get("sync") or {}).get("method_used") in PTP_METHODS
        a, b = (hi_ts, ti_ts) if use_dev else (hi_host, ti_host)
        how = "nearest device timestamp" if use_dev else "nearest HOST receive time (estimate)"
        period_ns = 1e9 / (fps_hint or _fps_from(a) or 10.0)
        order = np.argsort(b)
        bs = b[order]
        for i, x in enumerate(a):
            if not np.isfinite(x) or not ok_h[i]:
                continue
            j = int(np.searchsorted(bs, x))
            cands = [c for c in (j - 1, j) if 0 <= c < bs.size and np.isfinite(bs[c])]
            if not cands:
                continue
            c = min(cands, key=lambda c: abs(bs[c] - x))
            if abs(bs[c] - x) <= period_ns / 2 and ok_t[order[c]]:
                pairs_idx.append((i, int(order[c])))
    notes.append(f"pairs rebuilt from sync_frames.csv using {how}")

    hi = np.array([p[0] for p in pairs_idx], dtype=int)
    ti = np.array([p[1] for p in pairs_idx], dtype=int)
    n = hi.size
    nan = np.full(n, math.nan)
    pairs = {
        "pair_index": np.arange(n, dtype=float),
        "grid_k": h["grid_k"][hi] if n else nan,
        "helios_frame_id": h["frame_id"][hi] if n else nan,
        "triton_frame_id": t["frame_id"][ti] if n else nan,
        "helios_ts_ns": hi_ts[hi] if n else nan,
        "triton_ts_ns": ti_ts[ti] if n else nan,
        "offset_start_us_file": nan.copy(),
        "triton_exposure_us": t["exposure_us"][ti] if n else nan,
        "offset_center_model_us_file": nan.copy(),
        "helios_host_ns": hi_host[hi] if n else nan,
        "triton_host_ns": ti_host[ti] if n else nan,
        "host_delta_ms_file": nan.copy(),
        **_empty_optional(n),
    }
    if n:
        pairs["segment_id"] = h["segment_id"][hi]
    cap = Capture(folder, "sync_frames_only", "unknown", "", pairs, {}, run, ts_ref, host_ref,
                  column_map={"pairing": how}, notes=notes)
    _attach_sync_frames(cap, sf, {}, span_hint)       # timestamps already counted above
    return cap


def _method_domain(info: dict) -> str | None:
    if info.get("clock_domain") in ("ptp", "independent"):
        return info["clock_domain"]
    m = info.get("method_used")
    if not m:
        return None
    if m in PTP_METHODS:
        return "ptp"
    if m == "freerun" or str(m).startswith("legacy"):
        return "independent"
    return None                     # e.g. "software": PTP only if it was locked -> need the column


def _decide_clock_domain(cap: Capture, override: str | None) -> None:
    """Whether the device timestamps of the two cameras share one clock.

    Precedence: explicit override > clock_domain column > run.json (clock_domain / method_used) >
    assumption. Captures of the original capture.py are independent: it never enabled PTP.
    """
    col = cap.column_map.get("clock_domain_column")
    dom = _method_domain(cap.info)
    if override:
        cap.clock_domain, cap.clock_domain_source = override, "command-line override"
        cap.notes.append(f"clock domain forced to {override!r} on the command line")
    elif col:
        cap.clock_domain = max(col, key=col.get)
        cap.clock_domain_source = "frames.csv clock_domain column"
    elif dom:
        cap.clock_domain = dom
        cap.clock_domain_source = f"run.json (method {cap.info.get('method_used')})"
    elif cap.source_kind in ("legacy", "capture_py"):
        cap.clock_domain = "independent"
        cap.clock_domain_source = "assumed (original capture.py: no PTP)"
    else:
        cap.clock_domain = "unknown"
        cap.clock_domain_source = "no information; treated as independent"
    if cap.clock_domain not in ("ptp", "independent", "unknown"):
        cap.notes.append(f"unrecognised clock_domain {cap.clock_domain!r}; treated as independent")
        cap.clock_domain = "unknown"

    ptp = cap.info.get("ptp") or {}
    if cap.clock_domain == "ptp" and ptp:
        sts = [str(ptp.get(k, "")).lower() for k in ("helios_status", "triton_status")]
        if not all(s in ("master", "slave") for s in sts):
            cap.notes.append(f"WARNING: run.json PTP status is {sts}; expected one Master + one Slave. "
                             "Offsets may not be on a common clock.")


def _fps_from(t_ns: np.ndarray) -> float | None:
    t = np.asarray(t_ns, dtype=float)
    t = np.sort(t[np.isfinite(t)])
    if t.size < 3:
        return None
    d = np.diff(t)
    d = d[d > 0]
    return float(1e9 / np.median(d)) if d.size else None


def _estimate_fps(cap: Capture) -> float | None:
    for cam in CAMERAS:
        f = cap.frames.get(cam)
        if f is not None:
            for arr in (f.ts_ns, f.host_ns):
                v = _fps_from(arr)
                if v:
                    return v
    return None


# ----------------------------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------------------------
def _axis(cap: Capture) -> str:
    """The most shareable clock for plotting both cameras on one time axis."""
    if cap.clock_domain == "ptp" and any(np.isfinite(cap.frames[c].ts_ns).any() for c in CAMERAS):
        return "ptp"
    if any(np.isfinite(cap.frames[c].host_ns).any() for c in CAMERAS) or \
            np.isfinite(cap.pairs.get("pair_host_ns", np.array([np.nan]))).any():
        return "host"
    return "own"


AXIS_LABEL = {"ptp": "shared PTP clock", "host": "host receive time", "own": "each camera's own clock"}


def _frame_times(f: CameraFrames, axis: str) -> np.ndarray:
    """Absolute (relative-to-ref) ns of a camera's frames on the chosen axis; NaN when invalid."""
    if axis == "ptp":
        return f.ts_ns
    if axis == "host":
        return f.host_ns
    return f.ts_ns - np.nanmin(f.ts_ns) if np.isfinite(f.ts_ns).any() else f.host_ns


def count_drops(present_k: np.ndarray, window: tuple[int, int] | None, max_span: int) -> dict:
    """Grid-based drop count of one camera: expected slots k in `window` minus delivered slots.

    The single definition used everywhere (see DROP_WINDOW / INCOMPLETE_IS_DROP). Never allocates
    an array proportional to the k span unless that span is plausible (<= max_span)."""
    if window is None:
        return {"drops": None}
    lo, hi = window
    span = hi - lo + 1
    ks = np.unique(present_k[np.isfinite(present_k)].astype(np.int64))
    ks = ks[(ks >= lo) & (ks <= hi)]
    if span > max_span:
        return {"drops": None, "error": f"grid index span {span} is implausible (> {max_span}); "
                                        "a timestamp step? drops not counted"}
    missing = np.setdiff1d(np.arange(lo, hi + 1, dtype=np.int64), ks)
    return {"drops": int(missing.size), "missing_k": missing, "expected": int(span), "delivered": int(ks.size)}


def _per_camera(cap: Capture, cam: str, axis: str, t0: float, run_window: tuple[float, float] | None,
                period_s: float | None, k_window: tuple[int, int] | None, max_span: int) -> dict:
    """Intervals, jitter, fps and drops of ONE camera (valid even without PTP: own clock)."""
    f = cap.frames[cam]
    good = ~f.incomplete
    use_dev = np.isfinite(f.ts_ns[good]).sum() >= 3
    clock = f.ts_ns if use_dev else f.host_ns
    t_axis = (_frame_times(f, axis) - t0) / 1e9
    sel = good & np.isfinite(clock)
    order = np.argsort(clock[sel])
    tc, ta = clock[sel][order], t_axis[sel][order]
    res: dict[str, Any] = {"received": int(f.incomplete.size), "complete_valid": int(sel.sum()),
                           "frames_source": f.source, "incomplete": int(f.incomplete.sum()),
                           "interval_basis": BASIS_PER_CAMERA if use_dev else BASIS_HOST,
                           "_t": ta, "_drop_t": np.array([])}
    invalid_notes = sum(1 for n in f.notes if "invalid" in n.lower())
    if invalid_notes:
        res["engine_flagged_invalid_ts"] = invalid_notes
    fid = f.frame_id[np.isfinite(f.frame_id)]
    if fid.size >= 2:
        u = np.unique(fid.astype(np.int64))
        d_id = np.diff(u)
        res["frame_id_gaps"] = int(np.sum(d_id[d_id > 1] - 1)) if d_id.size else 0
    if tc.size < 3:
        res["drops"] = None
        return res
    d = np.diff(tc)
    med = float(np.median(d))
    normal = d[d <= DROP_GAP_FACTOR * med]
    res.update({
        "interval_median_ms": med / 1e6,
        "interval_mean_ms": float(np.mean(d)) / 1e6,
        "jitter_std_us": float(np.std(normal, ddof=1)) / 1e3 if normal.size > 1 else 0.0,
        "jitter_p1_p99_us": [float(np.percentile(normal, 1) - med) / 1e3,
                             float(np.percentile(normal, 99) - med) / 1e3] if normal.size else None,
        "fps_from_intervals": 1e9 / med,
        "first_s": float(np.nanmin(ta)) if np.isfinite(ta).any() else None,
        "last_s": float(np.nanmax(ta)) if np.isfinite(ta).any() else None,
    })

    # Drops - WHY two methods: with a PTP grid each frame has an exact slot k, so a missing k is an
    # exact drop. Without a grid, drops are inferred from long gaps (+ missing frames at the edges
    # of the run window on a shared clock, so a camera that stopped early is charged for it).
    present = f.grid_k if not INCOMPLETE_IS_DROP else np.where(good, f.grid_k, math.nan)
    if k_window is not None and np.isfinite(f.grid_k).any() and period_s:
        dr = count_drops(present, k_window, max_span)
        res["drops"] = dr["drops"]
        res["drop_method"] = (f"missing PTP grid index k in the {DROP_WINDOW} of both cameras' k ranges"
                              + ("" if INCOMPLETE_IS_DROP else "; incomplete frames count as delivered"))
        if dr.get("error"):
            res["drop_error"] = dr["error"]
        if dr.get("missing_k") is not None:
            res["_drop_t"] = (dr["missing_k"] - k_window[0]) * period_s
            res["expected"] = dr["expected"]
    else:
        gaps = d > DROP_GAP_FACTOR * med
        mult = np.maximum(np.round(d[gaps] / med).astype(int) - 1, 1)
        interior = int(mult.sum())
        drop_t = list(np.repeat(ta[:-1][gaps], mult))
        lead = tail = 0
        if run_window is not None and axis != "own" and period_s and np.isfinite(ta).any():
            lead = max(int(round((np.nanmin(ta) - run_window[0]) / period_s)), 0)
            tail = max(int(round((run_window[1] - np.nanmax(ta)) / period_s)), 0)
            drop_t += [run_window[0]] * lead + [float(np.nanmax(ta))] * tail
        res["drops"] = interior + lead + tail
        res["drop_gaps"] = int(gaps.sum())
        res["drop_method"] = (f"interval > {DROP_GAP_FACTOR} x median ({'device' if use_dev else 'host'} "
                              "clock) + missing frames at the run edges")
        res["_drop_t"] = np.array(drop_t, dtype=float)
    if f.source.startswith("frames.csv"):
        res["drop_note"] = ("only frames that ended up in a pair are known (no sync_frames.csv); "
                            "drops include frames lost at pairing / writer")
    return res


def _fps_windows(t_s: np.ndarray, width: float, t_end: float, shift: float = 0.0):
    """Frames per window. WHY `shift`: frames on a perfect grid land exactly on whole seconds, so
    integer-second bin edges would randomly put a frame left or right (9 / 11 fps aliasing).
    Starting the bins half a frame period early puts every frame mid-bin."""
    t = t_s[np.isfinite(t_s)]
    if t.size < 2 or t_end < width or (t_end + shift) / width > MAX_WINDOWS:
        return np.array([]), np.array([])
    edges = np.arange(-shift, t_end + 1e-9, width)
    if edges.size < 2:
        return np.array([]), np.array([])
    counts, _ = np.histogram(t, bins=edges)          # only FULL windows
    return edges[:-1], counts / width


def analyze(cap: Capture) -> dict:
    """Compute every sync metric of one capture. Returns a JSON-able dict (+ '_series' arrays)."""
    p, info = cap.pairs, cap.info
    ptp = cap.clock_domain == "ptp"
    period_s = 1.0 / cap.fps_setting if cap.fps_setting else None
    notes = list(cap.notes)
    warnings_: list[str] = []
    method = info.get("method_used")

    # --- run window on the shared axis, from COMPLETE frames with valid timestamps only ----------
    axis = _axis(cap)
    starts, ends = {}, {}
    for c in CAMERAS:
        f = cap.frames[c]
        t = _frame_times(f, axis)[~f.incomplete]
        t = t[np.isfinite(t)]
        if t.size:
            starts[c], ends[c] = float(t.min()), float(t.max())
    if not starts and axis == "host" and np.isfinite(p.get("pair_host_ns", np.array([np.nan]))).any():
        ph = p["pair_host_ns"]
        starts["pair"], ends["pair"] = float(np.nanmin(ph)), float(np.nanmax(ph))
    t0 = min(starts.values()) if starts else 0.0
    union_end = (max(ends.values()) - t0) / 1e9 if ends else 0.0
    run_window = (0.0, union_end) if axis != "own" else None

    # --- time of each pair -----------------------------------------------------------------------
    if axis == "ptp":
        t_pair = (p["helios_ts_ns"] - t0) / 1e9
    elif axis == "host":
        hh = p["helios_host_ns"] if np.isfinite(p["helios_host_ns"]).any() else p.get("pair_host_ns", p["helios_host_ns"])
        t_pair = (hh - t0) / 1e9
    else:
        hh = p["helios_ts_ns"]
        t_pair = (hh - np.nanmin(hh)) / 1e9 if np.isfinite(hh).any() else np.arange(cap.n_pairs) * (period_s or 0.1)
    tp = t_pair[np.isfinite(t_pair)]
    pair_span = float(tp.max() - tp.min()) if tp.size > 1 else 0.0

    # --- grid window for drops -------------------------------------------------------------------
    k_window = None
    n_frames_total = sum(int(np.isfinite(cap.frames[c].grid_k).sum()) for c in CAMERAS)
    max_span = MAX_K_SPAN_FACTOR * max(n_frames_total, 1) + 1000
    ranges = []
    for c in CAMERAS:
        f = cap.frames[c]
        k = f.grid_k[(~f.incomplete) & np.isfinite(f.grid_k)]
        if k.size:
            ranges.append((int(k.min()), int(k.max())))
    if ranges and period_s:
        k_window = ((min(r[0] for r in ranges), max(r[1] for r in ranges)) if DROP_WINDOW == "union"
                    else (max(r[0] for r in ranges), min(r[1] for r in ranges)))
        if k_window[1] < k_window[0]:
            k_window = None

    cams = {c: _per_camera(cap, c, axis, t0, run_window, period_s, k_window, max_span) for c in CAMERAS}

    # --- duration, yield, dead-camera detection --------------------------------------------------
    seconds_actual = info.get("seconds_actual")
    union_span = union_end if starts else pair_span
    duration = seconds_actual or union_span
    duration_source = "run.json elapsed time" if seconds_actual else f"union of both cameras' frames ({AXIS_LABEL[axis]})"
    if period_s and axis != "own":
        gap_lim = max(DEAD_CAMERA_GAP_S, 5 * period_s)
        for c in CAMERAS:
            cr = cams[c]
            if cr.get("first_s") is None:
                warnings_.append(f"{c.upper()} DELIVERED NO USABLE FRAMES.")
                continue
            if cr["first_s"] > gap_lim:
                warnings_.append(f"{c.upper()} STARTED LATE: first frame {cr['first_s']:.1f} s after the "
                                 "other camera's first frame.")
            if union_span - cr["last_s"] > gap_lim:
                warnings_.append(f"{c.upper()} STOPPED DELIVERING FRAMES at t = {cr['last_s']:.1f} s of a "
                                 f"{union_span:.1f} s run ({union_span - cr['last_s']:.1f} s missing).")
        if seconds_actual and seconds_actual - union_span > gap_lim + 1.0:
            warnings_.append(f"BOTH CAMERAS' frames end {seconds_actual - union_span:.1f} s before the end of "
                             f"the {seconds_actual:.1f} s run (run.json elapsed time).")
    expected_pairs = duration * cap.fps_setting if (duration and cap.fps_setting) else None

    # --- engine vs analysis drop numbers ---------------------------------------------------------
    for c in CAMERAS:
        eng = info["per_camera"][c]
        cams[c]["engine"] = {k: v for k, v in eng.items() if v is not None}
        if eng.get("dropped") is not None and cams[c].get("drops") is not None \
                and int(eng["dropped"]) != int(cams[c]["drops"]):
            notes.append(f"{c}: engine reports {eng['dropped']} dropped "
                         f"({eng.get('drop_definition') or 'engine definition'}), analysis counts "
                         f"{cams[c]['drops']} ({cams[c].get('drop_method')}).")

    # --- suspect pairs: engine flags + an independent host-delta check (PTP runs only) ----------
    host_delta = (p["triton_host_ns"] - p["helios_host_ns"]) / 1e6
    fill = ~np.isfinite(host_delta)
    host_delta[fill] = p["host_delta_ms_file"][fill]
    suspect = p["suspect"].copy()
    reasons = [r if s else "" for r, s in zip(p["suspect_reason"], suspect)]
    reasons = [r or ("engine flag" if s else "") for r, s in zip(reasons, suspect)]
    if ptp and period_s and cap.n_pairs >= 20 and np.isfinite(host_delta).sum() >= 20:
        # Global (not running) median: on one PTP clock the host delta is just the transfer-latency
        # difference and has no reason to drift, while a PTP step shifts ALL later pairs by n periods
        # - a running median would follow the step and only flag its edge.
        band = HOST_DELTA_BAND_PERIODS * period_s * 1e3
        jump = np.abs(host_delta - np.nanmedian(host_delta)) > band
        for i in np.flatnonzero(jump):
            suspect[i] = True
            reasons[i] = (reasons[i] + "; " if reasons[i] else "") + "host_delta_jump (analysis check)"
    reason_counts = Counter()
    for r, s in zip(reasons, suspect):
        if s:
            for part in r.split(";"):
                reason_counts[part.strip() or "unspecified"] += 1
    good_pair = ~suspect
    suspect_info = {"n": int(suspect.sum()), "reasons": dict(reason_counts),
                    "engine_rejected_suspect": info.get("rejected_suspect"),
                    "note": "suspect pairs are excluded from the offset / residual / centre statistics"}

    # --- MEASURED timestamp offset, residual vs plan ----------------------------------------------
    offset = residual = centre = None
    planned = None
    if ptp:
        offset = device_offset_us(p["helios_ts_ns"], p["triton_ts_ns"], cap.clock_domain)
        fill = ~np.isfinite(offset) & ~p["ts_excluded"]
        offset[fill] = p["offset_start_us_file"][fill]
        planned = p["planned_lag_us"].copy()
        if not np.isfinite(planned).any():
            planned[:] = info.get("planned_lag_us") if info.get("planned_lag_us") is not None else math.nan
        if not np.isfinite(planned).any() and method in SCHEDULED_METHODS:
            planned[:] = 0.0                                    # old start-aligned files: plan = 0
            notes.append("no planned lag recorded: residual computed against a plan of 0 µs (start alignment)")
        residual = offset - planned
        rf = p["residual_us_file"]
        fill_r = ~np.isfinite(residual) & np.isfinite(rf) & ~p["ts_excluded"]
        residual[fill_r] = rf[fill_r]
        if np.nanmedian(np.abs(offset)) > 1e6:
            warnings_.append("median |timestamp offset| > 1 s although clock_domain is 'ptp' - PTP was "
                             "probably not locked; the offsets are NOT trustworthy.")
    elif np.isfinite(p["offset_start_us_file"]).any():
        notes.append("frames.csv contains offset_start_us values but the clock domain is "
                     f"{cap.clock_domain!r}: they were IGNORED (two independent clocks cannot be compared).")

    helios_center_us = info["helios_center_us"]
    exp_source = info.get("triton_exposure_source")
    if offset is not None:
        centre = p["offset_center_model_us_file"].copy()
        centre[p["ts_excluded"] | ~np.isfinite(offset)] = math.nan
        need = ~np.isfinite(centre) & np.isfinite(p["triton_exposure_us"])
        centre[need] = offset[need] + p["triton_exposure_us"][need] / 2 - helios_center_us
    centre_basis = (f"MODELLED (helios_center_us = {helios_center_us:.0f} µs from {info['helios_center_source']}"
                    f"; Triton exposure from {exp_source or 'per-frame value in the CSV'}; unverified)")
    if exp_source and "cap" in str(exp_source).lower():
        warnings_.append("Triton exposure per frame unknown: the exposure CAP was used as an upper bound, "
                         "so the modelled centre offset may be off by up to cap/2.")

    def masked(x):
        return None if x is None else np.where(good_pair, x, math.nan)

    off_g, res_g, cen_g = masked(offset), masked(residual), masked(centre)
    use_res = res_g is not None and np.isfinite(res_g).any() and method in SCHEDULED_METHODS
    drift = drift_summary(t_pair, res_g if use_res else off_g, "µs") if offset is not None else None

    # --- segments (PTP re-lock etc.) -------------------------------------------------------------
    segments = None
    seg = p["segment_id"]
    if np.isfinite(seg).any() and np.unique(seg[np.isfinite(seg)]).size > 1:
        segments = []
        for s in np.unique(seg[np.isfinite(seg)]):
            m = seg == s
            entry = {"segment_id": int(s), "pairs": int(m.sum())}
            if offset is not None:
                entry["timestamp_offset_median_us"] = float(np.nanmedian(off_g[m])) if np.isfinite(off_g[m]).any() else None
            segments.append(entry)

    # --- host-based numbers, legacy estimate -----------------------------------------------------
    hd_drift = drift_summary(t_pair, host_delta, "ms") if np.isfinite(host_delta).sum() >= 3 else None
    legacy = None if ptp else _legacy_estimate(cap, host_delta, period_s)

    # --- fps over time ---------------------------------------------------------------------------
    t_end = union_span
    fps_win = {}
    half = 0.5 * (period_s or (pair_span / max(cap.n_pairs - 1, 1)) or 0.0)
    for w in FPS_WINDOWS_S:
        start, pf = _fps_windows(t_pair, w, t_end, half)
        entry = {"window_start_s": start.tolist(), "pairs": pf.tolist()}
        for c in CAMERAS:
            _, cf = _fps_windows(cams[c]["_t"], w, t_end, half)
            entry[c] = cf.tolist() if cf.size == start.size else []
        fps_win[f"{int(w)}s"] = entry
    if t_end / min(FPS_WINDOWS_S) > MAX_WINDOWS:
        notes.append(f"time axis spans {t_end:.0f} s - fps windows skipped (cap {MAX_WINDOWS})")

    gk = p["grid_k"][np.isfinite(p["grid_k"])]
    missing_pairs = None
    if gk.size >= 2 and (gk.max() - gk.min()) < max_span:
        missing_pairs = int((gk.max() - gk.min() + 1) - np.unique(gk).size)

    incomplete = info.get("incomplete")
    if incomplete is None and any(cams[c]["frames_source"] == "sync_frames.csv" for c in CAMERAS):
        incomplete = int(sum(cams[c]["incomplete"] for c in CAMERAS))
    if cap.excluded_ts:
        warnings_.append(f"{sum(cap.excluded_ts.values())} implausible timestamps (<= 0 or > "
                         f"{TS_PLAUSIBLE_RANGE_S:.0f} s from the median) were EXCLUDED: {cap.excluded_ts}")
    if suspect_info["n"]:
        warnings_.append(f"{suspect_info['n']} suspect pairs excluded from the sync statistics: "
                         f"{suspect_info['reasons']}")
    ptp_events = _ptp_events(info)
    warnings_ += ptp_events["warnings"]
    if info.get("action_skipped_late"):
        warnings_.append(f"{info['action_skipped_late']} action commands were skipped because they "
                         "would have arrived late.")

    n_valid = int(np.isfinite(off_g).sum()) if off_g is not None else None
    result: dict[str, Any] = {
        "meta": {
            "folder": cap.folder, "source_kind": cap.source_kind, "synthetic": cap.synthetic,
            "clock_domain": cap.clock_domain, "clock_domain_source": cap.clock_domain_source,
            "method_requested": info.get("method_requested"), "method_used": method,
            "legacy_sync_mode": info.get("legacy_sync_mode"),
            "fallback_chain": info.get("fallback_chain"), "fps_setting": cap.fps_setting,
            "seconds_setting": info.get("seconds"), "seconds_actual": seconds_actual,
            "stop_reason": info.get("stop_reason"),
            "alignment_requested": info.get("alignment_requested"),
            "alignment_used": info.get("alignment_used"),
            "planned_lag_us": (float(np.nanmedian(planned)) if planned is not None and np.isfinite(planned).any()
                               else info.get("planned_lag_us")),
            "exposure_cap_us": info.get("exposure_cap_us"),
            "helios_center_us": helios_center_us, "helios_center_source": info["helios_center_source"],
            "triton_trigger_delay_us": info.get("triton_trigger_delay_us"),
            "triton_exposure_source": exp_source,
            "ptp": info.get("ptp"), "ptp_settled": info.get("ptp_settled"),
            "action": info.get("action"), "action_skipped_late": info.get("action_skipped_late"),
            "cameras": info.get("cameras"), "sdk": info.get("sdk"),
            "missing_nodes": info.get("missing_nodes"), "failed_node_writes": info.get("failed_node_writes"),
            "column_map": cap.column_map, "time_axis": AXIS_LABEL[axis],
            "excluded_timestamps": cap.excluded_ts,
            "notes": notes, "warnings": warnings_,
        },
        "pairs": cap.n_pairs,
        "pairs_valid_for_sync": n_valid,
        "duration_s": duration,
        "duration_source": duration_source,
        "pair_span_s": pair_span,
        "pair_yield": (cap.n_pairs / expected_pairs) if expected_pairs else None,
        "fps_mean": (cap.n_pairs - 1) / pair_span if pair_span > 0 else None,
        "fps_windows": fps_win,
        "cameras": {c: {k: v for k, v in cams[c].items() if not k.startswith("_")} for c in CAMERAS},
        "missing_pairs": missing_pairs,
        "failed_grabs": info.get("failed_grabs"),
        "failed_grab_reasons": info.get("failed_grab_reasons"),
        "incomplete": incomplete,
        "pairs_saved": info.get("pairs_saved"),
        "pairs_dropped_disk_slow": info.get("pairs_dropped_disk_slow"),
        "suspect": suspect_info,
        "offset_start_us": ({"name": OFFSET_NAME, "basis": BASIS_MEASURED, "caveat": TIMESTAMP_CAVEAT,
                             **describe(off_g)} if offset is not None else
                            {"name": OFFSET_NAME, "basis": "NOT AVAILABLE: cameras not on a shared clock", "n": 0}),
        "residual_us": ({"name": "timestamp offset − planned lag", "basis": BASIS_MEASURED,
                         **describe(res_g)} if residual is not None and np.isfinite(res_g).any()
                        else {"basis": "NOT AVAILABLE", "n": 0}),
        "offset_center_model_us": ({"name": "exposure-centre offset (Triton − Helios)", "basis": centre_basis,
                                    "helios_center_us": helios_center_us, **describe(cen_g)}
                                   if cen_g is not None and np.isfinite(cen_g).any()
                                   else {"basis": "NOT AVAILABLE", "n": 0}),
        "drift": drift,
        "segments": segments,
        "host_delta_ms": {"basis": BASIS_HOST, **describe(host_delta)},
        "host_delta_drift": hd_drift,
        "legacy_estimate": legacy,
        "ptp_monitor": ptp_events["summary"],
    }
    result["headline"] = _headline(result)
    result["verdict"] = _verdict(result)
    result["_series"] = {
        "t_pair_s": t_pair, "offset_us": offset, "residual_us": residual, "centre_us": centre,
        "suspect": suspect, "host_delta_ms": host_delta,
        "est_offset_ms": legacy.pop("_series") if legacy else None,
        "bound_offset_ms": legacy.pop("_bound") if legacy else None,
        "cam_t": {c: cams[c]["_t"] for c in CAMERAS},
        "drop_t": {c: cams[c]["_drop_t"] for c in CAMERAS},
        "ptp_monitor": info.get("ptp_monitor"),
    }
    return result


def _ptp_events(info: dict) -> dict:
    """PTP health during the run: status changes and offset-from-master summary."""
    mon = info.get("ptp_monitor")
    warnings_, summary = [], {}
    if info.get("ptp_settled") is False:
        warnings_.append("PTP servo did NOT report settled before recording (run.json ptp.settled = false).")
    if mon:
        bad = [i for i, (h, t) in enumerate(zip(mon["helios_status"], mon["triton_status"]))
               if {str(h).lower(), str(t).lower()} != {"master", "slave"}]
        changes = []
        prev = None
        for t, h, s in zip(mon["t_s"], mon["helios_status"], mon["triton_status"]):
            if (h, s) != prev:
                changes.append([t, h, s])
                prev = (h, s)
        summary = {"samples": len(mon["t_s"]), "samples_not_master_slave": len(bad),
                   "status_changes": changes}
        offs = np.array([v for v in mon["triton_offset_ns"] if v is not None], dtype=float)
        if offs.size:
            summary["triton_offset_from_master_ns"] = describe(offs)
        if bad:
            warnings_.append(f"PTP monitor: {len(bad)} of {len(mon['t_s'])} samples were NOT Master/Slave "
                             f"(status changes: {changes}).")
    return {"warnings": warnings_, "summary": summary or None}


def _lower_envelope_fit(x: np.ndarray, y: np.ndarray, n_windows: int = 30):
    """Line under a noisy point cloud: y_env = a + b x.

    WHY not least squares: y = host_receive - device_ts = clock offset + drift * x + latency, and
    latency jitter is one-sided (a frame can be late, never early). The minimum of each window is the
    least-delayed frame, so a line through the window minima follows the true clock relation much
    more tightly. The line is finally shifted down so that no point lies below it.
    """
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 10:
        return None
    order = np.argsort(x)
    x, y = x[order], y[order]
    idx = [w[np.argmin(y[w])] for w in np.array_split(np.arange(x.size), min(n_windows, x.size // 3))]
    b, a = np.polyfit(x[idx], y[idx], 1)
    a += float(np.min(y - (a + b * x)))
    return a, b


def _legacy_estimate(cap: Capture, host_delta_ms: np.ndarray, period_s: float | None) -> dict:
    """Best-effort estimate of the camera-to-camera offset when the clocks are independent.

    Without a shared clock the offset of a pair is the PHASE difference of two free-running frame
    trains (freerun) or the latency difference of two host trigger commands (capture.py "trigger").
    1. 'device_ts_lower_envelope' (both cameras' device ts AND host times exist): map each camera's
       device timestamps onto the host clock via the lower envelope of (host - device), then
       offset = mapped(triton) - mapped(helios) - (assumed latency difference).
    2. 'host_delta_minus_latency': offset = host_delta - (assumed latency difference).
    The constant part depends on LEGACY_LATENCY_MS (an assumption); the drift does not.
    """
    p = cap.pairs
    lat_diff = LEGACY_LATENCY_MS["triton"] - LEGACY_LATENCY_MS["helios"]
    have_dev = all(np.isfinite(p[f"{c}_ts_ns"]).sum() >= 10 and np.isfinite(p[f"{c}_host_ns"]).sum() >= 10
                   for c in CAMERAS)
    method = "host_delta_minus_latency"
    est = host_delta_ms - lat_diff
    if have_dev:
        mapped = {}
        for c in CAMERAS:
            dev, host = p[f"{c}_ts_ns"], p[f"{c}_host_ns"]
            env = _lower_envelope_fit(dev, host - dev)
            if env is None:
                break
            a, b = env
            mapped[c] = dev + a + b * dev
        if len(mapped) == 2:
            method = "device_ts_lower_envelope"
            est = (mapped["triton"] - mapped["helios"]) / 1e6 - lat_diff
    period_ms = period_s * 1e3 if period_s else None
    t_min = None
    for key in ("helios_host_ns", "pair_host_ns", "helios_ts_ns"):
        if key in p and np.isfinite(p[key]).sum() >= 3:
            t_min = (p[key] - np.nanmin(p[key])) / 60e9
            break
    fit = linear_fit(t_min, est) if t_min is not None else None
    trigger = (cap.info.get("legacy_sync_mode") == "trigger")
    # capture.py: was the Triton frame already waiting when it was fetched (gap ~ fetch overhead)?
    waiting_frac, reliable, bound = None, True, None
    if cap.source_kind == "capture_py" and np.isfinite(host_delta_ms).any():
        hd = host_delta_ms[np.isfinite(host_delta_ms)]
        waiting = hd < CAPTURE_PY_WAITING_GAP_MS
        waiting_frac = float(waiting.mean())
        reliable = int((~waiting).sum()) >= MIN_PROMPT_FETCHES
        if not reliable and period_s:
            # The Triton frame arrived before the Helios fetch returned, i.e. it is OLDER than the
            # host times suggest, by up to one period: true offset in [est - P, est].
            bound = np.maximum(np.abs(est), np.abs(est - period_s * 1e3))
    caveats = ["Independent camera clocks: the camera-to-camera offset cannot be measured, only "
               "estimated from host receive times. The constant part depends on the assumed transfer "
               "latencies; the variation over time is real."]
    if cap.source_kind == "capture_py":
        caveats.append("capture.py fetched the Triton buffer only after the Helios get_buffer returned and "
                       "stored |gap|, so the Triton host time is max(arrival, Helios return).")
        if waiting_frac is not None and not reliable:
            caveats.append(f"In {100 * waiting_frac:.0f} % of pairs the Triton frame was already waiting (gap < "
                           f"{CAPTURE_PY_WAITING_GAP_MS:g} ms), so its arrival time is unknown: the estimate is only "
                           "an UPPER BOUND of the offset; the true offset lies between estimate − 1 period and "
                           "estimate (worst-case |offset| reported as the bound).")
    return {
        "basis": BASIS_ESTIMATE if reliable else BASIS_BOUND,
        "method": method,
        "reliable": reliable,
        "triton_waiting_fraction": waiting_frac,
        "worst_case_abs_offset_ms": describe(bound) if bound is not None else None,
        "latency_assumed_ms": dict(LEGACY_LATENCY_MS),
        "estimated_offset_ms": describe(est),
        # Phase drift (ms/min, ppm): independent of the latency assumption.
        "drift_ms_per_min": fit["slope"] if fit else None,
        "drift_se_ms_per_min": fit["slope_se"] if fit else None,
        "drift_ppm": fit["slope"] / 60e3 * 1e6 if fit else None,
        "triton_after_helios_median_ms": float(np.nanmedian(host_delta_ms)) if np.isfinite(host_delta_ms).any() else None,
        "pairing": ("software trigger: both cameras fired by two host commands (Triton first); offset = "
                    "command + exposure-start latency difference, no frame-period bound"
                    if trigger else "free-run, next Helios frame then newest Triton frame (NewestOnly)"),
        "bound_latest_frame_pairing_ms": None if trigger else period_ms,
        "bound_nearest_timestamp_pairing_ms": None if (trigger or not period_ms) else period_ms / 2,
        "note": " ".join(caveats),
        "_series": est,
        "_bound": bound,
    }


# ----------------------------------------------------------------------------------------------
# Headline + verdict
# ----------------------------------------------------------------------------------------------
def _fmt_us(v):
    if v is None or not math.isfinite(v):
        return "n/a"
    a = abs(v)
    if a >= 1e3:
        return f"{v / 1e3:.2f} ms"
    return f"{v:.1f} µs" if a >= 10 else f"{v:.2f} µs"


def _hl(key, label, basis, stats, unit, meaning) -> dict:
    to_ms = 1e-3 if unit == "µs" else 1.0
    p95_ms = stats["p95_abs"] * to_ms if stats.get("n") else None
    return {"key": key, "label": label, "basis": basis, "unit": unit, "n": stats.get("n", 0),
            "median": stats.get("median"), "median_abs": stats.get("median_abs"),
            "p95_abs": stats.get("p95_abs"), "max_abs": stats.get("max_abs"),
            "p95_abs_ms": p95_ms,
            "meets_motion_criterion": (p95_ms < MOTION_CRITERION_MS) if p95_ms is not None else None,
            "px_at_motion_p95": pixels_at_motion(p95_ms), "meaning": meaning}


def _headline(r: dict) -> list[dict]:
    """The three numbers side by side: (a) residual, (b) modelled centre offset, (c) offset."""
    m = r["meta"]
    out = []
    scheduled = m["method_used"] in SCHEDULED_METHODS
    if scheduled and r["residual_us"].get("n"):
        out.append(_hl("residual", f"(a) |{OFFSET_NAME} − planned lag ({_fmt_us(m['planned_lag_us'])})|",
                       BASIS_MEASURED, r["residual_us"], "µs",
                       "scheduling precision + clock lock; NOT proof of exposure overlap"))
    if r["offset_center_model_us"].get("n"):
        out.append(_hl("centre", "(b) |exposure-centre offset|", r["offset_center_model_us"]["basis"],
                       r["offset_center_model_us"], "µs",
                       "what matters for motion smear / colour misregistration (model unverified)"))
    if not scheduled:
        if r["offset_start_us"].get("n"):
            out.append(_hl("offset", f"(c) |{OFFSET_NAME}|", BASIS_MEASURED, r["offset_start_us"], "µs",
                           "free-running cameras on one clock: random phase, up to half a period"))
        elif r.get("legacy_estimate") and r["legacy_estimate"].get("worst_case_abs_offset_ms"):
            out.append(_hl("bound", f"(c) |{OFFSET_NAME}| worst case (bound)", BASIS_BOUND,
                           r["legacy_estimate"]["worst_case_abs_offset_ms"], "ms",
                           "independent clocks and unknown Triton arrival: only a bound is possible"))
        elif r.get("legacy_estimate") and r["legacy_estimate"]["estimated_offset_ms"].get("n"):
            out.append(_hl("estimate", f"(c) |{OFFSET_NAME}| (estimate)", BASIS_ESTIMATE,
                           r["legacy_estimate"]["estimated_offset_ms"], "ms",
                           "independent clocks: host-based estimate, constant part assumes latencies"))
    return out


def _verdict(r: dict) -> str:
    """Plain-language paragraph, one sentence per headline number with its own criterion."""
    m = r["meta"]
    parts = []
    for w in m["warnings"]:
        if w.isupper() or "STOPPED" in w or "STARTED LATE" in w or "NO USABLE" in w or "BOTH CAMERAS" in w:
            parts.append("WARNING: " + w)
    crit = f"the {MOTION_CRITERION_MS:g} ms motion criterion"

    def px(h):
        q = h.get("px_at_motion_p95")
        return (f" (≈ {q['triton_px']:.2g} Triton px / {q['helios_px']:.2g} Helios px at "
                f"{MOTION_DEG_PER_S:g} °/s)") if q else ""

    if m["clock_domain"] == "ptp":
        parts.append(f"Both cameras were timestamped on one PTP clock (method {m['method_used']}).")
    for h in r["headline"]:
        u = h["unit"]
        f = _fmt_us if u == "µs" else (lambda v: "n/a" if v is None else f"{v:.2f} ms")
        side = "below" if h["meets_motion_criterion"] else "ABOVE"
        if h["key"] == "residual":
            parts.append(f"(a) MEASURED: 95 % of pairs were within {f(h['p95_abs'])} of the planned Triton "
                         f"lag (max {f(h['max_abs'])}). This proves clock lock and scheduling precision only, "
                         "not exposure overlap: what each camera's timestamp marks is NOT CONFIRMED by LUCID.")
        elif h["key"] == "centre":
            parts.append(f"(b) MODELLED: the exposure centres differ by {f(h['median'])} median, |p95| "
                         f"{f(h['p95_abs'])}{px(h)} — {side} {crit}. This rests on the unverified Helios "
                         f"timing model (helios_center_us = {m['helios_center_us']:.0f} µs); verify with a scope/LED.")
        elif h["key"] == "offset":
            parts.append(f"(c) MEASURED: |{OFFSET_NAME}| median {f(h['median_abs'])}, p95 {f(h['p95_abs'])}"
                         f"{px(h)} — {side} {crit}.")
        elif h["key"] == "bound":
            le = r["legacy_estimate"]
            parts.append(f"(c) BOUND: the cameras were NOT on a shared clock, and in "
                         f"{100 * le['triton_waiting_fraction']:.0f} % of pairs the Triton frame was already "
                         f"waiting when fetched, so not even its arrival time is known. The offset is only bounded: "
                         f"|offset| up to {f(h['max_abs'])} (p95 {f(h['p95_abs'])}){px(h)} — {side} {crit}.")
        elif h["key"] == "estimate":
            le = r["legacy_estimate"]
            parts.append(f"(c) ESTIMATE: the cameras were NOT on a shared clock, so the offset cannot be "
                         f"measured. From host receive times it is estimated at {f(h['median'])} median, |p95| "
                         f"{f(h['p95_abs'])}{px(h)} — {side} {crit}.")
            if le.get("bound_latest_frame_pairing_ms"):
                parts.append(f"With this pairing the offset can be anything up to one frame period "
                             f"({le['bound_latest_frame_pairing_ms']:.0f} ms) and it wanders as the oscillators drift.")
    if not r["headline"]:
        hd = r["host_delta_ms"]
        parts.append("No cross-camera offset could be computed." +
                     (f" Host receive delta median {hd['median']:.1f} ms." if hd.get("n") else ""))
    d = r.get("drift") or {}
    if d:
        parts.append(f"Drift: {d['verdict']}" + (f" ({d['slope_per_min']:+.3f} ± {d['slope_se_per_min']:.3f} µs/min)."
                                                  if d.get("detected") else "."))
    drops = {c: r["cameras"][c].get("drops") for c in CAMERAS}
    y = r.get("pair_yield")
    parts.append(f"{r['pairs']} pairs in {r['duration_s']:.0f} s"
                 + (f" (yield {100 * y:.1f} % of {r['meta']['fps_setting']:g} fps)" if y else "")
                 + f"; dropped frames: Helios {drops['helios']}, Triton {drops['triton']}.")
    if r["suspect"]["n"]:
        parts.append(f"{r['suspect']['n']} suspect pairs were excluded.")
    return " ".join(parts)


# ----------------------------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------------------------
def _plt():
    """Import matplotlib lazily, headless (Agg), with a readable house style."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise SystemExit("matplotlib is needed for plots: pip install matplotlib "
                         "(or run with --no-plots)") from e
    plt.rcParams.update({
        "figure.figsize": (8.0, 4.4), "figure.dpi": 110, "savefig.dpi": 150,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#8a8984", "axes.labelcolor": "#2b2b2a",
        "xtick.color": "#52514e", "ytick.color": "#52514e",
        "axes.grid": True, "grid.color": COLOR_GRID, "grid.linewidth": 0.8,
        "legend.frameon": False,
    })
    return plt


def _title(result: dict, text: str, sub: str = "") -> str:
    t = ("SYNTHETIC DATA — " if result["meta"].get("synthetic") else "") + text
    return t + ("\n" + sub if sub else "")


def _save(plt, fig, outdir, name) -> str:
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, name))
    plt.close(fig)
    return name


def _scatter_with_suspects(ax, t, y, suspect, unit):
    """Good pairs as grey dots, suspect pairs as red x clipped to the good-data range."""
    good = ~suspect & np.isfinite(y)
    ax.plot(t[good], y[good], ".", ms=2.5, color=COLOR_PAIR, alpha=0.6)
    ax.plot([], [], ".", ms=8, color=COLOR_PAIR, label="pair")       # legend proxy (readable size)
    med = float(np.nanmedian(y[good])) if good.any() else math.nan
    if good.any():
        ax.axhline(med, color=COLOR_INK, lw=1.2, ls="--", label=f"median {med:.2f} {unit}")
    bad = suspect & np.isfinite(y)
    if bad.any() and good.any():
        lo, hi = np.percentile(y[good], [0.1, 99.9])
        pad = max(hi - lo, 1e-9) * 0.15
        lo, hi = lo - pad, hi + pad
        ax.set_ylim(lo, hi)
        ax.plot(t[bad], np.clip(y[bad], lo, hi), "x", ms=5, mew=1.2, color=COLOR_SUSPECT,
                label=f"suspect, excluded ({int(bad.sum())}; clipped to axis)")
    return med


def make_plots(result: dict, outdir: str) -> list[str]:
    """Write the PNGs into outdir; returns their file names (for the report)."""
    plt = _plt()
    os.makedirs(outdir, exist_ok=True)
    s = result["_series"]
    t = s["t_pair_s"]
    axis = result["meta"]["time_axis"]
    tlabel = f"time since start [s]  ({axis})"
    out = []
    ptp = s["offset_us"] is not None
    suspect = s["suspect"]
    m = result["meta"]

    # 1+2: residual vs plan (or offset) over time + histogram.
    if ptp:
        res_ok = s["residual_us"] is not None and np.isfinite(s["residual_us"]).any() \
            and m["method_used"] in SCHEDULED_METHODS
        if res_ok:
            y, unit = s["residual_us"], "µs"
            ylab = f"timestamp offset − planned lag ({_fmt_us(m['planned_lag_us'])}) [µs]"
            ttl = ("Timestamp offset vs plan per pair",
                   "MEASURED on the shared PTP clock — scheduling precision, not exposure overlap")
        else:
            y, unit, ylab = s["offset_us"], "µs", "Triton − Helios timestamp offset [µs]"
            ttl = ("Timestamp offset per pair", "MEASURED on the shared PTP clock (what the timestamp marks is unconfirmed)")
    elif s["est_offset_ms"] is not None and np.isfinite(s["est_offset_ms"]).any():
        y, unit, ylab = s["est_offset_ms"], "ms", "estimated Triton − Helios offset [ms]"
        reliable = (result.get("legacy_estimate") or {}).get("reliable", True)
        ttl = ("Estimated camera offset per pair",
               "HOST-BASED ESTIMATE (independent clocks; constant part assumes transfer latencies)" if reliable
               else "HOST-BASED UPPER BOUND — Triton frames were already waiting; true offset in [y − 1 period, y]")
    else:
        y = None
    if y is not None and np.isfinite(y).any():
        fig, ax = plt.subplots()
        _scatter_with_suspects(ax, t, y, suspect if ptp else np.zeros_like(suspect), unit)
        d = result.get("drift") or {}
        if ptp and d:
            ax.plot([], [], " ", label=f"slope {d['slope_per_min']:+.3f} ± {d['slope_se_per_min']:.3f} µs/min"
                                       f" — {d['verdict'].split(' (')[0]}")
        elif not ptp and not (result.get("legacy_estimate") or {}).get("reliable", True):
            P = 1e3 / m["fps_setting"] if m["fps_setting"] else None
            if P:
                ax.fill_between(t, y - P, y, color=COLOR_GRID, alpha=0.8, lw=0,
                                label="possible true offset (y − 1 period … y)")
                ax.set_ylim(np.nanmin(y) - P * 1.1, np.nanmax(y) + P * 0.1)
        elif not ptp and (result.get("legacy_estimate") or {}).get("drift_ms_per_min") is not None:
            le = result["legacy_estimate"]
            ax.plot([], [], " ", label=f"drift {le['drift_ms_per_min']:+.3f} ms/min ({le['drift_ppm']:+.1f} ppm)")
        ax.set(title=_title(result, *ttl), xlabel=tlabel, ylabel=ylab)
        ax.legend(loc="best", fontsize=8)
        out.append(_save(plt, fig, outdir, "offset_over_time.png"))

        fig, ax = plt.subplots()
        yy = y[np.isfinite(y) & (~suspect if ptp else True)]
        med = float(np.median(yy))
        ax.hist(yy, bins=min(80, max(10, int(np.sqrt(yy.size)))), color=COLOR_PAIR,
                edgecolor="white", linewidth=0.5)
        ax.axvline(med, color=COLOR_INK, lw=1.2, ls="--", label=f"median {med:.2f} {unit}")
        ax.set(title=_title(result, "Distribution" + (" (suspect pairs excluded)" if ptp else ""), ttl[1]),
               xlabel=ylab, ylabel="pairs")
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper right")
        out.append(_save(plt, fig, outdir, "offset_histogram.png"))

    # 2b: MODELLED centre offset over time.
    c = s["centre_us"]
    if c is not None and np.isfinite(c).any():
        fig, ax = plt.subplots()
        cm = c / 1e3
        _scatter_with_suspects(ax, t, cm, suspect, "ms")
        ax.axhspan(-MOTION_CRITERION_MS, MOTION_CRITERION_MS, color=COLOR_GRID, alpha=0.6, lw=0,
                   label=f"±{MOTION_CRITERION_MS:g} ms motion criterion")
        ax.set(title=_title(result, "Exposure-centre offset per pair",
                            f"MODELLED — helios_center_us = {m['helios_center_us']:.0f} µs (unverified model)"),
               xlabel=tlabel, ylabel="Triton − Helios exposure-centre offset [ms]")
        lo, hi = ax.get_ylim()
        ax.set_ylim(min(lo, -1.2 * MOTION_CRITERION_MS), max(hi, 1.2 * MOTION_CRITERION_MS))
        ax.legend(loc="best", fontsize=8)
        out.append(_save(plt, fig, outdir, "centre_offset_over_time.png"))

    # 3: fps over time.
    fw10, fw1 = result["fps_windows"].get("10s", {}), result["fps_windows"].get("1s", {})

    def _ext(x, y, w):
        return list(x) + [x[-1] + w], list(y) + [y[-1]]

    if fw1.get("window_start_s"):
        fig, ax = plt.subplots()
        ax.step(*_ext(fw1["window_start_s"], fw1["pairs"], 1.0), where="post", color=COLOR_PAIR, alpha=0.3,
                lw=1, label="pairs, 1 s windows")
        if fw10.get("window_start_s"):
            ax.step(*_ext(fw10["window_start_s"], fw10["pairs"], 10.0), where="post", color=COLOR_PAIR, lw=2,
                    label="pairs, 10 s windows")
            for cam, col in (("helios", COLOR_HELIOS), ("triton", COLOR_TRITON)):
                if fw10.get(cam):
                    ax.step(*_ext(fw10["window_start_s"], fw10[cam], 10.0), where="post", color=col, lw=1.5,
                            label=f"{cam.capitalize()} frames, 10 s windows")
        fs = m["fps_setting"]
        if fs:
            ax.axhline(fs, color=COLOR_INK, lw=1, ls=":", label=f"setting {fs:g} fps")
        ax.set(title=_title(result, "Frame rate over time", f"window counts on the {axis}"),
               xlabel=tlabel, ylabel="frames per second")
        ax.set_ylim(bottom=0)
        ax.legend(loc="lower right", fontsize=8)
        out.append(_save(plt, fig, outdir, "fps_over_time.png"))

    # 4: cumulative drops per camera.
    fig, ax = plt.subplots()
    end = result["duration_s"] or (float(np.nanmax(t)) if np.isfinite(t).any() else 1.0)
    for cam, col in (("helios", COLOR_HELIOS), ("triton", COLOR_TRITON)):
        dt = np.sort(np.asarray(s["drop_t"][cam], dtype=float))
        xs = np.concatenate([[0.0], dt, [max(end, dt[-1] if dt.size else 0.0)]])
        ys = np.concatenate([[0], np.arange(1, dt.size + 1), [dt.size]])
        ax.step(xs, ys, where="post", color=col, lw=2, label=f"{cam.capitalize()}: {dt.size} dropped")
    ax.set(title=_title(result, "Dropped frames (cumulative)",
                        result["cameras"]["helios"].get("drop_method", "")), xlabel=tlabel,
           ylabel="dropped frames so far")
    ax.set_ylim(bottom=0)
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.legend(loc="upper left")
    out.append(_save(plt, fig, outdir, "drops_over_time.png"))

    # 5: host receive delta.
    hd = s["host_delta_ms"]
    if np.isfinite(hd).any():
        fig, ax = plt.subplots()
        _scatter_with_suspects(ax, t, hd, suspect, "ms")
        sub = ("HOST-BASED — NOT the sync metric under PTP (includes network transfer)" if ptp
               else "HOST-BASED — the only cross-camera indicator without a shared clock")
        ax.set(title=_title(result, "Host receive time difference", sub),
               xlabel=tlabel, ylabel="Triton − Helios host receive time [ms]")
        ax.legend(loc="best", fontsize=8)
        out.append(_save(plt, fig, outdir, "host_delta_over_time.png"))

    # 6: PTP monitor.
    mon = s.get("ptp_monitor")
    if mon and any(v is not None for v in mon["triton_offset_ns"] + mon["helios_offset_ns"]):
        fig, ax = plt.subplots()
        tt = np.array([math.nan if v is None else v for v in mon["t_s"]], dtype=float)
        for cam, col in (("helios", COLOR_HELIOS), ("triton", COLOR_TRITON)):
            yv = np.array([math.nan if v is None else v for v in mon[f"{cam}_offset_ns"]], dtype=float)
            if np.isfinite(yv).any():
                ax.plot(tt, yv, "-", marker=".", ms=4, lw=1.2, color=col, label=f"{cam.capitalize()} PtpOffsetFromMaster")
        # shade every monitor interval in which the roles were not exactly one Master + one Slave
        shaded = False
        for k in range(len(mon["t_s"])):
            ok = {str(mon["helios_status"][k]).lower(), str(mon["triton_status"][k]).lower()} == {"master", "slave"}
            if not ok and mon["t_s"][k] is not None:
                t1 = mon["t_s"][k + 1] if k + 1 < len(mon["t_s"]) and mon["t_s"][k + 1] is not None else mon["t_s"][k] + 1
                ax.axvspan(mon["t_s"][k], t1, color=COLOR_SUSPECT, alpha=0.25, lw=0,
                           label=None if shaded else "not Master/Slave (see report timeline)")
                shaded = True
        ax.set_yscale("symlog", linthresh=1000)          # ns-level servo noise AND large steps visible
        ax.set(title=_title(result, "PTP offset from master during the run",
                            "camera-reported servo offset (symlog axis; linear within ±1 µs)"),
               xlabel="time since PTP monitor start [s]", ylabel="offset from master [ns]")
        ax.legend(loc="best", fontsize=8)
        out.append(_save(plt, fig, outdir, "ptp_offset_over_time.png"))
    return out


# ----------------------------------------------------------------------------------------------
# Reports
# ----------------------------------------------------------------------------------------------
def _json_ready(obj):
    if isinstance(obj, dict):
        return {k: _json_ready(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_ready(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        return None if not math.isfinite(float(obj)) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _f(v, fmt="{:.2f}", unit=""):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "n/a"
    return fmt.format(v) + (f" {unit}" if unit else "")


def _md(text) -> str:
    """Escape '|' (as in |offset|) so it does not split a markdown table cell."""
    return str(text).replace("|", "\\|")


def headline_rows(r: dict) -> list[tuple[str, ...]]:
    rows = []
    for h in r["headline"]:
        f = _fmt_us if h["unit"] == "µs" else (lambda v: "n/a" if v is None else f"{v:.2f} ms")
        q = h.get("px_at_motion_p95")
        rows.append((h["label"], f(h["median_abs"]), f(h["p95_abs"]), f(h["max_abs"]),
                     f"{q['triton_px']:.2g} / {q['helios_px']:.2g}" if q else "n/a",
                     "yes" if h["meets_motion_criterion"] else "NO", h["basis"], h["meaning"]))
    return rows


def summary_rows(r: dict) -> list[tuple[str, str, str]]:
    """(metric, value, basis) rows of the detail table (also printed by the CLI)."""
    rows = [("Pairs (valid for sync stats)", f"{r['pairs']} ({_f(r['pairs_valid_for_sync'], '{}')})", "count"),
            ("Run duration", _f(r["duration_s"], "{:.1f}", "s"), r["duration_source"]),
            ("Pair yield", _f(100 * r["pair_yield"] if r["pair_yield"] else None, "{:.1f}", "%"),
             "pairs / (duration × fps setting)"),
            ("Mean pair rate (over the paired span)", _f(r["fps_mean"], "{:.3f}", "fps"), r["meta"]["time_axis"])]
    for c in CAMERAS:
        cr = r["cameras"][c]
        rows.append((f"{c.capitalize()} frame interval (median / jitter SD)",
                     f"{_f(cr.get('interval_median_ms'), '{:.3f}', 'ms')} / {_f(cr.get('jitter_std_us'), '{:.1f}', 'µs')}",
                     cr["interval_basis"]))
        rows.append((f"{c.capitalize()} dropped frames", _f(cr.get("drops"), "{}"),
                     cr.get("drop_error") or cr.get("drop_method", "n/a")))
        if cr.get("frame_id_gaps") is not None:
            rows.append((f"{c.capitalize()} frame_id gaps", str(cr["frame_id_gaps"]),
                         "camera frame counter (with NewestOnly: frames the application skipped)"))
        eng = cr.get("engine") or {}
        if eng.get("received") is not None or eng.get("expected") is not None:
            rows.append((f"{c.capitalize()} frames received / expected (engine)",
                         f"{_f(eng.get('received'), '{}')} / {_f(eng.get('expected'), '{}')}", "run.json"))
    rows.append(("Missing pairs (grid gaps in frames.csv)", _f(r["missing_pairs"], "{}"), "grid index"))
    rows.append(("Failed grabs / incomplete buffers",
                 f"{_f(r['failed_grabs'], '{}')} / {_f(r['incomplete'], '{}')}",
                 json.dumps(r["failed_grab_reasons"]) if r.get("failed_grab_reasons") else "run.json / sync_frames.csv"))
    if r.get("pairs_saved") is not None:
        rows.append(("Pairs saved / dropped (disk slow)",
                     f"{r['pairs_saved']} / {_f(r.get('pairs_dropped_disk_slow'), '{}')}", "run.json result"))
    rows.append(("Suspect pairs excluded", str(r["suspect"]["n"]),
                 json.dumps(r["suspect"]["reasons"]) if r["suspect"]["n"] else "engine flag + host-delta check"))
    if r["meta"].get("action_skipped_late") is not None:
        rows.append(("Action commands skipped (late)", str(r["meta"]["action_skipped_late"]), "run.json"))
    o = r["offset_start_us"]
    if o.get("n"):
        rows += [(f"{OFFSET_NAME}: median", _fmt_us(o["median"]), "MEASURED"),
                 ("  mean ± SD", f"{_fmt_us(o['mean'])} ± {_fmt_us(o['std'])}", "MEASURED"),
                 ("  min / max", f"{_fmt_us(o['min'])} / {_fmt_us(o['max'])}", "MEASURED")]
        rr = r["residual_us"]
        if rr.get("n"):
            rows.append((f"  minus planned lag ({_fmt_us(r['meta']['planned_lag_us'])}): median / |p95| / |max|",
                         f"{_fmt_us(rr['median'])} / {_fmt_us(rr['p95_abs'])} / {_fmt_us(rr['max_abs'])}", "MEASURED"))
        d = r.get("drift") or {}
        if d:
            rows.append(("  drift (linear fit ± SE)",
                         f"{d['slope_per_min']:+.3f} ± {d['slope_se_per_min']:.3f} µs/min over {d['window_s']:.0f} s",
                         f"MEASURED — {d['verdict']}"))
    else:
        rows.append((OFFSET_NAME, "not measurable", "independent clocks"))
    c = r["offset_center_model_us"]
    if c.get("n"):
        rows.append(("Exposure-centre offset: median / |p95|", f"{_fmt_us(c['median'])} / {_fmt_us(c['p95_abs'])}",
                     "MODELLED"))
    if r.get("segments"):
        rows.append(("Segments (PTP re-lock etc.)", json.dumps(r["segments"]), "frames.csv segment_id"))
    le = r.get("legacy_estimate")
    if le and le["estimated_offset_ms"].get("n"):
        e = le["estimated_offset_ms"]
        rows.append(("Estimated offset: median / |p95| / |max|",
                     f"{e['median']:.2f} / {e['p95_abs']:.2f} / {e['max_abs']:.2f} ms",
                     f"HOST-BASED ESTIMATE ({le['method']}, latencies assumed {le['latency_assumed_ms']} ms)"))
        rows.append(("  phase drift between the cameras",
                     f"{_f(le.get('drift_ms_per_min'), '{:+.3f}', 'ms/min')} ({_f(le.get('drift_ppm'), '{:+.1f}', 'ppm')})",
                     "HOST-BASED (independent of the latency assumption)"))
        rows.append(("  pairing", le["pairing"], "capture.py logic"))
        if le.get("triton_waiting_fraction") is not None:
            rows.append(("  Triton frame already waiting when fetched",
                         f"{100 * le['triton_waiting_fraction']:.1f} % of pairs",
                         "estimate reliable" if le["reliable"] else "estimate is only an UPPER BOUND"))
        if le.get("worst_case_abs_offset_ms"):
            w = le["worst_case_abs_offset_ms"]
            rows.append(("  worst-case |offset| bound: median / p95 / max",
                         f"{w['median']:.1f} / {w['p95_abs']:.1f} / {w['max_abs']:.1f} ms", BASIS_BOUND))
        if le.get("bound_latest_frame_pairing_ms"):
            rows.append(("  theoretical bound (latest-frame / nearest pairing)",
                         f"{le['bound_latest_frame_pairing_ms']:.1f} / {le['bound_nearest_timestamp_pairing_ms']:.1f} ms",
                         "1 / ½ frame period"))
    h = r["host_delta_ms"]
    if h.get("n"):
        rows.append(("Host receive delta Triton − Helios: median / |p95| / |max|",
                     f"{h['median']:.2f} / {h['p95_abs']:.2f} / {h['max_abs']:.2f} ms", "HOST-BASED"))
    pm = r.get("ptp_monitor")
    if pm:
        rows.append(("PTP monitor samples (not Master/Slave)",
                     f"{pm['samples']} ({pm['samples_not_master_slave']})", "run.json ptp.monitor"))
        if pm.get("triton_offset_from_master_ns"):
            o2 = pm["triton_offset_from_master_ns"]
            rows.append(("  Triton offset from master: median / |max|",
                         f"{o2['median']:.0f} / {o2['max_abs']:.0f} ns", "camera-reported"))
    return rows


MEASURED_NOTE = f"""\
## What is measured, modelled and host-estimated

| Label | Source | What it shows |
|---|---|---|
| **MEASURED** | `buffer.timestamp_ns` of both cameras on the same PTP clock (`clock_domain = ptp`) | That the clocks are locked and the firmware scheduled both frames as planned. {TIMESTAMP_CAVEAT} Exposure overlap needs a physical test: scope on ExposureActive, LED, or a rotation test. |
| **MODELLED** | MEASURED offset + Triton exposure/2 − `helios_center_us` (centroid of the Helios ToF sub-frames, from a data-sheet model) | The number that matters for motion smear. It is only as good as the Helios timing model; verify with a scope. |
| **HOST-BASED** | PC time when `get_buffer()` returned | Includes GigE transfer (~20–25 ms per frame) and OS jitter. Never the sync metric under PTP; for the old capture.py runs it is the only cross-camera indicator. |
| **PER-CAMERA** | One camera's own device clock | Valid for that camera's frame intervals, fps and drops, even without PTP. Never subtracted across cameras unless both are on PTP; the analyser refuses to. |

Pixel equivalents assume {MOTION_DEG_PER_S:g} °/s rotation, Triton f ≈ {F_TRITON_PX:.0f} px (6 mm lens), Helios f ≈ {F_HELIOS_PX:.0f} px.
"""


def write_report(result: dict, outdir: str, plots: list[str] | None = None) -> dict[str, str]:
    """Write sync_report.md + sync_report.json into outdir."""
    os.makedirs(outdir, exist_ok=True)
    m = result["meta"]
    jpath = os.path.join(outdir, "sync_report.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(_json_ready(result), f, indent=2)

    L = [f"# Sync report{' — SYNTHETIC DATA' if m['synthetic'] else ''}\n",
         f"Capture: `{m['folder']}` — method **{m['method_used']}**, clock domain **{m['clock_domain']}**\n"]
    loud = [w for w in m["warnings"] if any(k in w for k in ("STOPPED", "STARTED LATE", "NO USABLE", "BOTH CAMERAS"))]
    for w in loud:
        L.append(f"> **WARNING: {w}**\n")
    L.append("## Headline\n")
    if result["headline"]:
        L.append("| Quantity | median | p95 | max | px @ "
                 f"{MOTION_DEG_PER_S:g} °/s (Triton / Helios, p95) | < {MOTION_CRITERION_MS:g} ms? | Basis | Meaning |")
        L.append("|---|---|---|---|---|---|---|---|")
        L += ["| " + " | ".join(_md(x) for x in row) + " |" for row in headline_rows(result)]
        L.append("")
    L.append(result["verdict"] + "\n")
    if m["clock_domain"] == "ptp":
        L.append(f"*Note on the timestamp offset:* {TIMESTAMP_CAVEAT}\n")
    L.append("## Run info\n")
    info = [
        ("Folder type", m["source_kind"]),
        ("Method used (requested)", f"{m['method_used']} ({m['method_requested']})"),
        ("Fallback chain", json.dumps(m["fallback_chain"]) if m["fallback_chain"] else "none"),
        ("Clock domain", f"{m['clock_domain']} — {m['clock_domain_source']}"),
        ("Alignment (requested / used)", f"{m['alignment_requested']} / {m['alignment_used']}"),
        ("Planned Triton lag", _fmt_us(m["planned_lag_us"])),
        ("Frame-rate setting / run length", f"{_f(m['fps_setting'], '{:g}', 'fps')} / "
                                            f"{_f(m['seconds_setting'], '{:g}', 's')} (actual {_f(m['seconds_actual'], '{:.1f}', 's')})"),
        ("Stop reason", m.get("stop_reason") or "n/a"),
        ("Helios exposure-centre model", f"{m['helios_center_us']:.0f} µs — {m['helios_center_source']}"),
        ("Triton exposure source", m.get("triton_exposure_source") or "per-frame CSV value / n/a"),
        ("Exposure cap / Triton trigger delay", f"{_f(m['exposure_cap_us'], '{:g}', 'µs')} / "
                                                f"{_f(m['triton_trigger_delay_us'], '{:g}', 'µs')}"),
    ]
    if m.get("ptp"):
        pt = m["ptp"]
        info.append(("PTP", f"Helios {pt.get('helios_status')}, Triton {pt.get('triton_status')}, "
                            f"master {pt.get('master')}, lock {pt.get('lock_seconds')} s, settled {m.get('ptp_settled')}"))
    if result.get("ptp_monitor") and result["ptp_monitor"].get("status_changes"):
        info.append(("PTP status timeline", json.dumps(result["ptp_monitor"]["status_changes"])))
    if m.get("action"):
        info.append(("Action scheduler", json.dumps({k: v for k, v in m["action"].items()
                                                     if not isinstance(v, (list, dict))})))
    for cam, d in (m.get("cameras") or {}).items():
        if isinstance(d, dict):
            info.append((f"{cam.capitalize()} camera", ", ".join(f"{k} {v}" for k, v in d.items() if v)))
    if m.get("sdk"):
        info.append(("SDK", json.dumps(m["sdk"])))
    if m.get("missing_nodes"):
        info.append(("Missing nodes", ", ".join(map(str, m["missing_nodes"]))))
    L.append("| | |\n|---|---|")
    L += [f"| {_md(k)} | {_md(v)} |" for k, v in info]
    L.append("")
    L.append("## Details\n")
    L.append("| Metric | Value | Basis |\n|---|---|---|")
    L += [f"| {_md(a)} | {_md(b)} | {_md(c)} |" for a, b, c in summary_rows(result)]
    L.append("")
    if m["warnings"] or m["notes"]:
        L.append("## Warnings and notes\n")
        L += [f"- **{w}**" for w in m["warnings"]]
        L += [f"- {n}" for n in m["notes"]]
        L.append("")
    if plots:
        L.append("## Plots\n")
        L += [f"![{p}]({p})\n" for p in plots]
    L.append(MEASURED_NOTE)
    mpath = os.path.join(outdir, "sync_report.md")
    with open(mpath, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return {"md": mpath, "json": jpath}


# ----------------------------------------------------------------------------------------------
# Before / after
# ----------------------------------------------------------------------------------------------
def _quantity(result: dict, which: str) -> tuple[np.ndarray | None, str]:
    """|quantity| per pair in ms, and its basis, for the comparison. Suspect pairs excluded.

    which = "offset": timestamp offset - MEASURED if on PTP, else the host-based ESTIMATE.
    which = "centre": MODELLED exposure-centre offset (PTP runs only)."""
    s = result["_series"]
    good = ~s["suspect"]
    if which == "offset":
        if s["offset_us"] is not None:
            return np.abs(s["offset_us"][good]) / 1e3, "MEASURED"
        if s.get("bound_offset_ms") is not None:
            return s["bound_offset_ms"], "HOST-BASED WORST-CASE BOUND"
        if s["est_offset_ms"] is not None and np.isfinite(s["est_offset_ms"]).any():
            return np.abs(s["est_offset_ms"]), "HOST-BASED ESTIMATE"
        return None, "n/a"
    if s["centre_us"] is not None and np.isfinite(s["centre_us"]).any():
        return np.abs(s["centre_us"][good]) / 1e3, "MODELLED"
    return None, "n/a"


def compare(before: dict, after: dict, outdir: str, plots: bool = True) -> dict[str, str]:
    """before_after.md / .json + overlay plots. Only like-for-like quantities are compared:
      * |timestamp offset|: a factor only if BOTH runs are on a PTP clock (MEASURED vs MEASURED) and
        neither has a planned lag (centre alignment); if BEFORE is a host-based estimate the ratio is
        labelled "estimate vs measured" (no bold).
      * |exposure-centre offset|: MODELLED vs MODELLED, only with the same helios_center_us.
    """
    os.makedirs(outdir, exist_ok=True)
    synthetic = before["meta"]["synthetic"] or after["meta"]["synthetic"]
    comps = []
    for which, name in (("offset", f"|{OFFSET_NAME}|"), ("centre", "|exposure-centre offset|")):
        b, bb = _quantity(before, which)
        a, ab = _quantity(after, which)
        b = b[np.isfinite(b)] if b is not None else None
        a = a[np.isfinite(a)] if a is not None else None
        entry = {"quantity": name, "before_basis": bb, "after_basis": ab,
                 "before": describe(b) if b is not None else {"n": 0},
                 "after": describe(a) if a is not None else {"n": 0}, "factor_p95": None,
                 "comparable": False, "_b": b, "_a": a}
        if entry["before"].get("n") and entry["after"].get("n") and entry["after"]["p95_abs"] > 0:
            entry["factor_p95"] = entry["before"]["p95_abs"] / entry["after"]["p95_abs"]
            entry["comparable"] = bb == ab
            why = []
            if bb != ab:
                why.append(f"{bb} vs {ab}: an estimate compared with a measurement — order of magnitude only")
            lags = [abs(x["meta"].get("planned_lag_us") or 0.0) for x in (before, after)]
            if which == "offset" and max(lags) > 1.0:
                why.append(f"a run has a deliberate planned Triton lag ({max(lags) / 1e3:.2f} ms, centre "
                           "alignment), so its timestamp offset is not like for like — compare the "
                           "exposure-centre offset instead")
            if which == "centre" and before["meta"]["helios_center_us"] != after["meta"]["helios_center_us"]:
                why.append(f"different helios_center_us ({before['meta']['helios_center_us']:.0f} vs "
                           f"{after['meta']['helios_center_us']:.0f} µs): Helios mode differs, not like for like")
            if why:
                entry["comparable"] = False
                entry["note"] = "; ".join(why)
        comps.append(entry)

    files = {}
    usable = [c for c in comps if c["_b"] is not None and c["_a"] is not None and c["_b"].size and c["_a"].size]
    if plots and usable:
        plt = _plt()
        fig, axes = plt.subplots(2, len(usable), figsize=(7.0 * len(usable), 7), squeeze=False, sharex="col")
        for j, c in enumerate(usable):
            floor = 1e-4                                   # 0.1 µs: a log axis cannot show 0
            bc, ac = np.maximum(c["_b"], floor), np.maximum(c["_a"], floor)
            lo = 10 ** math.floor(math.log10(min(bc.min(), ac.min())))
            hi = 10 ** math.ceil(math.log10(max(bc.max(), ac.max(), 2 * MOTION_CRITERION_MS)))
            bins = np.logspace(math.log10(lo), math.log10(hi), 80)
            ax1, ax2 = axes[0][j], axes[1][j]
            lab_b = f"BEFORE: {before['meta']['method_used']} — {c['before_basis']}"
            lab_a = f"AFTER: {after['meta']['method_used']} — {c['after_basis']}"
            ax1.hist(bc, bins=bins, color=COLOR_BEFORE, alpha=0.85, label=lab_b)
            ax1.hist(ac, bins=bins, color=COLOR_AFTER, alpha=0.85, label=lab_a)
            ax1.set(xscale="log", ylabel="pairs",
                    title=("SYNTHETIC DATA — " if synthetic else "") + c["quantity"]
                    + ("" if c["comparable"] else "\nNOT like for like — see before_after.md"))
            ax1.grid(axis="x", visible=False)
            ax1.legend(loc="upper center", fontsize=7)
            for x, col, lab in ((bc, COLOR_BEFORE, "BEFORE"), (ac, COLOR_AFTER, "AFTER")):
                xs = np.sort(x)
                ax2.step(xs, np.arange(1, xs.size + 1) / xs.size, where="post", color=col, lw=2, label=lab)
            ax2.axvline(MOTION_CRITERION_MS, color=COLOR_INK, ls=":", lw=1)
            ax2.text(MOTION_CRITERION_MS, 0.03, f" {MOTION_CRITERION_MS:g} ms motion criterion", fontsize=8)
            ax2.set(xscale="log", xlabel="per pair [ms]  (log scale)", ylabel="fraction of pairs ≤ x", ylim=(0, 1.02))
            ax2.legend(loc="center right")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "before_after_abs_offset.png"))
        plt.close(fig)
        files["png"] = os.path.join(outdir, "before_after_abs_offset.png")

    def ms(v):
        return "n/a" if v is None else _fmt_us(v * 1e3)

    def px(v):
        q = pixels_at_motion(v)
        return "n/a" if q is None else f"{q['triton_px']:.2g} / {q['helios_px']:.2g}"

    L = [f"# Before / after synchronisation{' — SYNTHETIC DATA' if synthetic else ''}\n",
         f"- BEFORE: `{before['meta']['folder']}` ({before['meta']['method_used']}, clock domain "
         f"{before['meta']['clock_domain']}, helios_center_us {before['meta']['helios_center_us']:.0f})",
         f"- AFTER: `{after['meta']['folder']}` ({after['meta']['method_used']}, clock domain "
         f"{after['meta']['clock_domain']}, helios_center_us {after['meta']['helios_center_us']:.0f})\n",
         f"Pixel columns: misregistration at {MOTION_DEG_PER_S:g} °/s, Triton px / Helios px "
         f"(f ≈ {F_TRITON_PX:.0f} / {F_HELIOS_PX:.0f} px).\n",
         "| Quantity | Run | Basis | median | p95 | max | px @ p95 (T / H) |",
         "|---|---|---|---|---|---|---|"]
    for c in comps:
        for tag, st, basis in (("BEFORE", c["before"], c["before_basis"]), ("AFTER", c["after"], c["after_basis"])):
            if st.get("n"):
                L.append(f"| {_md(c['quantity'])} | {tag} | {basis} | {ms(st['median'])} | {ms(st['p95_abs'])} | "
                         f"{ms(st['max_abs'])} | {px(st['p95_abs'])} |")
            else:
                L.append(f"| {_md(c['quantity'])} | {tag} | {basis} | n/a | n/a | n/a | n/a |")
    L.append("")
    for c in comps:
        if c["factor_p95"] is None:
            continue
        if c["comparable"]:
            L.append(f"**{_md(c['quantity'])}: the 95th percentile improved by a factor of "
                     f"{c['factor_p95']:,.0f}** ({c['before_basis']} vs {c['after_basis']}).")
        else:
            L.append(f"{_md(c['quantity'])}: p95 ratio {c['factor_p95']:,.0f} — {c.get('note', 'not like for like')}.")
    L.append("")
    L.append("| | BEFORE | AFTER |\n|---|---|---|")
    L.append(f"| Pairs / duration | {before['pairs']} / {before['duration_s']:.0f} s | {after['pairs']} / {after['duration_s']:.0f} s |")
    for c in CAMERAS:
        L.append(f"| {c.capitalize()} dropped frames | {_f(before['cameras'][c].get('drops'), '{}')} | "
                 f"{_f(after['cameras'][c].get('drops'), '{}')} |")
    L.append("")
    L.append("Notes: the timestamp offset proves clock lock and scheduling precision, not exposure overlap "
             "(what each timestamp marks is not confirmed by LUCID). The exposure-centre offset is MODELLED "
             "from an unverified Helios timing model; it is the quantity that matters for motion. A BEFORE "
             "run without PTP has only a host-based ESTIMATE (see its sync_report.md).\n")
    if files.get("png"):
        L.append("![before vs after](before_after_abs_offset.png)\n")
    md = os.path.join(outdir, "before_after.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    files["md"] = md
    with open(os.path.join(outdir, "before_after.json"), "w", encoding="utf-8") as f:
        json.dump(_json_ready({"comparisons": comps, "motion_deg_per_s": MOTION_DEG_PER_S,
                               "f_triton_px": F_TRITON_PX, "f_helios_px": F_HELIOS_PX}), f, indent=2)
    return files


def analyze_folder(folder: str, outdir: str | None = None, mapping=None, clock_domain=None,
                   fps=None, plots: bool = True) -> tuple[dict, dict]:
    """Convenience: load + analyze + plots + report. Returns (result, written files)."""
    cap = load_capture(folder, mapping=mapping, clock_domain=clock_domain, fps=fps)
    result = analyze(cap)
    outdir = outdir or os.path.join(cap.folder, "sync_analysis")
    pngs = make_plots(result, outdir) if plots else []
    files = write_report(result, outdir, pngs)
    files["plots"] = [os.path.join(outdir, p) for p in pngs]
    return result, files


# ----------------------------------------------------------------------------------------------
# Writing captures in the DESIGN.md schema (used by the synthetic generator)
# ----------------------------------------------------------------------------------------------
def write_capture(folder: str, pairs: list[dict], frames: list[dict] | None, run: dict,
                  lead_columns: tuple[str, ...] = (), old_schema: bool = False) -> None:
    """Write frames.csv / sync_frames.csv / run.json with the column names of DESIGN.md.
    lead_columns: original capture.py columns written first (capture_sync.py keeps them).
    old_schema: only the pre-round-1 columns (to test compatibility with early files)."""
    os.makedirs(folder, exist_ok=True)
    if old_schema:
        base_p, extra_pair_columns = DESIGN_V0_PAIR_COLUMNS, ()
        base_f, extra_frame_columns = DESIGN_V0_FRAME_COLUMNS, ()
    else:
        base_p, extra_pair_columns = PAIR_COLUMNS, PAIR_COLUMNS_OPTIONAL
        base_f, extra_frame_columns = FRAME_COLUMNS, FRAME_COLUMNS_OPTIONAL
    pcols = list(lead_columns) + list(base_p) + [c for c in extra_pair_columns if c not in base_p]
    with open(os.path.join(folder, "frames.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=pcols, extrasaction="ignore")
        w.writeheader()
        for r in pairs:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in pcols})
    if frames is not None:
        fcols = list(base_f) + [c for c in extra_frame_columns if c not in base_f]
        with open(os.path.join(folder, "sync_frames.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fcols, extrasaction="ignore")
            w.writeheader()
            for r in frames:
                w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fcols})
    with open(os.path.join(folder, "run.json"), "w", encoding="utf-8") as f:
        json.dump(run, f, indent=2)
