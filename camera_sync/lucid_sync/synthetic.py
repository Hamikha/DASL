"""Synthetic capture folders for tests and for `sync_test.py demo`.

Every folder written here is marked {"synthetic": true} in run.json, and all plots made from it
carry "SYNTHETIC DATA" in the title, so a synthetic figure can never be mistaken for a measurement.

Generators:
  make_ptp_capture         what capture_sync.py should produce (DESIGN.md schema + round-1
                           columns): ptpsync / action (start- or centre-aligned) or freerun_ptp,
                           with optional faults (PTP step, dead camera, bad timestamps)
  make_capture_py_capture  EXACTLY what the user's original capture.py writes (sync_mode
                           "freerun" or "trigger"; independent clocks)
  make_legacy_capture      other old layouts with guessed / opaque column names
"""
from __future__ import annotations

import csv
import json
import os
import time

import numpy as np

from .analysis import CAPTURE_PY_COLUMNS, helios_center_us_model, write_capture

PTP_BASE_NS = 1_761_480_000_123_456_789   # ~Oct 2025 in PTP ns: big on purpose (float64 trap)
HOST_BASE_NS = 5_123_456_789_012           # perf_counter_ns-like origin
EPOCH0 = 1_761_480_000.25                  # time.time()-like origin
BOOKKEEPING_US = 17.7                      # LUCID: Triton timestamp latches ~18 us after Helios'
CAPTURE_PY_ALL_COLUMNS = list(CAPTURE_PY_COLUMNS) + ["grab_ms", "copy_ms", "queue_depth", "queued"]


def _capture_py_run_json(sync_mode: str, fps: float, helios_mode=("Distance6000mmSingleFreq", "Exp250Us"),
                         duration_s=None, pairs=0, failed=None) -> dict:
    """run.json keys of the original capture.py (main(): info + result)."""
    return {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "run": "synthetic_run",
        "sync_mode": sync_mode, "frame_rate_target": fps,
        "helios": {"serial": "SYNTH-H", "model": "HTP003S-001", "width": 640, "height": 480,
                   "pixel_format": "Coord3D_ABCY16", "scale": 0.25, "offsets": [0, 0, 0],
                   "operating_mode": helios_mode[0], "exposure": helios_mode[1],
                   "conversion_gain": "High", "throughput": 48000000},
        "triton": {"serial": "SYNTH-T", "model": "TRI032S-C", "width": 2048, "height": 1536,
                   "pixel_format": "BayerRG8", "throughput": 60000000},
        "calibration": None,
        "files": {"helios": "raw/helios_{:06d}.npy", "triton": "raw/triton_{:06d}.npy"},
        "result": {"stop_reason": "time limit", "error": None, "duration_s": duration_s,
                   "pairs_captured": pairs, "pairs_saved": pairs, "pairs_dropped_disk_slow": 0,
                   "failed_grabs": failed or {}, "mean_fps": round(pairs / duration_s, 2) if duration_s else 0,
                   "mean_grab_ms": None, "mean_arrival_gap_ms": None, "max_arrival_gap_ms": None,
                   "disk_write_error": None},
        "synthetic": True,
    }


def make_ptp_capture(folder: str, n_pairs: int = 3000, fps: float = 10.0, method: str = "ptpsync",
                     alignment: str = "start", triton_exposure_us: float = 5000.0,
                     helios_mode=("Distance5000mmMultiFreq", "Exp250Us"),
                     bookkeeping_us: float = BOOKKEEPING_US, offset_jitter_us: float = 0.35,
                     helios_drop_k=(500, 1800), triton_drop_k=(2400,), drift_us_per_min: float = 0.0,
                     ptp_step_at_k: int | None = None, engine_flags_suspect: bool = True,
                     triton_dies_at_k: int | None = None, bad_ts: str | None = None,
                     freerun_phase_ms: float = 23.0, freerun_rel_ppm: float = 5.0,
                     new_columns: bool = True, lead_columns: bool = True, seed: int = 1) -> dict:
    """A PTP run in the new schema.

    method "ptpsync"/"action": exposures on a shared grid; Triton lagged by the planned lag
        (0 for alignment "start", helios_center - exposure/2 for "center"), plus LUCID's ~18 us
        bookkeeping and sub-µs jitter.
    method "freerun_ptp": both free-run on one PTP clock; phase = freerun_phase_ms drifting at
        freerun_rel_ppm; nearest-timestamp pairing; no grid index.
    Faults: ptp_step_at_k (Triton clock steps +1 period from k on: frames paired one period apart
    with perfect-looking timestamps), triton_dies_at_k, bad_ts "zero" (ts=0 on an incomplete frame
    and on one pair) or "jump" (one Helios timestamp +1e15 ns).
    """
    rng = np.random.default_rng(seed)
    period_ns = int(round(1e9 / fps))
    hc = helios_center_us_model(*helios_mode)
    planned = (hc - triton_exposure_us / 2) if (alignment == "center" and method != "freerun_ptp") else 0.0
    grid = method != "freerun_ptp"
    hdrop, tdrop = set(helios_drop_k), set(triton_drop_k)
    n_k = n_pairs + len(hdrop | tdrop)
    pairs, frames = [], []
    pi = 0
    for k in range(n_k):
        t_rel = k * period_ns
        h_ts = PTP_BASE_NS + t_rel + 23_000 + int(rng.normal(0, 50))
        if grid:
            off = planned + bookkeeping_us + float(np.clip(rng.normal(0, offset_jitter_us), -1, 1)) \
                + drift_us_per_min * t_rel / 60e9
        else:
            off = (freerun_phase_ms * 1e3 + freerun_rel_ppm * 1e-6 * t_rel / 1e3 + period_ns / 2e3) \
                % (period_ns / 1e3) - period_ns / 2e3
        t_phys = h_ts + int(round(off * 1000))                 # physical Triton time on the PTP clock
        stepped = ptp_step_at_k is not None and k >= ptp_step_at_k
        t_ts = t_phys + (period_ns if stepped else 0)          # what the stepped clock reports
        h_host = HOST_BASE_NS + t_rel + 20_000_000 + int(rng.exponential(400_000))
        t_host = HOST_BASE_NS + (t_phys - PTP_BASE_NS - 23_000) + 27_000_000 + int(rng.exponential(900_000))
        have_h = k not in hdrop
        have_t = k not in tdrop and (triton_dies_at_k is None or k < triton_dies_at_k)
        frames_k = {"helios": (have_h, h_ts, h_host, k), "triton": (have_t, t_ts, t_host, k + (1 if stepped else 0))}
        for cam, (have, ts, host, kk) in frames_k.items():
            if have:
                frames.append({"camera": cam, "frame_id": k + 1, "ts_ns": ts, "host_ns": host,
                               "grid_k": kk if grid else "", "incomplete": 0, "paired": 0,
                               "pair_index": "", "exposure_us": triton_exposure_us if cam == "triton" else "",
                               "note": "", "segment_id": 1 if stepped else 0, "_k": kk})
    # pairing as the engine would do it: same grid k (stepped Triton frames pair one period off)
    by = {c: {f["_k"]: f for f in frames if f["camera"] == c} for c in ("helios", "triton")}
    for k in sorted(set(by["helios"]) & set(by["triton"])):
        h, t = by["helios"][k], by["triton"][k]
        h["paired"] = t["paired"] = 1
        h["pair_index"] = t["pair_index"] = pi
        stepped = ptp_step_at_k is not None and k >= ptp_step_at_k
        mismatch = stepped and k > ptp_step_at_k              # physically different instants
        off_us = (t["ts_ns"] - h["ts_ns"]) / 1000
        pairs.append({
            "frame": pi, "host_time": f"{EPOCH0 + (h['host_ns'] - HOST_BASE_NS) / 1e9:.6f}",
            "hlt_frame_id": h["frame_id"], "tri_frame_id": t["frame_id"], "hlt_ts_ns": h["ts_ns"],
            "tri_ts_ns": t["ts_ns"], "arrival_gap_ms": f"{abs(t['host_ns'] - h['host_ns']) / 1e6:.1f}",
            "grab_ms": "30.0", "copy_ms": "3.0", "queue_depth": 0, "queued": 1,
            "pair_index": pi, "grid_k": k if grid else "", "helios_frame_id": h["frame_id"],
            "triton_frame_id": t["frame_id"], "helios_ts_ns": h["ts_ns"], "triton_ts_ns": t["ts_ns"],
            "offset_start_us": f"{off_us:.3f}", "triton_exposure_us": triton_exposure_us,
            "offset_center_model_us": f"{off_us + triton_exposure_us / 2 - hc:.3f}",
            "helios_host_ns": h["host_ns"], "triton_host_ns": t["host_ns"],
            "host_delta_ms": f"{(t['host_ns'] - h['host_ns']) / 1e6:.4f}", "clock_domain": "ptp",
            "planned_lag_us": f"{planned:.3f}" if grid else "",
            "residual_us": f"{off_us - planned:.3f}" if grid else "",
            "suspect": int(mismatch and engine_flags_suspect),
            "suspect_reason": "host_delta_jump" if (mismatch and engine_flags_suspect) else "",
            "segment_id": h["segment_id"],
        })
        pi += 1
    if not grid:          # freerun_ptp: nearest-timestamp pairing done above by k is equivalent here
        for p in pairs:
            p["grid_k"] = ""

    if bad_ts == "zero":
        frames.append({"camera": "triton", "frame_id": 99999, "ts_ns": 0, "host_ns": HOST_BASE_NS + 5_000_000,
                       "grid_k": "", "incomplete": 1, "paired": 0, "pair_index": "", "exposure_us": "",
                       "note": "invalid timestamp (incomplete buffer)", "segment_id": 0})
        pairs[10]["helios_ts_ns"] = 0
        pairs[10]["offset_start_us"] = ""
    elif bad_ts == "jump":
        j = 1234
        pairs[j]["helios_ts_ns"] += 10 ** 15
        for f in frames:
            if f["camera"] == "helios" and f["pair_index"] == j:
                f["ts_ns"] += 10 ** 15
                f["grid_k"] = (f["grid_k"] or 0) + 10 ** 7

    for f in frames:
        f.pop("_k", None)
    seconds = n_k / fps
    received = {c: sum(1 for f in frames if f["camera"] == c and not f["incomplete"]) for c in ("helios", "triton")}
    monitor = []
    for s in range(int(seconds) + 1):
        lost = ptp_step_at_k is not None and int(ptp_step_at_k / fps) == s
        monitor.append({"t_s": float(s), "helios_status": "Master",
                        "triton_status": "Uncalibrated" if lost else "Slave",
                        "triton_offset_ns": float(rng.normal(0, 30)) if not lost else 2.5e8})
    sync = {
        "synthetic": True, "method_requested": method, "method_used": method, "fallback_chain": [],
        "clock_domain": "ptp", "frame_rate": fps, "seconds": seconds, "seconds_actual": seconds,
        "exposure_cap_us": 8000.0, "helios_center_us": hc,
        "triton_trigger_delay_us": planned if method == "action" else 0,
        "ptp": {"helios_status": "Master", "triton_status": "Slave", "master": "helios",
                "lock_seconds": 12.0, "offset_from_master_ns_samples": [35, -20, 12],
                "status_changes": [[0.0, "Initializing", "Initializing"], [12.0, "Master", "Slave"]]},
        "cameras": {"helios": {"model": "HTP003S-001", "serial": "SYNTH-H", "firmware": "1.16.0.0"},
                    "triton": {"model": "TRI032S-C", "serial": "SYNTH-T", "firmware": "1.x"}},
        "missing_nodes": [], "failed_node_writes": [],
        "stats": {"pairs": len(pairs), "failed_grabs": 0, "incomplete": 1 if bad_ts == "zero" else 0},
        "sdk": {"arena_api_version": "synthetic", "mock": True},
    }
    if new_columns:
        sync.update({
            "alignment_requested": alignment, "alignment_used": alignment if grid else None,
            "planned_lag_us": planned if grid else None, "triton_exposure_source": "chunk",
        })
        sync["ptp"].update({"settled": True, "monitor": monitor})
        sync["stats"].update({
            "rejected_suspect": sum(p["suspect"] for p in pairs),
            "per_camera": {c: {"frames_received": received[c], "frames_expected": n_k} for c in ("helios", "triton")},
        })
        if method == "action":
            sync["action_skipped_late"] = 0
            sync["action"] = {"fired": n_k, "skipped_late": 0, "errors": 0}
    run = _capture_py_run_json(method, fps, helios_mode, duration_s=seconds, pairs=len(pairs))
    run.pop("sync_mode")
    run["sync"] = sync
    if new_columns:
        write_capture(folder, pairs, frames, run,
                      lead_columns=tuple(CAPTURE_PY_ALL_COLUMNS) if lead_columns else ())
    else:   # pre-round-1 schema: only the DESIGN.md columns
        write_capture(folder, pairs, frames, run, old_schema=True)
    return {"pairs": len(pairs), "helios_drops": len(hdrop), "triton_drops": len(tdrop),
            "period_ns": period_ns, "planned_lag_us": planned, "helios_center_us": hc, "n_k": n_k}


def make_capture_py_capture(folder: str, sync_mode: str = "freerun", n_pairs: int = 600, fps: float = 10.0,
                            rel_ppm: float = 20.0, phase0_ms: float = 30.0, seed: int = 3) -> dict:
    """A folder EXACTLY as the user's original capture.py writes it (no PTP, independent clocks).

    freerun: both cameras free-run at fps (Triton oscillator rel_ppm slower); StreamBufferHandling
      NewestOnly; each loop takes the next Helios frame, then the newest Triton frame (waiting for a
      new one if none arrived since the last pair) -> pairs up to ~1 period apart.
    trigger: per pair the host fires Triton then Helios by software trigger ~0.3 ms apart.
    Host times: time.time() when get_buffer returned; Triton fetched only after the Helios buffer.
    """
    rng = np.random.default_rng(seed)
    T = 1.0 / fps
    a_h, a_t = +rel_ppm / 2 * 1e-6, -rel_ppm / 2 * 1e-6
    lat_h, lat_t = 0.020, 0.025
    rows, true_off = [], []
    if sync_mode == "freerun":
        nt = n_pairs + 5
        true_h = np.arange(n_pairs + 2) * T / (1 + a_h)
        true_t = phase0_ms / 1e3 - T + np.arange(nt + 2) * T / (1 + a_t)
        arr_t = true_t + lat_t + rng.exponential(0.0012, true_t.size)
        last_j = -1
        for i in range(n_pairs):
            h_rx = true_h[i] + lat_h + rng.exponential(0.0008)
            avail = np.flatnonzero((arr_t <= h_rx) & (np.arange(arr_t.size) > last_j))
            if avail.size:
                j = int(avail[-1])                         # newest already waiting
                t_rx = h_rx + 0.0002
            else:
                j = last_j + 1                             # wait for the next one
                t_rx = max(arr_t[j], h_rx + 0.0002)
            last_j = j
            rows.append((i, h_rx, t_rx, i + 1, j + 1, true_h[i] * (1 + a_h), (true_t[j] - true_t[0]) * (1 + a_t)))
            true_off.append((true_t[j] - true_h[i]) * 1e3)
    else:
        for i in range(n_pairs):
            fire = i * T
            t_true = fire + 0.00005
            h_true = fire + 0.0003 + 0.00005 + rng.normal(0, 0.00003)
            h_rx = h_true + lat_h + rng.exponential(0.0008)
            t_rx = max(t_true + lat_t + rng.exponential(0.0012), h_rx + 0.0002)
            rows.append((i, h_rx, t_rx, i + 1, i + 1, h_true * (1 + a_h), t_true * (1 + a_t)))
            true_off.append((t_true - h_true) * 1e3)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "frames.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CAPTURE_PY_ALL_COLUMNS)
        for i, h_rx, t_rx, hid, tid, h_dev, t_dev in rows:
            w.writerow([i, f"{EPOCH0 + h_rx:.6f}", hid, tid, 3_600_000_000_000 + int(round(h_dev * 1e9)),
                        900_000_000_000 + int(round(t_dev * 1e9)), f"{abs(t_rx - h_rx) * 1000:.1f}",
                        "30.0", "3.0", 0, 1])
    dur = rows[-1][1] - rows[0][1] + T
    with open(os.path.join(folder, "run.json"), "w") as f:
        json.dump(_capture_py_run_json(sync_mode, fps, duration_s=round(dur, 2), pairs=n_pairs,
                                       failed={"triton incomplete": 2}), f, indent=2)
    to = np.array(true_off)
    return {"rows": n_pairs, "true_offset_ms": to, "true_offset_median_ms": float(np.median(to))}


def make_legacy_capture(folder: str, n_frames: int = 3000, fps: float = 10.0, rel_ppm: float = 20.0,
                        phase0_ms: float = 30.0, style: str = "host_only", skip_rows=(1000,),
                        seed: int = 2, write_run_json: bool = True) -> dict:
    """Other old free-run layouts: independent clocks, Triton phase drifting rel_ppm against Helios.

    Row k pairs Helios frame k with Triton frame k; skip_rows = loop hiccups (1 drop per camera).
    style "host_only" (frame_idx, helios_host_time, triton_host_time), "guessed_names"
    (frame_no, depth_timestamp, color_timestamp, depth_recv_time, color_recv_time, ...), or
    "opaque" (a..g, needs --map).
    """
    rng = np.random.default_rng(seed)
    T = 1.0 / fps
    a_h, a_t = +rel_ppm / 2 * 1e-6, -rel_ppm / 2 * 1e-6
    k = np.arange(n_frames)
    true_h = k * T / (1 + a_h)
    true_t = phase0_ms / 1e3 + k * T / (1 + a_t)
    dev_h = 3_600_123_456_789 + np.round(true_h * (1 + a_h) * 1e9).astype(np.int64)
    dev_t = 912_345_678_901 + np.round((true_t - phase0_ms / 1e3) * (1 + a_t) * 1e9).astype(np.int64)
    host_h = EPOCH0 + true_h + 0.020 + rng.exponential(0.0008, n_frames)
    host_t = EPOCH0 + true_t + 0.025 + rng.exponential(0.0012, n_frames)
    keep = np.ones(n_frames, bool)
    keep[list(skip_rows)] = False
    os.makedirs(folder, exist_ok=True)
    if style == "host_only":
        header = ["frame_idx", "helios_host_time", "triton_host_time"]
        rows = [[i, f"{host_h[i]:.6f}", f"{host_t[i]:.6f}"] for i in k[keep]]
    else:
        header = (["frame_no", "depth_timestamp", "color_timestamp", "depth_recv_time",
                   "color_recv_time", "depth_file", "color_file"] if style == "guessed_names"
                  else ["a", "b", "c", "d", "e", "f", "g"])
        rows = [[i, dev_h[i], dev_t[i], f"{host_h[i]:.6f}", f"{host_t[i]:.6f}",
                 f"depth/{i:06d}.npy", f"color/{i:06d}.png"] for i in k[keep]]
    with open(os.path.join(folder, "frames.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    if write_run_json:
        with open(os.path.join(folder, "run.json"), "w") as f:
            json.dump({"synthetic": True, "fps": fps, "note": "synthetic legacy free-run capture"}, f)
    true_off_ms = (true_t - true_h)[keep] * 1e3
    return {"rows": int(keep.sum()), "true_offset_ms": true_off_ms,
            "true_offset_median_ms": float(np.median(true_off_ms)),
            "drift_ms_per_min": rel_ppm * 1e-6 * 60 * 1e3}
