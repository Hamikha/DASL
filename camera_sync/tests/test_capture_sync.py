"""capture_sync.py end to end on the mock, and the user's process.py UNCHANGED on its output."""
from __future__ import annotations

import csv
import glob
import json
import os
import subprocess
import sys
import threading

import numpy as np
import pytest

import capture_sync as cs

PROCESS_PY = "/home/user/dasl/process.py"
ORIGINAL_HEADER = ["frame", "host_time", "hlt_frame_id", "tri_frame_id", "hlt_ts_ns",
                   "tri_ts_ns", "arrival_gap_ms", "grab_ms", "copy_ms", "queue_depth", "queued"]
RUN_KEYS = {"created", "run", "sync_mode", "frame_rate_target", "helios", "triton",
            "calibration", "files", "result", "sync"}
RESULT_KEYS = {"stop_reason", "error", "duration_s", "pairs_captured", "pairs_saved",
               "pairs_dropped_disk_slow", "failed_grabs", "mean_fps", "mean_grab_ms",
               "mean_arrival_gap_ms", "max_arrival_gap_ms", "disk_write_error"}


def write_calibration(path):
    cv2 = pytest.importorskip("cv2")
    fs = cv2.FileStorage(str(path), cv2.FileStorage_WRITE)
    fs.write("cameraMatrix", np.array([[1739.0, 0, 1024], [0, 1739.0, 768], [0, 0, 1]]))
    fs.write("distCoeffs", np.zeros((1, 5)))
    fs.write("rotationVector", np.zeros((3, 1)))
    fs.write("translationVector", np.array([[-50.0], [0.0], [0.0]]))
    fs.release()


@pytest.fixture
def configured(monkeypatch, tmp_path):
    calib = tmp_path / "orientation.yml"
    write_calibration(calib)
    for name, value in dict(OUTPUT_ROOT=str(tmp_path / "captures"), CALIB_FILE=str(calib),
                            WARMUP_SECONDS=0.5, MIN_FREE_GB_START=0.1, MIN_FREE_GB_STOP=0.01,
                            ADVANCED=dict(PTP_POLL_S=0.05, PTP_MIN_SETTLE_S=0.2,
                                          PTP_MONITOR_S=0.3)).items():
        monkeypatch.setattr(cs, name, value)
    return tmp_path


def run_dir(tmp_path):
    (d,) = glob.glob(str(tmp_path / "captures" / "run_*"))
    return d


def test_layout_and_stop_file(mock, configured, monkeypatch, capsys):
    monkeypatch.setattr(cs, "SECONDS", 0)                         # no time limit
    timer = threading.Timer(3.0, lambda: open(os.path.join(run_dir(configured), "STOP"),
                                              "w").close())
    timer.start()
    assert cs.main() == 0
    timer.cancel()
    d = run_dir(configured)
    info = json.load(open(os.path.join(d, "run.json")))
    assert RUN_KEYS <= set(info) and RESULT_KEYS <= set(info["result"])
    assert info["result"]["stop_reason"] == "STOP file"
    assert info["sync_mode"] == info["sync"]["method_used"] == "ptpsync"
    assert info["helios"]["scale"] == 0.25 and info["helios"]["offsets"] == [-8192.0, -8192.0, 0.0]
    assert info["helios"]["operating_mode"] == "Distance5000mmMultiFreq"
    assert info["helios"]["exposure"] == "Exp250Us" and info["helios"]["throughput"] == 40_000_000
    assert info["sync"]["helios_center_us"] == pytest.approx(6405.0)
    rows = list(csv.reader(open(os.path.join(d, "frames.csv"))))
    assert rows[0][:11] == ORIGINAL_HEADER
    assert rows[0][11:13] == ["pair_index", "grid_k"]
    saved = info["result"]["pairs_saved"]
    assert saved == len(rows) - 1 >= 20
    assert [int(r[0]) for r in rows[1:]] == list(range(saved))
    assert len(glob.glob(os.path.join(d, "raw", "helios_*.npy"))) == saved
    assert np.load(os.path.join(d, "raw", "helios_000000.npy")).shape == (120, 160, 4)
    assert os.path.exists(os.path.join(d, "sync_frames.csv"))
    assert os.path.exists(os.path.join(d, "orientation.yml"))
    out = capsys.readouterr().out
    assert "Triton arrived after Helios by" in out and "Residual vs plan" in out
    assert info["sync"]["restore"]["failed"] == []


@pytest.mark.skipif(not os.path.exists(PROCESS_PY), reason="user's process.py not available")
def test_full_frames_then_users_process_py_unchanged(mock, configured, monkeypatch, capsys):
    pytest.importorskip("cv2")
    mock(small_frames=False)
    monkeypatch.setattr(cs, "SECONDS", 1.5)
    assert cs.main() == 0
    summary = capsys.readouterr().out
    d = run_dir(configured)
    info = json.load(open(os.path.join(d, "run.json")))
    saved = info["result"]["pairs_saved"]
    assert saved >= 12
    assert np.load(os.path.join(d, "raw", "helios_000000.npy")).shape == (480, 640, 4)
    assert np.load(os.path.join(d, "raw", "triton_000000.npy")).shape == (1536, 2048)
    proc = subprocess.run([sys.executable, PROCESS_PY, d], capture_output=True, text=True,
                          timeout=600)
    print(summary)
    print(proc.stdout[-3000:], proc.stderr[-3000:])
    assert proc.returncode == 0, proc.stderr
    report = json.load(open(os.path.join(d, "process_report.json")))
    assert report["frames_processed"] == saved and not report["frames_skipped"]
    assert report["rgbd_camera"]["lens_fit_rms_px"] < 0.5
    assert len(glob.glob(os.path.join(d, "ply", "frame_*.ply"))) == saved
    for f in ("rgb.txt", "depth.txt", "associations.txt", "camera.yaml"):
        assert os.path.exists(os.path.join(d, "tum", f))
    assert len(glob.glob(os.path.join(d, "tum", "depth", "*.png"))) == saved
