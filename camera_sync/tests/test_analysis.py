"""Tests of lucid_sync/analysis.py and sync_test.py on synthetic captures (no camera, no mock SDK).

Run:  python -m pytest camera_sync/tests/test_analysis.py -q
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import time
import types

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lucid_sync import analysis as A  # noqa: E402
from lucid_sync.synthetic import (make_capture_py_capture, make_legacy_capture,  # noqa: E402
                                  make_ptp_capture)
import sync_test  # noqa: E402

MULTI250 = A.helios_center_us_model("Distance5000mmMultiFreq", "Exp250Us")


@pytest.fixture(scope="module")
def ptp_folder(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("ptp"))
    info = make_ptp_capture(d)            # start-aligned ptpsync, 3000 pairs, 2+1 drops
    return d, info


@pytest.fixture(scope="module")
def ptp_result(ptp_folder):
    return A.analyze(A.load_capture(ptp_folder[0]))


@pytest.fixture(scope="module")
def center_result(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("center"))
    info = make_ptp_capture(d, alignment="center")
    return A.analyze(A.load_capture(d)), info


def _rewrite_csv(src, dst, fn):
    with open(src) as f:
        rows = list(csv.DictReader(f))
    fn(rows)
    with open(dst, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------- helpers / models
def test_helios_center_model():
    assert A.helios_center_us_model("Distance6000mmSingleFreq", "Exp250Us") == pytest.approx(2815, abs=1)
    assert MULTI250 == pytest.approx(6405, abs=1)
    assert A.helios_center_us_model("Distance5000mmMultiFreq", "Exp1000Us") == pytest.approx(9345, abs=1)
    assert A.helios_center_us_model(None, "Exp250Us") is None


def test_device_offset_int_exact_and_refusal():
    h, t = 1_761_480_000_123_456_789, 1_761_480_000_123_474_489          # 17.7 µs apart
    assert A.device_offset_us(np.array([h]), np.array([t]), "ptp")[0] == 17.7
    assert A.device_offset_us([h], [t], "ptp")[0] == 17.7                 # python ints
    with pytest.raises(A.ClockDomainError):
        A.device_offset_us([0, 100], [17_700, 100_017_700], clock_domain="independent")
    with pytest.raises(A.ClockDomainError):
        A.device_offset_us([0], [1], clock_domain="unknown")


# ---------------------------------------------------------------- (i) PTP, start-aligned
def test_ptp_basic_numbers(ptp_result):
    r = ptp_result
    assert r["meta"]["clock_domain"] == "ptp" and r["meta"]["source_kind"] == "new"
    assert r["pairs"] == 3000 and r["pairs_valid_for_sync"] == 3000
    assert r["duration_s"] == pytest.approx(300.3, abs=0.05)
    assert r["duration_source"] == "run.json elapsed time"
    assert r["pair_yield"] == pytest.approx(3000 / 3003, abs=1e-3)
    o = r["offset_start_us"]
    assert o["name"] == A.OFFSET_NAME and o["basis"] == A.BASIS_MEASURED
    assert o["median"] == pytest.approx(17.7, abs=0.05)
    assert 0.2 < o["std"] < 0.5 and o["max_abs"] <= 18.7 + 1e-6
    rr = r["residual_us"]                                   # plan = 0 for start alignment
    assert rr["median"] == pytest.approx(17.7, abs=0.05)


def test_ptp_drops_exact_from_grid(ptp_result):
    cams = ptp_result["cameras"]
    assert cams["helios"]["drops"] == 2 and cams["triton"]["drops"] == 1
    assert "grid index" in cams["helios"]["drop_method"]
    assert ptp_result["missing_pairs"] == 3
    assert cams["helios"]["received"] == 3001 and cams["triton"]["received"] == 3002
    assert cams["helios"]["engine"]["expected"] == 3003


def test_start_alignment_headline_is_honest(ptp_result):
    """Start-aligned: residual tiny, but the MODELLED centre offset is ~ -3.9 ms -> must say ABOVE."""
    keys = [h["key"] for h in ptp_result["headline"]]
    assert keys == ["residual", "centre"]
    c = ptp_result["offset_center_model_us"]
    assert c["median"] == pytest.approx(17.7 + 5000 / 2 - MULTI250, abs=0.2)
    centre = ptp_result["headline"][1]
    assert centre["meets_motion_criterion"] is False
    assert centre["px_at_motion_p95"]["triton_px"] > 1
    v = ptp_result["verdict"]
    assert "ABOVE the 1 ms motion criterion" in v
    assert "start-of-exposure" not in v and "well below" not in v
    assert "NOT CONFIRMED" in v


def test_center_alignment(center_result):
    r, info = center_result
    assert info["planned_lag_us"] == pytest.approx(MULTI250 - 2500)
    assert r["meta"]["planned_lag_us"] == pytest.approx(MULTI250 - 2500, abs=1e-3)
    assert r["meta"]["alignment_used"] == "center"
    assert r["residual_us"]["median"] == pytest.approx(17.7, abs=0.05)
    assert r["offset_center_model_us"]["median"] == pytest.approx(17.7, abs=0.1)
    assert all(h["meets_motion_criterion"] for h in r["headline"])
    assert r["meta"]["helios_center_source"] == "run.json sync.helios_center_us"


def test_ptp_fps_windows(ptp_result):
    vals = np.array(ptp_result["fps_windows"]["1s"]["pairs"])
    assert len(vals) == 300
    assert (vals == 10).sum() == 297 and (vals == 9).sum() == 3        # no bin-edge aliasing


def test_ptp_offsets_exact_despite_huge_timestamps(ptp_folder):
    off = A.analyze(A.load_capture(ptp_folder[0]))["_series"]["offset_us"]
    with open(os.path.join(ptp_folder[0], "frames.csv")) as f:
        exact = np.array([(int(r["triton_ts_ns"]) - int(r["helios_ts_ns"])) / 1000 for r in csv.DictReader(f)])
    assert np.max(np.abs(off - exact)) < 1e-6


def test_drift_detection_rules(ptp_result, tmp_path):
    d = ptp_result["drift"]
    assert d["detected"] is False and "within noise" in d["verdict"]
    assert d["slope_se_per_min"] > 0
    make_ptp_capture(str(tmp_path / "drift"), drift_us_per_min=2.0, seed=5)
    d2 = A.analyze(A.load_capture(str(tmp_path / "drift")))["drift"]
    assert d2["detected"] is True
    assert d2["slope_per_min"] == pytest.approx(2.0, abs=0.03)
    assert d2["ci95_per_min"][0] < 2.0 < d2["ci95_per_min"][1]
    assert d2["change_first_to_last"] == pytest.approx(8.0, abs=0.3)
    make_ptp_capture(str(tmp_path / "short"), n_pairs=300, helios_drop_k=(), triton_drop_k=(),
                     drift_us_per_min=2.0)
    d3 = A.analyze(A.load_capture(str(tmp_path / "short")))["drift"]
    assert d3["detected"] is False and "too short" in d3["verdict"]


def test_old_schema_without_round1_columns(tmp_path):
    make_ptp_capture(str(tmp_path), new_columns=False)
    r = A.analyze(A.load_capture(str(tmp_path)))
    assert r["residual_us"]["median"] == pytest.approx(17.7, abs=0.05)   # plan assumed 0
    assert any("plan of 0" in n for n in r["meta"]["notes"])
    assert r["suspect"]["n"] == 0


def test_freerun_ptp_headline(tmp_path):
    make_ptp_capture(str(tmp_path), method="freerun_ptp", helios_drop_k=(), triton_drop_k=())
    r = A.analyze(A.load_capture(str(tmp_path)))
    keys = [h["key"] for h in r["headline"]]
    assert keys == ["centre", "offset"]
    off = r["headline"][1]
    assert off["basis"] == A.BASIS_MEASURED and off["median_abs"] > 10_000     # ~23 ms random phase
    assert off["meets_motion_criterion"] is False


# ---------------------------------------------------------------- faults
def test_ptp_step_pairs_are_suspect(tmp_path):
    """A +1 period PTP step on the Triton pairs frames 100 ms apart with perfect-looking offsets."""
    for flags in (True, False):
        d = str(tmp_path / f"step_{flags}")
        make_ptp_capture(d, n_pairs=600, helios_drop_k=(), triton_drop_k=(), ptp_step_at_k=400,
                         engine_flags_suspect=flags)
        r = A.analyze(A.load_capture(d))
        assert r["suspect"]["n"] >= 190                      # ~199 mismatched pairs
        assert r["offset_start_us"]["max_abs"] < 19           # stats unaffected
        if not flags:
            assert "host_delta_jump (analysis check)" in r["suspect"]["reasons"]
        assert any("PTP monitor" in w for w in r["meta"]["warnings"])
        assert r["segments"] and len(r["segments"]) == 2
    files = A.make_plots(r, str(tmp_path / "plots"))
    assert "ptp_offset_over_time.png" in files and "centre_offset_over_time.png" in files


def test_dead_camera_is_loud(tmp_path):
    make_ptp_capture(str(tmp_path), n_pairs=500, helios_drop_k=(), triton_drop_k=(), triton_dies_at_k=20)
    r = A.analyze(A.load_capture(str(tmp_path)))
    assert r["pairs"] == 20
    assert r["cameras"]["triton"]["drops"] == 480
    assert r["duration_s"] == pytest.approx(50.0, abs=0.1)
    assert r["pair_yield"] < 0.05
    assert "TRITON STOPPED DELIVERING FRAMES" in r["verdict"]
    A.write_report(r, str(tmp_path / "out"))
    assert "WARNING: TRITON STOPPED" in (tmp_path / "out" / "sync_report.md").read_text()


@pytest.mark.parametrize("kind", ["zero", "jump"])
def test_bad_timestamps_excluded_not_fatal(tmp_path, kind):
    make_ptp_capture(str(tmp_path), bad_ts=kind)
    t = time.perf_counter()
    r = A.analyze(A.load_capture(str(tmp_path)))
    A.make_plots(r, str(tmp_path / "plots"))
    assert time.perf_counter() - t < 30
    assert sum(r["meta"]["excluded_timestamps"].values()) >= 1
    assert r["pairs_valid_for_sync"] == 2999
    assert r["offset_start_us"]["max_abs"] < 19
    assert r["duration_s"] < 400
    assert r["cameras"]["helios"]["drops"] in (2, 3) and r["cameras"]["triton"]["drops"] == 1
    assert any("implausible timestamps" in w for w in r["meta"]["warnings"])


def test_independent_clock_domain_ignores_device_offsets(ptp_folder, tmp_path):
    def mark(rows):
        for r in rows:
            r["clock_domain"] = "independent"
    _rewrite_csv(os.path.join(ptp_folder[0], "frames.csv"), tmp_path / "frames.csv", mark)
    r = A.analyze(A.load_capture(str(tmp_path)))
    assert r["meta"]["clock_domain"] == "independent"
    assert r["offset_start_us"]["n"] == 0 and r["offset_center_model_us"]["n"] == 0
    assert r["drift"] is None
    assert any("IGNORED" in n for n in r["meta"]["notes"])
    assert r["legacy_estimate"] is not None


# ---------------------------------------------------------------- original capture.py format
@pytest.mark.parametrize("mode", ["freerun", "trigger"])
def test_capture_py_native_format(tmp_path, mode):
    truth = make_capture_py_capture(str(tmp_path), sync_mode=mode, n_pairs=600)
    cap = A.load_capture(str(tmp_path))                     # zero --map arguments
    assert cap.source_kind == "capture_py" and cap.clock_domain == "independent"
    assert cap.info["helios_center_us"] == pytest.approx(2815, abs=1)
    r = A.analyze(cap)
    assert r["offset_start_us"]["n"] == 0
    assert r["meta"]["method_used"].startswith(f"legacy {'software trigger' if mode == 'trigger' else 'freerun'}")
    assert r["failed_grabs"] == 2 and r["duration_source"] == "run.json elapsed time"
    le = r["legacy_estimate"]
    est = r["_series"]["est_offset_ms"]
    if mode == "trigger":
        assert le["reliable"] is True and le["bound_latest_frame_pairing_ms"] is None
        assert np.median(est - truth["true_offset_ms"]) == pytest.approx(0, abs=0.2)
        assert r["headline"][0]["key"] == "estimate"
    else:
        # NewestOnly: the Triton frame was usually already waiting -> only a bound is honest
        assert le["reliable"] is False and le["triton_waiting_fraction"] > 0.9
        o = truth["true_offset_ms"]
        assert np.all((o <= est + 0.01) & (o >= est - 100.01))
        assert r["headline"][0]["key"] == "bound"
        assert "BOUND" in r["verdict"]
    assert r["cameras"]["helios"]["interval_median_ms"] == pytest.approx(100.0, abs=0.01)


def test_capture_py_columns_ahead_of_sync_columns(ptp_folder):
    """New captures keep the original columns first; the analysis uses the sync columns."""
    with open(os.path.join(ptp_folder[0], "frames.csv")) as f:
        header = f.readline().strip().split(",")
    assert header[:3] == ["frame", "host_time", "hlt_frame_id"] and "pair_index" in header


# ---------------------------------------------------------------- other legacy layouts
def test_legacy_host_only(tmp_path):
    truth = make_legacy_capture(str(tmp_path), style="host_only")
    cap = A.load_capture(str(tmp_path))
    assert cap.source_kind == "legacy" and cap.clock_domain == "independent"
    assert cap.column_map["units"]["helios_host"] == "s"
    r = A.analyze(cap)
    le = r["legacy_estimate"]
    assert le["method"] == "host_delta_minus_latency"
    assert le["estimated_offset_ms"]["median"] == pytest.approx(truth["true_offset_median_ms"], abs=1.0)
    assert le["drift_ppm"] == pytest.approx(20.0, abs=2.0)
    assert r["host_delta_drift"]["slope_per_min"] == pytest.approx(1.2, abs=0.08)
    assert le["bound_latest_frame_pairing_ms"] == pytest.approx(100.0)
    assert r["cameras"]["helios"]["drops"] == 1 and r["cameras"]["triton"]["drops"] == 1


def test_legacy_guessed_column_names(tmp_path):
    truth = make_legacy_capture(str(tmp_path), style="guessed_names", write_run_json=False)
    cap = A.load_capture(str(tmp_path))
    m = cap.column_map
    assert (m["helios_ts"], m["triton_ts"]) == ("depth_timestamp", "color_timestamp")
    assert (m["helios_host"], m["triton_host"]) == ("depth_recv_time", "color_recv_time")
    assert m["units"] == {"helios_ts": "ns", "triton_ts": "ns", "helios_host": "s", "triton_host": "s"}
    assert cap.fps_setting == pytest.approx(10.0, rel=1e-3)
    r = A.analyze(cap)
    assert r["legacy_estimate"]["method"] == "device_ts_lower_envelope"
    assert np.max(np.abs(r["_series"]["est_offset_ms"] - truth["true_offset_ms"])) < 0.05
    assert r["legacy_estimate"]["drift_ppm"] == pytest.approx(20.0, abs=0.5)


def test_legacy_opaque_names_need_map(tmp_path):
    truth = make_legacy_capture(str(tmp_path), style="opaque")
    with pytest.raises(ValueError, match="--map"):
        A.load_capture(str(tmp_path))
    spec = "helios_ts=b,triton_ts=c,helios_host=d,triton_host=e,ts_units=ns,host_units=s"
    r = A.analyze(A.load_capture(str(tmp_path), mapping=spec))
    assert np.max(np.abs(r["_series"]["est_offset_ms"] - truth["true_offset_ms"])) < 0.05
    with pytest.raises(ValueError, match="not in frames.csv"):
        A.load_capture(str(tmp_path), mapping="helios_ts=nope")


def test_parse_map_and_units():
    assert A.parse_map("helios_ts=a, triton_ts=b,units=ms") == {"helios_ts": "a", "triton_ts": "b", "units": "ms"}
    with pytest.raises(ValueError):
        A.parse_map("bogus=a")
    with pytest.raises(ValueError):
        A.parse_map("units=minutes")
    assert A._guess_unit([str(i * 100_000_000) for i in range(10)], 10) == "ns"
    assert A._guess_unit([str(i * 100_000) for i in range(10)], 10) == "us"
    assert A._guess_unit([str(i * 100) for i in range(10)], 10) == "ms"
    assert A._guess_unit([f"{1.7e9 + i * 0.1:.6f}" for i in range(10)], 10) == "s"
    assert A._guess_unit(["1", "2", "3"], 10, column="hlt_ts_ns") == "ns"


# ---------------------------------------------------------------- sync_frames.csv only
def test_sync_frames_only_folder(ptp_folder, tmp_path):
    shutil.copy(os.path.join(ptp_folder[0], "sync_frames.csv"), tmp_path / "sync_frames.csv")
    shutil.copy(os.path.join(ptp_folder[0], "run.json"), tmp_path / "run.json")
    cap = A.load_capture(str(tmp_path))
    assert cap.source_kind == "sync_frames_only" and cap.clock_domain == "ptp"
    r = A.analyze(cap)
    assert r["pairs"] == 3000
    assert r["offset_start_us"]["median"] == pytest.approx(17.7, abs=0.05)
    assert r["cameras"]["helios"]["drops"] == 2 and r["cameras"]["triton"]["drops"] == 1


def test_sync_frames_only_grid_pairing(ptp_folder, tmp_path):
    def unpair(rows):
        for r in rows:
            r["pair_index"], r["paired"] = "", ""
    _rewrite_csv(os.path.join(ptp_folder[0], "sync_frames.csv"), tmp_path / "sync_frames.csv", unpair)
    shutil.copy(os.path.join(ptp_folder[0], "run.json"), tmp_path / "run.json")
    cap = A.load_capture(str(tmp_path))
    assert "grid index" in cap.column_map["pairing"]
    assert A.analyze(cap)["pairs"] == 3000


# ---------------------------------------------------------------- reports, compare, CLI
def test_report_and_plots_written(ptp_folder, tmp_path):
    result, files = A.analyze_folder(ptp_folder[0], str(tmp_path / "out"))
    for name in ("offset_over_time.png", "offset_histogram.png", "centre_offset_over_time.png",
                 "fps_over_time.png", "drops_over_time.png", "host_delta_over_time.png",
                 "ptp_offset_over_time.png", "sync_report.md", "sync_report.json"):
        p = tmp_path / "out" / name
        assert p.is_file() and p.stat().st_size > 1000, name
    md = (tmp_path / "out" / "sync_report.md").read_text()
    assert "SYNTHETIC DATA" in md and "## Headline" in md
    assert "MEASURED" in md and "MODELLED" in md and "HOST-BASED" in md and "NOT CONFIRMED" in md
    assert "start-of-exposure" not in md
    j = json.loads((tmp_path / "out" / "sync_report.json").read_text())
    assert j["offset_start_us"]["n"] == 3000 and j["meta"]["synthetic"] is True
    assert "_series" not in j and len(j["headline"]) == 2


def test_compare_like_with_like(tmp_path):
    make_capture_py_capture(str(tmp_path / "legacy"), sync_mode="freerun", n_pairs=600)
    make_ptp_capture(str(tmp_path / "frptp"), method="freerun_ptp", n_pairs=600, helios_drop_k=(), triton_drop_k=())
    make_ptp_capture(str(tmp_path / "after"), n_pairs=600, alignment="center", helios_drop_k=(), triton_drop_k=())
    res = {k: A.analyze(A.load_capture(str(tmp_path / k))) for k in ("legacy", "frptp", "after")}

    A.compare(res["frptp"], res["after"], str(tmp_path / "c1"))
    j = json.loads((tmp_path / "c1" / "before_after.json").read_text())
    off, cen = j["comparisons"]
    assert off["comparable"] is False and "planned Triton lag" in off["note"]     # centre-aligned AFTER
    assert cen["comparable"] is True and cen["before_basis"] == "MODELLED" and cen["factor_p95"] > 100
    md = (tmp_path / "c1" / "before_after.md").read_text()
    assert md.count("improved by a factor") == 1 and "px @ p95" in md

    make_ptp_capture(str(tmp_path / "after_start"), n_pairs=600, helios_drop_k=(), triton_drop_k=())
    A.compare(res["frptp"], A.analyze(A.load_capture(str(tmp_path / "after_start"))), str(tmp_path / "c3"))
    off = json.loads((tmp_path / "c3" / "before_after.json").read_text())["comparisons"][0]
    assert off["comparable"] is True and off["factor_p95"] > 100            # MEASURED vs MEASURED

    A.compare(res["legacy"], res["after"], str(tmp_path / "c2"))
    j = json.loads((tmp_path / "c2" / "before_after.json").read_text())
    off, cen = j["comparisons"]
    assert off["comparable"] is False and "BOUND" in off["before_basis"]
    assert cen["before"]["n"] == 0 and cen["factor_p95"] is None
    md = (tmp_path / "c2" / "before_after.md").read_text()
    assert "improved by a factor" not in md
    assert "an estimate compared with a measurement" in md
    assert os.path.isfile(tmp_path / "c2" / "before_after_abs_offset.png")


def test_cli_analyze_and_demo(ptp_folder, tmp_path, capsys):
    assert sync_test.main(["analyze", ptp_folder[0], "--out", str(tmp_path / "a")]) == 0
    out = capsys.readouterr().out
    assert "HEADLINE" in out and "MEASURED" in out
    assert sync_test.main(["demo", "--out", str(tmp_path / "demo")]) == 0
    for tag in ("legacy", "ptp"):
        assert (tmp_path / "demo" / f"analysis_{tag}" / "before_after_abs_offset.png").is_file()
    assert sync_test.main(["analyze", str(tmp_path / "does_not_exist")]) == 2


def test_live_glue_with_fake_engine(tmp_path, monkeypatch):
    """run_live's contract with the engine: CaptureEngine(settings, writer=None) context manager,
    .configure(), .run(seconds) -> result with .write_sync_csvs(pairs_csv, frames_csv) and .run_info."""
    calls = []

    class FakeResult:
        def __init__(self, folder):
            self.folder = folder
            with open(os.path.join(folder, "run.json")) as f:
                self.run_info = json.load(f)["sync"]

        def write_sync_csvs(self, pairs_csv, frames_csv):
            shutil.copy(os.path.join(self.folder, "frames.csv"), pairs_csv)
            shutil.copy(os.path.join(self.folder, "sync_frames.csv"), frames_csv)

    class FakeEngine:
        def __init__(self, settings, writer=None):
            assert writer is None
            self.s = settings

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            calls.append("closed")

        def configure(self):
            calls.append(("configure", self.s.SYNC_METHOD, self.s.FRAME_RATE))

        def run(self, seconds):
            src = str(tmp_path / "engine_tmp")
            make_ptp_capture(src, n_pairs=int(seconds * self.s.FRAME_RATE), helios_drop_k=(), triton_drop_k=())
            return FakeResult(src)

    fake = types.ModuleType("lucid_sync.engine")
    fake.CaptureEngine = FakeEngine
    monkeypatch.setitem(sys.modules, "lucid_sync.engine", fake)
    out = tmp_path / "live"
    assert sync_test.main(["live", "--seconds", "20", "--out", str(out), "--no-plots"]) == 0
    assert calls == [("configure", "ptpsync", 10.0), "closed"]
    run = json.loads((out / "run.json").read_text())
    assert list(run.keys()) == ["sync"]                       # settings not duplicated
    j = json.loads((out / "sync_analysis" / "sync_report.json").read_text())
    assert j["pairs"] == 200 and j["offset_start_us"]["n"] == 200
