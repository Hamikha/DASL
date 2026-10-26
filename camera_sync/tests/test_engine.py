"""Integration tests: CaptureEngine end to end on the mock SDK (real time)."""
from __future__ import annotations

import json
import threading
import time

import numpy as np
import pytest

from conftest import SDK_INFO, SYSTEM
from lucid_sync import engine as engine_mod
from lucid_sync import triggers as triggers_mod
from lucid_sync.engine import CaptureEngine
from lucid_sync.settings import SyncSettings

BIAS_US = 41.1 - 23.4          # mock = LUCID's measured Helios/Triton timestamp points
CENTER_US = 6405.0             # timing model: Distance5000mmMultiFreq + Exp250Us
LAG_US = CENTER_US - 8000 / 2  # default centre alignment with the 8 ms cap


def settings(**kw) -> SyncSettings:
    base = dict(FRAME_RATE=10.0, PTP_TIMEOUT_S=5.0, PTP_POLL_S=0.05, PTP_MIN_SETTLE_S=0.2,
                PTP_SETTLE_MAX_S=2.0, PTP_MONITOR_S=0.2)
    base.update(kw)
    return SyncSettings(**base)


def capture(s: SyncSettings, seconds: float, writer=None, **run_kw):
    with CaptureEngine(s, writer=writer, system=SYSTEM, sdk_info=SDK_INFO) as eng:
        eng.configure()
        return eng.run(seconds, **run_kw)


def assert_good_grid_run(res, seconds, method, lag_us=LAG_US):
    st, info = res.stats, res.run_info
    assert info["method_used"] == method and info["clock_domain"] == "ptp"
    expected = seconds * 10
    assert expected - 4 <= st["pairs"] <= expected + 6, st["pairs"]
    assert st["dropped_helios"] == 0 and st["dropped_triton"] == 0
    assert st["incomplete"] == 0 and st["failed_grabs"] == 0 and st["rejected_suspect"] == 0
    assert st["pair_yield"] == pytest.approx(1.0, abs=0.05)
    assert st["fps"] == pytest.approx(10.0, rel=0.002)
    assert st["residual_us"]["median"] == pytest.approx(BIAS_US, abs=1.5)
    assert st["residual_us"]["max_abs"] < BIAS_US + 10
    assert st["offset_start_us"]["median"] == pytest.approx(lag_us + BIAS_US, abs=1.5)
    assert info["planned_lag_us"] == pytest.approx(lag_us)
    assert all(p.helios.grid_k == p.triton.grid_k for p in res.pairs)
    json.dumps(info)                                        # run.json-ready


def test_ptpsync_10fps_center_default(mock):
    res = capture(settings(SYNC_METHOD="ptpsync"), 6.0)
    assert_good_grid_run(res, 6.0, "ptpsync")
    info = res.run_info
    assert info["fallback_chain"] == [] and info["ptpsync_rate_check"]["ok"]
    assert info["alignment_requested"] == info["alignment_used"] == "center"
    assert info["helios_center_us"] == pytest.approx(CENTER_US)
    assert info["ptp"]["master"] == "helios" and info["ptp"]["settled"] is True
    mon = info["ptp"]["monitor"]
    assert mon["samples"] >= 25 and mon["bad_samples"] == 0 and mon["timeline"]
    assert info["state_after_start"]["triton"]["AcquisitionStartMode"] == "PTPSync"
    assert info["stats"]["pairing"]["window_source"] == "PTP-latched capture window"
    # bright scene: chunk exposure ~1 ms recorded per frame
    assert {p.triton.exposure_us for p in res.pairs} == {1000.0}
    assert info["triton_exposure_source"] == {"chunk": len(
        [f for f in res.frames if f.camera == "triton"])}
    p = res.pairs[0]
    assert p.offset_center_model_us == pytest.approx(LAG_US + BIAS_US + 500 - CENTER_US, abs=3)
    assert info["restore"]["failed"] == [] and info["restore"]["restored"]


def test_action_10fps(mock):
    res = capture(settings(SYNC_METHOD="action"), 6.0)
    assert_good_grid_run(res, 6.0, "action")
    act = res.run_info["action"]
    assert act["fired"] >= 59 and act["skipped_late"] == 0 and act["error"] == 0
    assert act["margin_ms"]["p50"] > 20 and res.stats["action_skipped_late"] == 0
    assert res.run_info["triton_trigger_delay_us"] == pytest.approx(LAG_US)
    # with a known T0 the grid residual IS the trigger->timestamp latency (23.4 / 41.1 us)
    pc = res.stats["per_camera"]
    assert pc["helios"]["grid_residual_us"]["median"] == pytest.approx(23.4, abs=2)
    assert pc["triton"]["grid_residual_us"]["median"] == pytest.approx(41.1, abs=2)  # lag in grid


def test_software_trigger_with_ptp(mock):
    res = capture(settings(SYNC_METHOD="software"), 4.0)
    st, info = res.stats, res.run_info
    assert info["method_used"] == "software" and info["clock_domain"] == "ptp"
    assert st["pairs"] >= 36 and st["dropped_helios"] == 0 and st["dropped_triton"] == 0
    assert abs(st["residual_us"]["median"]) < 2000          # GigE command skew, measured
    assert info["trigger"]["fired"] >= 38 and info["trigger"]["not_armed"] == 0
    assert all(p.helios.frame_id - p.triton.frame_id ==
               res.pairs[0].helios.frame_id - res.pairs[0].triton.frame_id for p in res.pairs)


def test_fallback_to_action_when_acquisition_start_mode_missing(mock):
    mock(drop_nodes={"triton": ["AcquisitionStartMode"]})
    res = capture(settings(SYNC_METHOD="ptpsync"), 3.0)
    assert_good_grid_run(res, 3.0, "action")
    chain = res.run_info["fallback_chain"]
    assert chain[0]["method"] == "ptpsync" and "AcquisitionStartMode" in chain[0]["reason"]


def test_ptp_never_locks_falls_back_to_software_without_ptp(mock):
    mock(ptp_never_locks=True)
    res = capture(settings(SYNC_METHOD="ptpsync", PTP_TIMEOUT_S=1.0), 3.0)
    info, st = res.run_info, res.stats
    assert info["method_used"] == "software" and info["clock_domain"] == "independent"
    assert [c["method"] for c in info["fallback_chain"]] == ["ptpsync", "action"]
    assert st["offset_start_us"]["n"] == 0                  # unmeasurable without PTP
    assert st["host_delta_ms"]["n"] == st["pairs"] >= 25
    assert st["dropped_helios"] == 0 and st["dropped_triton"] == 0


def test_legacy_freerun(mock):
    res = capture(settings(SYNC_METHOD="freerun"), 3.0)
    st = res.stats
    assert res.run_info["method_used"] == "freerun" and st["pairs"] >= 25
    assert st["host_delta_ms"]["max_abs"] <= 50.0           # nearest within half a period
    assert res.run_info["alignment_used"] == "none" and res.run_info["ptp"] is None
    assert all(p.csv_row()["clock_domain"] == "independent" for p in res.pairs)


def test_dark_scene_exposure_cap_keeps_10fps(mock):
    mock(scene="dark")
    res = capture(settings(SYNC_METHOD="ptpsync", EXPOSURE_CAP_US=8000), 3.0)
    assert_good_grid_run(res, 3.0, "ptpsync")
    assert {p.triton.exposure_us for p in res.pairs} == {8000.0}
    assert res.run_info["exposure"]["mode"] == "auto_capped"
    assert res.stats["offset_center_model_us"]["median"] == pytest.approx(BIAS_US, abs=2)


def test_packet_loss_counted_and_no_bad_pairs(mock):
    mock(packet_loss_prob=0.15)
    res = capture(settings(SYNC_METHOD="ptpsync"), 4.0)
    st = res.stats
    assert st["incomplete"] > 0
    assert all(not p.helios.incomplete and not p.triton.incomplete for p in res.pairs)
    assert all(abs(p.residual_us - BIAS_US) < 10 for p in res.pairs)
    assert st["dropped_helios"] == 0 and st["dropped_triton"] == 0   # incomplete != dropped


# --------------------------------------------------------------------------- review MAJORs
def test_major1_ptp_step_one_period_flagged_and_not_written(mock):
    """Triton slave clock silently steps +1 period mid-run (exposures stay on the grid).
    Grid pairing alone would combine frames 100 ms apart with a perfect offset."""
    mock(ptp_step={"camera": "triton", "after_lock_s": 3.0, "step_ns": 100_000_000})
    written = []
    res = capture(settings(SYNC_METHOD="ptpsync"), 5.0, writer=lambda p: written.append(p))
    # Pairs exposed after the recording stopped (collected while draining) are never
    # checked, saved or counted, so they are left out of this test.
    pairs = [p for p in res.pairs if p.status != "after_stop"]
    first = min(pairs, key=lambda p: p.pair_index)          # before the step: ground truth
    d0 = first.helios.frame_id - first.triton.frame_id
    assert not first.suspect and abs(first.residual_us - BIAS_US) < 10
    wrong = [p for p in pairs if p.helios.frame_id - p.triton.frame_id != d0]
    assert len(wrong) >= 10                                 # the step really mis-pairs
    escaped = [(p.pair_index, p.grid_k, p.helios.frame_id, p.triton.frame_id,
                round(p.host_delta_ms, 1), p.suspect_reason) for p in wrong
               if not (p.suspect and "host_delta" in p.suspect_reason)]
    assert not escaped, escaped
    assert all(p.status == "rejected_suspect" for p in wrong)
    assert not any(p.suspect for p in pairs if p not in wrong)
    assert {id(p) for p in written}.isdisjoint({id(p) for p in wrong})
    assert res.stats["rejected_suspect"] == len(wrong)
    assert res.stats["pairs"] == len(pairs) - len(wrong)
    rows = [r for r in res.pair_rows() if r["suspect"] == 1]
    assert rows and rows[0]["suspect_reason"].startswith("host_delta")


def test_major1_ptp_monitor_marks_offset_glitch_window(mock):
    """Servo glitch: slave clock 50 us off for 1 s, status stays Slave."""
    mock(ptp_glitch={"camera": "triton", "after_lock_s": 2.0, "duration_s": 1.0,
                     "lock_lost": False, "offset_ns": 50_000})
    written = []
    res = capture(settings(SYNC_METHOD="ptpsync"), 5.0, writer=lambda p: written.append(p))
    bad = [p for p in res.pairs if abs(p.residual_us - BIAS_US) > 30]
    assert len(bad) >= 7
    assert all(p.suspect and "ptp_monitor" in p.suspect_reason for p in bad)
    assert not set(map(id, written)) & set(map(id, bad))
    mon = res.run_info["ptp"]["monitor"]
    assert mon["bad_samples"] >= 3 and mon["bad_intervals_ptp_ns"]
    good = [p for p in res.pairs if not p.suspect]
    assert all(abs(p.residual_us - BIAS_US) < 10 for p in good)


def test_major1_lock_lost_window_counted_as_drops(mock):
    mock(ptp_glitch={"camera": "triton", "after_lock_s": 2.0, "duration_s": 1.0,
                     "lock_lost": True})
    res = capture(settings(SYNC_METHOD="ptpsync"), 5.0)
    st, mon = res.stats, res.run_info["ptp"]["monitor"]
    assert st["dropped_triton"] >= 8 and st["dropped_helios"] == 0     # no PTPSync frames
    assert any(s["triton"] == "Listening" for s in mon["timeline"])
    assert all(abs(p.residual_us - BIAS_US) < 10 for p in res.pairs if not p.suspect)


def test_major3_dead_triton_counts_drops(mock):
    mock(host_stall={"camera": "triton", "after_s": 1.5, "duration_s": 100})
    res = capture(settings(SYNC_METHOD="ptpsync"), 4.0)
    st = res.stats
    exp_t = st["frames_expected"]["triton"]
    assert exp_t >= 38 and st["frames_received"]["triton"] <= 16
    assert st["dropped_triton"] == exp_t - st["frames_received"]["triton"]
    assert st["dropped_helios"] == 0 and st["pair_yield"] < 0.5
    assert st["per_camera"]["triton"]["timeouts"] >= 1


def test_major3_dead_camera_in_freerun_is_estimated(mock):
    mock(host_stall={"camera": "triton", "after_s": 1.5, "duration_s": 100})
    res = capture(settings(SYNC_METHOD="freerun"), 3.0)
    assert res.stats["dropped_triton"] >= 10 and res.stats["dropped_helios"] == 0


def test_major4_zero_timestamp_ignored(mock):
    mock(timestamp_override={"triton": {6: 0}})
    res = capture(settings(SYNC_METHOD="ptpsync"), 3.0)
    st = res.stats
    assert st["invalid_ts"] == 1 and st["dropped_triton"] == 1
    bad = [f for f in res.frames if f.ts_ns == 0]
    assert len(bad) == 1 and not bad[0].paired and "invalid_ts" in bad[0].note
    assert all(p.triton.ts_ns > 0 for p in res.pairs)


def test_major6_restore_on_exit_returns_nodes(mock):
    names = ("TriggerMode", "TriggerSource", "AcquisitionStartMode", "PtpEnable", "PixelFormat",
             "ExposureAutoUpperLimit", "DeviceLinkThroughputLimitMode", "ChunkModeActive",
             "AcquisitionFrameRateEnable", "PTPSyncFrameRate", "ExposureTimeSelector")

    def snap():
        devs = SYSTEM.create_device()
        out = {}
        for d in devs:
            for n in names:
                node = d.nodemap.get_node(n)
                if node is not None:
                    out[(d.nodemap["DeviceModelName"].value, n)] = node.value
            out[(d.nodemap["DeviceModelName"].value, "buf")] = \
                d.tl_stream_nodemap["StreamBufferHandlingMode"].value
        SYSTEM.destroy_device()
        return out

    before = snap()
    res = capture(settings(SYNC_METHOD="action"), 1.0)
    assert res.run_info["restore"]["failed"] == []
    assert snap() == before
    res = capture(settings(SYNC_METHOD="ptpsync", RESTORE_ON_EXIT=False), 1.0)
    after = snap()
    assert after[("TRI032S-C", "AcquisitionStartMode")] == "PTPSync"   # left as configured


def test_scheduler_death_aborts_run_loudly(mock, monkeypatch):
    real = triggers_mod.ptp_now_ns
    calls = {"n": 0}

    def stepping(master):
        calls["n"] += 1
        return real(master) + (10**9 if calls["n"] > 3 else 0)       # PTP time step

    monkeypatch.setattr(triggers_mod, "ptp_now_ns", stepping)
    res = capture(settings(SYNC_METHOD="action", ACTION_RESYNC_S=0.3), 6.0)
    info = res.run_info
    assert info["aborted_reason"] and "PTP time step" in info["aborted_reason"]
    assert info["stop_reason"] == info["aborted_reason"] and info["seconds_actual"] < 4


# --------------------------------------------------------------------------- writer & stop
def test_writer_gets_pixels_and_csv(mock, tmp_path):
    got = []

    def writer(pair):
        got.append((pair.save_index, pair.images["helios"].shape, pair.images["helios"].dtype,
                    pair.images["triton"].shape, pair.images["triton"].dtype))

    res = capture(settings(SYNC_METHOD="ptpsync"), 2.0, writer=writer)
    assert len(got) == res.stats["pairs"] > 10
    assert [g[0] for g in got] == list(range(len(got)))     # file index = save order
    assert got[0][1:] == ((120, 160, 4), np.dtype("<u2"), (384, 512), np.dtype("uint8"))
    assert all(p.status == "written" and p.images is None for p in res.pairs)
    res.write_sync_csvs(tmp_path / "frames.csv", tmp_path / "sync_frames.csv")
    lines = (tmp_path / "frames.csv").read_text().splitlines()
    assert lines[0].split(",")[-5:] == ["planned_lag_us", "residual_us", "suspect",
                                        "suspect_reason", "segment_id"]
    assert len(lines) == len(got) + 1


def test_slow_writer_drops_pairs_without_blocking_acquisition(mock):
    def slow(pair):
        time.sleep(0.6)

    res = capture(settings(SYNC_METHOD="ptpsync", WRITER_QUEUE_PAIRS=2), 3.0, writer=slow)
    w = res.stats["writer"]
    assert w["dropped_queue_full"] > 10 and w["written"] >= 2
    assert res.stats["pairs"] >= 27                         # acquisition kept going
    assert res.stats["dropped_helios"] == 0 and res.stats["dropped_triton"] == 0
    dropped = [p for p in res.pairs if p.status == "writer_dropped"]
    assert len(dropped) == w["dropped_queue_full"]


def test_should_stop_and_warmup(mock):
    calls = []

    def stop_after_15(status):
        calls.append(status)
        return "frame limit" if status["pairs"] >= 15 else None

    res = capture(settings(SYNC_METHOD="ptpsync"), 0, should_stop=stop_after_15, warmup_s=1.0)
    assert res.run_info["stop_reason"] == "frame limit"
    assert 15 <= res.stats["pairs"] <= 22
    assert res.stats["per_camera"]["helios"]["warmup_frames_discarded"] >= 8


def test_ctrl_c_twice_still_returns_stats(mock, monkeypatch):
    real_sleep = time.sleep
    calls = {"n": 0}

    class FakeTime:
        def __getattr__(self, name):
            return getattr(time, name)

        @staticmethod
        def sleep(sec):
            if threading.current_thread() is threading.main_thread():
                calls["n"] += 1
                if calls["n"] in (30, 31):                  # 2nd one hits the drain
                    raise KeyboardInterrupt
            real_sleep(sec)

    monkeypatch.setattr(engine_mod, "time", FakeTime())
    res = capture(settings(SYNC_METHOD="ptpsync"), 30.0)
    assert res.interrupted and res.run_info["stop_reason"] == "Ctrl+C"
    assert 5 <= res.stats["pairs"] <= 25 and res.run_info["seconds_actual"] < 5
    assert SYSTEM.mock_world.cameras["helios"].engine is None      # stream stopped
    assert res.run_info["restore"]["failed"] == []
