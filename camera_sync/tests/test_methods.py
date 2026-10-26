"""Tests of lucid_sync.methods against the mock (configuration only, no streaming)."""
from __future__ import annotations

import pytest

from conftest import SYSTEM
from lucid_sync.methods import (CameraNotFound, PtpNotLocked, SyncSetupError,
                                clamp_action_lead_ms, configure_common, configure_with_fallback,
                                desired_triton_lag_ns, enable_ptp, find_cameras, open_cameras,
                                plan_ptpsync_offsets, ptp_now_ns, read_helios_scale_offsets,
                                system_tl_keeper)
from lucid_sync.nodes import ALLOWED_WRITES, NodeReport
from lucid_sync.settings import SyncSettings
from lucid_sync.timing import helios_light_window

FAST = dict(PTP_TIMEOUT_S=5.0, PTP_POLL_S=0.05, PTP_MIN_SETTLE_S=0.2, PTP_SETTLE_MAX_S=2.0)
PTP_KW = dict(timeout_s=5, poll_s=0.05, min_settle_s=0.2, settle_max_s=2.0)


def opened(report=None):
    report = report or NodeReport()
    h, t = open_cameras(SYSTEM, report)
    return h, t, system_tl_keeper(SYSTEM, report), report


def test_find_cameras_and_identity(mock):
    h_info, t_info = find_cameras(SYSTEM)
    assert h_info["model"].startswith("HTP") and t_info["model"].startswith("TRI")
    h, t, tl, _ = opened()
    assert h.identity == dict(model="HTP003S-001", serial="213500017", firmware="1.16.0.0",
                              ip="169.254.3.11")
    assert t.identity["model"] == "TRI032S-C" and tl is not None


def test_find_cameras_missing_lists_found(mock):
    mock(device_missing="triton")
    with pytest.raises(CameraNotFound) as e:
        find_cameras(SYSTEM)
    assert "Triton" in str(e.value) and "HTP003S-001" in str(e.value)


def test_configure_common_caps_exposure_and_bandwidth(mock):
    h, t, _, rep = opened()
    exp = configure_common(h, t, SyncSettings(EXPOSURE_CAP_US=8000))
    assert exp["mode"] == "auto_capped" and exp["chunk_exposure"] is True
    assert t.nm.get("ExposureAuto") == "Continuous"
    assert t.nm.get("ExposureAutoUpperLimit") == 8000.0
    assert t.nm.get("DeviceLinkThroughputLimit") == 60_000_000
    assert h.nm.get("DeviceLinkThroughputLimit") == 40_000_000          # capture.py value
    assert h.sn.get("StreamBufferHandlingMode") == "NewestOnly"         # capture.py value
    assert h.sn.get("StreamAutoNegotiatePacketSize") is True
    assert h.nm.get("Scan3dOperatingMode") == "Distance5000mmMultiFreq"  # user choice
    assert h.nm.get("ExposureTimeSelector") == "Exp250Us"
    assert h.nm.get("ConversionGain") == "High"
    assert t.nm.get("BalanceWhiteAuto") == "Continuous"
    assert exp["helios_center_us"] == pytest.approx(6405.0)
    assert not rep.failed
    assert all(w["node"] in ALLOWED_WRITES for w in rep.writes)
    scale, offsets = read_helios_scale_offsets(h.nm)
    assert scale == 0.25 and offsets == [-8192.0, -8192.0, 0.0]


def test_exposure_fallback_to_fixed_when_limit_nodes_missing(mock):
    mock(drop_nodes={"triton": ["ExposureAutoLimitAuto", "ExposureAutoUpperLimit",
                                "ExposureAutoLowerLimit"]})
    h, t, _, rep = opened()
    exp = configure_common(h, t, SyncSettings(EXPOSURE_CAP_US=6000))
    assert exp["mode"] == "fixed_cap_fallback" and exp["fixed_us"] == 6000.0
    assert t.nm.get("ExposureAuto") == "Off" and t.nm.get("ExposureTime") == 6000.0


def test_exposure_cap_without_limit_auto_node(mock):
    mock(triton_has_exposure_auto_limit_auto=False)
    h, t, _, _ = opened()
    exp = configure_common(h, t, SyncSettings(EXPOSURE_CAP_US=7000))
    assert exp["mode"] == "auto_capped" and t.nm.get("ExposureAutoUpperLimit") == 7000.0


def test_enable_ptp_locks_helios_master(mock):
    h, t, _, _ = opened()
    rep = enable_ptp(h.nm, t.nm, **PTP_KW)
    assert (rep.helios_status, rep.triton_status, rep.master) == ("Master", "Slave", "helios")
    assert 0.3 < rep.lock_seconds < 2.0
    assert rep.settled is True and rep.settle_method == "offset_threshold"
    assert len(rep.offset_from_master_ns_samples) >= 5
    assert ptp_now_ns(h.nm) > 0


def test_settle_gate_not_settled_and_fixed_wait(mock):
    mock(ptp_jitter_ns_std=200_000)                  # |offset| mostly > 10 us
    h, t, _, _ = opened()
    rep = enable_ptp(h.nm, t.nm, **{**PTP_KW, "settle_max_s": 0.5})
    assert rep.settled is False and rep.settle_seconds >= 0.5      # continues, flagged
    mock(drop_nodes={"triton": ["PtpOffsetFromMaster"]})
    h, t, _, _ = opened()
    rep = enable_ptp(h.nm, t.nm, **{**PTP_KW, "min_settle_s": 0.3})
    assert rep.settled is None and rep.settle_method == "fixed_wait"
    assert rep.settle_seconds >= 0.3


def test_ptp_role_change_toggles_enable(mock):
    h, t, _, rep_nodes = opened()
    t.nm.set("PtpSlaveOnly", False)                 # earlier session: Triton not slave-only
    t.nm.set("PtpEnable", True)
    rep = enable_ptp(h.nm, t.nm, **PTP_KW)
    assert rep.roles_reset == ["triton"] and rep.master == "helios"
    seq = [(w["node"], w["new"]) for w in rep_nodes.writes if w["camera"] == "triton"]
    i = seq.index(("PtpEnable", False))
    assert seq[i + 1:i + 3] == [("PtpSlaveOnly", True), ("PtpEnable", True)]


def test_enable_ptp_timeout_raises(mock):
    mock(ptp_never_locks=True)
    h, t, _, _ = opened()
    with pytest.raises(PtpNotLocked) as e:
        enable_ptp(h.nm, t.nm, **{**PTP_KW, "timeout_s": 0.8})
    assert e.value.report.triton_status == "Listening"


def test_plan_ptpsync_offsets_all_cases():
    P = 100_000_000
    assert plan_ptpsync_offsets(0, P, True, False) == (0, None, 0, "start alignment")
    h, t, lag, _ = plan_ptpsync_offsets(6_850_000, P, True, True)
    assert (h, t, lag) == (0, 6_850_000, 6_850_000)
    h, t, lag, _ = plan_ptpsync_offsets(6_850_000, P, True, False)      # mock default
    assert (h, t, lag) == (93_150_000, None, 6_850_000)
    h, t, lag, _ = plan_ptpsync_offsets(-650_000, P, True, False)
    assert (h, t, lag) == (650_000, None, -650_000)
    h, t, lag, _ = plan_ptpsync_offsets(-650_000, P, False, True)
    assert (h, t, lag) == (None, 99_350_000, -650_000)
    assert plan_ptpsync_offsets(5_000, P, False, False)[2] == 0


def test_timing_model_values():
    assert helios_light_window("Distance5000mmMultiFreq", "Exp250Us")["center_us"] == \
        pytest.approx(6405.0)
    assert helios_light_window("Distance6000mmSingleFreq", "Exp250Us")["center_us"] == \
        pytest.approx(2815.0)
    w = helios_light_window("Distance5000mmMultiFreq", "Exp1000Us")
    assert w["center_us"] == pytest.approx(9345.0) and w["n_subframes"] == 8
    assert w["light_end_us"] == pytest.approx(7 * 2530 + 980)
    with pytest.raises(ValueError):
        helios_light_window("Distance5000mm", "Exp250Us")
    assert SyncSettings().helios_center_us() == pytest.approx(6405.0)
    assert SyncSettings(HELIOS_CENTER_US=7000).helios_center_us() == 7000


def test_desired_lag_and_lead_clamp():
    assert desired_triton_lag_ns(SyncSettings(ALIGNMENT="start"), 5000)[0] == 0
    assert SyncSettings().ALIGNMENT == "center"                              # default
    assert desired_triton_lag_ns(SyncSettings(), 5000)[0] == 6405_000 - 2_500_000
    s = SyncSettings(ALIGNMENT="center", HELIOS_CENTER_US=9350)
    assert desired_triton_lag_ns(s, 5000)[0] == 6_850_000
    assert desired_triton_lag_ns(s, 20000)[0] == -650_000
    bad = SyncSettings(HELIOS_OPERATING_MODE="UnknownMode")
    lag, note = desired_triton_lag_ns(bad, 5000)
    assert lag == 0 and "timing model" in note
    assert clamp_action_lead_ms(50, 1e8, 4) == 50
    assert clamp_action_lead_ms(500, 1e8, 4) == 300
    assert clamp_action_lead_ms(80, 1e8, 1) == 50


def test_fallback_ptpsync_to_action_when_start_mode_missing(mock):
    mock(drop_nodes={"triton": ["AcquisitionStartMode"]})
    h, t, tl, _ = opened()
    s = SyncSettings(SYNC_METHOD="ptpsync", **FAST)
    res = configure_with_fallback("ptpsync", h, t, tl, s, configure_common(h, t, s))
    assert res.method_used == "action"
    assert res.fallback_chain[0]["method"] == "ptpsync"
    assert "AcquisitionStartMode" in res.fallback_chain[0]["reason"]
    assert t.nm.get("TriggerSource") == "Action0" and t.nm.get("TriggerMode") == "On"
    assert tl.get("ActionCommandTargetIP") == 0xFFFFFFFF
    assert res.plan.state["triton"]["TriggerSource"] == "Action0"        # read back
    assert res.plan.alignment_used == "center" and t.nm.get("TriggerDelay") > 0


def test_fallback_when_ptp_never_locks_costs_one_timeout(mock):
    mock(ptp_never_locks=True)
    h, t, tl, _ = opened()
    s = SyncSettings(SYNC_METHOD="ptpsync", PTP_TIMEOUT_S=0.8, PTP_POLL_S=0.05)
    import time
    t0 = time.monotonic()
    res = configure_with_fallback("ptpsync", h, t, tl, s, configure_common(h, t, s))
    assert time.monotonic() - t0 < 2.5                       # one timeout, not several
    # software trigger still works without PTP (independent clocks), better than freerun
    assert res.method_used == "software" and res.plan.clock_domain == "independent"
    assert [c["method"] for c in res.fallback_chain] == ["ptpsync", "action"]
    assert all("PtpNotLocked" in c["reason"] for c in res.fallback_chain)
    assert h.nm.get("PtpEnable") is False and t.nm.get("PtpEnable") is False
    assert res.ptp is not None and res.ptp.triton_status == "Listening"


def test_requested_freerun_never_enables_ptp(mock):
    h, t, tl, rep = opened()
    s = SyncSettings(SYNC_METHOD="freerun")
    res = configure_with_fallback("freerun", h, t, tl, s, configure_common(h, t, s))
    assert res.method_used == "freerun" and res.fallback_chain == [] and res.ptp is None
    assert not any(w["node"] == "PtpEnable" for w in rep.writes)
    assert h.nm.get("AcquisitionFrameRate") == 10.0


def test_action_unavailable_without_tl_nodemap_falls_to_software_with_ptp(mock):
    mock(drop_nodes={"triton": ["AcquisitionStartMode"]})
    h, t, _, _ = opened()
    s = SyncSettings(SYNC_METHOD="ptpsync", **FAST)
    res = configure_with_fallback("ptpsync", h, t, None, s, configure_common(h, t, s))
    assert res.method_used == "software" and res.plan.clock_domain == "ptp"
    assert "tl_system_nodemap" in res.fallback_chain[1]["reason"]
    assert t.nm.get("TriggerSource") == "Software"


def test_readback_failure_makes_method_fail_with_reason(mock):
    """Review item 7: a write that silently does not take effect is caught by read-back."""
    mock(stuck_nodes={"triton": ["TriggerMode"]})   # writes accepted but ignored
    h, t, tl, _ = opened()
    t.nm.nodemap["TriggerMode"].raw_set("On", "FrameStart")   # left On by an earlier run
    s = SyncSettings(SYNC_METHOD="ptpsync", **FAST)
    res = configure_with_fallback("ptpsync", h, t, tl, s, configure_common(h, t, s))
    assert res.fallback_chain[0]["method"] == "ptpsync"
    assert "read-back" in res.fallback_chain[0]["reason"]
    assert "triton TriggerMode reads 'On'" in res.fallback_chain[0]["reason"]
    assert res.method_used == "action"                   # needs TriggerMode On: works


def test_stuck_in_ptpsync_fails_every_other_method(mock):
    mock(enum_entries={"triton": {"AcquisitionStartMode": ["PTPSync"]}},
         drop_nodes={"helios": ["PTPSyncFrameRate"]})
    h, t, tl, _ = opened()
    assert t.nm.get("AcquisitionStartMode") == "PTPSync"
    s = SyncSettings(SYNC_METHOD="ptpsync", **FAST)
    with pytest.raises(SyncSetupError) as e:
        configure_with_fallback("ptpsync", h, t, tl, s, configure_common(h, t, s))
    msg = str(e.value)
    for m in ("ptpsync", "action", "software", "freerun_ptp", "freerun"):
        assert f"{m}: " in msg
    assert "AcquisitionStartMode is PTPSync and cannot be changed" in msg


def test_trigger_write_order_matches_lucid_sample(mock):
    """[HT] Cpp_HLTRGB_3_Overlay_PTP_SAC_v7.cpp lines 131-133: Selector, Source, Mode."""
    h, t, tl, rep = opened()
    t.nm.set("TriggerSelector", "FrameBurstStart")       # force the selector write too
    s = SyncSettings(SYNC_METHOD="action", **FAST)
    configure_with_fallback("action", h, t, tl, s, configure_common(h, t, s))
    seq = [w["node"] for w in rep.writes if w["camera"] == "triton"
           and w["node"] in ("TriggerSelector", "TriggerSource", "TriggerMode")]
    assert seq[-3:] == ["TriggerSelector", "TriggerSource", "TriggerMode"]


def test_ptpsync_center_alignment_offsets_written(mock):
    h, t, tl, _ = opened()
    s = SyncSettings(ALIGNMENT="center", EXPOSURE_CAP_US=5000, **FAST)
    res = configure_with_fallback("ptpsync", h, t, tl, s, configure_common(h, t, s))
    assert res.method_used == "ptpsync"
    lag = 6_405_000 - 2_500_000                     # timing-model centre - cap/2
    assert h.nm.get("PTPSyncOffset") == 100_000_000 - lag   # only Helios has the node
    assert res.plan.lags_ns == {"helios": 0, "triton": lag}
    assert res.plan.planned_lag_ns == lag and res.plan.alignment_used == "center"


def test_ptpsync_without_offset_node_falls_back_to_start(mock):
    mock(helios_has_ptpsync_offset=False)
    h, t, tl, _ = opened()
    s = SyncSettings(**FAST)
    res = configure_with_fallback("ptpsync", h, t, tl, s, configure_common(h, t, s))
    assert res.method_used == "ptpsync" and res.plan.alignment_used == "start"
    assert res.plan.alignment_requested == "center" and res.plan.planned_lag_ns == 0
    assert "no PTPSyncOffset" in res.plan.alignment_note
    assert h.nm.get("AcquisitionStartMode") == "PTPSync" and t.nm.get("PTPSyncFrameRate") == 10.0
