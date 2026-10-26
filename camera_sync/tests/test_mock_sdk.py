"""Tests of the mock arena_api (camera_sync/mock/arena_api). Runtime target: < 60 s.

Run:  python -m pytest camera_sync/tests/test_mock_sdk.py -q
"""
from __future__ import annotations

import os
import statistics
import sys
import threading
import time

import numpy as np
import pytest

_MOCK_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mock")
if _MOCK_DIR not in sys.path:
    sys.path.insert(0, _MOCK_DIR)

import arena_api  # noqa: E402
from arena_api.buffer import BufferFactory  # noqa: E402
from arena_api.enums import PixelFormat  # noqa: E402
from arena_api.system import system  # noqa: E402

LOCK_S = 0.5


@pytest.fixture(autouse=True)
def fresh_mock():
    arena_api.mock_reset(seed=1234, small_frames=True, ptp_lock_s=LOCK_S)
    yield
    system.destroy_device()
    arena_api.mock_reset()


def open_pair():
    devs = system.create_device()
    by_model = {d.nodemap["DeviceModelName"].value: d for d in devs}
    return by_model["HTP003S-001"], by_model["TRI032S-C"]


def lock_ptp(h, t, helios_master=True, timeout=5.0):
    h.nodemap["PtpSlaveOnly"].value = not helios_master
    t.nodemap["PtpSlaveOnly"].value = helios_master
    h.nodemap["PtpEnable"].value = True
    t.nodemap["PtpEnable"].value = True
    want = ("Master", "Slave") if helios_master else ("Slave", "Master")
    t_end = time.monotonic() + timeout
    while time.monotonic() < t_end:
        if (h.nodemap["PtpStatus"].value, t.nodemap["PtpStatus"].value) == want:
            return
        time.sleep(0.05)
    raise AssertionError("PTP did not lock")


def ptp_now(dev):
    dev.nodemap["PtpDataSetLatch"].execute()
    return dev.nodemap["PtpDataSetLatchValue"].value


def collect(dev, n, timeout_ms=3000):
    out = []
    for _ in range(n):
        b = dev.get_buffer(timeout=timeout_ms)
        out.append((b.frame_id, b.timestamp_ns, b.is_incomplete))
        dev.requeue_buffer(b)
    return out


def collect_for(devs, seconds):
    res = {id(d): [] for d in devs}
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        for d in devs:
            try:
                b = d.get_buffer(timeout=5)
            except TimeoutError:
                continue
            res[id(d)].append((b.frame_id, b.timestamp_ns, b.is_incomplete))
            d.requeue_buffer(b)
    return [res[id(d)] for d in devs]


# ------------------------------------------------------------------------- enumeration
def test_enumeration_and_marker():
    assert arena_api.__mock__ is True and arena_api.__version__.startswith("mock-")
    infos = system.device_infos
    assert {i["model"] for i in infos} == {"HTP003S-001", "TRI032S-C"}
    for i in infos:
        assert {"model", "serial", "ip", "mac", "vendor"} <= set(i)
    assert len({i["serial"] for i in infos}) == 2 and len({i["ip"] for i in infos}) == 2
    h, t = open_pair()
    assert h.nodemap["DeviceFirmwareVersion"].value == "1.16.0.0"
    assert (h.nodemap["Width"].value, h.nodemap["Height"].value) == (160, 120)   # small
    assert t.nodemap["PixelFormat"].value == "BayerRG8"
    with pytest.raises(RuntimeError):
        system.create_device(device_infos=[infos[0]])        # already open


def test_env_knobs(monkeypatch):
    from arena_api import _mock_config
    monkeypatch.setenv("LUCID_MOCK_SMALL", "1")
    monkeypatch.setenv("LUCID_MOCK_FAULTS", '{"scene": "dim", "drop_node": ["PTPSyncOffset"], '
                                            '"drift_ppm": {"triton": -30}}')
    cfg = _mock_config.fresh_config()
    assert cfg.small_frames and cfg.scene == "dim"
    assert cfg.drop_nodes == {"helios": ["PTPSyncOffset"], "triton": ["PTPSyncOffset"]}
    assert cfg.drift_ppm == {"helios": 12.0, "triton": -30}
    with pytest.raises(KeyError):
        _mock_config.fresh_config({"no_such_knob": 1})


def test_device_missing_fault():
    arena_api.mock_reset(device_missing="triton")
    infos = system.device_infos
    assert [i["model"] for i in infos] == ["HTP003S-001"]
    assert len(system.create_device()) == 1


# ------------------------------------------------------------------------- nodes
def test_node_read_write_limits_and_missing():
    h, t = open_pair()
    nm = t.nodemap
    fr = nm["AcquisitionFrameRate"]
    with pytest.raises(RuntimeError):                    # not writable until Enable
        fr.value = 5.0
    nm["AcquisitionFrameRateEnable"].value = True
    fr.value = 5.0
    assert fr.value == 5.0 and fr.unit == "Hz" and fr.min <= 5.0 <= fr.max
    with pytest.raises(ValueError):
        fr.value = fr.max + 1
    with pytest.raises(ValueError):
        nm["TriggerSource"].value = "Line7"
    assert "Action0" in nm["TriggerSource"].enumentry_names
    with pytest.raises(ValueError):
        nm["GevSCPSPacketSize"].value = 1501               # increment 4
    with pytest.raises(RuntimeError):
        nm["DeviceSerialNumber"].value = "x"               # read-only
    with pytest.raises(RuntimeError):
        nm["ExposureTime"].value = 1000.0                  # auto exposure owns it
    nm["ExposureAuto"].value = "Off"
    nm["ExposureTime"].value = 1000.0
    with pytest.raises(ValueError):
        nm["ExposureTime"].value = 10.0                    # < 30.25 us
    with pytest.raises(KeyError):
        nm["NoSuchNode"]
    assert nm.get_node("NoSuchNode") is None
    d = nm.get_node(["Width", "NoSuchNode"])
    assert d["Width"].value == 512 and d["NoSuchNode"] is None
    # selector semantics: TriggerMode is stored per TriggerSelector entry
    nm["TriggerSelector"].value = "FrameBurstStart"
    nm["TriggerMode"].value = "On"
    nm["TriggerSelector"].value = "FrameStart"
    assert nm["TriggerMode"].value == "Off"
    # Triton has no PTPSyncOffset by default (NOT CONFIRMED on Triton); Helios has it
    assert nm.get_node("PTPSyncOffset") is None and h.nodemap.get_node("PTPSyncOffset")
    assert h.nodemap["Scan3dOperatingMode"].value == "Distance5000mmMultiFreq"
    assert t.tl_stream_nodemap["StreamBufferHandlingMode"].value == "OldestFirst"
    # locked while streaming
    t.start_stream(2)
    with pytest.raises(RuntimeError):
        nm["AcquisitionStartMode"].value = "PTPSync"
    with pytest.raises(RuntimeError):
        nm["TriggerMode"].value = "On"
    assert nm["TriggerMode"].is_writable is False
    t.stop_stream()
    nm["TriggerMode"].value = "On"


def test_missing_node_fault_injection():
    arena_api.mock_reset(small_frames=True,
                         drop_nodes={"triton": ["AcquisitionStartMode", "PTPSyncFrameRate"]})
    h, t = open_pair()
    assert t.nodemap.get_node("AcquisitionStartMode") is None
    with pytest.raises(KeyError):
        t.nodemap["PTPSyncFrameRate"]
    assert h.nodemap["AcquisitionStartMode"].value == "Normal"
    # Triton can still free-run without the dropped nodes
    t.start_stream(3)
    assert collect(t, 1)[0][0] == 1


# ------------------------------------------------------------------------- PTP
def test_ptp_lock_roles_and_clock_domain():
    h, t = open_pair()
    assert h.nodemap["PtpStatus"].value == "Disabled"
    # Before PTP the device clocks are unrelated (random boot offsets, seconds apart)
    assert abs(ptp_now(h) - ptp_now(t)) > 1e9
    t.nodemap["PtpDataSetLatch"].execute()
    assert t.nodemap["PtpOffsetFromMaster"].value == -1
    lock_ptp(h, t, helios_master=True)
    for _ in range(5):
        dh = ptp_now(t) - ptp_now(h)            # slave now on master's clock
        assert abs(dh) < 1_000_000              # jitter ~1 us + Python call latency
    assert t.nodemap["PtpServoStatus"].value == "Locked"
    t.nodemap["PtpDataSetLatch"].execute()
    assert abs(t.nodemap["PtpOffsetFromMaster"].value) < 10_000


def test_ptp_bmca_lower_serial_and_never_locks():
    h, t = open_pair()
    for d in (h, t):                             # both may be master -> BMCA
        d.nodemap["PtpSlaveOnly"].value = False
        d.nodemap["PtpEnable"].value = True
    time.sleep(LOCK_S + 0.2)
    lower_is_helios = h.nodemap["DeviceSerialNumber"].value < t.nodemap["DeviceSerialNumber"].value
    assert (h.nodemap["PtpStatus"].value, t.nodemap["PtpStatus"].value) == \
        (("Master", "Slave") if lower_is_helios else ("Slave", "Master"))
    arena_api.mock_reset(ptp_never_locks=True, ptp_lock_s=0.2)
    h, t = open_pair()
    h.nodemap["PtpEnable"].value = True
    t.nodemap["PtpSlaveOnly"].value = True
    t.nodemap["PtpEnable"].value = True
    time.sleep(0.5)
    assert h.nodemap["PtpStatus"].value == "Master"
    assert t.nodemap["PtpStatus"].value == "Listening"


# ------------------------------------------------------------------------- PTPSync
def setup_ptpsync(h, t, fps=10.0, limits=True, triton_exposure_us=5000.0):
    for d in (h, t):
        nm = d.nodemap
        nm["AcquisitionMode"].value = "Continuous"
        nm["AcquisitionStartMode"].value = "PTPSync"
        nm["AcquisitionFrameRateEnable"].value = True
        nm["AcquisitionFrameRate"].value = nm["AcquisitionFrameRate"].max
        nm["PTPSyncFrameRate"].value = fps
        d.tl_stream_nodemap["StreamAutoNegotiatePacketSize"].value = True
        if limits:
            nm["DeviceLinkThroughputLimitMode"].value = "On"
    if limits:
        h.nodemap["DeviceLinkThroughputLimit"].value = 48_000_000
        t.nodemap["DeviceLinkThroughputLimit"].value = 60_000_000
    t.nodemap["ExposureAuto"].value = "Off"
    t.nodemap["ExposureTime"].value = triton_exposure_us


def test_ptpsync_frames_aligned_at_configured_fps():
    h, t = open_pair()
    lock_ptp(h, t)
    setup_ptpsync(h, t, fps=10.0)
    h.start_stream(10)
    t.start_stream(10)
    assert h.nodemap["AcquisitionFrameRate"].value == pytest.approx(10.0)
    hs, ts = collect_for([h, t], 1.6)
    P = 100_000_000
    hk = {ts_ // P: ts_ for _, ts_, _ in hs}
    tk = {ts_ // P: ts_ for _, ts_, _ in ts}
    common = sorted(set(hk) & set(tk))
    assert len(common) >= 12
    offsets_us = [(tk[k] - hk[k]) / 1000 for k in common]
    # Both expose on the same PTP grid; timestamps differ by LUCID's documented ~18 us
    # bookkeeping offset (TRI ~+41 us, HLT ~+23 us after the grid time) + jitter.
    assert abs(statistics.median(offsets_us) - 17.7) < 3
    assert max(abs(o - 17.7) for o in offsets_us) < 8
    assert all(abs(hk[k] % P - 23_400) < 5_000 for k in common)
    periods = np.diff(sorted(hk.values()))
    assert np.all(np.abs(periods - P) < 10_000)              # 10 fps on the PTP clock
    assert not any(inc for _, _, inc in hs + ts)             # throttled: no loss


def test_ptpsync_no_frames_without_lock():
    arena_api.mock_reset(small_frames=True, ptp_never_locks=True, ptp_lock_s=0.3)
    h, t = open_pair()
    h.nodemap["PtpEnable"].value = True
    t.nodemap["PtpSlaveOnly"].value = True
    t.nodemap["PtpEnable"].value = True
    setup_ptpsync(h, t)
    h.start_stream(4)
    t.start_stream(4)
    time.sleep(0.6)
    assert len(collect(h, 2)) == 2                           # master runs on its own grid
    with pytest.raises(TimeoutError):
        t.get_buffer(timeout=300)                            # slave never locks -> no frames


def test_ptpsync_offset_shifts_phase():
    h, t = open_pair()
    lock_ptp(h, t)
    setup_ptpsync(h, t)
    h.nodemap["PTPSyncOffset"].value = 2_000_000             # 2 ms
    h.start_stream(4)
    ts = [x[1] for x in collect(h, 3)]
    assert all(abs(x % 100_000_000 - 2_023_400) < 5_000 for x in ts)


# ------------------------------------------------------------------------- action commands
def setup_action(d, delay_us=0.0):
    nm = d.nodemap
    nm["TriggerSelector"].value = "FrameStart"
    nm["TriggerMode"].value = "On"
    nm["TriggerSource"].value = "Action0"
    nm["TriggerDelay"].value = delay_us
    nm["ActionUnconditionalMode"].value = "On"
    nm["ActionSelector"].value = 0
    nm["ActionDeviceKey"].value = 1
    nm["ActionGroupKey"].value = 1
    nm["ActionGroupMask"].value = 1
    nm["DeviceLinkThroughputLimitMode"].value = "On"


def fire(execute_time):
    tl = system.tl_system_nodemap
    tl["ActionCommandDeviceKey"].value = 1
    tl["ActionCommandGroupKey"].value = 1
    tl["ActionCommandGroupMask"].value = 1
    tl["ActionCommandTargetIP"].value = 0xFFFFFFFF
    tl["ActionCommandExecuteTime"].value = execute_time
    tl["ActionCommandFireCommand"].execute()


def test_action_command_timing_with_trigger_delay():
    h, t = open_pair()
    lock_ptp(h, t)
    setup_action(h)
    setup_action(t, delay_us=6850.0)
    t.nodemap["ExposureAuto"].value = "Off"
    t.nodemap["ExposureTime"].value = 5000.0
    h.start_stream(8)
    t.start_stream(8)
    T0 = (ptp_now(h) // 1_000_000 + 50) * 1_000_000          # 50 ms ahead, on a ms grid
    times = [T0 + k * 100_000_000 for k in range(4)]
    for T in times:                                          # queue of 4 fits ActionQueueSize
        fire(T)
    hb = [x[1] for x in collect(h, 4)]
    tb = [x[1] for x in collect(t, 4)]
    dh = [(ts - T) / 1000 for ts, T in zip(hb, times)]
    dt = [(ts - T) / 1000 for ts, T in zip(tb, times)]
    assert all(abs(x - 23.4) < 5 for x in dh), dh             # LUCID: HLT ~ T + 23 us
    assert all(abs(x - (41.1 + 6850)) < 6 for x in dt), dt    # TRI ~ T + 41 us + delay
    assert h.mock_stats().get("actions_dropped_queue_full", 0) == 0


def test_action_queue_overflow_and_late_actions():
    h, t = open_pair()
    lock_ptp(h, t)
    setup_action(h)
    h.start_stream(8)
    T0 = ptp_now(h) + 300_000_000
    for k in range(6):                                       # ActionQueueSize = 4
        fire(T0 + k * 50_000_000)
    assert h.mock_stats()["actions_dropped_queue_full"] == 2
    assert len(collect(h, 4)) == 4
    fire(ptp_now(h) - 10_000_000)                            # already in the past
    collect(h, 1)                                            # default: fires immediately
    assert h.mock_stats()["actions_late"] == 1
    arena_api.mock_config().late_action_behavior = "drop"
    fire(ptp_now(h) - 10_000_000)
    with pytest.raises(TimeoutError):
        h.get_buffer(timeout=200)
    assert h.mock_stats()["actions_dropped_late"] == 1


def test_trigger_while_busy_is_dropped_and_software_trigger():
    h, t = open_pair()
    nm = h.nodemap
    nm["TriggerMode"].value = "On"
    nm["TriggerSource"].value = "Software"
    h.start_stream(4)
    nm["TriggerSoftware"].execute()
    nm["TriggerSoftware"].execute()                          # Helios frame takes 20 ms: busy
    b = h.get_buffer(timeout=1000)
    assert b.frame_id == 1
    h.requeue_buffer(b)
    with pytest.raises(TimeoutError):
        h.get_buffer(timeout=150)
    assert h.mock_stats()["busy_drops"] == 1


# ------------------------------------------------------------------------- exposure drag
def test_dark_scene_auto_exposure_drags_fps_and_upper_limit_fixes_it():
    arena_api.mock_reset(small_frames=True, scene="dark", device_missing="helios")
    (t,) = system.create_device()
    nm = t.nodemap
    nm["AcquisitionFrameRateEnable"].value = True
    nm["AcquisitionFrameRate"].value = 10.0
    nm["ChunkModeActive"].value = True
    nm["ChunkSelector"].value = "ExposureTime"
    nm["ChunkEnable"].value = True
    assert nm["ExposureAuto"].value == "Continuous"
    t.start_stream(4)
    b1 = t.get_buffer(timeout=4000)
    exp1 = b1.get_chunk("ChunkExposureTime").value
    ts1 = b1.timestamp_ns
    t.requeue_buffer(b1)
    b2 = t.get_buffer(timeout=4000)
    fps_dark = 1e9 / (b2.timestamp_ns - ts1)
    t.requeue_buffer(b2)
    t.stop_stream()
    assert exp1 > 1e6 and fps_dark < 1.0                     # the user's ~0.6 fps bug
    # Fix: cap auto exposure. Switching the limit auto Off alone keeps the full range.
    with pytest.raises(RuntimeError):
        nm["ExposureAutoUpperLimit"].value = 20_000.0
    nm["ExposureAutoLimitAuto"].value = "Off"
    assert nm["ExposureAutoUpperLimit"].value == 10_000_000.0
    nm["ExposureAutoUpperLimit"].value = 20_000.0
    t.start_stream(4)
    frames = collect(t, 6)
    fps = 1e9 / statistics.median(np.diff([f[1] for f in frames]))
    assert fps == pytest.approx(10.0, rel=0.01)
    assert nm["ChunkExposureTime"].value == 20_000.0
    assert nm["Gain"].value > 0                              # gain auto takes up the rest


# ------------------------------------------------------------------------- bandwidth
def test_incomplete_buffers_without_throughput_limit():
    arena_api.mock_reset(small_frames=True, ptp_lock_s=LOCK_S, switch_overflow_incomplete_prob=1.0)
    h, t = open_pair()
    lock_ptp(h, t)
    setup_ptpsync(h, t, limits=False, triton_exposure_us=2000.0)
    h.start_stream(10)
    t.start_stream(10)
    hs, ts = collect_for([h, t], 1.2)
    n_inc = sum(inc for _, _, inc in hs + ts)
    assert len(hs) >= 8 and len(ts) >= 8
    assert n_inc >= 0.5 * min(len(hs), len(ts))              # simultaneous bursts overflow
    for d in (h, t):
        d.stop_stream()
    for d in (h, t):                                         # throttled (108 MB/s total)
        d.nodemap["DeviceLinkThroughputLimitMode"].value = "On"
    h.nodemap["DeviceLinkThroughputLimit"].value = 48_000_000
    t.nodemap["DeviceLinkThroughputLimit"].value = 60_000_000
    h.start_stream(10)
    t.start_stream(10)
    hs, ts = collect_for([h, t], 0.8)
    assert len(hs) >= 5 and not any(inc for _, _, inc in hs + ts)


def test_packet_loss_fault():
    arena_api.mock_reset(small_frames=True, packet_loss_prob=1.0, device_missing="triton")
    (h,) = system.create_device()
    h.start_stream(3)
    b = h.get_buffer(timeout=1000)
    assert b.is_incomplete and b.size_filled < b.payload_size
    h.requeue_buffer(b)


# ------------------------------------------------------------------------- buffers
def test_buffer_contents_requeue_copy_and_pool_loss():
    h, t = open_pair()
    h.start_stream(2)                                        # freerun 30 fps, 2 buffers
    b = h.get_buffer(timeout=1000)
    assert b.pixel_format == PixelFormat.Coord3D_ABCY16 and b.bits_per_pixel == 64
    assert b.size_filled == 160 * 120 * 8
    arr = np.ctypeslib.as_array(b.pdata, (b.size_filled,)).view("<u2").reshape(120, 160, 4)
    h.nodemap["Scan3dCoordinateSelector"].value = "CoordinateC"
    z_mm = arr[60, 80, 2] * h.nodemap["Scan3dCoordinateScale"].value \
        + h.nodemap["Scan3dCoordinateOffset"].value
    assert 2500 < z_mm < 3500                                # plane ~3 m away
    cp = BufferFactory.copy(b)
    h.requeue_buffer(b)
    with pytest.raises(RuntimeError):
        b.timestamp_ns                                       # use after requeue
    assert cp.frame_id >= 1 and len(cp.data) == 160 * 120 * 8
    BufferFactory.destroy(cp)
    time.sleep(0.4)                                          # ~12 frames, 2 buffers
    ids = [x[0] for x in collect(h, 3)]
    assert h.mock_stats()["lost_no_buffer"] > 0
    # OldestFirst keeps the two oldest frames; the newer ones were lost -> frame_id gap
    assert ids[1] == ids[0] + 1 and ids[2] > ids[1] + 1
    assert h.tl_stream_nodemap["StreamLostFrameCount"].value > 0


def test_clean_shutdown_joins_threads():
    h, t = open_pair()
    h.start_stream(3)
    t.start_stream(3)
    assert any(th.name.startswith("arena-mock-") for th in threading.enumerate())
    system.destroy_device()
    assert not any(th.name.startswith("arena-mock-") for th in threading.enumerate())
    with pytest.raises(RuntimeError):
        h.nodemap
