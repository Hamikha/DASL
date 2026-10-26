"""Unit tests: NodeKeeper (against mock node maps) and stats."""
from __future__ import annotations

import math

import pytest

from conftest import SYSTEM
from lucid_sync import stats
from lucid_sync.nodes import (ALLOWED_WRITES, MissingNodeError, NodeKeeper, NodeReport,
                              NodeWriteError, restore_writes)


class RaisingNodeMap:
    """A node map whose get_node RAISES for unknown names (real-SDK behaviour unknown)."""

    def get_node(self, name):
        raise KeyError(name)


@pytest.fixture
def triton(mock):
    devs = SYSTEM.create_device()
    by = {d.nodemap["DeviceModelName"].value: d for d in devs}
    rep = NodeReport()
    # GevSCPSPacketSize is NOT on the production allow-list; allowed here for the snap test
    return NodeKeeper(by["TRI032S-C"].nodemap, "triton", "device", rep,
                      allow=ALLOWED_WRITES | {"GevSCPSPacketSize"})


def test_get_and_missing_both_styles(triton):
    assert triton.get("DeviceModelName") == "TRI032S-C"
    assert triton.get("NoSuchNode", default=42) == 42            # get_node -> None
    triton.get("NoSuchNode")                                      # logged once only
    other = NodeKeeper(RaisingNodeMap(), "x", "device", triton.report)
    assert other.get("Anything", default="d") == "d"              # get_node raises
    assert not other.has("Anything")
    names = [(m["camera"], m["node"]) for m in triton.missing]
    assert names == [("triton", "NoSuchNode"), ("x", "Anything")]
    with pytest.raises(MissingNodeError):
        triton.get("NoSuchNode", required=True)


def test_set_enum_checks_entries(triton):
    assert triton.set("TriggerSource", "Action0")
    assert triton.get("TriggerSource") == "Action0"
    assert not triton.set("TriggerSource", "Line7")               # not an entry: refused
    assert triton.get("TriggerSource") == "Action0"
    f = triton.failed[-1]
    assert f["node"] == "TriggerSource" and "not available" in f["error"]
    with pytest.raises(NodeWriteError):
        triton.set("TriggerSource", "Line7", required=True)
    assert "Action0" in triton.entries("TriggerSource")
    assert triton.entries("NoSuchNode") == []


def test_set_range_and_clamp(triton):
    assert not triton.set("AcquisitionFrameRate", 5.0)            # Enable is False -> RO
    assert "not writable" in triton.failed[-1]["error"]
    assert triton.set("AcquisitionFrameRateEnable", True)
    lo, hi, _ = triton.limits("AcquisitionFrameRate")
    assert not triton.set("AcquisitionFrameRate", hi + 100)        # out of range, no clamp
    assert "outside" in triton.failed[-1]["error"]
    assert triton.set("AcquisitionFrameRate", hi + 100, clamp=True)
    assert math.isclose(triton.get("AcquisitionFrameRate"), hi)
    assert triton.report.clamped[-1]["written"] == hi
    # integer increment snapping (GevSCPSPacketSize inc 4 from min 576)
    assert triton.set("GevSCPSPacketSize", 9001, clamp=True)
    assert (triton.get("GevSCPSPacketSize") - 576) % 4 == 0


def test_allow_list_refuses_everything_else(mock):
    devs = SYSTEM.create_device()
    keeper = NodeKeeper(devs[0].nodemap, "x", "device", NodeReport())   # production list
    for name in ("UserSetSave", "UserSetSelector", "GevPersistentIPAddress",
                 "GevCurrentIPAddress", "GevSCPSPacketSize", "DeviceReset",
                 "DeviceFirmwareVersion", "LineSource"):
        assert not keeper.set(name, 1)
        assert "allow-list" in keeper.failed[-1]["error"]
    assert not keeper.execute("UserSetSave") and not keeper.execute("TriggerArmed")
    assert keeper.execute("PtpDataSetLatch")
    assert not keeper.report.writes


def test_enum_member_values_and_restore(triton):
    from arena_api.enums import PixelFormat
    assert triton.set("PixelFormat", PixelFormat.Mono8)          # enum member, as capture.py
    assert triton.get("PixelFormat") == "Mono8"
    assert triton.set("PixelFormat", "Mono8")                     # equal -> skipped
    triton.set("AcquisitionFrameRateEnable", True)
    triton.set("AcquisitionFrameRate", 5.0)
    triton.set("TriggerSelector", "FrameStart")
    triton.set("TriggerSource", "Software")
    res = restore_writes(triton.report)
    assert not res["failed"]
    assert triton.get("PixelFormat") == "BayerRG8"
    assert triton.get("AcquisitionFrameRateEnable") is False
    assert triton.get("TriggerSource") == "Line0"
    assert restore_writes(triton.report) is res                  # idempotent
    # 4 writes: TriggerSelector already read FrameStart, so that write was skipped
    assert len(triton.report.to_dict()["node_writes"]) == 4      # kept for run.json


def test_execute_try_set_first_and_skip_equal(triton):
    assert triton.execute("PtpDataSetLatch")
    assert not triton.execute("NoSuchCommand")
    assert triton.try_set_first("AcquisitionStartMode", ["Bogus", "PTPSync", "Normal"]) == "PTPSync"
    assert triton.try_set_first("AcquisitionStartMode", ["Bogus"]) is None
    n_writes = len(triton.report.writes)
    assert triton.set("AcquisitionStartMode", "PTPSync")         # already equal -> skipped
    assert len(triton.report.writes) == n_writes
    assert triton.report.to_dict()["node_writes"][-1]["new"] == "PTPSync"


# --------------------------------------------------------------------------- stats
def test_summarize_empty_and_values():
    e = stats.summarize([])
    assert e["n"] == 0 and e["mean"] is None and e["p95_abs"] is None
    assert stats.summarize([None, float("nan")])["n"] == 0
    s = stats.summarize([1.0, -2.0, 3.0, None])
    assert s["n"] == 3 and s["min"] == -2.0 and s["max"] == 3.0 and s["max_abs"] == 3.0
    assert math.isclose(s["mean"], 2 / 3, abs_tol=1e-3) and s["median"] == 1.0
    big = stats.summarize(list(range(-100, 101)))
    assert big["p95_abs"] == 95.0 and big["std"] > 0


def test_fps_duration_missing_gaps():
    ts = [10**18 + i * 100_000_000 for i in range(11)]
    assert math.isclose(stats.fps_from_timestamps(ts), 10.0)
    assert stats.fps_from_timestamps([1]) is None and stats.fps_from_timestamps([]) is None
    assert math.isclose(stats.duration_s(ts), 1.0)
    assert stats.count_missing({1, 2, 4, 7}, 1, 7) == 3
    assert stats.count_missing({1, 2}, 5, 4) == 0
    assert stats.count_missing({1, 2, 4}, 1, 4, expected=[1, 3, 4, 9]) == 1
    assert stats.frame_id_gaps([1, 2, 3, 6, 7, 7, 10]) == 4
    assert stats.frame_id_gaps([]) == 0
