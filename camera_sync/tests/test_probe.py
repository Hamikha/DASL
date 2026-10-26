"""probe_cameras.py must be strictly READ-ONLY: no node write, no command execution."""
from __future__ import annotations

import json

from conftest import SDK_INFO, SYSTEM

import probe_cameras
from arena_api import _nodes


def snapshot() -> dict:
    """Stored values of every node of every node map (mock internals)."""
    maps = {"tl_system": SYSTEM.tl_system_nodemap}
    for key, cam in SYSTEM.mock_world.cameras.items():
        maps[f"{key}/device"] = cam.nodemap
        maps[f"{key}/stream"] = cam.tl_stream_nodemap
        maps[f"{key}/tl_device"] = cam.tl_device_nodemap
    return {m: {n.name: dict(n._values) for n in nm} for m, nm in maps.items()}


def test_probe_writes_nothing(mock, monkeypatch, tmp_path):
    # Make the cameras non-default first, so "unchanged" is a meaningful check.
    devs = SYSTEM.create_device()
    for d in devs:
        d.nodemap["AcquisitionFrameRateEnable"].value = True
        d.nodemap["PtpEnable"].value = True
    SYSTEM.destroy_device()

    writes: list[str] = []
    orig = _nodes.Node.value
    monkeypatch.setattr(_nodes.Node, "value", property(
        orig.fget, lambda self, v: writes.append(self.name)))
    monkeypatch.setattr(_nodes.CommandNode, "execute",
                        lambda self: writes.append(f"execute {self.name}"))
    before = snapshot()
    result = probe_cameras.probe(SYSTEM, SDK_INFO, dump_all=True)
    assert writes == []
    assert snapshot() == before

    cams = result["cameras"]
    assert set(cams) == {"helios", "triton"}
    acq = cams["helios"]["nodemap"]["acquisition"]
    assert "PTPSync" in acq["AcquisitionStartMode"]["entries"]
    assert acq["PTPSyncOffset"]["exists"] and not cams["triton"]["nodemap"]["acquisition"][
        "PTPSyncOffset"]["exists"]
    assert cams["triton"]["nodemap"]["ptp"]["PtpDataSetLatch"]["kind"] == "command"
    assert result["tl_system_nodemap"]["exists"]
    assert result["tl_system_nodemap"]["nodes"]["ActionCommandFireCommand"]["exists"]
    assert "tl_system_nodemap" in result["dir_system"]
    json.dumps(result)


def test_probe_cli_saves_json_and_txt(mock, tmp_path):
    out = tmp_path / "probe"
    assert probe_cameras.main(["--out", str(out)]) == 0
    data = json.loads((out / "probe.json").read_text())
    assert data["sdk"]["mock"] is True
    assert "AcquisitionStartMode" in (out / "probe.txt").read_text()
