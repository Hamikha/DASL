"""READ-ONLY probe of both cameras - step 1 of the on-rig checklist.

For every node named in research/A_lucid_capabilities.md and DESIGN.md this prints and saves
(JSON + text) whether it exists, its type, access (readable / writable), value, range and
enum entries. It also records ``dir(system)`` and whether ``system.tl_system_nodemap`` and
the ActionCommand* host nodes exist (A, open question 4).

It NEVER writes a node and NEVER executes a command (not even PtpDataSetLatch), so it cannot
change the cameras. Consequence: selector-dependent nodes (TriggerMode per TriggerSelector,
ChunkEnable per ChunkSelector, LineMode per LineSelector, ...) are shown only for the
currently selected entry.

Usage:
    python probe_cameras.py                 # real cameras
    LUCID_MOCK=1 python probe_cameras.py    # mock SDK
    python probe_cameras.py --out probe_out --all   # --all also dumps every node in each map
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lucid_sync.methods import find_cameras  # noqa: E402
from lucid_sync.nodes import node_kind  # noqa: E402

log = logging.getLogger("probe_cameras")

DEVICE_NODES = {
    "identity": ["DeviceVendorName", "DeviceModelName", "DeviceSerialNumber",
                 "DeviceFirmwareVersion", "DeviceLinkSpeed"],
    "image": ["Width", "Height", "PixelFormat"],
    "helios_tof": ["Scan3dOperatingMode", "ExposureTimeSelector", "Scan3dCoordinateSelector",
                   "Scan3dCoordinateScale", "Scan3dCoordinateOffset"],
    "acquisition": ["AcquisitionMode", "AcquisitionStartMode", "AcquisitionFrameRateEnable",
                    "AcquisitionFrameRate", "PTPSyncFrameRate", "PTPSyncOffset",
                    "AcquisitionBurstFrameCount", "LinkAcquisitionFrameRateLimit"],
    "ptp": ["PtpEnable", "PtpSlaveOnly", "PtpStatus", "PtpClockAccuracy", "PtpServoStatus",
            "PtpDataSet", "PtpDataSetLatch", "PtpDataSetLatchValue", "PtpOffsetFromMaster",
            "PtpClockID", "PtpParentClockID", "PtpGrandmasterClockID",
            "Timestamp", "TimestampReset", "TimestampLatch", "TimestampLatchValue"],
    "trigger": ["TriggerSelector", "TriggerMode", "TriggerSource", "TriggerActivation",
                "TriggerDelay", "TriggerOverlap", "TriggerLatency", "TriggerArmed",
                "TriggerSoftware"],
    "action": ["ActionUnconditionalMode", "ActionSelector", "ActionDeviceKey",
               "ActionGroupKey", "ActionGroupMask", "ActionQueueSize", "ActionMode",
               "ActionSendTriggerSource", "ActionPTPTimestampMode", "ActionSendPTPTimestamp"],
    "transfer_control": ["TransferControlMode", "TransferOperationMode", "TransferStart",
                         "TransferStop", "TransferQueueMaxBlockCount",
                         "TransferQueueCurrentBlockCount"],
    "gpio": ["LineSelector", "LineMode", "LineSource", "LineInverter", "LineStatus",
             "VoltageExternalEnable"],
    "exposure": ["ExposureAuto", "ExposureTime", "ExposureAutoLimitAuto",
                 "ExposureAutoLowerLimit", "ExposureAutoUpperLimit", "ExposureAutoAlgorithm",
                 "ExposureAutoDamping", "TargetBrightness", "GainAuto", "Gain"],
    "bandwidth": ["DeviceLinkThroughputLimitMode", "DeviceLinkThroughputLimit",
                  "DeviceLinkThroughputReserve", "DeviceStreamChannelPacketSize",
                  "GevSCPSPacketSize", "GevSCPD", "GevSCFTD"],
    "chunks": ["ChunkModeActive", "ChunkSelector", "ChunkEnable", "ChunkExposureTime",
               "ChunkCRC", "ChunkGain", "ChunkTimestamp"],
    "events": ["EventExposureStartTimestamp", "EventExposureEndTimestamp"],
}
STREAM_NODES = ["StreamAutoNegotiatePacketSize", "StreamPacketResendEnable",
                "StreamBufferHandlingMode", "PacketResendWindowFrameCount",
                "StreamDeliveredFrameCount", "StreamLostFrameCount",
                "StreamIncompleteFrameCount"]
TL_DEVICE_NODES = ["DeviceModelName", "DeviceSerialNumber", "GevDeviceIPAddress",
                   "GevDeviceSubnetMask", "GevDeviceMACAddress"]
TL_SYSTEM_NODES = ["ActionCommandDeviceKey", "ActionCommandGroupKey", "ActionCommandGroupMask",
                   "ActionCommandTargetIP", "ActionCommandExecuteTime",
                   "ActionCommandFireCommand"]


def _safe(fn: Any) -> Any:
    try:
        v = fn()
    except Exception as exc:
        return f"<error: {type(exc).__name__}: {exc}>"
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        return [x if isinstance(x, (bool, int, float, str)) else str(x) for x in v]
    return str(getattr(v, "name", v))


def describe_node(nodemap: Any, name: str) -> dict:
    """Everything readable about one node, WITHOUT writing or executing anything."""
    try:
        node = nodemap.get_node(name)
    except Exception as exc:
        return dict(exists=False, error=f"get_node raised {type(exc).__name__}: {exc}")
    if node is None:
        return dict(exists=False)
    kind = node_kind(node)
    d: dict = dict(exists=True, kind=kind,
                   interface_type=_safe(lambda: node.interface_type),
                   readable=_safe(lambda: node.is_readable),
                   writable=_safe(lambda: node.is_writable),
                   access_mode=_safe(lambda: node.access_mode))
    if kind != "command" and d["readable"] is True:
        d["value"] = _safe(lambda: node.value)
    if kind in ("int", "float"):
        d.update(min=_safe(lambda: node.min), max=_safe(lambda: node.max),
                 inc=_safe(lambda: node.inc), unit=_safe(lambda: node.unit))
    if kind == "enum":
        d["entries"] = _safe(lambda: node.enumentry_names)
    return d


def describe_map(nodemap: Any, names: list[str], dump_all: bool = False) -> dict:
    out = {n: describe_node(nodemap, n) for n in names}
    if dump_all:
        try:
            extra = [n for n in nodemap.feature_names if n not in out]
        except Exception:
            extra = []
        for n in extra:
            out[n] = describe_node(nodemap, n)
    return out


def probe(system: Any, sdk_info: dict, dump_all: bool = False) -> dict:
    """Collect the probe data (opens and releases the devices; writes no node)."""
    result: dict = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), sdk=sdk_info,
                        dir_system=[a for a in dir(system) if not a.startswith("_")])
    infos = list(system.device_infos)
    result["device_infos"] = [{k: _safe(lambda v=v: v) for k, v in i.items()} for i in infos]
    try:
        tl = system.tl_system_nodemap
        result["tl_system_nodemap"] = dict(
            exists=True, nodes=describe_map(tl, TL_SYSTEM_NODES, dump_all))
    except Exception as exc:
        result["tl_system_nodemap"] = dict(exists=False,
                                           error=f"{type(exc).__name__}: {exc}")
    try:
        h_info, t_info = find_cameras(system)
        wanted = {"helios": h_info, "triton": t_info}
    except Exception as exc:          # probe whatever is there
        log.error("%s", exc)
        result["find_cameras_error"] = str(exc)
        wanted = {f"device{i}": info for i, info in enumerate(infos)}
    result["cameras"] = {}
    for name, info in wanted.items():
        devs = system.create_device(device_infos=[info])
        dev = devs[0]
        try:
            cam: dict = dict(info={k: _safe(lambda v=v: v) for k, v in info.items()})
            cam["nodemap"] = {grp: describe_map(dev.nodemap, names)
                              for grp, names in DEVICE_NODES.items()}
            if dump_all:
                listed = {n for names in DEVICE_NODES.values() for n in names}
                try:
                    rest = [n for n in dev.nodemap.feature_names if n not in listed]
                except Exception:
                    rest = []
                cam["nodemap"]["other"] = describe_map(dev.nodemap, rest)
            cam["tl_stream_nodemap"] = describe_map(dev.tl_stream_nodemap, STREAM_NODES,
                                                    dump_all)
            try:
                cam["tl_device_nodemap"] = describe_map(dev.tl_device_nodemap,
                                                        TL_DEVICE_NODES, dump_all)
            except Exception as exc:
                cam["tl_device_nodemap"] = {"error": str(exc)}
            result["cameras"][name] = cam
        finally:
            system.destroy_device(dev)
    return result


def to_text(result: dict) -> str:
    lines = [f"LUCID probe {result['time']}  sdk={result['sdk']}", "",
             "device_infos:"]
    lines += [f"  {i}" for i in result["device_infos"]]
    tl = result["tl_system_nodemap"]
    lines += ["", f"system.tl_system_nodemap exists: {tl['exists']}"]
    for n, d in tl.get("nodes", {}).items():
        lines.append(f"  {n:34s} {_fmt(d)}")
    lines += ["", "dir(system): " + ", ".join(result["dir_system"])]
    for cam, data in result["cameras"].items():
        lines += ["", "=" * 78, f"{cam}: {data['info'].get('model')} "
                  f"sn={data['info'].get('serial')} ip={data['info'].get('ip')}", "=" * 78]
        for grp, nodes in data["nodemap"].items():
            lines.append(f"[{grp}]")
            lines += [f"  {n:34s} {_fmt(d)}" for n, d in nodes.items()]
        for mp in ("tl_stream_nodemap", "tl_device_nodemap"):
            lines.append(f"[{mp}]")
            lines += [f"  {n:34s} {_fmt(d)}" for n, d in data[mp].items()
                      if isinstance(d, dict)]
    return "\n".join(lines) + "\n"


def _fmt(d: dict) -> str:
    if not d.get("exists"):
        return "MISSING" + (f" ({d['error']})" if d.get("error") else "")
    acc = ("R" if d.get("readable") is True else "-") + ("W" if d.get("writable") is True else "-")
    s = f"{d.get('kind', '?'):7s} {acc}"
    if "value" in d:
        s += f" = {d['value']!r}"
    if "min" in d:
        s += f"  [{d['min']} .. {d['max']}] inc={d['inc']} {d.get('unit') or ''}"
    if "entries" in d:
        s += f"  entries={d['entries']}"
    return s


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=None, help="output folder (default probe_<time>)")
    ap.add_argument("--all", action="store_true", help="also dump every node of each map")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from lucid_sync.sdk import load_system   # noqa: PLC0415
    system, sdk_info = load_system()
    result = probe(system, sdk_info, dump_all=args.all)
    out = Path(args.out or time.strftime("probe_%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "probe.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    text = to_text(result)
    (out / "probe.txt").write_text(text, encoding="utf-8")
    print(text)
    log.info("saved %s and %s", out / "probe.json", out / "probe.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
