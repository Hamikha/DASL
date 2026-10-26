"""Camera discovery, common configuration and the five sync methods with fallback.

Source tags in comments refer to research/A_lucid_capabilities.md:
  [HT]   LUCID app note "Helios2 and Triton Sync" + its C++ sample
         research/dl/Cpp_HLTRGB_3_Overlay_PTP_SAC_v7.cpp (cited by line numbers)
  [SAC]  LUCID app note "Multi Camera Synchronization using PTP and Scheduled Action Commands"
  [BW]   LUCID app note "Bandwidth Sharing in Multi-Camera Systems" (contains PTPSync code)
  [MH]   LUCID KB "Using Multiple Helios Cameras Simultaneously" (PTPSyncOffset, TriggerDelay)
  [PTPKB] LUCID KB "Precision Time Protocol (PTP)"
  [TRM-T] LUCID Triton Technical Reference Manual (2019)
  [AE]   LUCID KB "Using Auto Exposure and Auto Gain"
  [C §n] research/C_constraints.md section n (bandwidth, ToF timing, exposure numbers)
  [CAP]  the user's original capture.py (settings it wrote are reproduced here)

Methods, best first:
  ptpsync      PTP + AcquisitionStartMode=PTPSync: the camera firmware exposes on a PTP grid.
  action       PTP + host-scheduled action commands on a fixed PTP grid (ActionScheduler).
  software     host-paced TriggerSoftware, as in capture.py ("trigger" mode); PTP timestamps
               if PTP is locked, otherwise independent clocks.
  freerun_ptp  both cameras free-run at FRAME_RATE, PTP on (shared clock, not phase locked).
  freerun      legacy baseline: free-run, no PTP (independent clocks).

Camera nodes survive between sessions while the camera stays powered (a previous run may have
left PTPSync or trigger mode on). Each method therefore writes the acquisition/trigger nodes
it depends on and then READS BACK the critical ones (verify_state); a mismatch makes the
method fail with a reason and the fallback continues. Other per-selector trigger state (e.g.
TriggerMode under FrameBurstStart) is NOT reset - only FrameStart is used.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from .nodes import NodeError, NodeKeeper, NodeReport
from .ptp import (HELIOS, TRITON, MethodUnavailable, PtpMonitor, PtpNotLocked, PtpReport,
                  SyncSetupError, disable_ptp, enable_ptp, ptp_now_ns)
from .settings import METHODS, SyncSettings
from .triggers import ActionScheduler, ScheduledAction, SoftwareTrigger, _ClockMap

log = logging.getLogger(__name__)

__all__ = [  # re-exported for backwards compatibility of imports
    "ActionScheduler", "ScheduledAction", "SoftwareTrigger", "_ClockMap", "PtpMonitor",
    "PtpReport", "PtpNotLocked", "MethodUnavailable", "SyncSetupError", "enable_ptp",
    "disable_ptp", "ptp_now_ns",
]

HELIOS_MODEL_PREFIXES = ("HTP", "HLT")      # Helios2+ / Helios2 [HT: "either ... can be used"]
TRITON_MODEL_PREFIXES = ("TRI",)
BROADCAST_IP = 0xFFFFFFFF                    # ActionCommandTargetIP used by LUCID [SAC][HT]


class CameraNotFound(RuntimeError):
    pass


# =========================================================================== cameras
@dataclass
class Cam:
    """One opened camera with keepers for its node maps."""

    name: str                 # "helios" | "triton"
    device: Any
    info: dict
    nm: NodeKeeper            # device.nodemap
    sn: NodeKeeper            # device.tl_stream_nodemap
    td: NodeKeeper | None     # device.tl_device_nodemap
    identity: dict = field(default_factory=dict)


def find_cameras(system: Any, helios_serial: str = "",
                 triton_serial: str = "") -> tuple[dict, dict]:
    """Return (helios_info, triton_info) from ``system.device_infos``.

    Helios: model starts with "HTP" (Helios2+) or "HLT" (Helios2). Triton: model starts with
    "TRI". Optional serial filters (capture.py HELIOS_SERIAL / TRITON_SERIAL). Raises
    CameraNotFound listing every device that WAS found.
    """
    infos = list(system.device_infos)
    found = [f"{i.get('model', '?')} sn={i.get('serial', '?')} ip={i.get('ip', '?')}"
             for i in infos]

    def pick(prefixes: tuple, serial: str) -> list[dict]:
        return [i for i in infos if str(i.get("model", "")).upper().startswith(prefixes)
                and (not serial or str(i.get("serial")) == str(serial))]

    helios, triton = pick(HELIOS_MODEL_PREFIXES, helios_serial), pick(TRITON_MODEL_PREFIXES,
                                                                       triton_serial)
    missing = [n for n, lst in (("Helios (HTP*/HLT*)", helios), ("Triton (TRI*)", triton))
               if not lst]
    if missing:
        raise CameraNotFound(f"camera(s) not found: {', '.join(missing)}. "
                             f"Devices found ({len(infos)}): {found or 'none'}")
    for name, lst in (("Helios", helios), ("Triton", triton)):
        if len(lst) > 1:
            log.warning("%d %s cameras found, using the first: %s", len(lst), name, found)
    return helios[0], triton[0]


def read_identity(keeper: NodeKeeper, device_info: dict | None = None) -> dict:
    """model, serial, firmware (DeviceFirmwareVersion) and ip of a camera (read only)."""
    info = device_info or {}
    return dict(model=keeper.get("DeviceModelName", info.get("model")),
                serial=keeper.get("DeviceSerialNumber", info.get("serial")),
                firmware=keeper.get("DeviceFirmwareVersion", info.get("version")),
                ip=info.get("ip"))


def open_cameras(system: Any, report: NodeReport, helios_serial: str = "",
                 triton_serial: str = "", retries: int = 1,
                 retry_s: float = 5.0) -> tuple[Cam, Cam]:
    """Find both cameras (retrying like capture.py), create the devices, wrap node maps."""
    import time
    for attempt in range(max(1, retries)):
        try:
            h_info, t_info = find_cameras(system, helios_serial, triton_serial)
            break
        except CameraNotFound as exc:
            if attempt + 1 >= retries:
                raise
            log.warning("%s - retrying in %.0f s (%d/%d)", exc, retry_s, attempt + 1, retries)
            time.sleep(retry_s)
    devices = list(system.create_device(device_infos=[h_info, t_info]))
    by_serial: dict[str, Any] = {}
    for d in devices:
        try:
            by_serial[str(d.nodemap.get_node("DeviceSerialNumber").value)] = d
        except Exception:
            pass
    h_dev = by_serial.get(str(h_info.get("serial")), devices[0])
    t_dev = by_serial.get(str(t_info.get("serial")), devices[1])

    def wrap(name: str, dev: Any, info: dict) -> Cam:
        td = None
        try:
            td = NodeKeeper(dev.tl_device_nodemap, name, "tl_device", report)
        except Exception:
            pass
        cam = Cam(name, dev, info, NodeKeeper(dev.nodemap, name, "device", report),
                  NodeKeeper(dev.tl_stream_nodemap, name, "stream", report), td)
        cam.identity = read_identity(cam.nm, info)
        log.info("%s: %s", name, cam.identity)
        return cam

    return wrap(HELIOS, h_dev, h_info), wrap(TRITON, t_dev, t_info)


def system_tl_keeper(system: Any, report: NodeReport) -> NodeKeeper | None:
    """Keeper for the host-side Arena TL System node map (ActionCommand* nodes).

    C++: pSystem->GetTLSystemNodeMap() [SAC][HT]. The Python name ``tl_system_nodemap`` is
    NOT CONFIRMED (A, open question 4) - if it does not exist, the action method is
    unavailable and the fallback chain moves on (to "software").
    """
    try:
        nm = system.tl_system_nodemap
    except Exception as exc:
        report.add_missing("host", "tl_system", "system.tl_system_nodemap", "get",
                           error=f"{type(exc).__name__}: {exc}")
        return None
    return NodeKeeper(nm, "host", "tl_system", report)


# =========================================================================== common config
def pixel_format_value(name: str) -> Any:
    """arena_api.enums.PixelFormat.<name> if available (what capture.py writes), else name."""
    try:
        from arena_api.enums import PixelFormat   # noqa: PLC0415
        return getattr(PixelFormat, name)
    except Exception:
        return name


def set_pixel_format(k: NodeKeeper, name: str) -> None:
    """Write PixelFormat like capture.py (enum member), falling back to the entry string."""
    if not k.set("PixelFormat", pixel_format_value(name)):
        k.set("PixelFormat", name, required=True)


def configure_common(helios: Cam, triton: Cam, s: SyncSettings) -> dict:
    """Settings that apply to every sync method (as capture.py [CAP] plus the exposure cap,
    exposure chunk and buffer handling). Returns the exposure/link info dict.
    Never touches IP settings, UserSets (persistence) or firmware (allow-list in nodes.py).
    """
    h, t = helios.nm, triton.nm
    # ---- Helios [CAP configure_helios]
    h.set("AcquisitionMode", "Continuous", required=True)
    set_pixel_format(h, s.HELIOS_PIXEL_FORMAT)
    h.set("Scan3dOperatingMode", s.HELIOS_OPERATING_MODE)
    h.set("ExposureTimeSelector", s.HELIOS_EXPOSURE_SELECTOR)
    if s.HELIOS_CONVERSION_GAIN is not None:
        h.set("ConversionGain", s.HELIOS_CONVERSION_GAIN)
    # ---- Triton [CAP configure_triton]
    t.set("AcquisitionMode", "Continuous", required=True)
    if s.TRITON_FULL_SENSOR:
        # Full sensor, no binning: the calibration was made at full resolution; an ROI or
        # binning left over from an ArenaView session would silently break it [CAP].
        for name in ("BinningHorizontal", "BinningVertical"):
            if t.has(name):
                t.set(name, 1)
        for name in ("OffsetX", "OffsetY"):
            if t.has(name):
                t.set(name, 0)
        for name in ("Width", "Height"):
            mx = t.limits(name)[1]
            if mx is not None:
                t.set(name, int(mx))
    set_pixel_format(t, s.TRITON_PIXEL_FORMAT)
    exposure = configure_triton_exposure(t, s)
    if s.TRITON_BALANCE_WHITE_AUTO is not None:
        t.set("BalanceWhiteAuto", s.TRITON_BALANCE_WHITE_AUTO)
    exposure["chunk_exposure"] = enable_exposure_chunk(t) if s.TRITON_EXPOSURE_CHUNK else False
    # ---- bandwidth: two cameras bursting at line rate overflow a 128-192 KB switch buffer
    # in ~1 ms, so a per-camera throughput limit is mandatory [C §2].
    link = {}
    for cam, limit in ((helios, s.THROUGHPUT_LIMIT_HELIOS), (triton, s.THROUGHPUT_LIMIT_TRITON)):
        if s.THROUGHPUT_LIMIT_ENABLE:
            if cam.nm.set("DeviceLinkThroughputLimitMode", "On"):
                cam.nm.set("DeviceLinkThroughputLimit", int(limit), clamp=True)
        else:
            log.warning("%s: THROUGHPUT_LIMIT_ENABLE is False - link limit left untouched",
                        cam.name)
        # Host stream side [HT][BW][CAP]. capture.py used NewestOnly ("a pair can never be
        # made of one fresh and one old image"); OldestFirst is the alternative [C §2].
        cam.sn.set("StreamAutoNegotiatePacketSize", True)
        cam.sn.set("StreamPacketResendEnable", True)
        cam.sn.set("StreamBufferHandlingMode", s.STREAM_BUFFER_HANDLING)
        speed = cam.nm.get("DeviceLinkSpeed")
        if speed is not None and int(speed) < 125_000_000:
            # Helios2+ needs 802.3at PoE+ and a 1 Gb/s link; some injectors are 10/100 [C §1].
            log.warning("%s: DeviceLinkSpeed = %s B/s (< 125,000,000): not a 1 GbE link!",
                        cam.name, speed)
        link[cam.name] = dict(link_speed_Bps=speed,
                              throughput_limit=cam.nm.get("DeviceLinkThroughputLimit"),
                              max_frame_rate=cam.nm.limits("AcquisitionFrameRate")[1])
    exposure["link"] = link
    exposure["helios_window"] = s.helios_window()
    exposure["helios_center_us"] = s.helios_center_us()
    return exposure


def read_helios_scale_offsets(h: NodeKeeper) -> tuple[float | None, list]:
    """Scan3d scale and per-axis offsets, read exactly as capture.py configure_helios does:
    scale with the current selector, then Scan3dCoordinateSelector = CoordinateA/B/C and
    Scan3dCoordinateOffset for each (the selector writes are recorded and restored)."""
    scale = h.get("Scan3dCoordinateScale")
    offsets = []
    for axis in ("CoordinateA", "CoordinateB", "CoordinateC"):
        h.set("Scan3dCoordinateSelector", axis)
        offsets.append(h.get("Scan3dCoordinateOffset"))
    return (None if scale is None else float(scale)), [
        None if o is None else float(o) for o in offsets]


def configure_triton_exposure(t: NodeKeeper, s: SyncSettings) -> dict:
    """Cap (or fix) the Triton exposure so a dark scene cannot drag the frame rate.

    The user's bug: auto exposure in a dark forest went to ~1.6 s -> ~0.6 fps. Preferred fix
    keeps auto exposure (as capture.py) but caps it: ExposureAutoLimitAuto Off, then
    ExposureAutoUpperLimit = cap [AE] (exact node spellings NOT CONFIRMED, display names
    "Exposure Auto Lower/Upper Limit"). If the limit nodes are not there, fall back to a
    FIXED exposure = cap. Returns dict(mode, cap_us, exposure_us_for_alignment, ...).
    """
    cap = float(s.EXPOSURE_CAP_US)
    info: dict = dict(cap_us=cap, fixed_us=None, mode=None)
    if s.TRITON_FIXED_EXPOSURE_US is not None:
        fixed = float(s.TRITON_FIXED_EXPOSURE_US)
        if fixed > cap:
            log.warning("TRITON_FIXED_EXPOSURE_US %.0f us is above EXPOSURE_CAP_US %.0f us",
                        fixed, cap)
        t.set("ExposureAuto", "Off")
        t.set("ExposureTime", fixed, clamp=True)
        info.update(mode="fixed", fixed_us=t.get("ExposureTime", fixed))
        info["exposure_us_for_alignment"] = info["fixed_us"]
        return info

    t.set("ExposureAuto", "Continuous")       # capture.py; also undoes an earlier "Off"
    capped = False
    if t.has("ExposureAutoLimitAuto"):
        t.set("ExposureAutoLimitAuto", "Off")  # limits are only writable when this is Off
    if t.has("ExposureAutoUpperLimit"):
        lower = t.get("ExposureAutoLowerLimit")
        if lower is not None and float(lower) > cap:
            t.set("ExposureAutoLowerLimit", t.limits("ExposureAutoLowerLimit")[0])
        if t.set("ExposureAutoUpperLimit", cap, clamp=True):
            readback = t.get("ExposureAutoUpperLimit")
            capped = readback is not None and float(readback) <= cap * 1.001
    if capped:
        info["mode"] = "auto_capped"
    else:
        # FALLBACK: no usable auto-exposure limit -> fixed exposure at the cap.
        log.warning("triton: auto-exposure upper limit unavailable -> FALLBACK to fixed "
                    "ExposureTime = %.0f us (ExposureAuto Off)", cap)
        t.set("ExposureAuto", "Off")
        t.set("ExposureTime", cap, clamp=True)
        info.update(mode="fixed_cap_fallback", fixed_us=t.get("ExposureTime", cap))
    # For centre alignment we need ONE exposure value: the fixed one, or the cap (in a dark
    # forest auto exposure sits at the cap; brighter scenes give shorter exposures and a
    # few ms of mis-centring, small against the 6-19 ms ToF window [C §4]).
    info["exposure_us_for_alignment"] = info["fixed_us"] if info["fixed_us"] else cap
    return info


def enable_exposure_chunk(t: NodeKeeper) -> bool:
    """Turn on the ExposureTime chunk so each Triton frame carries its real exposure.

    Nodes ChunkModeActive / ChunkSelector / ChunkEnable / ChunkExposureTime [TRM-T]; the
    Triton lists "exposure time" chunk data [P-TRI]. The ChunkSelector entry name
    "ExposureTime" is NOT CONFIRMED, so we only use it if the camera lists it.
    """
    if not t.has("ChunkModeActive") or "ExposureTime" not in t.entries("ChunkSelector"):
        log.warning("triton: no ExposureTime chunk -> per-frame exposure is NOT measured; "
                    "the configured value (or the cap) is recorded instead and flagged")
        return False
    ok = (t.set("ChunkModeActive", True) and t.set("ChunkSelector", "ExposureTime")
          and t.set("ChunkEnable", True))
    return bool(ok)


# =========================================================================== plans
@dataclass
class SyncPlan:
    """What the chosen method configured; the engine needs it for pairing and run.json."""

    method: str
    clock_domain: str              # "ptp" | "independent"
    pairing: str                   # "grid" | "trigger" | "nearest_ts" | "nearest_host"
    period_ns: float
    lags_ns: dict = field(default_factory=lambda: {HELIOS: 0, TRITON: 0})
    anchor_ns: int | None = None   # grid origin (action: the scheduler's T0)
    alignment_requested: str = "start"
    alignment_used: str = "start"  # "center" | "start" | "none" (free-run)
    alignment_note: str = ""
    planned_lag_ns: int = 0        # commanded Triton-minus-Helios exposure-start lag
    trigger_delay_us: dict = field(default_factory=dict)
    ptpsync_offset_ns: dict = field(default_factory=dict)
    action: dict = field(default_factory=dict)
    state: dict = field(default_factory=dict)      # read-back after configuration
    notes: list = field(default_factory=list)

    @property
    def triton_lag_us(self) -> float:
        return self.planned_lag_ns / 1000.0

    def to_dict(self) -> dict:
        return asdict(self)


def desired_triton_lag_ns(s: SyncSettings, exposure_us: float | None) -> tuple[int, str]:
    """(lag_ns, note): how much later the Triton exposure should start than the Helios frame.

    "center": t_d = Helios centre - exposure/2, so the Triton exposure centre lands on the
              modelled centre of the Helios light collection [C §4: TriggerDelay ~ C - E/2].
              The centre comes from HELIOS_CENTER_US or the (unverified) lucid_sync.timing
              model for the configured mode + exposure selector.
    "start":  0.
    """
    if s.ALIGNMENT != "center":
        return 0, "start alignment requested"
    hc = s.helios_center_us()
    if hc is None:
        return 0, "no Helios timing model for this mode -> start alignment"
    e = float(exposure_us or 0.0)
    lag_ns = int(round((hc - e / 2.0) * 1000.0))
    if abs(lag_ns) >= s.period_ns / 2:
        return 0, f"centre lag {lag_ns / 1e6:.1f} ms >= half a period -> start alignment"
    return lag_ns, f"centre alignment: {hc:.0f} us - {e:.0f}/2 us"


def _alignment(s: SyncSettings, lag_used: int, note: str) -> tuple[str, str]:
    if lag_used:
        return "center", note
    if s.ALIGNMENT == "center":
        log.warning("centre alignment not possible (%s) -> start alignment", note)
    return "start", note


def plan_ptpsync_offsets(lag_ns: int, period_ns: float, helios_has: bool,
                         triton_has: bool) -> tuple[int | None, int | None, int, str]:
    """Translate a desired Triton lag into PTPSyncOffset values.

    Returns (helios_offset_ns, triton_offset_ns, lag_used_ns, note); None = do not write.

    PTPSyncOffset delays a camera's exposures on the PTP grid by that many ns [MH]. It is
    confirmed on Helios2+ FW >= 1.16.0.0 but NOT on the Triton, so both cases are handled:

    * Triton has the node: Triton offset = lag (Helios 0). Triton frame k starts lag after
      Helios frame k.
    * Only the Helios has it: delay the HELIOS instead by (period - lag) mod period. Helios
      frame k then starts at k*P + (P - lag), and Triton frame k+1 at (k+1)*P = Helios frame
      k start + lag. So every Triton exposure still starts `lag` after a Helios exposure; only
      the firmware slot index differs by one, which the pairer handles because it pairs on
      (timestamp - lag). This relies on the grid sitting at PTP multiples of the period on
      BOTH cameras - unverified (review item 20.1).
    * A negative lag (Triton exposure longer than 2 x the Helios centre) is the mirror image.
    * Neither camera has the node: alignment impossible -> lag 0 ("start").

    Offsets that are not needed but exist are written as 0, so a value left over from an
    earlier session cannot silently shift the grid.
    """
    P = int(round(period_ns))
    h0 = 0 if helios_has else None
    t0 = 0 if triton_has else None
    if lag_ns == 0:
        return h0, t0, 0, "start alignment"
    if lag_ns > 0:
        if triton_has:
            return h0, lag_ns, lag_ns, "Triton PTPSyncOffset = lag"
        if helios_has:
            return (P - lag_ns) % P, None, lag_ns, \
                "only Helios has PTPSyncOffset: Helios delayed by (period - lag)"
    else:
        if helios_has:
            return -lag_ns, t0, lag_ns, "negative lag: Helios PTPSyncOffset = -lag"
        if triton_has:
            return None, (P + lag_ns) % P, lag_ns, \
                "negative lag, only Triton has PTPSyncOffset: Triton delayed by period + lag"
    return h0, t0, 0, "no PTPSyncOffset node on either camera: centre alignment impossible"


# =========================================================================== shared helpers
def set_trigger_on(nm: NodeKeeper, source: str) -> None:
    """FrameStart trigger from `source`, in LUCID's order: TriggerSelector -> TriggerSource
    -> TriggerMode ([HT] Cpp_HLTRGB_3_Overlay_PTP_SAC_v7.cpp lines 131-133; same in A
    §(b)(iii)). Choosing the source before arming is also the SFNC recommendation."""
    nm.set("TriggerSelector", "FrameStart", required=True)
    nm.set("TriggerSource", source, required=True)
    nm.set("TriggerMode", "On", required=True)


def set_trigger_off(nm: NodeKeeper) -> None:
    """TriggerMode Off for FrameStart (required: a camera stuck in trigger mode gives no
    frames in free-run / PTPSync)."""
    if nm.has("TriggerSelector"):
        nm.set("TriggerSelector", "FrameStart", required=True)
    if nm.has("TriggerMode"):
        nm.set("TriggerMode", "Off", required=True)


def leave_ptpsync(cam: Cam) -> None:
    """Make sure AcquisitionStartMode is not PTPSync (any method except ptpsync).

    "Normal" is written only if the camera lists it (the entry name is not in the LUCID
    sources; the mock has it). If the camera is in PTPSync and no other entry can be set,
    the method fails: with PTPSync on "you lose the ability to trigger" [PTPKB] and the
    frames would be mislabelled (review item 7).
    """
    nm = cam.nm
    if not nm.has("AcquisitionStartMode"):
        return
    entries = nm.entries("AcquisitionStartMode")
    current = nm.get("AcquisitionStartMode")
    if "Normal" in entries and nm.set("AcquisitionStartMode", "Normal"):
        return
    if current == "PTPSync":
        others = [e for e in entries if e != "PTPSync"]
        chosen = nm.try_set_first("AcquisitionStartMode", others) if others else None
        if chosen is None:
            raise MethodUnavailable(f"{cam.name}: AcquisitionStartMode is PTPSync and cannot "
                                    f"be changed (entries {entries})")
        log.warning("%s: AcquisitionStartMode 'Normal' unavailable, used %r", cam.name, chosen)


EXPECTED_STATE = {
    # method: (AcquisitionStartMode must be PTPSync?, TriggerMode, TriggerSource or None)
    "ptpsync": (True, "Off", None),
    "action": (False, "On", "Action0"),
    "software": (False, "On", "Software"),
    "freerun_ptp": (False, "Off", None),
    "freerun": (False, "Off", None),
}


def read_state(cam: Cam) -> dict:
    nm = cam.nm
    return {n: nm.get(n) for n in ("AcquisitionStartMode", "TriggerSelector", "TriggerMode",
                                   "TriggerSource", "TriggerDelay", "PtpEnable")
            if nm.has(n)}


def verify_state(method: str, helios: Cam, triton: Cam) -> dict:
    """READ BACK the critical acquisition state on both cameras (review item 7).

    Raises MethodUnavailable listing every mismatch, so the fallback chain moves on with a
    clear reason instead of capturing in an unintended mode.
    """
    want_ptpsync, want_mode, want_source = EXPECTED_STATE[method]
    problems, state = [], {}
    for cam in (helios, triton):
        st = state[cam.name] = read_state(cam)
        asm = st.get("AcquisitionStartMode")
        if "AcquisitionStartMode" in st and (asm == "PTPSync") != want_ptpsync:
            problems.append(f"{cam.name} AcquisitionStartMode reads {asm!r}")
        if want_ptpsync and "AcquisitionStartMode" not in st:
            problems.append(f"{cam.name} AcquisitionStartMode missing")
        if "TriggerMode" in st and st["TriggerMode"] != want_mode:
            problems.append(f"{cam.name} TriggerMode reads {st['TriggerMode']!r}, "
                            f"expected {want_mode!r}")
        if want_source and st.get("TriggerSource") != want_source:
            problems.append(f"{cam.name} TriggerSource reads {st.get('TriggerSource')!r}, "
                            f"expected {want_source!r}")
    if problems:
        raise MethodUnavailable(f"read-back after configuring {method}: " + "; ".join(problems))
    return state


# =========================================================================== ptpsync
def precheck_ptpsync(helios: Cam, triton: Cam) -> None:
    """Raise MethodUnavailable unless both cameras support PTPSync (nothing is written)."""
    for cam in (helios, triton):
        if not cam.nm.has("AcquisitionStartMode"):
            cam.nm.get("AcquisitionStartMode")               # record as missing
            raise MethodUnavailable(f"{cam.name}: node AcquisitionStartMode missing")
        entries = cam.nm.entries("AcquisitionStartMode")
        if "PTPSync" not in entries:
            raise MethodUnavailable(f"{cam.name}: AcquisitionStartMode has no 'PTPSync' entry "
                                    f"(entries={entries})")
        if not cam.nm.has("PTPSyncFrameRate"):
            cam.nm.get("PTPSyncFrameRate")
            raise MethodUnavailable(f"{cam.name}: node PTPSyncFrameRate missing")


def setup_ptpsync(helios: Cam, triton: Cam, s: SyncSettings, exposure: dict) -> SyncPlan:
    """Configure LUCID PTPSync on both cameras (PTP must already be locked) [BW][HT].

    The firmware then "schedules 4 advance action commands in the firmware itself" [BW] on
    a grid of period 1/PTPSyncFrameRate on the shared PTP clock. Must run before start_stream
    (AcquisitionStartMode and TriggerMode are locked while streaming).
    """
    precheck_ptpsync(helios, triton)
    lag, note = desired_triton_lag_ns(s, exposure.get("exposure_us_for_alignment"))
    h_off, t_off, lag_used, off_note = plan_ptpsync_offsets(
        lag, s.period_ns, helios.nm.has("PTPSyncOffset"), triton.nm.has("PTPSyncOffset"))
    used, why = _alignment(s, lag_used, off_note if lag and not lag_used else note)
    offsets = {HELIOS: h_off, TRITON: t_off}
    for cam in (helios, triton):
        nm = cam.nm
        # With PTP Sync "you lose the ability to trigger cameras" [PTPKB]: trigger off.
        set_trigger_off(nm)
        if nm.has("TriggerDelay"):
            # A TriggerDelay left by an earlier centre-aligned action run must not apply;
            # whether PTPSync honours TriggerDelay is unknown (review item 11).
            nm.set("TriggerDelay", 0.0)
        nm.set("AcquisitionMode", "Continuous")                                   # [BW]
        nm.set("AcquisitionStartMode", "PTPSync", required=True)                  # [BW][HT]
        # AcquisitionFrameRate "won't be controlling the actual frame rate but it might cap
        # the max achievable frame rate" -> set it to its maximum [BW].
        nm.set("AcquisitionFrameRateEnable", True)
        fr_max = nm.limits("AcquisitionFrameRate")[1]
        if fr_max is not None:
            nm.set("AcquisitionFrameRate", float(fr_max), clamp=True)
            if float(fr_max) < s.FRAME_RATE:
                log.warning("%s: AcquisitionFrameRate max %.2f < FRAME_RATE %.2f - the link "
                            "limit cannot sustain this rate, frames will be skipped",
                            cam.name, fr_max, s.FRAME_RATE)
        # [HT] says set it "on one camera"; the [BW] code writes every camera -> both, same.
        nm.set("PTPSyncFrameRate", float(s.FRAME_RATE), required=True)
        if offsets[cam.name] is not None:
            nm.set("PTPSyncOffset", int(offsets[cam.name]), required=offsets[cam.name] != 0)
    state = verify_state("ptpsync", helios, triton)
    return SyncPlan(method="ptpsync", clock_domain="ptp", pairing="grid",
                    period_ns=s.period_ns, lags_ns={HELIOS: 0, TRITON: lag_used},
                    alignment_requested=s.ALIGNMENT, alignment_used=used, alignment_note=why,
                    planned_lag_ns=lag_used,
                    ptpsync_offset_ns={k: v for k, v in offsets.items() if v is not None},
                    state=state, notes=[off_note])


def verify_ptpsync_rate(helios: Cam, triton: Cam, s: SyncSettings, tol: float = 0.01) -> dict:
    """After start_stream, AcquisitionFrameRate should read PTPSyncFrameRate [HT step 5]."""
    out: dict = {}
    for cam in (helios, triton):
        v = cam.nm.get("AcquisitionFrameRate")
        out[cam.name] = v
        if v is None or abs(float(v) - s.FRAME_RATE) > tol * s.FRAME_RATE:
            log.warning("%s: AcquisitionFrameRate reads %s after start, expected %.3f",
                        cam.name, v, s.FRAME_RATE)
    out["ok"] = all(out[c] is not None and abs(float(out[c]) - s.FRAME_RATE)
                    <= tol * s.FRAME_RATE for c in (HELIOS, TRITON))
    return out


# =========================================================================== triggered modes
HOST_ACTION_NODES = ("ActionCommandDeviceKey", "ActionCommandGroupKey", "ActionCommandGroupMask",
                     "ActionCommandTargetIP", "ActionCommandExecuteTime",
                     "ActionCommandFireCommand")
DEVICE_ACTION_NODES = ("ActionSelector", "ActionDeviceKey", "ActionGroupKey", "ActionGroupMask")


def _precheck_trigger_source(helios: Cam, triton: Cam, source: str) -> None:
    for cam in (helios, triton):
        if source not in cam.nm.entries("TriggerSource"):
            raise MethodUnavailable(f"{cam.name}: TriggerSource has no {source!r} entry "
                                    f"(entries={cam.nm.entries('TriggerSource')})")
        if "FrameStart" not in cam.nm.entries("TriggerSelector"):
            raise MethodUnavailable(f"{cam.name}: TriggerSelector has no 'FrameStart' entry")


def precheck_action(helios: Cam, triton: Cam, tl: NodeKeeper | None) -> None:
    """Raise MethodUnavailable unless scheduled action commands can work (nothing written)."""
    if tl is None:
        raise MethodUnavailable("host: system.tl_system_nodemap not available")
    miss = [n for n in HOST_ACTION_NODES if not tl.has(n)]
    if miss:
        for n in miss:
            tl.get(n)
        raise MethodUnavailable(f"host: TL system nodes missing: {miss}")
    _precheck_trigger_source(helios, triton, "Action0")
    for cam in (helios, triton):
        miss = [n for n in DEVICE_ACTION_NODES if not cam.nm.has(n)]
        if miss:
            for n in miss:
                cam.nm.get(n)
            raise MethodUnavailable(f"{cam.name}: action nodes missing: {miss}")
    # The scheduler reads PTP time from the master (Helios) [SAC][HT].
    for n in ("PtpDataSetLatch", "PtpDataSetLatchValue"):
        if not helios.nm.has(n):
            helios.nm.get(n)
            raise MethodUnavailable(f"helios: node {n} missing (cannot read PTP time)")


def _trigger_delays(s: SyncSettings, exposure: dict) -> tuple[dict, int, str, str]:
    """TriggerDelay (us) per camera for the desired lag [MH][TRM-T]: Triton for a positive
    lag, Helios for a negative one. Returns (delays_us, lag_ns, alignment_used, note)."""
    lag, note = desired_triton_lag_ns(s, exposure.get("exposure_us_for_alignment"))
    used, why = _alignment(s, lag, note)
    return {HELIOS: max(-lag, 0) / 1000.0, TRITON: max(lag, 0) / 1000.0}, lag, used, why


def _write_delays(cam: Cam, delays_us: dict) -> None:
    d = float(delays_us[cam.name])
    if cam.nm.has("TriggerDelay") or d:
        cam.nm.set("TriggerDelay", d, required=bool(d))


def setup_action(helios: Cam, triton: Cam, tl: NodeKeeper | None, s: SyncSettings,
                 exposure: dict) -> SyncPlan:
    """Configure scheduled action commands on both cameras and the host [SAC][HT].

    Device side ([HT] cpp lines 131-139): TriggerSelector=FrameStart, TriggerSource=Action0,
    TriggerMode=On, ActionUnconditionalMode=On, ActionSelector=0, ActionDeviceKey/GroupKey/
    GroupMask. Host side (TL system node map, cpp lines 505-508): ActionCommandDeviceKey/
    GroupKey/GroupMask and ActionCommandTargetIP=0xFFFFFFFF (broadcast).
    """
    precheck_action(helios, triton, tl)
    assert tl is not None
    delays_us, lag, used, why = _trigger_delays(s, exposure)
    keys = dict(device=int(s.ACTION_DEVICE_KEY), group=int(s.ACTION_GROUP_KEY),
                mask=int(s.ACTION_GROUP_MASK))
    queue_sizes = {}
    for cam in (helios, triton):
        nm = cam.nm
        leave_ptpsync(cam)                       # undo a PTPSync left by earlier runs
        nm.set("AcquisitionMode", "Continuous")
        # AcquisitionFrameRate has no effect with TriggerMode On [TRM-T]; capture.py also
        # disabled it in trigger mode, so it can never reject a slightly early trigger.
        nm.set("AcquisitionFrameRateEnable", False)
        set_trigger_on(nm, "Action0")
        _write_delays(cam, delays_us)
        nm.set("ActionUnconditionalMode", "On")
        nm.set("ActionSelector", 0, required=True)
        nm.set("ActionDeviceKey", keys["device"], required=True)
        nm.set("ActionGroupKey", keys["group"], required=True)
        nm.set("ActionGroupMask", keys["mask"], required=True)
        q = nm.get("ActionQueueSize")
        queue_sizes[cam.name] = None if q is None else int(q)
    tl.set("ActionCommandDeviceKey", keys["device"], required=True)
    tl.set("ActionCommandGroupKey", keys["group"], required=True)
    tl.set("ActionCommandGroupMask", keys["mask"], required=True)
    tl.set("ActionCommandTargetIP", BROADCAST_IP, required=True)
    state = verify_state("action", helios, triton)
    known = [q for q in queue_sizes.values() if q]
    lead_ms = clamp_action_lead_ms(s.ACTION_LEAD_MS, s.period_ns, min(known) if known else 1)
    return SyncPlan(method="action", clock_domain="ptp", pairing="grid", period_ns=s.period_ns,
                    lags_ns={HELIOS: int(delays_us[HELIOS] * 1000),
                             TRITON: int(delays_us[TRITON] * 1000)},
                    alignment_requested=s.ALIGNMENT, alignment_used=used, alignment_note=why,
                    planned_lag_ns=lag, trigger_delay_us=delays_us, state=state,
                    action=dict(keys=keys, queue_size=queue_sizes, lead_ms=lead_ms,
                                target_ip=hex(BROADCAST_IP)))


def clamp_action_lead_ms(lead_ms: float, period_ns: float, queue_size: int) -> float:
    """Never schedule more than ActionQueueSize-1 periods ahead [TRM-T: ActionQueueSize].

    With lead L and period P, about floor(L/P)+1 commands are pending in the camera at once;
    more than ActionQueueSize pending ones are dropped. With a queue of 1 we keep L < P/2.
    """
    period_ms = period_ns / 1e6
    max_ms = (queue_size - 1) * period_ms if queue_size >= 2 else 0.5 * period_ms
    if lead_ms > max_ms:
        log.warning("ACTION_LEAD_MS %.1f > %.1f ms (ActionQueueSize %d) -> clamped",
                    lead_ms, max_ms, queue_size)
        return max_ms
    return float(lead_ms)


def precheck_software(helios: Cam, triton: Cam) -> None:
    _precheck_trigger_source(helios, triton, "Software")
    for cam in (helios, triton):
        if not cam.nm.has("TriggerSoftware"):
            cam.nm.get("TriggerSoftware")
            raise MethodUnavailable(f"{cam.name}: command TriggerSoftware missing")


def setup_software(helios: Cam, triton: Cam, s: SyncSettings, exposure: dict,
                   ptp_locked: bool) -> SyncPlan:
    """Software trigger as capture.py's "trigger" mode [CAP configure_trigger]:
    AcquisitionFrameRateEnable False, TriggerSelector FrameStart, TriggerSource Software,
    TriggerMode On. The host paces TriggerSoftware at FRAME_RATE (SoftwareTrigger).
    Centre alignment via TriggerDelay as in the action method. Pairing uses the index of
    the fired trigger; offsets are measured if PTP is locked (clock_domain "ptp")."""
    precheck_software(helios, triton)
    delays_us, lag, used, why = _trigger_delays(s, exposure)
    for cam in (helios, triton):
        leave_ptpsync(cam)
        cam.nm.set("AcquisitionMode", "Continuous")
        cam.nm.set("AcquisitionFrameRateEnable", False)
        set_trigger_on(cam.nm, "Software")
        _write_delays(cam, delays_us)
    state = verify_state("software", helios, triton)
    return SyncPlan(method="software", clock_domain="ptp" if ptp_locked else "independent",
                    pairing="trigger", period_ns=s.period_ns,
                    lags_ns={HELIOS: 0, TRITON: 0}, alignment_requested=s.ALIGNMENT,
                    alignment_used=used, alignment_note=why, planned_lag_ns=lag,
                    trigger_delay_us=delays_us, state=state)


# =========================================================================== freerun
def setup_freerun(helios: Cam, triton: Cam, s: SyncSettings, ptp: bool) -> SyncPlan:
    """Both cameras free-run at FRAME_RATE (TriggerMode Off, not PTPSync).

    ptp=True  -> "freerun_ptp": PTP stays on, timestamps share one clock, pairs by nearest
                 timestamp; the offset is honest but can be up to half a period.
    ptp=False -> "freerun": legacy baseline, PTP switched off, independent clocks; only host
                 receive-time deltas are available.
    """
    for cam in (helios, triton):
        nm = cam.nm
        set_trigger_off(nm)
        leave_ptpsync(cam)
        nm.set("AcquisitionMode", "Continuous")
        nm.set("AcquisitionFrameRateEnable", True)
        nm.set("AcquisitionFrameRate", float(s.FRAME_RATE), clamp=True)
    if not ptp:
        disable_ptp(helios.nm, triton.nm)
    method = "freerun_ptp" if ptp else "freerun"
    state = verify_state(method, helios, triton)
    return SyncPlan(method=method, clock_domain="ptp" if ptp else "independent",
                    pairing="nearest_ts" if ptp else "nearest_host", period_ns=s.period_ns,
                    alignment_requested=s.ALIGNMENT, alignment_used="none",
                    alignment_note="free-run: phase not controlled", state=state)


# =========================================================================== fallback
@dataclass
class MethodResult:
    method_requested: str
    method_used: str
    fallback_chain: list
    plan: SyncPlan
    ptp: PtpReport | None


def configure_with_fallback(requested: str, helios: Cam, triton: Cam, tl: NodeKeeper | None,
                            s: SyncSettings, exposure: dict,
                            ptp_enabler: Callable[..., PtpReport] = enable_ptp) -> MethodResult:
    """Configure the requested method, walking down ptpsync -> action -> software ->
    freerun_ptp -> freerun until one works. Each failure (missing node, missing enum entry,
    PTP not locked, failed required write, failed read-back) is recorded in fallback_chain.

    Cheap node checks run BEFORE the slow PTP negotiation, and a PTP failure is remembered,
    so a PTP that never locks costs one timeout, not several. "software" uses PTP if it is
    (or can be) locked and otherwise runs on independent clocks; it does not fail on PTP.
    Requesting "freerun" uses no PTP at all (the legacy baseline). If every method fails,
    SyncSetupError is raised with the whole chain.
    """
    if requested not in METHODS:
        raise ValueError(f"unknown sync method {requested!r}; choose from {METHODS}")
    chain: list[dict] = []
    ptp_report: PtpReport | None = None      # set once PTP is locked
    ptp_failed: PtpReport | None = None      # statuses of a negotiation that timed out
    ptp_error: str | None = None             # remembered PTP failure

    def need_ptp() -> None:
        nonlocal ptp_report, ptp_failed, ptp_error
        if ptp_error is not None:
            raise PtpNotLocked(f"PTP unavailable (see above): {ptp_error}")
        if ptp_report is not None:
            return
        try:
            ptp_report = ptp_enabler(
                helios.nm, triton.nm, timeout_s=s.PTP_TIMEOUT_S, poll_s=s.PTP_POLL_S,
                threshold_ns=s.PTP_LOCK_THRESHOLD_NS, settle_samples=s.PTP_SETTLE_SAMPLES,
                settle_max_s=s.PTP_SETTLE_MAX_S, min_settle_s=s.PTP_MIN_SETTLE_S)
        except (PtpNotLocked, MethodUnavailable, NodeError) as exc:
            ptp_error = str(exc)
            ptp_failed = getattr(exc, "report", None)
            raise

    plan: SyncPlan | None = None
    for method in METHODS[METHODS.index(requested):]:
        try:
            if method == "ptpsync":
                precheck_ptpsync(helios, triton)
                need_ptp()
                plan = setup_ptpsync(helios, triton, s, exposure)
            elif method == "action":
                precheck_action(helios, triton, tl)
                need_ptp()
                plan = setup_action(helios, triton, tl, s, exposure)
            elif method == "software":
                precheck_software(helios, triton)
                try:
                    need_ptp()
                except (SyncSetupError, NodeError) as exc:
                    log.warning("software trigger without PTP (independent clocks): %s", exc)
                    disable_ptp(helios.nm, triton.nm)
                plan = setup_software(helios, triton, s, exposure, ptp_report is not None)
            elif method == "freerun_ptp":
                need_ptp()
                plan = setup_freerun(helios, triton, s, ptp=True)
            else:
                plan = setup_freerun(helios, triton, s, ptp=False)
            break
        except (SyncSetupError, NodeError) as exc:
            reason = f"{type(exc).__name__}: {exc}"
            chain.append(dict(method=method, reason=reason))
            log.warning("sync method %s unavailable -> falling back. Reason: %s", method, reason)
    if plan is None:
        raise SyncSetupError("no sync method could be configured: " +
                             " | ".join(f"{c['method']}: {c['reason']}" for c in chain))
    if ptp_report is None:
        ptp_report = ptp_failed                      # keep the failed negotiation's states
    log.info("sync method used: %s (requested %s), alignment %s, planned lag %.1f us",
             plan.method, requested, plan.alignment_used, plan.planned_lag_ns / 1000)
    return MethodResult(requested, plan.method, chain, plan, ptp_report)

