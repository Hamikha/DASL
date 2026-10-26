"""Simulation core of the mock: clocks, PTP, cameras (node maps) and stream engines.

Time model
----------
* "true time" = time.monotonic_ns() of the host process; every model event (exposure start,
  readout end, delivery) is computed in true ns, so timestamps are exact even though the
  engine threads wake up with OS jitter (only *delivery* to the host is jittery, as in reality).
* each camera has a free-running oscillator:  own(t) = boot_ns + (t - t0) * (1 + ppm*1e-6)
* with PTP locked as Slave, the camera's clock = master's own clock + Gaussian residual.
  Before lock (or without PTP) a camera keeps its own clock.

Locking order: camera lock -> world lock (never the reverse). The world never touches nodes.
"""
from __future__ import annotations

import heapq
import itertools
import math
import random
import threading
import time
from collections import deque
from typing import Any

from . import _mock_config as mc
from ._images import SCAN3D_OFFSET_MM, SCAN3D_SCALE_MM, frame_bytes, template
from ._mock_errors import MockStreamError, MockTimeoutError
from ._nodes import (BooleanNode, CommandNode, EnumerationNode, FloatNode, IntegerNode,
                     NodeContext, NodeMap, StringNode)
from .buffer import Buffer, _Slot
from .enums import PixelFormat

HELIOS, TRITON = mc.HELIOS, mc.TRITON


def now_ns() -> int:
    return time.monotonic_ns()


# --------------------------------------------------------------------------- profiles
PROFILES: dict[str, dict[str, Any]] = {
    HELIOS: dict(model="HTP003S-001", serial="213500017", ip="169.254.3.11",
                 mac="1c:0f:af:0d:4a:11", firmware="1.16.0.0", width=640, height=480,
                 pixel_format="Coord3D_ABCY16",
                 # ABC16 / Mono16 entries are NOT CONFIRMED for HTP003S (mock only).
                 pixel_formats=["Coord3D_ABCY16", "Coord3D_ABC16", "Coord3D_C16", "Mono16"]),
    TRITON: dict(model="TRI032S-C", serial="223600482", ip="169.254.3.12",
                 mac="1c:0f:af:0e:7b:12", firmware="1.10.0.0", width=2048, height=1536,
                 pixel_format="BayerRG8",
                 # Only BayerRG8 is required by the design; others NOT CONFIRMED (mock only).
                 pixel_formats=["BayerRG8", "BayerRG16", "RGB8", "Mono8"]),
}

PTP_STATES = ["Initializing", "Faulty", "Disabled", "Listening", "PreMaster", "Master",
              "Passive", "Uncalibrated", "Slave"]          # IEEE 1588 port states
EXP_MIN_US, EXP_MAX_US = 30.25, 10_000_000.0                # Triton ExposureTime range
GAIN_MAX_DB = 48.0                                          # NOT CONFIRMED (IMX265 typical)
HELIOS_EXPOSURES_US = {"Exp1000Us": 1000.0, "Exp250Us": 250.0, "Exp62_5Us": 62.5}
SCAN3D_MODES = ["Distance8333mmMultiFreq", "Distance6000mmSingleFreq",
                "Distance5000mmMultiFreq", "Distance4000mmSingleFreq",
                "Distance3000mmSingleFreq", "Distance1250mmSingleFreq"]
HOST_LATENCY_NS = 300_000     # NIC + driver + SDK hand-off after the last packet (est.)
ETH_OVERHEAD = 38             # preamble 8 + header 14 + FCS 4 + IFG 12
GVSP_HEADERS = 48             # IP 20 + UDP 8 + GVSP extended-ID 20


def ip_to_int(ip: str) -> int:
    a, b, c, d = (int(x) for x in ip.split("."))
    return (a << 24) | (b << 16) | (c << 8) | d


class _PtpPort:
    __slots__ = ("enable", "slave_only", "since")

    def __init__(self, t: int) -> None:
        self.enable = False
        self.slave_only = False
        self.since = t


# =========================================================================== world
class SimWorld:
    """Shared state: the cameras, the PTP domain and the shared GigE link to the host."""

    def __init__(self, cfg: mc.MockConfig) -> None:
        self.cfg = cfg
        self.lock = threading.RLock()
        self.seed = cfg.seed if cfg.seed is not None else random.randrange(2**31)
        rng = random.Random(self.seed)
        self.t0 = now_ns()
        self.ports = {k: _PtpPort(self.t0) for k in mc.CAMERA_KEYS}
        self.master_key: str | None = None
        self.master_since = self.t0
        self._transfers: dict[str, deque] = {k: deque(maxlen=512) for k in mc.CAMERA_KEYS}
        self.cameras: dict[str, SimCamera] = {}
        for key in mc.CAMERA_KEYS:
            self.cameras[key] = SimCamera(self, key, PROFILES[key],
                                          random.Random(rng.randrange(2**31)))

    def visible_cameras(self) -> list["SimCamera"]:
        return [c for k, c in self.cameras.items() if k != self.cfg.device_missing]

    # ------------------------------------------------------------------ PTP domain
    def ptp_params_changed(self, cam: "SimCamera", enable: bool, slave_only: bool) -> None:
        with self.lock:
            t = now_ns()
            p = self.ports[cam.key]
            if enable != p.enable or (enable and slave_only != p.slave_only):
                p.since = t                    # (re)start negotiation for this port
            p.enable, p.slave_only = enable, slave_only
            # BMCA stand-in: among enabled, non-slave-only ports the lowest serial wins.
            cands = [k for k, pp in self.ports.items() if pp.enable and not pp.slave_only]
            new_master = min(cands, key=lambda k: self.cameras[k].serial) if cands else None
            if new_master != self.master_key:
                self.master_key, self.master_since = new_master, t

    def _master_lock_start(self) -> int:
        return max(self.ports[self.master_key].since, self.master_since)

    def ptp_status(self, key: str, t: int) -> str:
        cfg = self.cfg
        lock_ns = int(cfg.ptp_lock_s * 1e9)
        with self.lock:
            p = self.ports[key]
            if not p.enable:
                return "Disabled"
            if t < p.since + int(cfg.ptp_init_s * 1e9):
                return "Initializing"
            if self.master_key is None:
                return "Listening"          # everyone slave-only: nobody announces
            m_start = self._master_lock_start()
            if key == self.master_key:
                return "Master" if t >= m_start + lock_ns else "Listening"
            if cfg.ptp_never_locks:
                return "Listening"
            s_lock = max(p.since, m_start) + lock_ns
            if t >= s_lock:
                g = self._fault_window(key, t, self.cfg.ptp_glitch)
                if g is not None and g.get("lock_lost", True):
                    return "Listening"          # fault knob ptp_glitch: lock lost
                return "Slave"
            if t >= s_lock - 0.3 * lock_ns:
                return "Uncalibrated"
            return "Listening"

    def _slave_lock_ns(self, key: str) -> int | None:
        """True time at which slave `key` (first) reached Slave in this negotiation."""
        with self.lock:
            p = self.ports[key]
            if not p.enable or self.master_key is None or key == self.master_key:
                return None
            return max(p.since, self._master_lock_start()) + int(self.cfg.ptp_lock_s * 1e9)

    def _fault_window(self, key: str, t: int, knob: dict | None) -> dict | None:
        """The knob dict if a ptp_glitch-style window is active for camera `key` at t."""
        if not knob or knob.get("camera") != key:
            return None
        s_lock = self._slave_lock_ns(key)
        if s_lock is None:
            return None
        a = s_lock + int(float(knob.get("after_lock_s", 0.0)) * 1e9)
        b = a + int(float(knob.get("duration_s", 1e9)) * 1e9)
        return knob if a <= t < b else None

    def fault_offset_ns(self, key: str, t: int, visible_only: bool = False) -> int:
        """Extra slave-clock error from the ptp_step / ptp_glitch knobs (0 normally).

        visible_only=True: only the part PtpOffsetFromMaster would report (the glitch
        offset; a ptp_step is a silent relock and is not reported)."""
        off = 0
        st = self.cfg.ptp_step
        if not visible_only and st and st.get("camera") == key:
            s_lock = self._slave_lock_ns(key)
            if s_lock is not None and t >= s_lock + int(float(st.get("after_lock_s", 0)) * 1e9):
                off += int(st.get("step_ns", 0))
        g = self._fault_window(key, t, self.cfg.ptp_glitch)
        if g is not None and not g.get("lock_lost", True):
            off += int(g.get("offset_ns", 0))
        return off

    def is_ptp_locked(self, key: str, t: int) -> bool:
        return self.ptp_status(key, t) in ("Master", "Slave")

    # ------------------------------------------------------------------ clocks
    def own_clock(self, cam: "SimCamera", t: int) -> int:
        return cam.boot_ns + int((t - self.t0) * cam.rate)

    def own_inv(self, cam: "SimCamera", T: int) -> int:
        return self.t0 + int((T - cam.boot_ns) / cam.rate)

    def clock_base(self, key: str, t: int) -> "SimCamera":
        """Whose oscillator defines camera `key`'s timestamps at true time t."""
        with self.lock:
            if self.ptp_status(key, t) == "Slave":
                return self.cameras[self.master_key]
            return self.cameras[key]

    def device_clock(self, cam: "SimCamera", t: int, jitter: bool = True) -> int:
        base = self.clock_base(cam.key, t)
        T = self.own_clock(base, t)
        if jitter and base is not cam and self.cfg.ptp_jitter_ns_std > 0:
            T += int(round(cam.rng.gauss(0.0, self.cfg.ptp_jitter_ns_std)))
        if base is not cam:
            T += self.fault_offset_ns(cam.key, t)   # fault knobs (0 by default)
        return T

    def device_inv(self, cam: "SimCamera", T: int, t_ref: int) -> int:
        """True time at which cam's clock (in the domain it uses at t_ref) reads T."""
        return self.own_inv(self.clock_base(cam.key, t_ref), T)

    # ------------------------------------------------------------------ shared link
    def register_transfer(self, key: str, start: int, end: int, rate: float,
                          capacity: float) -> bool:
        """Record a frame transfer; True if it overflowed the switch buffer.

        Model: while two cameras send simultaneously, the PC-facing switch port receives
        rate1 + rate2 but drains `capacity`; the excess fills the switch's shared buffer.
        Overflow (packet loss) happens when excess * overlap > switch_buffer_bytes.
        """
        with self.lock:
            worst = 0.0
            for other, dq in self._transfers.items():
                if other == key:
                    continue
                for s, e, r in dq:
                    ov = min(end, e) - max(start, s)
                    excess = rate + r - capacity
                    if ov > 0 and excess > 0:
                        worst = max(worst, excess * ov / 1e9)
            self._transfers[key].append((start, end, rate))
            return worst > self.cfg.switch_buffer_bytes

    # ------------------------------------------------------------------ action commands
    def fire_action(self, device_key: int, group_key: int, group_mask: int,
                    target_ip: int, execute_time: int) -> int:
        t_send = now_ns()
        n = 0
        for cam in self.visible_cameras():
            n += cam.receive_action(device_key, group_key, group_mask, target_ip,
                                    execute_time, t_send)
        return n

    def hardware_pulse(self, line: str) -> None:
        t = now_ns()
        for cam in self.visible_cameras():
            cam.line_pulse(line, t)

    def shutdown(self) -> None:
        for cam in self.cameras.values():
            cam.stop_stream()


# =========================================================================== camera
class SimCamera:
    """One physical camera: oscillator, node maps and (while streaming) a StreamEngine.

    Node values persist across create_device/destroy_device (camera stays powered), as on
    real hardware; arena_api.mock_reset() "power-cycles" everything.
    """

    def __init__(self, world: SimWorld, key: str, profile: dict, rng: random.Random) -> None:
        self.world, self.key, self.profile, self.rng = world, key, profile, rng
        self.cfg = world.cfg
        self.serial = profile["serial"]
        self.ip = profile["ip"]
        self.ip_int = ip_to_int(profile["ip"])
        self.lock = threading.RLock()
        self.rate = 1.0 + float(self.cfg.drift_ppm[key]) * 1e-6
        lo, hi = self.cfg.initial_offset_s_range
        self.boot_ns = int(rng.uniform(lo, hi) * 1e9)
        self.engine: StreamEngine | None = None
        self.opened = False
        self.current_exposure_us = 0.0
        self.current_gain_db = 0.0
        self.stats: dict[str, int] = {}
        self.ctx = NodeContext(self.lock, streaming=lambda: self.engine is not None)
        self._build_nodes()
        for name in self.cfg.drop_nodes.get(key, []) or []:
            self.nodemap._remove(name)
            self.tl_stream_nodemap._remove(name)
        for name in self.cfg.stuck_nodes.get(key, []) or []:      # fault knob
            node = self.nodemap.get_node(name)
            if node is not None:
                node._stuck = True
        for name, entries in (self.cfg.enum_entries.get(key, {}) or {}).items():   # fault knob
            node = self.nodemap.get_node(name)
            if node is not None:
                node._entries = list(entries)
                if node.raw_get() not in node._entries:
                    node.raw_set(node._entries[0])
        if key == TRITON:
            self.current_exposure_us = self._auto_exposure_us()

    def bump(self, stat: str, n: int = 1) -> None:
        self.stats[stat] = self.stats.get(stat, 0) + n

    def has(self, name: str) -> bool:
        return name in self.nodemap

    def raw(self, name: str, sel: Any = None, default: Any = None) -> Any:
        """Stored node value (mock-internal; tolerant of dropped nodes)."""
        node = self.nodemap.get_node(name)
        return default if node is None else node.raw_get(sel)

    # ------------------------------------------------------------------ node maps
    def _build_nodes(self) -> None:
        c, cfg, P = self.ctx, self.cfg, self.profile
        is_h = self.key == HELIOS
        nm = self.nodemap = NodeMap(f"{P['model']} device nodemap")
        add = nm._add
        small = 4 if cfg.small_frames else 1

        # ---- device information (read-only). DeviceVendorName: standard SFNC name, not in
        # the LUCID research tables -> NAME NOT CONFIRMED (requested by the design).
        for name, val in (("DeviceVendorName", "Lucid Vision Labs"),
                          ("DeviceModelName", P["model"]),
                          ("DeviceSerialNumber", P["serial"]),
                          ("DeviceFirmwareVersion", P["firmware"])):
            add(StringNode(name, c, value=val, writable=False))
        add(IntegerNode("DeviceLinkSpeed", c, value=cfg.link_speed_Bps, writable=False,
                        min=cfg.link_speed_Bps, max=cfg.link_speed_Bps, unit="Bps"))

        # ---- image format
        W, H = P["width"] // small, P["height"] // small
        add(IntegerNode("Width", c, value=W, min=W if is_h else 64, max=W, inc=4,
                        writable=not is_h, lock_while_streaming=True, unit="px"))
        add(IntegerNode("Height", c, value=H, min=H if is_h else 64, max=H, inc=2,
                        writable=not is_h, lock_while_streaming=True, unit="px"))
        add(EnumerationNode("PixelFormat", c, value=P["pixel_format"],
                            entries=P["pixel_formats"], lock_while_streaming=True))

        # ---- acquisition
        # Entry "SingleFrame" NOT CONFIRMED (standard SFNC); only "Continuous" is sourced.
        add(EnumerationNode("AcquisitionMode", c, value="Continuous",
                            entries=["Continuous", "SingleFrame"],
                            lock_while_streaming=True))
        add(BooleanNode("AcquisitionFrameRateEnable", c, value=False))
        fr = add(FloatNode("AcquisitionFrameRate", c, value=None, min=0.1,
                           max=self.max_frame_rate, unit="Hz",
                           writable=lambda: bool(nm["AcquisitionFrameRateEnable"].raw_get())))
        fr._getter = lambda: self.effective_frame_rate()
        # LowLatency entry NOT CONFIRMED (mock includes it; behaves like Normal).
        add(EnumerationNode("AcquisitionStartMode", c, value="Normal",
                            entries=["Normal", "LowLatency", "PTPSync"],
                            lock_while_streaming=True))
        add(FloatNode("PTPSyncFrameRate", c, value=10.0, min=0.1,
                      max=cfg.helios_max_fps if is_h else cfg.triton_sensor_max_fps,
                      unit="Hz"))
        if (cfg.helios_has_ptpsync_offset if is_h else cfg.triton_has_ptpsync_offset):
            # Range NOT CONFIRMED; LUCID [MH] only says "delay in ns".
            add(IntegerNode("PTPSyncOffset", c, value=0, min=0, max=999_999_999, unit="ns"))

        # ---- trigger (values stored per TriggerSelector entry, like GenICam)
        add(EnumerationNode("TriggerSelector", c, value="FrameStart",
                            entries=["FrameStart", "FrameBurstStart"]))
        sel = dict(selector="TriggerSelector")
        add(EnumerationNode("TriggerMode", c, value="Off", entries=["Off", "On"],
                            lock_while_streaming=True, **sel))
        add(EnumerationNode("TriggerSource", c, value="Line0",
                            entries=["Software", "Line0", "Line2", "Line3", "Action0"], **sel))
        add(EnumerationNode("TriggerActivation", c, value="RisingEdge",
                            entries=["RisingEdge", "FallingEdge", "AnyEdge"], **sel))
        add(FloatNode("TriggerDelay", c, value=0.0, min=0.0, max=10_000_000.0, unit="us",
                      **sel))
        add(CommandNode("TriggerSoftware", c, action=self._software_trigger, **sel))
        add(BooleanNode("TriggerArmed", c, writable=False, getter=self._trigger_armed))

        # ---- scheduled action commands (device side)
        add(EnumerationNode("ActionUnconditionalMode", c, value="Off", entries=["Off", "On"]))
        add(IntegerNode("ActionSelector", c, value=0, min=0, max=0))
        add(IntegerNode("ActionDeviceKey", c, value=0, min=0, max=0xFFFFFFFF))
        add(IntegerNode("ActionGroupKey", c, value=0, min=0, max=0xFFFFFFFF,
                        selector="ActionSelector"))
        add(IntegerNode("ActionGroupMask", c, value=0, min=0, max=0xFFFFFFFF,
                        selector="ActionSelector"))
        q = int(cfg.action_queue_size[self.key])
        add(IntegerNode("ActionQueueSize", c, value=q, min=q, max=q, writable=False))

        # ---- PTP
        def _ptp_changed(node, old, new):
            self.world.ptp_params_changed(self, bool(nm["PtpEnable"].raw_get()),
                                          bool(nm["PtpSlaveOnly"].raw_get()))
        add(BooleanNode("PtpEnable", c, value=False, on_change=_ptp_changed))
        add(BooleanNode("PtpSlaveOnly", c, value=False, on_change=_ptp_changed))
        add(EnumerationNode("PtpStatus", c, entries=PTP_STATES, writable=False,
                            getter=lambda: self.world.ptp_status(self.key, now_ns())))
        # Entry names of PtpServoStatus NOT CONFIRMED.
        add(EnumerationNode("PtpServoStatus", c, entries=["Unknown", "Locked"], writable=False,
                            getter=lambda: "Locked" if self.world.ptp_status(
                                self.key, now_ns()) == "Slave" else "Unknown"))
        add(IntegerNode("PtpOffsetFromMaster", c, value=-1, writable=False, unit="ns"))
        add(CommandNode("PtpDataSetLatch", c, action=self._ptp_latch))
        add(IntegerNode("PtpDataSetLatchValue", c, value=0, writable=False, unit="ns"))

        # ---- transport / bandwidth
        add(EnumerationNode("DeviceLinkThroughputLimitMode", c, value="Off",
                            entries=["Off", "On"]))
        add(IntegerNode("DeviceLinkThroughputLimit", c, value=cfg.link_speed_Bps,
                        min=1_000_000, max=cfg.link_speed_Bps, unit="Bps",
                        writable=lambda: nm["DeviceLinkThroughputLimitMode"].raw_get() == "On"))
        add(IntegerNode("DeviceLinkThroughputReserve", c, value=10, min=0, max=50, unit="%"))
        add(IntegerNode("GevSCPD", c, value=0, min=0, max=10_000_000, unit="ns"))
        add(IntegerNode("GevSCFTD", c, value=0, min=0, max=100_000_000, unit="ns"))
        add(IntegerNode("GevSCPSPacketSize", c, value=1500, min=576, max=9000, inc=4,
                        lock_while_streaming=True, unit="B"))

        # ---- chunks (ChunkSelector entry names NOT CONFIRMED; ChunkTimestamp NOT CONFIRMED)
        add(BooleanNode("ChunkModeActive", c, value=False, lock_while_streaming=True))
        chunk_entries = ["Image", "CRC", "Timestamp"] + ([] if is_h else ["ExposureTime", "Gain"])
        add(EnumerationNode("ChunkSelector", c, value="Image", entries=chunk_entries))
        add(BooleanNode("ChunkEnable", c, value=False, selector="ChunkSelector"))
        add(IntegerNode("ChunkTimestamp", c, value=0, writable=False, unit="ns"))  # NAME NOT CONFIRMED

        # ---- GPIO lines
        add(EnumerationNode("LineSelector", c, value="Line0",
                            entries=["Line0", "Line1", "Line2", "Line3", "Line4"]))
        line_default = {"Line0": "Input", "Line1": "Output", "Line2": "Input",
                        "Line3": "Input", "Line4": "Output"}
        lm = add(EnumerationNode("LineMode", c, value="Input", entries=["Input", "Output"],
                                 selector="LineSelector",
                                 writable=lambda: nm["LineSelector"].raw_get() in ("Line2", "Line3")))
        for ln, mode in line_default.items():
            lm.raw_set(mode, ln)
        # ExposureActive on Helios2+ NOT CONFIRMED.
        add(EnumerationNode("LineSource", c, value="Off", entries=["Off", "ExposureActive"],
                            selector="LineSelector"))

        if is_h:
            self._build_helios(nm, c)
        else:
            self._build_triton(nm, c)

        # ---- stream TL node map
        sn = self.tl_stream_nodemap = NodeMap(f"{P['model']} TL stream nodemap")
        sn._add(BooleanNode("StreamAutoNegotiatePacketSize", c, value=False,
                            lock_while_streaming=True))
        sn._add(BooleanNode("StreamPacketResendEnable", c, value=True))
        sn._add(EnumerationNode("StreamBufferHandlingMode", c, value="OldestFirst",
                                entries=["OldestFirst", "OldestFirstOverwrite", "NewestOnly"],
                                lock_while_streaming=True))
        # Counter names below are NAME NOT CONFIRMED (mock diagnostics).
        for name, stat in (("StreamDeliveredFrameCount", "delivered"),
                           ("StreamLostFrameCount", "lost_no_buffer"),
                           ("StreamIncompleteFrameCount", "incomplete")):
            sn._add(IntegerNode(name, c, writable=False,
                                getter=lambda s=stat: self.stats.get(s, 0)))

        # ---- TL device node map: Gev* names are NAME NOT CONFIRMED (standard GenTL SFNC)
        dn = self.tl_device_nodemap = NodeMap(f"{P['model']} TL device nodemap")
        dn._add(StringNode("DeviceModelName", c, value=P["model"], writable=False))
        dn._add(StringNode("DeviceSerialNumber", c, value=P["serial"], writable=False))
        dn._add(IntegerNode("GevDeviceIPAddress", c, value=self.ip_int, writable=False))
        dn._add(IntegerNode("GevDeviceSubnetMask", c, value=0xFFFF0000, writable=False))
        dn._add(StringNode("GevDeviceMACAddress", c, value=P["mac"], writable=False))
        self.tl_interface_nodemap = NodeMap(f"{P['model']} TL interface nodemap")

    def _build_helios(self, nm: NodeMap, c: NodeContext) -> None:
        add = nm._add
        add(EnumerationNode("Scan3dOperatingMode", c, value="Distance5000mmMultiFreq",
                            entries=SCAN3D_MODES, lock_while_streaming=True))
        add(EnumerationNode("ExposureTimeSelector", c, value="Exp1000Us",
                            entries=list(HELIOS_EXPOSURES_US)))
        # ConversionGain: used by the user's capture.py on the real HTP003S (NC in research).
        add(EnumerationNode("ConversionGain", c, value="Low", entries=["Low", "High"]))
        # Entry names CoordinateA/B/C and the scale/offset values are NOT CONFIRMED.
        add(EnumerationNode("Scan3dCoordinateSelector", c, value="CoordinateA",
                            entries=["CoordinateA", "CoordinateB", "CoordinateC"]))
        sc = add(FloatNode("Scan3dCoordinateScale", c, value=SCAN3D_SCALE_MM, writable=False,
                           selector="Scan3dCoordinateSelector", unit="mm"))
        of = add(FloatNode("Scan3dCoordinateOffset", c, value=0.0, writable=False,
                           selector="Scan3dCoordinateSelector", unit="mm"))
        for coord, off in SCAN3D_OFFSET_MM.items():
            sc.raw_set(SCAN3D_SCALE_MM, coord)
            of.raw_set(off, coord)

    def _build_triton(self, nm: NodeMap, c: NodeContext) -> None:
        add = nm._add
        auto_on = lambda name: nm[name].raw_get() != "Off"   # noqa: E731
        add(EnumerationNode("ExposureAuto", c, value="Continuous",
                            entries=["Off", "Once", "Continuous"]))
        et = add(FloatNode("ExposureTime", c, value=10_000.0, min=EXP_MIN_US,
                           max=EXP_MAX_US, unit="us",
                           writable=lambda: not auto_on("ExposureAuto")))
        et._getter = lambda: (et.raw_get() if not auto_on("ExposureAuto")
                              else self.current_exposure_us)

        def limits_auto() -> bool:
            n = nm.get_node("ExposureAutoLimitAuto")
            return n is not None and n.raw_get() == "Continuous"

        if self.cfg.triton_has_exposure_auto_limit_auto:
            def _limit_auto_changed(node, old, new):
                # GenICam convention: switching an Auto feature Off keeps the last value it
                # computed, i.e. the full range here -> switching Off alone caps nothing.
                if old == "Continuous" and new == "Off":
                    nm["ExposureAutoLowerLimit"].raw_set(EXP_MIN_US)
                    nm["ExposureAutoUpperLimit"].raw_set(EXP_MAX_US)
            add(EnumerationNode("ExposureAutoLimitAuto", c, value="Continuous",  # NAME NOT CONFIRMED
                                entries=["Off", "Continuous"], on_change=_limit_auto_changed))
        lo = add(FloatNode("ExposureAutoLowerLimit", c, value=EXP_MIN_US, unit="us",
                           min=EXP_MIN_US, max=lambda: hi.value,
                           writable=lambda: not limits_auto()))
        hi = add(FloatNode("ExposureAutoUpperLimit", c, value=EXP_MAX_US, unit="us",
                           min=lambda: lo.value, max=EXP_MAX_US,
                           writable=lambda: not limits_auto()))
        lo._getter = lambda: EXP_MIN_US if limits_auto() else lo.raw_get()
        hi._getter = lambda: EXP_MAX_US if limits_auto() else hi.raw_get()
        # GainAuto / Gain: standard SFNC names; only ChunkGain is in the LUCID TRM table ->
        # NAME NOT CONFIRMED (requested by the design). Default values NOT CONFIRMED.
        add(EnumerationNode("GainAuto", c, value="Continuous",
                            entries=["Off", "Once", "Continuous"]))
        g = add(FloatNode("Gain", c, value=0.0, min=0.0, max=GAIN_MAX_DB, unit="dB",
                          writable=lambda: not auto_on("GainAuto")))
        g._getter = lambda: g.raw_get() if not auto_on("GainAuto") else self.current_gain_db
        add(IntegerNode("TargetBrightness", c, value=70, min=0, max=255))  # default NOT CONFIRMED
        # BalanceWhiteAuto: used by the user's capture.py (NC in research; no effect here).
        add(EnumerationNode("BalanceWhiteAuto", c, value="Off",
                            entries=["Off", "Once", "Continuous"]))
        add(FloatNode("ChunkExposureTime", c, value=0.0, writable=False, unit="us"))
        add(FloatNode("ChunkGain", c, value=0.0, writable=False, unit="dB"))

    # ------------------------------------------------------------------ derived quantities
    def nominal_link_bytes(self, fmt: PixelFormat, w: int, h: int) -> int:
        f = 4 if (self.cfg.small_frames and self.cfg.small_frames_keep_link_timing) else 1
        return frame_bytes(fmt, w * f, h * f)

    def link_payload_Bps(self) -> float:
        """Payload bytes/s this camera achieves on the wire with current settings."""
        pkt = int(self.raw("GevSCPSPacketSize", default=1500))
        eff = (pkt - GVSP_HEADERS) / (pkt + ETH_OVERHEAD)
        speed = float(self.cfg.link_speed_Bps)
        if self.raw("DeviceLinkThroughputLimitMode", default="Off") == "On":
            wire = min(float(self.raw("DeviceLinkThroughputLimit")), speed)
        else:
            scpd = float(self.raw("GevSCPD", default=0)) * 1e-9
            per_pkt = (pkt + ETH_OVERHEAD) / speed + scpd
            wire = (pkt + ETH_OVERHEAD) / per_pkt
        return wire * eff

    def link_capacity_Bps(self) -> float:
        pkt = int(self.raw("GevSCPSPacketSize", default=1500))
        return self.cfg.link_speed_Bps * (pkt - GVSP_HEADERS) / (pkt + ETH_OVERHEAD)

    def helios_window_ns(self) -> tuple[int, int]:
        """(light-collection span, total frame window) for the current Helios mode."""
        integ_us = (HELIOS_EXPOSURES_US[self.raw("ExposureTimeSelector")]
                    * self.cfg.helios_integration_ratio)
        n_sub = 8 if self.raw("Scan3dOperatingMode").endswith("MultiFreq") else 4
        readout_us = self.cfg.helios_readout_ms * 1000.0
        window = int(n_sub * (integ_us + readout_us) * 1000)
        return window - int(readout_us * 1000), window

    def max_frame_rate(self) -> float:
        fmt = PixelFormat[self.raw("PixelFormat")]
        nbytes = self.nominal_link_bytes(fmt, self.raw("Width"), self.raw("Height"))
        reserve = float(self.raw("DeviceLinkThroughputReserve", default=10)) / 100.0
        link_fps = self.link_payload_Bps() * (1.0 - reserve) / nbytes
        if self.key == HELIOS:
            sensor = min(self.cfg.helios_max_fps, 1e9 / self.helios_window_ns()[1])
        else:
            sensor = self.cfg.triton_sensor_max_fps
            if self.raw("ExposureAuto") == "Off":
                sensor = min(sensor, 1e6 / (self.raw("ExposureTime")
                                            + self.cfg.triton_readout_ms * 1000.0))
        return math.floor(min(sensor, link_fps) * 1000.0) / 1000.0

    def effective_frame_rate(self) -> float:
        """What AcquisitionFrameRate reads: stored value clamped to the current max."""
        stored = self.nodemap["AcquisitionFrameRate"].raw_get()
        mx = self.max_frame_rate()
        return mx if stored is None else min(float(stored), mx)

    # ------------------------------------------------------------------ Triton exposure
    def _auto_exposure_us(self) -> float:
        wanted = self._scene_wanted_us()
        n = self.nodemap.get_node("ExposureAutoLimitAuto")
        if n is not None and n.raw_get() == "Continuous":
            lo, hi = EXP_MIN_US, EXP_MAX_US
        else:
            lo = float(self.raw("ExposureAutoLowerLimit", default=EXP_MIN_US))
            hi = float(self.raw("ExposureAutoUpperLimit", default=EXP_MAX_US))
        return min(max(wanted, lo, EXP_MIN_US), hi, EXP_MAX_US)

    def _scene_wanted_us(self) -> float:
        base = float(self.cfg.scene_exposure_us[self.cfg.scene])
        tb = self.raw("TargetBrightness", default=70) or 1
        return base * tb / 70.0

    def triton_exposure_for_frame(self) -> float:
        """Exposure (us) the camera uses for a frame starting now; applies auto modes."""
        nm = self.nodemap
        mode = nm["ExposureAuto"].raw_get()
        if mode == "Off":
            e = float(nm["ExposureTime"].raw_get())
        else:
            e = self._auto_exposure_us()
            if mode == "Once":
                nm["ExposureTime"].raw_set(e)
                nm["ExposureAuto"].raw_set("Off")
        self.current_exposure_us = e
        gmode = nm["GainAuto"].raw_get()
        if gmode == "Off":
            self.current_gain_db = float(nm["Gain"].raw_get())
        else:
            ratio = self._scene_wanted_us() / e
            self.current_gain_db = min(max(20 * math.log10(max(ratio, 1e-9)), 0.0), GAIN_MAX_DB)
            if gmode == "Once":
                nm["Gain"].raw_set(self.current_gain_db)
                nm["GainAuto"].raw_set("Off")
        return e

    def frame_timing(self) -> tuple[int, int, float | None]:
        """(exposure/light span ns, sensor busy window ns, exposure_us for chunk or None)."""
        if self.key == HELIOS:
            light, window = self.helios_window_ns()
            return light, window, None
        e = self.triton_exposure_for_frame()
        return int(e * 1000), int(e * 1000 + self.cfg.triton_readout_ms * 1e6), e

    # ------------------------------------------------------------------ commands / events
    def _ptp_latch(self) -> None:
        with self.lock:
            t = now_ns()
            nm = self.nodemap
            nm["PtpDataSetLatchValue"].raw_set(self.world.device_clock(self, t))
            st = self.world.ptp_status(self.key, t)
            if st == "Slave":
                off = int(round(self.rng.gauss(0.0, self.cfg.ptp_jitter_ns_std)))
                off += self.world.fault_offset_ns(self.key, t, visible_only=True)
            elif st == "Master":
                off = 0
            else:
                off = -1          # TRM: -1 = synchronization has not started
            nm["PtpOffsetFromMaster"].raw_set(off)

    def _trigger_armed(self) -> bool:
        e = self.engine
        return bool(e is not None and e.mode == "trigger" and now_ns() >= e.busy_until)

    def _software_trigger(self) -> None:
        with self.lock:
            e = self.engine
            if (e is not None and e.mode == "trigger"
                    and self.raw("TriggerSelector") == "FrameStart"
                    and self.raw("TriggerSource", "FrameStart") == "Software"):
                e.push_trigger(now_ns(), "software")
            else:
                self.bump("software_trigger_ignored")

    def line_pulse(self, line: str, t: int) -> None:
        with self.lock:
            e = self.engine
            if (e is not None and e.mode == "trigger"
                    and self.raw("TriggerSource", "FrameStart") == line
                    and self.nodemap["LineMode"].raw_get(line) == "Input"):
                e.push_trigger(t, "line")

    def receive_action(self, device_key: int, group_key: int, group_mask: int,
                       target_ip: int, execute_time: int, t_send: int) -> int:
        """Handle a (possibly scheduled) action command. Returns 1 if accepted."""
        with self.lock:
            if target_ip not in (0xFFFFFFFF, self.ip_int):
                return 0
            if (device_key != self.raw("ActionDeviceKey")
                    or group_key != self.raw("ActionGroupKey", 0)
                    or (group_mask & self.raw("ActionGroupMask", 0)) == 0):
                return 0
            self.bump("actions_received")
            e = self.engine
            if (e is None or e.mode != "trigger"
                    or self.raw("TriggerSource", "FrameStart") != "Action0"):
                self.bump("actions_ignored_not_armed")
                return 0
            t_rx = t_send + int(self.cfg.action_network_latency_us * 1000)
            if execute_time == 0:                 # unscheduled action: execute on receipt
                t_trig = t_rx
            else:
                t_trig = self.world.device_inv(self, int(execute_time), t_rx)
                if t_trig < t_rx:
                    self.bump("actions_late")
                    if self.cfg.late_action_behavior == "drop":
                        self.bump("actions_dropped_late")
                        return 0
                    t_trig = t_rx
                else:
                    pending = sum(1 for (t, _s, k) in e.triggers if k == "action" and t > t_rx)
                    if pending >= int(self.raw("ActionQueueSize")):
                        self.bump("actions_dropped_queue_full")
                        return 0
            e.push_trigger(t_trig, "action")
            return 1

    # ------------------------------------------------------------------ streaming
    def start_stream(self, n_buffers: int) -> "StreamEngine":
        with self.lock:
            if self.engine is not None:
                raise MockStreamError("stream already started")
            sn = self.tl_stream_nodemap
            neg = sn.get_node("StreamAutoNegotiatePacketSize")
            if neg is not None and neg.raw_get() and self.has("GevSCPSPacketSize"):
                self.nodemap["GevSCPSPacketSize"].raw_set(int(self.cfg.host_mtu))
            eng = StreamEngine(self, n_buffers)
            self.engine = eng
            eng.start()
            return eng

    def stop_stream(self) -> None:
        with self.lock:
            eng = self.engine
            if eng is None:
                return
            eng.request_stop()
        eng.thread.join(timeout=5.0)        # join OUTSIDE the lock (thread needs it)
        with self.lock:
            eng.release_buffers()
            self.engine = None


# =========================================================================== stream
class _Frame:
    __slots__ = ("frame_id", "t_exp", "exposure_ns", "window_ns", "ts", "t_tx_start",
                 "t_deliver", "incomplete", "chunks", "source")


class StreamEngine:
    """Background thread producing frames for one camera while its stream runs."""

    POLL_NS = 50_000_000     # re-evaluate state (PTP lock, frame rate) at least every 50 ms

    def __init__(self, cam: SimCamera, n_buffers: int) -> None:
        if n_buffers < 1:
            raise ValueError("number_of_buffers must be >= 1")
        self.cam, self.world, self.cfg = cam, cam.world, cam.cfg
        nm = cam.nodemap
        self.wake = threading.Condition(cam.lock)
        self.out_cond = threading.Condition(cam.lock)
        # Snapshot of settings that are locked while streaming.
        self.fmt = PixelFormat[nm["PixelFormat"].raw_get()]
        self.width, self.height = int(nm["Width"].raw_get()), int(nm["Height"].raw_get())
        self.nbytes = frame_bytes(self.fmt, self.width, self.height)
        self.link_bytes = cam.nominal_link_bytes(self.fmt, self.width, self.height)
        if cam.raw("AcquisitionStartMode", default="Normal") == "PTPSync":
            self.mode = "ptpsync"
        elif cam.raw("TriggerMode", "FrameStart") == "On":
            self.mode = "trigger"
        else:
            self.mode = "freerun"
        self.single_frame = cam.raw("AcquisitionMode") == "SingleFrame"
        self.handling = cam.tl_stream_nodemap["StreamBufferHandlingMode"].raw_get() \
            if "StreamBufferHandlingMode" in cam.tl_stream_nodemap else "OldestFirst"
        self.chunks_on: set[str] = set()
        if cam.raw("ChunkModeActive"):
            ce = nm["ChunkEnable"]
            self.chunks_on = {e for e in nm["ChunkSelector"].enumentry_names if ce.raw_get(e)}
        # PTPSync: the AcquisitionFrameRate set before start caps the achievable rate [BW];
        # afterwards the camera reports PTPSyncFrameRate in AcquisitionFrameRate [HT].
        self.fps_cap = cam.effective_frame_rate()
        if self.mode == "ptpsync":
            nm["AcquisitionFrameRate"].raw_set(float(nm["PTPSyncFrameRate"].raw_get()))
        content = template(self.fmt, self.width, self.height)
        self.slots = [_Slot(self.nbytes, content, self) for _ in range(n_buffers)]
        self.free: deque[_Slot] = deque(self.slots)
        self.out: deque[_Slot] = deque()
        self.triggers: list[tuple[int, int, str]] = []
        self._seq = itertools.count()
        self.inflight: deque[_Frame] = deque()
        self.busy_until = 0
        self.link_free = 0
        self.last_exp: int | None = None
        self.next_frame_id = 1
        self.t_start = now_ns()
        self.next_auto: int | None = None
        self.next_auto_g: int | None = None   # PTPSync: PTP time of the next grid point
        self.last_g: int | None = None
        self.running = True
        self.error: Exception | None = None
        self.thread = threading.Thread(target=self._run, daemon=True,
                                       name=f"arena-mock-{cam.key}-stream")
        stall = self.cfg.host_stall
        self.stall = None
        if stall and stall.get("camera") == cam.key:
            a = self.t_start + int(float(stall.get("after_s", 0)) * 1e9)
            self.stall = (a, a + int(float(stall.get("duration_s", 1)) * 1e9))

    # ------------------------------------------------------------------ control
    def start(self) -> None:
        if self.mode == "freerun":
            period = self._freerun_period_dev_ns(self._guess_window_ns())
            # Arbitrary phase: free-running cameras start whenever they start.
            self.next_auto = self.t_start + 2_000_000 + int(self.cam.rng.uniform(0, period))
        self.thread.start()

    def request_stop(self) -> None:
        self.running = False
        self.wake.notify_all()
        self.out_cond.notify_all()

    def release_buffers(self) -> None:
        for s in self.slots:            # invalidate every outstanding Buffer view
            s.generation += 1
            s.outstanding = False
        self.free.clear()
        self.out.clear()

    def push_trigger(self, t: int, kind: str) -> None:
        heapq.heappush(self.triggers, (t, next(self._seq), kind))
        self.wake.notify_all()

    # ------------------------------------------------------------------ thread
    def _run(self) -> None:
        with self.wake:
            try:
                while self.running:
                    self._advance(now_ns())
                    t_next = self._next_event_time()
                    now = now_ns()
                    wait_ns = self.POLL_NS if t_next is None else min(self.POLL_NS, t_next - now)
                    if wait_ns > 0:
                        self.wake.wait(wait_ns / 1e9)
            except Exception as exc:          # a mock bug must not look like a silent timeout
                self.error = exc
                self.running = False
                self.out_cond.notify_all()
                raise

    def _next_event_time(self) -> int | None:
        ts = [t for t in (self.next_auto,
                          self.triggers[0][0] if self.triggers else None,
                          self.inflight[0].t_deliver if self.inflight else None)
              if t is not None]
        return min(ts) if ts else None

    def _advance(self, now: int) -> None:
        """Process every model event with time <= now, in time order."""
        while self.running:
            if self.mode == "ptpsync" and self.next_auto is None:
                self._ptpsync_schedule(now)
            t_del = self.inflight[0].t_deliver if self.inflight else None
            t_trig = self.triggers[0][0] if self.triggers else None
            t_auto = self.next_auto
            cands = [t for t in (t_del, t_trig, t_auto) if t is not None]
            if not cands or min(cands) > now:
                return
            t_min = min(cands)
            if t_del == t_min:
                self._deliver(self.inflight.popleft())
            elif t_trig == t_min:
                t, _, kind = heapq.heappop(self.triggers)
                self._on_trigger(t, kind)
            else:
                self._on_auto(t_auto)

    # ------------------------------------------------------------------ frame starts
    def _guess_window_ns(self) -> int:
        cam = self.cam
        if cam.key == HELIOS:
            return cam.helios_window_ns()[1]
        e = cam._auto_exposure_us() if cam.raw("ExposureAuto") != "Off" \
            else float(cam.raw("ExposureTime"))
        return int((e + self.cfg.triton_readout_ms * 1000) * 1000)

    def _freerun_period_dev_ns(self, window_ns: int) -> float:
        cam = self.cam
        if cam.raw("AcquisitionFrameRateEnable"):
            fps = cam.effective_frame_rate()
        else:
            fps = cam.max_frame_rate()
        # A long exposure DRAGS the frame rate: the sensor cannot start the next frame
        # before exposure + readout of this one finished.
        return max(1e9 / fps, float(window_ns))

    def _ptpsync_schedule(self, now: int) -> None:
        cam, world = self.cam, self.world
        if not world.is_ptp_locked(cam.key, now):
            return                                  # PTPSync without lock: no frames
        P = 1e9 / float(cam.raw("PTPSyncFrameRate"))
        off = int(cam.raw("PTPSyncOffset", default=0))
        T_now = world.device_clock(cam, now, jitter=False)
        if self.last_g is None or self.last_g < T_now - 500_000_000:
            ref = T_now + 1_000_000                 # first grid point >= 1 ms ahead
            k = math.floor((ref - off) / P) + 1
        else:                                       # continue the grid without skipping
            k = math.floor((self.last_g - off) / P + 0.5) + 1
        g = off + int(round(k * P))
        self.next_auto_g = g
        self.next_auto = world.device_inv(cam, g, now)

    def _on_auto(self, t: int) -> None:
        cam, cfg = self.cam, self.cfg
        if self.mode == "freerun":
            fr = self._make_frame(t, "freerun")
            if self.single_frame:
                self.next_auto = None
                self.mode = "done"
                return
            period_true = self._freerun_period_dev_ns(fr.window_ns) / \
                self.world.clock_base(cam.key, t).rate
            self.next_auto = t + int(period_true)
            return
        # PTPSync grid point (firmware-scheduled action command)
        self.last_g = self.next_auto_g
        self.next_auto = None
        if not self.world.is_ptp_locked(cam.key, t):
            return
        if cam.rng.random() < cfg.missed_trigger_prob:
            cam.bump("missed_triggers")
            return
        t_exp = t + self._latency_ns()
        if self.last_exp is not None and (t_exp - self.last_exp) < 0.999e9 / self.fps_cap:
            cam.bump("ptpsync_skipped_fps_cap")
            return
        if t_exp < self.busy_until:
            cam.bump("busy_drops")
            return
        self._make_frame(t_exp, "ptpsync")

    def _on_trigger(self, t: int, kind: str) -> None:
        cam, cfg = self.cam, self.cfg
        if self.mode != "trigger":
            return
        if cam.rng.random() < cfg.missed_trigger_prob:
            cam.bump("missed_triggers")
            return
        delay_ns = int(float(cam.raw("TriggerDelay", "FrameStart", 0.0)) * 1000)
        t_exp = t + self._latency_ns() + delay_ns
        if t_exp < self.busy_until:
            # "If the input trigger frequency is greater than the sensor framerate, the
            # trigger is discarded, not delayed" [LUCID GPIO app note]
            cam.bump("busy_drops")
            return
        self._make_frame(t_exp, kind)
        if self.single_frame:
            self.mode = "done"

    def _latency_ns(self) -> int:
        lat = self.cfg.trigger_latency_us[self.cam.key] * 1000.0
        if self.cfg.trigger_jitter_ns_std > 0:
            lat += self.cam.rng.gauss(0.0, self.cfg.trigger_jitter_ns_std)
        return int(round(max(lat, 0.0)))

    # ------------------------------------------------------------------ frame creation
    def _make_frame(self, t_exp: int, source: str) -> _Frame:
        cam, cfg, world = self.cam, self.cfg, self.world
        exposure_ns, window_ns, exp_us = cam.frame_timing()
        self.busy_until = t_exp + window_ns
        self.last_exp = t_exp
        fr = _Frame()
        fr.frame_id = self.next_frame_id
        self.next_frame_id += 1
        fr.t_exp, fr.exposure_ns, fr.window_ns, fr.source = t_exp, exposure_ns, window_ns, source
        ev = cfg.timestamp_event
        t_stamp = t_exp + (exposure_ns if ev == "exposure_end"
                           else window_ns if ev == "readout_end" else 0)
        fr.ts = world.device_clock(cam, t_stamp) + \
            int(cfg.timestamp_extra_offset_us[cam.key] * 1000)
        override = cfg.timestamp_override.get(cam.key) or {}       # fault knob
        for fid_key, ts_value in override.items():
            if int(fid_key) == fr.frame_id:
                fr.ts = int(ts_value)
        # Transfer: starts after readout (+ GevSCFTD), serialised on this camera's link.
        rate = cam.link_payload_Bps()
        scftd = int(cam.raw("GevSCFTD", default=0))
        fr.t_tx_start = max(t_exp + window_ns + scftd, self.link_free)
        t_tx_end = fr.t_tx_start + int(self.link_bytes / rate * 1e9)
        self.link_free = t_tx_end
        overflow = world.register_transfer(cam.key, fr.t_tx_start, t_tx_end, rate,
                                           cam.link_capacity_Bps())
        fr.incomplete = bool((overflow and cam.rng.random() < cfg.switch_overflow_incomplete_prob)
                             or cam.rng.random() < cfg.packet_loss_prob)
        if overflow:
            cam.bump("switch_overflows")
        t_del = t_tx_end + HOST_LATENCY_NS
        if self.stall and self.stall[0] <= t_del < self.stall[1]:
            t_del = self.stall[1]
        fr.t_deliver = t_del
        chunks: dict[str, Any] = {}
        if "Timestamp" in self.chunks_on:
            chunks["ChunkTimestamp"] = fr.ts
        if exp_us is not None and "ExposureTime" in self.chunks_on:
            chunks["ChunkExposureTime"] = exp_us
        if cam.key == TRITON and "Gain" in self.chunks_on:
            chunks["ChunkGain"] = cam.current_gain_db
        fr.chunks = chunks
        self.inflight.append(fr)
        cam.bump("frames_exposed")
        return fr

    def _deliver(self, fr: _Frame) -> None:
        cam = self.cam
        if self.handling == "NewestOnly":
            while self.out:                       # older undelivered frames are discarded
                self.free.append(self.out.popleft())
                cam.bump("lost_no_buffer")
        if self.free:
            slot = self.free.popleft()
        elif self.handling != "OldestFirst" and self.out:
            slot = self.out.popleft()             # overwrite the oldest waiting frame
            cam.bump("lost_no_buffer")
        else:
            cam.bump("lost_no_buffer")            # OldestFirst: no free buffer -> frame lost
            return
        size = self.nbytes
        if fr.incomplete:
            size = int(size * cam.rng.uniform(0.3, 0.97))
            cam.bump("incomplete")
        slot.meta = dict(
            width=self.width, height=self.height, pixel_format=self.fmt, size_filled=size,
            timestamp_ns=fr.ts, frame_id=fr.frame_id, is_incomplete=fr.incomplete,
            chunks=fr.chunks,
            mock=dict(camera=cam.key, source=fr.source, t_exposure_start_true_ns=fr.t_exp,
                      exposure_ns=fr.exposure_ns, window_ns=fr.window_ns,
                      t_deliver_true_ns=fr.t_deliver, t_tx_start_true_ns=fr.t_tx_start))
        nm = cam.nodemap
        for name, val in fr.chunks.items():
            if name in nm:
                nm[name].raw_set(val)
        self.out.append(slot)
        cam.bump("delivered")
        self.out_cond.notify_all()

    # ------------------------------------------------------------------ host side
    def get_buffer(self, timeout_ms: float | None) -> Buffer:
        deadline = None if timeout_ms is None else now_ns() + int(timeout_ms * 1e6)
        with self.out_cond:
            while not self.out:
                if self.error is not None:
                    raise MockStreamError(f"mock stream engine crashed: {self.error!r}") \
                        from self.error
                if not self.running:
                    raise MockStreamError("stream stopped")
                if deadline is None:
                    self.out_cond.wait(0.1)
                    continue
                remaining = deadline - now_ns()
                if remaining <= 0:
                    raise MockTimeoutError(
                        f"get_buffer timed out after {timeout_ms} ms ({self.cam.profile['model']})")
                self.out_cond.wait(remaining / 1e9)
            slot = self.out.popleft()
            slot.outstanding = True
            return Buffer(slot)

    def requeue(self, buf: Buffer) -> None:
        slot = buf._slot
        if buf._standalone:
            raise MockStreamError("requeue_buffer() got a BufferFactory copy; "
                                  "use BufferFactory.destroy()")
        if slot.owner is not self:
            raise MockStreamError("buffer does not belong to this device's stream")
        if not slot.outstanding or slot.generation != buf._gen:
            raise MockStreamError("buffer already requeued")
        slot.generation += 1
        slot.outstanding = False
        self.free.append(slot)
