"""``from arena_api.system import system`` - the mock's entry point.

Mirrors the arena_api surface used by the capture scripts:
  system.device_infos, system.create_device(device_infos=...), system.destroy_device(device=None),
  system.tl_system_nodemap (ActionCommand* nodes; Python accessor name NOT CONFIRMED on the
  real SDK), Device.nodemap / tl_stream_nodemap / tl_device_nodemap, start_stream,
  stop_stream, get_buffer, requeue_buffer.
"""
from __future__ import annotations

import threading
from typing import Any, Iterable

from . import _mock_config as mc
from ._mock_errors import MockStreamError
from ._nodes import CommandNode, IntegerNode, NodeContext, NodeMap
from ._sim import SimCamera, SimWorld
from .buffer import Buffer


class Device:
    """An opened camera (returned by system.create_device)."""

    def __init__(self, cam: SimCamera) -> None:
        self._cam = cam
        self._open = True

    def _c(self) -> SimCamera:
        if not self._open:
            raise MockStreamError("device was destroyed (system.destroy_device)")
        return self._cam

    # node maps
    @property
    def nodemap(self) -> NodeMap:
        return self._c().nodemap

    @property
    def tl_stream_nodemap(self) -> NodeMap:
        return self._c().tl_stream_nodemap

    @property
    def tl_device_nodemap(self) -> NodeMap:
        return self._c().tl_device_nodemap

    @property
    def tl_interface_nodemap(self) -> NodeMap:
        return self._c().tl_interface_nodemap

    # streaming
    def start_stream(self, number_of_buffers: int = 10) -> None:
        self._c().start_stream(int(number_of_buffers))

    def stop_stream(self) -> None:
        self._c().stop_stream()

    def get_buffer(self, timeout: float | None = None, number_of_buffers: int | None = None
                   ) -> Buffer | list[Buffer]:
        """Block up to `timeout` ms (None = forever). Raises TimeoutError subclass."""
        eng = self._c().engine
        if eng is None:
            raise MockStreamError("get_buffer() called while the stream is not started")
        if number_of_buffers is None:
            return eng.get_buffer(timeout)
        return [eng.get_buffer(timeout) for _ in range(int(number_of_buffers))]

    def requeue_buffer(self, buffers: Buffer | Iterable[Buffer]) -> None:
        cam = self._c()
        bufs = [buffers] if isinstance(buffers, Buffer) else list(buffers)
        with cam.lock:
            eng = cam.engine
            for b in bufs:
                if eng is None:
                    # Stream already stopped: buffers were released with it; nothing to do.
                    b._valid = False
                    continue
                eng.requeue(b)

    # mock-only helpers (NOT in the real SDK)
    def mock_stats(self) -> dict[str, int]:
        with self._cam.lock:
            return dict(self._cam.stats)

    @property
    def mock_key(self) -> str:
        return self._cam.key

    def __repr__(self) -> str:
        p = self._cam.profile
        return f"<mock Device {p['model']} sn={p['serial']} ip={p['ip']}>"


class System:
    DEVICE_INFOS_TIMEOUT_MILLISEC = 100

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._world: SimWorld | None = None
        self._devices: list[Device] = []
        self._tl_nodemap: NodeMap | None = None

    # ------------------------------------------------------------------ internal
    def _w(self) -> SimWorld:
        with self._lock:
            if self._world is None:
                self._world = SimWorld(mc.CONFIG)
            return self._world

    def _mock_reset(self) -> None:
        """Power-cycle the simulated cameras (destroys devices, re-reads the config)."""
        with self._lock:
            self.destroy_device()
            if self._world is not None:
                self._world.shutdown()
            self._world = None
            self._tl_nodemap = None

    # ------------------------------------------------------------------ public API
    @property
    def device_infos(self) -> list[dict[str, Any]]:
        out = []
        for cam in self._w().visible_cameras():
            p = cam.profile
            out.append({
                "model": p["model"], "vendor": "Lucid Vision Labs", "serial": p["serial"],
                "ip": p["ip"], "subnetmask": "255.255.0.0", "defaultgateway": "0.0.0.0",
                "mac": p["mac"], "name": p["model"], "dhcp": False, "presistentip": False,
                "lla": True,
                "version": p["firmware"],   # key name NOT CONFIRMED in the real SDK
            })
        return out

    def create_device(self, device_infos: Iterable[dict] | dict | None = None) -> list[Device]:
        world = self._w()
        visible = {c.serial: c for c in world.visible_cameras()}
        if device_infos is None:
            wanted = list(visible.values())
        else:
            if isinstance(device_infos, dict):
                device_infos = [device_infos]
            wanted = []
            for info in device_infos:
                cam = visible.get(str(info.get("serial")))
                if cam is None:
                    raise ValueError(f"device not found: {info!r}")
                wanted.append(cam)
        devices = []
        with self._lock:
            for cam in wanted:
                if cam.opened:
                    raise MockStreamError(f"device {cam.serial} is already open")
            for cam in wanted:
                cam.opened = True
                d = Device(cam)
                self._devices.append(d)
                devices.append(d)
        return devices

    def destroy_device(self, device: Device | Iterable[Device] | None = None) -> None:
        with self._lock:
            if device is None:
                targets = list(self._devices)
            elif isinstance(device, Device):
                targets = [device]
            else:
                targets = list(device)
            for d in targets:
                if not d._open:
                    continue
                d._cam.stop_stream()          # clean shutdown: joins the stream thread
                d._cam.opened = False
                d._open = False
                if d in self._devices:
                    self._devices.remove(d)

    @property
    def tl_system_nodemap(self) -> NodeMap:
        with self._lock:
            if self._tl_nodemap is None:
                self._tl_nodemap = self._build_tl_nodemap()
            return self._tl_nodemap

    def _build_tl_nodemap(self) -> NodeMap:
        ctx = NodeContext(threading.RLock())
        nm = NodeMap("Arena TL system nodemap")
        u32 = dict(min=0, max=0xFFFFFFFF)
        nm._add(IntegerNode("ActionCommandDeviceKey", ctx, value=0, **u32))
        nm._add(IntegerNode("ActionCommandGroupKey", ctx, value=0, **u32))
        nm._add(IntegerNode("ActionCommandGroupMask", ctx, value=0, **u32))
        nm._add(IntegerNode("ActionCommandTargetIP", ctx, value=0xFFFFFFFF, **u32))
        nm._add(IntegerNode("ActionCommandExecuteTime", ctx, value=0, min=0, max=2**63 - 1,
                            unit="ns"))

        def fire() -> None:
            g = lambda n: nm[n].raw_get()   # noqa: E731
            self._w().fire_action(g("ActionCommandDeviceKey"), g("ActionCommandGroupKey"),
                                  g("ActionCommandGroupMask"), g("ActionCommandTargetIP"),
                                  g("ActionCommandExecuteTime"))
        nm._add(CommandNode("ActionCommandFireCommand", ctx, action=fire))
        return nm

    # mock-only: simulate an electrical edge on a GPIO line of every camera
    def mock_hardware_trigger(self, line: str = "Line0") -> None:
        self._w().hardware_pulse(line)

    @property
    def mock_world(self) -> SimWorld:
        return self._w()


system = System()
