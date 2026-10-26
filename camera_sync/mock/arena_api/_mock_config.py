"""Knobs of the mock arena_api (NOT part of the real SDK).

Three ways to set them, applied in this order:
  1. defaults below (dataclass field defaults),
  2. environment: LUCID_MOCK_SMALL=1 and LUCID_MOCK_FAULTS='{"knob": value, ...}' (JSON),
  3. programmatically: ``arena_api.mock_reset(knob=value, ...)`` (rebuilds the simulated
     cameras so construction-time knobs such as drift or dropped nodes take effect), or
     ``arena_api._mock_config.CONFIG.knob = value`` for knobs read live (probabilities, scene).

Every number here that models real hardware is listed with its provenance in
camera_sync/mock/README_MOCK.md ("assumptions NOT verified on real hardware").
"""
from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Any

# Camera keys used everywhere in the mock config.
HELIOS = "helios"
TRITON = "triton"
CAMERA_KEYS = (HELIOS, TRITON)


def _per_cam(helios: Any, triton: Any) -> Any:
    return field(default_factory=lambda: {HELIOS: helios, TRITON: triton})


@dataclass
class MockConfig:
    # ---------------------------------------------------------------- general
    seed: int | None = None              # RNG seed (None = random each reset)
    small_frames: bool = False           # LUCID_MOCK_SMALL=1: Width/Height / 4 on both cameras
    # Keep the bandwidth model at full-resolution byte counts even with small frames, so
    # transfer times / link congestion stay realistic while files stay small.
    small_frames_keep_link_timing: bool = True
    device_missing: str | None = None    # "helios" | "triton" | None: hide one camera

    # ---------------------------------------------------------------- clocks
    drift_ppm: dict = _per_cam(12.0, -18.0)          # free-running oscillator error
    # Random device-clock value at mock start, uniform in this range (seconds). Real cameras
    # count from power-up, so their clocks are unrelated before PTP.
    initial_offset_s_range: tuple = (50.0, 5000.0)

    # ---------------------------------------------------------------- PTP
    ptp_lock_s: float = 3.0              # time from PtpEnable to Master / Slave state
    ptp_init_s: float = 0.2              # "Initializing" phase before "Listening"
    ptp_never_locks: bool = False        # slaves stay "Listening" forever (master still Master)
    ptp_jitter_ns_std: float = 1000.0    # residual slave-vs-master clock error, 1 sigma
    # BMCA tie-break when several cameras may be master: lowest serial wins (stand-in for
    # the clock identity comparison of real BMCA).

    # ---------------------------------------------------------------- trigger / timestamps
    # Trigger event (action time / software trigger / PTPSync grid point) -> exposure start.
    # Chosen so that default timestamps reproduce LUCID's measurement (HLT ~T+23us,
    # TRI ~T+41us) with timestamp_event="exposure_start".
    trigger_latency_us: dict = _per_cam(23.4, 41.1)
    trigger_jitter_ns_std: float = 500.0
    # Which event buffer.timestamp_ns marks: "exposure_start" | "exposure_end" | "readout_end"
    timestamp_event: str = "exposure_start"
    timestamp_extra_offset_us: dict = _per_cam(0.0, 0.0)  # extra constant per camera
    action_queue_size: dict = _per_cam(4, 4)               # default ActionQueueSize
    action_network_latency_us: float = 100.0  # host -> camera delivery of an action command
    late_action_behavior: str = "fire_immediately"  # or "drop" (ExecuteTime already past)

    # ---------------------------------------------------------------- sensor timing
    triton_readout_ms: float = 18.0      # IMX265 frame readout (1 / 55.6 fps)
    triton_sensor_max_fps: float = 55.6
    helios_max_fps: float = 30.0
    helios_readout_ms: float = 1.55      # per sub-frame
    helios_integration_ratio: float = 0.98  # 1000 us selector -> 0.98 ms real integration

    # ---------------------------------------------------------------- Triton auto exposure
    scene: str = "bright"                # "bright" | "dim" | "dark"
    scene_exposure_us: dict = field(default_factory=lambda: {
        "bright": 1_000.0, "dim": 15_000.0, "dark": 1_650_000.0})
    triton_has_exposure_auto_limit_auto: bool = True   # ExposureAutoLimitAuto NOT CONFIRMED
    triton_has_ptpsync_offset: bool = False            # PTPSyncOffset on Triton NOT CONFIRMED
    helios_has_ptpsync_offset: bool = True             # HTP FW >= 1.16.0.0

    # ---------------------------------------------------------------- network
    link_speed_Bps: int = 125_000_000
    host_mtu: int = 9000                 # what StreamAutoNegotiatePacketSize negotiates
    switch_buffer_bytes: int = 192 * 1024
    switch_overflow_incomplete_prob: float = 0.9
    packet_loss_prob: float = 0.0        # extra random incomplete buffers (both cameras)

    # ---------------------------------------------------------------- faults
    missed_trigger_prob: float = 0.0     # each trigger / PTPSync grid event lost with this p
    drop_nodes: dict = _per_cam([], [])  # node names removed from that camera's nodemap
    # Delay all deliveries of a camera during a window, e.g.
    # {"camera": "triton", "after_s": 5.0, "duration_s": 2.0} (seconds after stream start).
    host_stall: dict | None = None
    # PTP faults (times are seconds after the slave's lock; see README_MOCK.md):
    # Silent step of the slave clock, e.g. {"camera": "triton", "after_lock_s": 2.0,
    # "step_ns": 100_000_000}. Timestamps jump; exposures stay physically on the grid;
    # PtpOffsetFromMaster does NOT show it (worst case: relock to a shifted time).
    ptp_step: dict | None = None
    # Servo glitch window, e.g. {"camera": "triton", "after_lock_s": 2.0, "duration_s": 1.0,
    # "lock_lost": True} (status Listening, own clock, no PTPSync frames) or
    # "lock_lost": False with "offset_ns": 50_000 (status stays Slave, clock and
    # PtpOffsetFromMaster both off by offset_ns).
    ptp_glitch: dict | None = None
    # Replace buffer.timestamp_ns of given frame_ids, e.g. {"triton": {5: 0}}.
    timestamp_override: dict = _per_cam({}, {})
    # Node writes that are accepted but silently ignored, e.g. {"triton": ["TriggerMode"]}.
    stuck_nodes: dict = _per_cam([], [])
    # Replace enum entry lists, e.g. {"triton": {"AcquisitionStartMode": ["PTPSync"]}}.
    enum_entries: dict = _per_cam({}, {})

    # ------------------------------------------------------------------ helpers
    def update(self, overrides: dict | None) -> None:
        """Apply a dict of knob overrides (validates the names)."""
        if not overrides:
            return
        names = {f.name for f in dataclasses.fields(self)}
        for key, value in overrides.items():
            if key == "drop_node":            # accept the singular spelling too
                key = "drop_nodes"
            if key not in names:
                raise KeyError(f"unknown mock knob {key!r}; known: {sorted(names)}")
            current = getattr(self, key)
            if key == "drop_nodes" and isinstance(value, (list, tuple)):
                value = {HELIOS: list(value), TRITON: list(value)}
            if isinstance(current, dict) and isinstance(value, dict) and key not in (
                    "host_stall", "ptp_step", "ptp_glitch"):
                merged = dict(current)
                merged.update(value)
                value = merged
            if key == "initial_offset_s_range":
                value = tuple(value)
            setattr(self, key, value)


def _env_overrides() -> dict:
    out: dict = {}
    if os.environ.get("LUCID_MOCK_SMALL", "").strip() in ("1", "true", "True", "yes"):
        out["small_frames"] = True
    raw = os.environ.get("LUCID_MOCK_FAULTS", "").strip()
    if raw:
        out.update(json.loads(raw))
    return out


def fresh_config(overrides: dict | None = None) -> MockConfig:
    cfg = MockConfig()
    cfg.update(_env_overrides())
    cfg.update(overrides)
    return cfg


# The live configuration object. Always the SAME object (reset in place), so modules may
# keep a reference to it.
CONFIG: MockConfig = fresh_config()


def reset(overrides: dict | None = None) -> MockConfig:
    """Defaults + environment + overrides, applied in place to CONFIG."""
    CONFIG.__dict__.update(fresh_config(overrides).__dict__)
    return CONFIG


def configure(**overrides: Any) -> MockConfig:
    """Mutate the live config in place (no camera rebuild). Use for live knobs."""
    CONFIG.update(overrides)
    return CONFIG
