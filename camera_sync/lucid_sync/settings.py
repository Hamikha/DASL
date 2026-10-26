"""User-facing settings of the sync library, collected in one dataclass.

The capture script keeps its "settings at the top of the file" style; it simply builds a
`SyncSettings(...)` from those constants and hands it to the engine. Defaults for the camera
image settings equal the user's original capture.py, except where noted. Every number has a
source in the research reports (tags in brackets) or is marked as an estimate.
"""
from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass

from .timing import helios_light_window

log = logging.getLogger(__name__)

#: The sync methods, best first. Automatic fallback walks DOWN this list.
METHODS = ("ptpsync", "action", "software", "freerun_ptp", "freerun")


@dataclass
class SyncSettings:
    # ------------------------------------------------------------------ what to do
    SYNC_METHOD: str = "ptpsync"       # one of METHODS; fallback starts here
    FRAME_RATE: float = 10.0           # pair rate (Hz). 15 Hz is the 1 GbE maximum [C §2]
    SECONDS: float = 10.0              # capture duration

    # ------------------------------------------------------------------ Triton
    TRITON_PIXEL_FORMAT: str = "BayerRG8"          # capture.py default
    TRITON_FULL_SENSOR: bool = True                # capture.py: Binning 1, Offset 0, max W/H
    TRITON_BALANCE_WHITE_AUTO: str | None = "Continuous"   # capture.py default
    # Upper limit of Triton auto exposure. 8 ms keeps hand-held motion blur at ~3 px at
    # 10 deg/s and can never over-run a 10 or 15 fps period [C §5].
    EXPOSURE_CAP_US: float = 8000.0
    # None = auto exposure capped at EXPOSURE_CAP_US; a number = fixed ExposureTime (us).
    TRITON_FIXED_EXPOSURE_US: float | None = None
    # Record the per-frame exposure as chunk data if the camera offers it [TRM-T][P-TRI].
    TRITON_EXPOSURE_CHUNK: bool = True

    # ------------------------------------------------------------------ Helios
    HELIOS_PIXEL_FORMAT: str = "Coord3D_ABCY16"
    # user choice 2026-10-26: multi-frequency for fewer range-wrap errors;
    # capture.py used Distance6000mmSingleFreq
    HELIOS_OPERATING_MODE: str = "Distance5000mmMultiFreq"
    HELIOS_EXPOSURE_SELECTOR: str = "Exp250Us"     # capture.py default
    HELIOS_CONVERSION_GAIN: str | None = "High"    # capture.py default (node ConversionGain)

    # ------------------------------------------------------------------ phase alignment
    # "center" (default): delay the Triton so its exposure centre sits on the modelled
    #   centre of the Helios light collection. The Helios frame collects light for ~6-19 ms
    #   (mode dependent), so "start" alignment puts the Triton exposure several ms early;
    #   even an unverified centroid estimate is closer than 0.
    # "start": both exposures start on the same grid point (only the ~18 us timestamp
    #   bookkeeping offset remains).
    ALIGNMENT: str = "center"
    # Centre of the Helios light collection after the Helios frame start (us). None =
    # computed from mode + exposure selector by lucid_sync.timing (an UNVERIFIED model;
    # ~6405 us for Distance5000mmMultiFreq + Exp250Us). Set a number after a scope check.
    HELIOS_CENTER_US: float | None = None

    # ------------------------------------------------------------------ bandwidth [C §2]
    THROUGHPUT_LIMIT_TRITON: int = 60_000_000      # bytes/s, capture.py default
    THROUGHPUT_LIMIT_HELIOS: int = 40_000_000      # bytes/s, capture.py default
    THROUGHPUT_LIMIT_ENABLE: bool = True

    # ------------------------------------------------------------------ PTP
    PTP_TIMEOUT_S: float = 60.0        # LUCID: negotiation "can take about 40 seconds" [BW]
    PTP_POLL_S: float = 0.5
    # Lock quality: |PtpOffsetFromMaster| must stay below this. A few us are expected
    # through an unmanaged switch [C §1]; 10 us leaves margin and is still 100x below
    # anything that matters for a 6-19 ms ToF frame.
    PTP_LOCK_THRESHOLD_NS: int = 10_000
    PTP_SETTLE_SAMPLES: int = 5        # consecutive good offset samples before recording
    PTP_SETTLE_MAX_S: float = 30.0     # give up waiting (continue, record settled=False)
    PTP_MIN_SETTLE_S: float = 10.0     # fixed wait when PtpOffsetFromMaster is missing
    PTP_MONITOR_S: float = 1.0         # in-run PTP monitor period (0 = off)

    # ------------------------------------------------------------------ triggers
    ACTION_LEAD_MS: float = 50.0       # how far ahead each action command is sent
    ACTION_RESYNC_S: float = 1.0       # how often the host<->PTP clock mapping is refreshed
    ACTION_DEVICE_KEY: int = 1         # LUCID's examples use 1 for all keys [HT][SAC]
    ACTION_GROUP_KEY: int = 1
    ACTION_GROUP_MASK: int = 1
    SOFTWARE_ARMED_TIMEOUT_S: float = 1.0   # capture.py waited 1 s for TriggerArmed

    # ------------------------------------------------------------------ host side
    BUFFERS: int = 10                  # capture.py STREAM_BUFFERS
    STREAM_BUFFER_HANDLING: str = "NewestOnly"     # capture.py default
    GET_BUFFER_TIMEOUT_MS: int | None = None   # None = max(1000, 3 periods)
    WRITER_QUEUE_PAIRS: int = 30       # full queue -> pair dropped, counted
    PAIR_HORIZON_FRAMES: int = 30      # unmatched frames are given up after this many periods
    RESTORE_ON_EXIT: bool = True       # put every node we wrote back (capture.py did too)

    # ------------------------------------------------------------------ helpers
    @property
    def period_ns(self) -> float:
        return 1e9 / float(self.FRAME_RATE)

    def get_buffer_timeout_ms(self) -> int:
        if self.GET_BUFFER_TIMEOUT_MS is not None:
            return int(self.GET_BUFFER_TIMEOUT_MS)
        return int(max(1000.0, 3.0 * 1000.0 / float(self.FRAME_RATE)))

    def helios_window(self) -> dict | None:
        """Modelled Helios light window for the configured mode (None if not modelled)."""
        try:
            return helios_light_window(self.HELIOS_OPERATING_MODE, self.HELIOS_EXPOSURE_SELECTOR)
        except ValueError as exc:
            log.warning("Helios timing model unavailable: %s", exc)
            return None

    def helios_center_us(self) -> float | None:
        """HELIOS_CENTER_US if set, else the timing model's centre (None if unknown)."""
        if self.HELIOS_CENTER_US is not None:
            return float(self.HELIOS_CENTER_US)
        w = self.helios_window()
        return None if w is None else float(w["center_us"])

    def validate(self) -> None:
        """Raise ValueError for settings that cannot work (cheap sanity checks only)."""
        if self.SYNC_METHOD not in METHODS:
            raise ValueError(f"SYNC_METHOD must be one of {METHODS}, got {self.SYNC_METHOD!r}")
        if self.ALIGNMENT not in ("start", "center"):
            raise ValueError(f"ALIGNMENT must be 'start' or 'center', got {self.ALIGNMENT!r}")
        if not 0.1 <= float(self.FRAME_RATE) <= 30.0:
            raise ValueError("FRAME_RATE must be within 0.1 .. 30 Hz (Helios2+ max 30 fps)")
        if self.EXPOSURE_CAP_US <= 0:
            raise ValueError("EXPOSURE_CAP_US must be > 0")
        if self.BUFFERS < 2:
            raise ValueError("BUFFERS must be >= 2")

    def as_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["helios_center_us_used"] = self.helios_center_us()
        return d
