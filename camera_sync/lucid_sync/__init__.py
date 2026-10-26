"""lucid_sync - camera synchronisation for a LUCID Helios2+ (ToF) + Triton (colour) rig.

Modules:
  settings  SyncSettings dataclass (all user settings)
  timing    modelled Helios light-collection window (unverified model)
  sdk       loads arena_api, or the mock when LUCID_MOCK=1
  nodes     NodeKeeper: allow-listed node writes, logging, restore of original values
  ptp       PTP enable / settle gate / in-run monitor
  triggers  action-command scheduler and software-trigger pacer threads
  methods   camera discovery, common config, the five sync methods, fallback
  pairing   grid / trigger-index / nearest pairing, frame and pair records, PairGuard
  stats     summary statistics
  engine    CaptureEngine: threads, pairing, writer callback, run.json "sync" object

Importing this package does not import arena_api; only sdk.load_system() / the engine do.
"""
from .engine import CaptureEngine, CaptureResult, buffer_to_array, write_csv
from .methods import (configure_common, configure_with_fallback, find_cameras, read_identity,
                      setup_action, setup_freerun, setup_ptpsync, setup_software, SyncPlan)
from .nodes import (ALLOWED_COMMANDS, ALLOWED_WRITES, MissingNodeError, NodeError, NodeKeeper,
                    NodeReport, NodeWriteError, restore_writes)
from .pairing import (FRAME_COLUMNS, PAIR_COLUMNS, FrameRecord, GridPairer, NearestPairer, Pair,
                      PairGuard, TriggerPairer)
from .ptp import (MethodUnavailable, PtpMonitor, PtpNotLocked, PtpReport, SyncSetupError,
                  enable_ptp, ptp_now_ns)
from .settings import METHODS, SyncSettings
from .timing import helios_light_window
from .triggers import ActionScheduler, SoftwareTrigger

__all__ = [
    "CaptureEngine", "CaptureResult", "buffer_to_array", "write_csv",
    "configure_common", "configure_with_fallback", "find_cameras", "read_identity",
    "setup_action", "setup_freerun", "setup_ptpsync", "setup_software", "SyncPlan",
    "ALLOWED_COMMANDS", "ALLOWED_WRITES", "MissingNodeError", "NodeError", "NodeKeeper",
    "NodeReport", "NodeWriteError", "restore_writes",
    "FRAME_COLUMNS", "PAIR_COLUMNS", "FrameRecord", "GridPairer", "NearestPairer", "Pair",
    "PairGuard", "TriggerPairer",
    "MethodUnavailable", "PtpMonitor", "PtpNotLocked", "PtpReport", "SyncSetupError",
    "enable_ptp", "ptp_now_ns", "METHODS", "SyncSettings", "helios_light_window",
    "ActionScheduler", "SoftwareTrigger",
]
