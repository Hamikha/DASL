"""Load LUCID's arena_api - the real SDK, or the mock when LUCID_MOCK=1.

The mock lives in ``camera_sync/mock/arena_api``. With ``LUCID_MOCK=1`` in the environment we
put ``camera_sync/mock`` at the FRONT of sys.path before the first ``import arena_api``, so
the mock shadows an installed real SDK. Every run logs which one was loaded, because a
capture made on the mock must never be mistaken for a real one.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

MOCK_DIR = Path(__file__).resolve().parent.parent / "mock"


def mock_requested() -> bool:
    return os.environ.get("LUCID_MOCK", "").strip().lower() in ("1", "true", "yes")


def _activate_mock() -> None:
    path = str(MOCK_DIR)
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)


def load_arena_api() -> tuple[Any, dict]:
    """Import arena_api and return ``(arena_api_module, sdk_info)``.

    sdk_info = {"arena_api_version": str, "mock": bool, "path": str} (goes into run.json).
    """
    if mock_requested():
        _activate_mock()
    import arena_api  # noqa: PLC0415  (import on purpose after the sys.path change)

    info = {
        "arena_api_version": str(getattr(arena_api, "__version__", "unknown")),
        "mock": bool(getattr(arena_api, "__mock__", False)),
        "path": str(getattr(arena_api, "__file__", "")),
    }
    if info["mock"]:
        log.warning("RUNNING ON THE MOCK SDK arena_api %s (%s) - NO REAL CAMERAS",
                    info["arena_api_version"], info["path"])
    else:
        log.info("arena_api %s loaded from %s", info["arena_api_version"], info["path"])
    if mock_requested() and not info["mock"]:
        # arena_api was imported before LUCID_MOCK could take effect.
        log.error("LUCID_MOCK=1 but the REAL arena_api is loaded (imported earlier?)")
    return arena_api, info


def load_system() -> tuple[Any, dict]:
    """``from arena_api.system import system`` with the mock switch; returns (system, info)."""
    _, info = load_arena_api()
    from arena_api.system import system  # noqa: PLC0415
    return system, info
