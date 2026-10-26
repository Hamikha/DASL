"""MOCK of LUCID's ``arena_api`` Python SDK (simulated Helios2+ HTP003S-001 + Triton TRI032S-C).

This package is NOT the real SDK. Scripts can detect it::

    import arena_api
    if getattr(arena_api, "__mock__", False):
        log.warning("running on the MOCK SDK %s", arena_api.__version__)

Activate by putting ``camera_sync/mock`` first on sys.path (or PYTHONPATH). Knobs: see
``arena_api._mock_config`` and camera_sync/mock/README_MOCK.md.
"""
from __future__ import annotations

from typing import Any

from . import _mock_config

__mock__ = True
__version__ = "mock-0.1.0"


def mock_reset(**knobs: Any) -> None:
    """Power-cycle the simulated cameras with fresh config (defaults + env + knobs).

    Example: ``arena_api.mock_reset(seed=1, scene="dark", drop_nodes={"triton": ["PTPSyncOffset"]})``
    """
    from .system import system
    system._mock_reset()
    _mock_config.reset(knobs)


def mock_config() -> _mock_config.MockConfig:
    """The live config object (live knobs such as probabilities or scene may be changed)."""
    return _mock_config.CONFIG
