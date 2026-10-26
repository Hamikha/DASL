"""Shared pytest setup: run everything against the mock SDK (LUCID_MOCK=1)."""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ["LUCID_MOCK"] = "1"

from lucid_sync.sdk import load_system  # noqa: E402

SYSTEM, SDK_INFO = load_system()
import arena_api  # noqa: E402  (the mock, loaded by load_system)

assert getattr(arena_api, "__mock__", False), "tests must run on the mock SDK"

DEFAULT_KNOBS = dict(seed=7, small_frames=True, ptp_lock_s=0.5)


@pytest.fixture
def mock():
    """Power-cycle the mock with test defaults; call mock(**knobs) to re-reset with knobs."""
    def reset(**knobs):
        SYSTEM.destroy_device()
        arena_api.mock_reset(**{**DEFAULT_KNOBS, **knobs})
        return SYSTEM
    reset()
    yield reset
    SYSTEM.destroy_device()
    arena_api.mock_reset()
