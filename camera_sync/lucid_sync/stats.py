"""Small, dependency-light statistics used for run.json and the sync report.

All functions accept empty input and then return None values instead of raising, because a
failed run (no pairs at all) must still produce a complete run.json.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

SUMMARY_KEYS = ("n", "mean", "median", "std", "p95_abs", "max_abs", "min", "max")


def _clean(values: Iterable[float | None]) -> np.ndarray:
    arr = np.array([float(v) for v in values if v is not None], dtype=np.float64)
    return arr[np.isfinite(arr)]


def summarize(values: Iterable[float | None], ndigits: int = 3) -> dict:
    """n, mean, median, std (population), p95 of |x|, max |x|, min, max.

    None and non-finite entries are ignored. Empty input -> n = 0 and all others None.
    """
    a = _clean(values)
    if a.size == 0:
        return {k: (0 if k == "n" else None) for k in SUMMARY_KEYS}
    absa = np.abs(a)
    out = dict(n=int(a.size), mean=float(a.mean()), median=float(np.median(a)),
               std=float(a.std()), p95_abs=float(np.percentile(absa, 95)),
               max_abs=float(absa.max()), min=float(a.min()), max=float(a.max()))
    return {k: (round(v, ndigits) if isinstance(v, float) else v) for k, v in out.items()}


def fps_from_timestamps(ts_ns: Sequence[int | None]) -> float | None:
    """Mean rate from first to last timestamp: (n - 1) / (last - first). None if < 2."""
    ts = sorted(int(t) for t in ts_ns if t is not None)
    if len(ts) < 2 or ts[-1] == ts[0]:
        return None
    return (len(ts) - 1) * 1e9 / (ts[-1] - ts[0])


def duration_s(ts_ns: Sequence[int | None]) -> float | None:
    ts = [int(t) for t in ts_ns if t is not None]
    if len(ts) < 2:
        return None
    return (max(ts) - min(ts)) / 1e9


def count_missing(present: Iterable[int], lo: int, hi: int,
                  expected: Iterable[int] | None = None) -> int:
    """How many grid indices in [lo, hi] (or in ``expected`` within [lo, hi]) are absent."""
    have = set(present)
    if hi < lo:
        return 0
    if expected is None:
        return sum(1 for k in range(lo, hi + 1) if k not in have)
    return sum(1 for k in set(expected) if lo <= k <= hi and k not in have)


def frame_id_gaps(frame_ids: Iterable[int]) -> int:
    """Frames skipped according to the camera's frame counter (sum of gaps - 1).

    The camera increments frame_id for every frame it produced, so a gap means a frame that
    was produced but never reached us (lost buffer / transfer). Duplicates are ignored.
    """
    ids = sorted(set(int(i) for i in frame_ids))
    return int(sum(b - a - 1 for a, b in zip(ids, ids[1:]) if b - a > 1))
