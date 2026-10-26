"""Pairing of Helios and Triton frames, and the per-frame / per-pair records.

Pairers, chosen by the sync method:

* `GridPairer` (ptpsync, action). Both cameras expose on ONE known time grid on the shared PTP
  clock: grid point k is at  anchor + k * period  (+ a deliberate per-camera lag, e.g. the
  Triton's TriggerDelay for exposure-centre alignment). Each frame gets
      k = round((timestamp - anchor - lag[camera]) / period)
  and frames with equal k form a pair.
* `TriggerPairer` (software trigger). k = index of the host-fired trigger the frame belongs
  to, found from the frame's host arrival time minus that camera's learned delivery latency.
* `NearestPairer` (freerun_ptp, freerun). Mutual-nearest timestamps within half a period
  (PTP time for freerun_ptp; HOST receive time for legacy freerun).

DROP DEFINITION (grid / trigger pairers; analysis.py uses the same):
  For each camera, dropped = number of EXPECTED grid indices k for which that camera
  delivered no buffer with a valid timestamp (complete OR incomplete). The expected set is
    - action / software: every trigger index k that was actually fired inside the capture
      window (skipped / not-armed triggers are not expected; they are reported separately);
    - ptpsync: every k whose planned exposure time lies in the capture window
      [PTP time latched when recording started + 1 period, PTP time latched when recording
      stopped]; if PTP time could not be latched: from the first to the last k received
      from EITHER camera (union, not overlap).
  Frames whose k lies outside the expected set are "edge_frames" (start-up / drain).
  Incomplete buffers count as delivered (not dropped) but are never paired; they are
  reported as "incomplete". Frames with an invalid timestamp, off-grid frames and
  duplicates have no usable k and are reported separately ("invalid_ts", "off_grid",
  "duplicates"); their k therefore counts as dropped if it was expected.
  frames_expected = |expected set|, frames_received = expected k present for that camera.

Pairers are not thread-safe; the engine calls them under one lock. They keep only a bounded
number of unmatched frames (images included) and give up on a frame after a horizon.
"""
from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterable

from . import stats

CAMERAS = ("helios", "triton")

#: Sync columns of frames.csv (one row per saved pair). The first 13 are DESIGN.md's.
PAIR_COLUMNS = ("pair_index", "grid_k", "helios_frame_id", "triton_frame_id", "helios_ts_ns",
                "triton_ts_ns", "offset_start_us", "triton_exposure_us",
                "offset_center_model_us", "helios_host_ns", "triton_host_ns", "host_delta_ms",
                "clock_domain", "planned_lag_us", "residual_us", "suspect", "suspect_reason",
                "segment_id")
#: Columns of sync_frames.csv (one row per received frame).
FRAME_COLUMNS = ("camera", "frame_id", "ts_ns", "host_ns", "grid_k", "incomplete", "paired",
                 "pair_index", "exposure_us", "note", "segment_id")

IMPLAUSIBLE_NS = 3600 * 10**9     # |ts - predicted| above one hour = implausible timestamp
TIME_BASE_ACCEPT = 5              # consecutive consistent implausible frames = new time base
SEGMENT_JUMP_PERIODS = 10         # a re-anchor jumping more than this starts a new segment


@dataclass(slots=True)
class FrameRecord:
    """Metadata of one received buffer (the pixels travel separately, see Pair.images)."""

    camera: str                      # "helios" | "triton"
    frame_id: int
    ts_ns: int                       # buffer.timestamp_ns as received (raw, never altered)
    host_ns: int                     # time.perf_counter_ns() when get_buffer() returned
    host_wall_ns: int = 0            # time.time_ns() at the same moment (for humans)
    incomplete: bool = False
    exposure_us: float | None = None
    exposure_source: str = ""        # "chunk" | "fixed_setting" | "cap_upper_bound" | ""
    width: int = 0
    height: int = 0
    pixel_format: str = ""
    size_filled: int = 0
    grab_ms: float = 0.0             # time blocked in get_buffer (waiting for the frame)
    copy_ms: float = 0.0             # time to copy the pixels out of the SDK buffer
    valid_ts: bool = True
    grid_k: int | None = None
    grid_residual_ns: float | None = None   # ts - expected grid time (grid pairers)
    segment_id: int = 0
    paired: bool = False
    pair_index: int | None = None
    note: str = ""

    def add_note(self, text: str) -> None:
        self.note = text if not self.note else f"{self.note};{text}"

    def csv_row(self) -> dict:
        return dict(camera=self.camera, frame_id=self.frame_id, ts_ns=self.ts_ns,
                    host_ns=self.host_ns, grid_k=_blank(self.grid_k),
                    incomplete=int(self.incomplete), paired=int(self.paired),
                    pair_index=_blank(self.pair_index), exposure_us=_blank(self.exposure_us),
                    note=self.note, segment_id=self.segment_id)


@dataclass(slots=True)
class Pair:
    """One Helios frame + one Triton frame taken for the same instant."""

    pair_index: int
    grid_k: int | None
    helios: FrameRecord
    triton: FrameRecord
    clock_domain: str                # "ptp" (one shared clock) | "independent"
    helios_center_us: float | None
    planned_lag_ns: int = 0          # commanded Triton-minus-Helios exposure-start lag
    segment_id: int = 0
    suspect: bool = False
    suspect_reason: str = ""
    images: dict[str, Any] | None = None   # {"helios": ndarray, "triton": ndarray}; freed
    # "pending" -> "written" | "writer_dropped" | "writer_error" | "not_saved" |
    #              "rejected_suspect"
    status: str = "pending"
    save_index: int | None = None    # file index given by the engine (capture.py "frame")
    queue_depth: int = 0
    queued: bool = False

    @property
    def offset_start_us(self) -> float | None:
        """MEASURED timestamp offset triton_ts - helios_ts on the shared PTP clock (None
        without PTP). What each timestamp marks (exposure start?) is NOT confirmed."""
        if self.clock_domain != "ptp":
            return None
        return (self.triton.ts_ns - self.helios.ts_ns) / 1000.0

    @property
    def planned_lag_us(self) -> float:
        return self.planned_lag_ns / 1000.0

    @property
    def residual_us(self) -> float | None:
        """MEASURED: offset_start_us - planned lag = how precisely the cameras hit the plan."""
        o = self.offset_start_us
        return None if o is None else o - self.planned_lag_us

    @property
    def triton_exposure_us(self) -> float | None:
        return self.triton.exposure_us

    @property
    def offset_center_model_us(self) -> float | None:
        """MODELLED (not measured) offset between the exposure centres."""
        e = self.triton.exposure_us
        if self.clock_domain != "ptp" or e is None or self.helios_center_us is None:
            return None
        # Subtract the integer timestamps FIRST: PTP ns values (~1e18) exceed float64's exact
        # range, so adding a float to them would round to 256 ns steps.
        start_diff_ns = self.triton.ts_ns - self.helios.ts_ns
        return (start_diff_ns + e * 1000.0 / 2.0 - self.helios_center_us * 1000.0) / 1000.0

    @property
    def host_delta_ms(self) -> float:
        """Host receive-time difference; NOT a sync metric when PTP is on."""
        return (self.triton.host_ns - self.helios.host_ns) / 1e6

    def csv_row(self) -> dict:
        return dict(pair_index=self.pair_index, grid_k=_blank(self.grid_k),
                    helios_frame_id=self.helios.frame_id, triton_frame_id=self.triton.frame_id,
                    helios_ts_ns=self.helios.ts_ns, triton_ts_ns=self.triton.ts_ns,
                    offset_start_us=_blank(_r(self.offset_start_us)),
                    triton_exposure_us=_blank(self.triton_exposure_us),
                    offset_center_model_us=_blank(_r(self.offset_center_model_us)),
                    helios_host_ns=self.helios.host_ns, triton_host_ns=self.triton.host_ns,
                    host_delta_ms=_r(self.host_delta_ms), clock_domain=self.clock_domain,
                    planned_lag_us=_r(self.planned_lag_us),
                    residual_us=_blank(_r(self.residual_us)), suspect=int(self.suspect),
                    suspect_reason=self.suspect_reason, segment_id=self.segment_id)


def _blank(v: Any) -> Any:
    return "" if v is None else v


def _r(v: float | None, nd: int = 3) -> float | None:
    return None if v is None else round(v, nd)


# =========================================================================== helpers
class _TsSanity:
    """Per-camera timestamp plausibility (review item 4).

    A timestamp <= 0 is invalid. Otherwise (ts - host_ns) of a camera changes only by clock
    drift, so a value more than IMPLAUSIBLE_NS away from the running median of recent frames
    is invalid - unless TIME_BASE_ACCEPT consecutive frames agree on the new value (within
    1 s): then the camera's time base really changed and the new base is accepted.
    """

    def __init__(self) -> None:
        self.hist: deque[int] = deque(maxlen=15)
        self.pending: list[int] = []

    def check(self, rec: FrameRecord) -> bool:
        if rec.ts_ns <= 0:
            return False
        d = int(rec.ts_ns) - int(rec.host_ns)
        if self.hist and abs(d - statistics.median(self.hist)) > IMPLAUSIBLE_NS:
            self.pending.append(d)
            if (len(self.pending) >= TIME_BASE_ACCEPT
                    and max(self.pending) - min(self.pending) < 10**9):
                self.hist.clear()
                self.hist.extend(self.pending)
                self.pending = []
                rec.add_note("time_base_step_accepted")
                return True
            return False
        self.pending = []
        self.hist.append(d)
        return True


@dataclass
class _Entry:
    rec: FrameRecord
    image: Any


class _PairerBase:
    def __init__(self, clock_domain: str, helios_center_us: float | None,
                 planned_lag_ns: int = 0, cameras: tuple[str, str] = CAMERAS) -> None:
        self.clock_domain = clock_domain
        self.helios_center_us = None if helios_center_us is None else float(helios_center_us)
        self.planned_lag_ns = int(planned_lag_ns)
        self.cameras = cameras
        self.frames: list[FrameRecord] = []     # every frame given to add(), arrival order
        self.pairs: list[Pair] = []
        self._next_index = 0
        self.unpaired: dict[str, int] = {c: 0 for c in cameras}
        self.invalid_ts: dict[str, int] = {c: 0 for c in cameras}
        self._sanity = {c: _TsSanity() for c in cameras}
        self.segment = 0

    def _valid(self, rec: FrameRecord) -> bool:
        """Timestamp sanity; invalid frames are recorded but never paired."""
        if self._sanity[rec.camera].check(rec):
            return True
        rec.valid_ts = False
        rec.add_note("invalid_ts")
        self.invalid_ts[rec.camera] += 1
        return False

    def _make_pair(self, k: int | None, a: _Entry, b: _Entry) -> Pair:
        ents = {a.rec.camera: a, b.rec.camera: b}
        h, t = ents["helios"], ents["triton"]
        images = None
        if h.image is not None or t.image is not None:
            images = {"helios": h.image, "triton": t.image}
        p = Pair(pair_index=self._next_index, grid_k=k, helios=h.rec, triton=t.rec,
                 clock_domain=self.clock_domain, helios_center_us=self.helios_center_us,
                 planned_lag_ns=self.planned_lag_ns, segment_id=h.rec.segment_id,
                 images=images)
        self._next_index += 1
        for e in (h, t):
            e.rec.paired = True
            e.rec.pair_index = p.pair_index
        self.pairs.append(p)
        return p

    def _give_up(self, e: _Entry, why: str) -> None:
        e.rec.add_note(why)
        e.image = None
        self.unpaired[e.rec.camera] += 1


# =========================================================================== grid
class GridPairer(_PairerBase):
    """Pair frames by their index on a known PTP time grid (ptpsync / action).

    Parameters
    ----------
    period_ns: grid period (1e9 / FRAME_RATE).
    lags_ns: deliberate per-camera delay relative to the grid, e.g. {"helios": 0,
        "triton": 5_350_000} for centre alignment. Small fixed offsets (trigger latency,
        LUCID's ~18 us timestamp bookkeeping offset) need no entry: rounding absorbs them.
    anchor_ns: PTP time of grid point k = 0. Known for action commands (the scheduler's T0).
        None = learn it from the first complete frame (ptpsync: the firmware's grid origin
        is not documented, so we do not assume it).
    horizon: an unmatched frame is given up once a frame with k > its k + horizon arrived.
    reanchor_after: only for a LEARNED anchor - after this many consecutive off-grid frames
        the grid phase is re-learned. The new k continues from that camera's last on-grid
        frame plus the elapsed HOST time in periods, so a PTP time step never fabricates
        thousands of drops. A jump of more than SEGMENT_JUMP_PERIODS periods starts a new
        segment (segment_id). A known anchor (action: T0) is never moved.
    """

    def __init__(self, period_ns: float, lags_ns: dict[str, int] | None = None,
                 anchor_ns: int | None = None, horizon: int = 30, clock_domain: str = "ptp",
                 helios_center_us: float | None = 9350.0, cameras: tuple[str, str] = CAMERAS,
                 reanchor_after: int = 3, planned_lag_ns: int = 0) -> None:
        super().__init__(clock_domain, helios_center_us, planned_lag_ns, cameras)
        # Exact rational period: 1e9/15 is not an integer number of ns, and PTP timestamps
        # are ~1e18 ns, too large for float arithmetic at ns resolution.
        self.period = Fraction(period_ns).limit_denominator(1000)
        self.lags = {c: int((lags_ns or {}).get(c, 0)) for c in cameras}
        self.anchor = None if anchor_ns is None else int(anchor_ns)
        self.anchor_learned = anchor_ns is None
        self.horizon = int(horizon)
        self.pending: dict[int, dict[str, _Entry]] = {}
        self.seen: dict[str, set[int]] = {c: set() for c in cameras}          # complete
        self.seen_incomplete: dict[str, set[int]] = {c: set() for c in cameras}
        self.max_k: int | None = None
        self.given_up_below: int | None = None   # every k below this was flushed
        self.off_grid: dict[str, int] = {c: 0 for c in cameras}
        self.duplicates: dict[str, int] = {c: 0 for c in cameras}
        self.reanchor_after = int(reanchor_after)
        self.reanchors: list[dict] = []
        self._offgrid_run = 0
        self._last_good: dict[str, tuple[int, int]] = {}      # cam -> (k, host_ns)

    # ------------------------------------------------------------------ grid maths
    def grid_index(self, camera: str, ts_ns: int) -> tuple[int, float]:
        """(k, residual_ns) of a timestamp. Requires a known anchor."""
        d = Fraction(int(ts_ns) - self.anchor - self.lags[camera])
        k = math.floor(d / self.period + Fraction(1, 2))
        return k, float(d - k * self.period)

    def _index(self, rec: FrameRecord) -> tuple[int, float] | None:
        """(k, residual_ns) of a frame; None if it cannot be placed yet."""
        if self.anchor is None:
            if rec.incomplete:
                return None
            self.anchor = int(rec.ts_ns) - self.lags[rec.camera]
        return self.grid_index(rec.camera, rec.ts_ns)

    def grid_time(self, camera: str, k: int) -> int:
        return self.anchor + self.lags[camera] + int(round(k * self.period))

    @property
    def grid_phase_ns(self) -> float | None:
        """anchor modulo period: where the grid sits relative to PTP-time multiples."""
        if self.anchor is None:
            return None
        return float(Fraction(self.anchor) % self.period)

    # ------------------------------------------------------------------ input
    def add(self, rec: FrameRecord, image: Any = None) -> list[Pair]:
        """Add one frame; return the pairs completed by it (possibly empty)."""
        self.frames.append(rec)
        cam = rec.camera
        rec.segment_id = self.segment
        if not self._valid(rec):
            return []
        placed = self._index(rec)
        if placed is None:                           # incomplete before any anchor
            rec.add_note("incomplete")
            return []
        k, resid = placed
        if abs(resid) > float(self.period) / 4:
            self._offgrid_run += 1
            if self.anchor_learned and self._offgrid_run >= self.reanchor_after:
                k, resid = self._reanchor(rec)
            else:
                # Not near any grid point (e.g. a late action command executed at once):
                # pairing it would produce a wrong pair, so it is recorded, never paired.
                rec.add_note("off_grid")
                self.off_grid[cam] += 1
                self.unpaired[cam] += 1
                return []
        self._offgrid_run = 0
        rec.grid_k, rec.grid_residual_ns, rec.segment_id = k, resid, self.segment
        if rec.incomplete:
            rec.add_note("incomplete")
            self.seen_incomplete[cam].add(k)
            return []
        if k in self.seen[cam]:
            rec.add_note("duplicate_k")
            self.duplicates[cam] += 1
            self.unpaired[cam] += 1
            return []
        self.seen[cam].add(k)
        self._last_good[cam] = (k, rec.host_ns)
        if self.given_up_below is not None and k < self.given_up_below:
            self._give_up(_Entry(rec, None), "late_partner_given_up")
            return []
        slot = self.pending.setdefault(k, {})
        slot[cam] = _Entry(rec, image)
        out: list[Pair] = []
        if all(c in slot for c in self.cameras):
            del self.pending[k]
            out.append(self._make_pair(k, slot[self.cameras[0]], slot[self.cameras[1]]))
        self.max_k = k if self.max_k is None else max(self.max_k, k)
        self._flush(self.max_k - self.horizon)
        return out

    def _reanchor(self, rec: FrameRecord) -> tuple[int, float]:
        """Re-learn the grid phase so this frame sits on a grid point, keeping k continuous
        via host time (review item 8)."""
        cam, P = rec.camera, float(self.period)
        last = self._last_good.get(cam) or next(iter(self._last_good.values()), None)
        if last is not None:
            k_new = last[0] + int(round((rec.host_ns - last[1]) / P))
        else:
            k_new = self.grid_index(cam, rec.ts_ns)[0]
        old = self.anchor
        self.anchor = int(rec.ts_ns) - self.lags[cam] - int(round(k_new * self.period))
        shift = self.anchor - old
        new_segment = abs(shift) > SEGMENT_JUMP_PERIODS * P
        if new_segment:
            self.segment += 1
        self.reanchors.append(dict(at_frame=len(self.frames) - 1, camera=cam, k=k_new,
                                   shift_ns=shift, segment=self.segment))
        rec.add_note("reanchor")
        return self.grid_index(cam, rec.ts_ns)

    def _flush(self, below: int) -> None:
        """Give up on every pending k < below (its partner will not come any more)."""
        for k in sorted(kk for kk in self.pending if kk < below):
            for e in self.pending.pop(k).values():
                self._give_up(e, "no_partner")
        if self.given_up_below is None or below > self.given_up_below:
            self.given_up_below = below

    # ------------------------------------------------------------------ end of run
    def k_window(self, ptp_lo: int | None, ptp_hi: int | None) -> dict[str, tuple[int, int]] | None:
        """Per-camera [k_lo, k_hi] whose grid time lies in [ptp_lo, ptp_hi] (None if unknown
        or if the grid moved by more than a quarter period during the run)."""
        if self.anchor is None or ptp_lo is None or ptp_hi is None:
            return None
        if any(abs(r["shift_ns"]) > float(self.period) / 4 for r in self.reanchors):
            return None
        out = {}
        for c in self.cameras:
            lo = math.ceil(Fraction(ptp_lo - self.anchor - self.lags[c]) / self.period)
            hi = math.floor(Fraction(ptp_hi - self.anchor - self.lags[c]) / self.period)
            out[c] = (lo, hi)
        return out

    def finalize(self, expected_k: Iterable[int] | None = None,
                 k_window: dict[str, tuple[int, int]] | None = None) -> tuple[list[Pair], dict]:
        """Flush everything; return (pairs completed now (always []), drop accounting).

        expected_k: grid indices actually requested (action/software: fired triggers).
        k_window: per-camera [lo, hi] of expected k (ptpsync, see k_window()).
        Neither: union of the k received from either camera. See the module docstring.
        """
        self._flush(10**18)
        have = {c: self.seen[c] | self.seen_incomplete[c] for c in self.cameras}
        out: dict = dict(dropped={c: None for c in self.cameras},
                         frames_expected={c: None for c in self.cameras},
                         frames_received={c: 0 for c in self.cameras},
                         edge_frames={c: 0 for c in self.cameras},
                         window_k=None, window_source=None, off_grid=dict(self.off_grid),
                         duplicates=dict(self.duplicates), unpaired=dict(self.unpaired),
                         invalid_ts=dict(self.invalid_ts), reanchors=list(self.reanchors[:50]),
                         segments=self.segment + 1,
                         anchor_ns=self.anchor, anchor_learned=self.anchor_learned,
                         grid_phase_ns=self.grid_phase_ns, period_ns=float(self.period),
                         lags_ns=dict(self.lags))
        all_k = set().union(*have.values())
        if expected_k is not None:
            exp = set(expected_k)
            windows = {c: exp for c in self.cameras}
            out["window_source"] = "fired triggers"
            out["window_k"] = [min(exp), max(exp)] if exp else None
        else:
            if k_window is not None:
                out["window_source"] = "PTP-latched capture window"
            elif all_k:
                lo, hi = min(all_k), max(all_k)
                k_window = {c: (lo, hi) for c in self.cameras}
                out["window_source"] = "union of received frames"
            if k_window is None:
                return [], out
            # Guard against absurd ranges (review item 8): never loop over millions of k.
            span = max(hi - lo for lo, hi in k_window.values())
            if span > 10 * (len(self.frames) + 100):
                out["window_source"] = "invalid (span too large)"
                return [], out
            windows = {c: set(range(lo, hi + 1)) for c, (lo, hi) in k_window.items()}
            out["window_k"] = {c: list(v) for c, v in k_window.items()}
        out["_expected"] = windows            # internal (engine): sets, not written to JSON
        for c in self.cameras:
            exp_c = windows[c]
            got = have[c] & exp_c
            out["frames_expected"][c] = len(exp_c)
            out["frames_received"][c] = len(got)
            out["dropped"][c] = len(exp_c) - len(got)
            out["edge_frames"][c] = len(have[c] - exp_c)
        return [], out


# =========================================================================== trigger
class TriggerPairer(GridPairer):
    """Pair frames of host-paced software triggers by trigger index k.

    `fires` is the SoftwareTrigger (first_fired() / nearest_fired(host_ns)). A camera's
    first frame is assumed to belong to the first fired trigger (the pacer waits for
    TriggerArmed before every trigger); after that each frame is assigned to the fired
    trigger nearest to (host arrival - that camera's median delivery latency). This stays
    correct when the latency exceeds a period (e.g. 78 ms at 15 fps).
    """

    def __init__(self, period_ns: float, fires: Any, **kw: Any) -> None:
        kw.setdefault("reanchor_after", 10**9)
        super().__init__(period_ns, anchor_ns=0, **kw)
        self.anchor_learned = False
        self.fires = fires
        self._latency: dict[str, deque[int]] = {c: deque(maxlen=25) for c in self.cameras}
        self.latency_ms: dict[str, float | None] = {c: None for c in self.cameras}

    def _index(self, rec: FrameRecord) -> tuple[int, float] | None:
        lat = self._latency[rec.camera]
        if not lat:
            first = self.fires.first_fired()
            if first is None:
                return None
            k, t_fire = first
        else:
            near = self.fires.nearest_fired(rec.host_ns - int(statistics.median(lat)))
            if near is None:
                return None
            k, t_fire = near
        latency = rec.host_ns - t_fire
        resid = float(latency - (statistics.median(lat) if lat else latency))
        if abs(resid) <= float(self.period) / 4:      # incomplete frames teach latency too
            lat.append(latency)
            self.latency_ms[rec.camera] = round(statistics.median(lat) / 1e6, 3)
        return k, resid

    def grid_time(self, camera: str, k: int) -> int:     # not a PTP grid
        raise NotImplementedError

    def finalize(self, expected_k: Iterable[int] | None = None,
                 k_window: dict | None = None) -> tuple[list[Pair], dict]:
        new, out = super().finalize(expected_k, None)
        out["anchor_ns"] = out["grid_phase_ns"] = None
        out["delivery_latency_ms"] = dict(self.latency_ms)
        return new, out


# =========================================================================== nearest
class NearestPairer(_PairerBase):
    """Mutual-nearest-timestamp pairing for free-running cameras.

    key = "ts_ns" (freerun_ptp: device timestamps on the shared PTP clock) or "host_ns"
    (legacy freerun: host receive time, the only common clock). A pair needs
    |t_a - t_b| <= tolerance (default: half a period) and must be mutually nearest, i.e.
    neither frame has a closer candidate in the other camera's stream.
    """

    def __init__(self, period_ns: float, key: str = "ts_ns", clock_domain: str = "ptp",
                 tolerance_ns: float | None = None, max_pending: int = 30,
                 helios_center_us: float | None = 9350.0, cameras: tuple[str, str] = CAMERAS,
                 planned_lag_ns: int = 0) -> None:
        super().__init__(clock_domain, helios_center_us, planned_lag_ns, cameras)
        if key not in ("ts_ns", "host_ns"):
            raise ValueError("key must be 'ts_ns' or 'host_ns'")
        self.key = key
        self.period_ns = float(period_ns)
        self.tol = float(tolerance_ns) if tolerance_ns is not None else self.period_ns / 2
        self.max_pending = int(max_pending)
        self.queues: dict[str, deque[_Entry]] = {c: deque() for c in cameras}

    def _t(self, e: _Entry) -> int:
        return getattr(e.rec, self.key)

    def add(self, rec: FrameRecord, image: Any = None) -> list[Pair]:
        self.frames.append(rec)
        if not self._valid(rec):
            return []
        if rec.incomplete:
            rec.add_note("incomplete")
            return []
        q = self.queues[rec.camera]
        e = _Entry(rec, image)
        # Keep each queue sorted by time (frames of one camera normally arrive in order).
        if q and self._t(q[-1]) > self._t(e):
            items = sorted(list(q) + [e], key=self._t)
            q.clear()
            q.extend(items)
        else:
            q.append(e)
        out = self._match(final=False)
        for c in self.cameras:                 # bounded memory if one camera stalls/dies
            while len(self.queues[c]) > self.max_pending:
                self._give_up(self.queues[c].popleft(), "no_partner")
        return out

    def _match(self, final: bool) -> list[Pair]:
        out: list[Pair] = []
        c0, c1 = self.cameras
        while self.queues[c0] and self.queues[c1]:
            x = c0 if self._t(self.queues[c0][0]) <= self._t(self.queues[c1][0]) else c1
            y = c1 if x == c0 else c0
            qx, qy = self.queues[x], self.queues[y]
            a = qx[0]                          # earliest unmatched frame overall
            if not final and self._t(qy[-1]) < self._t(a) + self.tol:
                break                          # y may still deliver a candidate for a
            b = qy[0]                          # nearest y frame at or after a
            gap = self._t(b) - self._t(a)
            if gap > self.tol:
                self._give_up(qx.popleft(), "no_partner")
                continue
            if len(qx) >= 2:
                a2 = qx[1]
                if abs(self._t(a2) - self._t(b)) < gap:
                    self._give_up(qx.popleft(), "no_partner")   # b prefers the next x frame
                    continue
            elif not final:
                break                          # x's next frame might be nearer to b
            qx.popleft()
            qy.popleft()
            out.append(self._make_pair(None, a, b))
        return out

    def finalize(self, expected_k: Iterable[int] | None = None,
                 k_window: dict | None = None) -> tuple[list[Pair], dict]:
        """Pair what can still be paired, give up the rest; return (new pairs, summary).
        Drops for free-run are estimated by the engine (frame_id gaps + head/tail)."""
        new = self._match(final=True)
        for c in self.cameras:
            while self.queues[c]:
                self._give_up(self.queues[c].popleft(), "no_partner")
        return new, dict(dropped={c: None for c in self.cameras},
                         edge_frames={c: 0 for c in self.cameras}, window_k=None,
                         unpaired=dict(self.unpaired), invalid_ts=dict(self.invalid_ts),
                         tolerance_ns=self.tol, key=self.key)


# =========================================================================== guard
class PairGuard:
    """Consistency check that does NOT depend on the PTP clock (review item 1).

    A PTP step of n periods on one camera shifts its timestamps by n*P while its exposures
    stay on the grid; grid pairing would then silently combine frames n*P apart with a
    perfect-looking offset. Two independent checks catch that:

    * host receive-time difference: must stay within half a period of its running median
      (median of the last `window` accepted pairs, checked once `min_samples` exist);
    * frame counters (grid modes): between two accepted pairs, neither camera's frame_id may
      advance by more than the grid index (a camera cannot expose more often than the grid);
      free-run: the two cameras' frame_id steps may differ by at most 1.

    Only used when the clock domain is "ptp" (legacy freerun pairs BY host time already).
    """

    def __init__(self, period_ns: float, window: int = 25, min_samples: int = 5) -> None:
        self.period_ns = float(period_ns)
        self.hd: deque[float] = deque(maxlen=window)
        self.min_samples = min_samples
        self.last: Pair | None = None
        self.flagged = 0

    def check(self, p: Pair) -> str:
        reasons = []
        hd = (p.triton.host_ns - p.helios.host_ns)
        if len(self.hd) >= self.min_samples:
            med = statistics.median(self.hd)
            if abs(hd - med) > self.period_ns / 2:
                reasons.append(f"host_delta {hd / 1e6:.1f} ms vs median {med / 1e6:.1f} ms")
        last = self.last
        if last is not None:
            dh = p.helios.frame_id - last.helios.frame_id
            dt = p.triton.frame_id - last.triton.frame_id
            if p.grid_k is not None and last.grid_k is not None:
                dk = p.grid_k - last.grid_k
                if dk > 0:
                    for cam, d in (("helios", dh), ("triton", dt)):
                        if d > dk:
                            reasons.append(f"{cam} frame_id step {d} > grid step {dk}")
            elif dh > 0 and dt > 0 and abs(dh - dt) > 1:
                reasons.append(f"frame_id steps differ (helios {dh}, triton {dt})")
        if reasons:
            self.flagged += 1
            return "; ".join(reasons)
        self.hd.append(hd)
        if last is None or p.grid_k is None or last.grid_k is None or p.grid_k > last.grid_k:
            self.last = p
        return ""


def frame_id_gaps(frames: list[FrameRecord]) -> int:
    return stats.frame_id_gaps(f.frame_id for f in frames)
