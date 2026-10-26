"""Unit tests of GridPairer / NearestPairer with synthetic timestamps (no SDK needed)."""
from __future__ import annotations

import random

import pytest

from lucid_sync.pairing import (FRAME_COLUMNS, PAIR_COLUMNS, FrameRecord, GridPairer,
                                NearestPairer, PairGuard, TriggerPairer)

P = 100_000_000                    # 10 fps
BASE = 1_761_000_000_000_000_000   # ~2025 in PTP ns: far beyond float64 ns resolution
TRI_BIAS = 17_700                  # LUCID's ~18 us Triton-after-Helios timestamp offset


def frame(cam, fid, ts, host=None, inc=False, exp=None):
    return FrameRecord(camera=cam, frame_id=fid, ts_ns=int(ts),
                       host_ns=int(host if host is not None else ts % 10**12), incomplete=inc,
                       exposure_us=exp)


def stream(cam, ks, t0=BASE, period=P, bias=0, jitter_ns=1000, rng=None, lag=0):
    rng = rng or random.Random(1)
    return [frame(cam, k + 1, t0 + lag + round(k * period) + bias + rng.gauss(0, jitter_ns))
            for k in ks]


def feed(pairer, frames):
    pairs = []
    for f in frames:
        pairs += pairer.add(f)
    new, summary = pairer.finalize()
    return pairs + new, summary


def interleave(a, b):
    out = []
    for x, y in zip(a, b):
        out += [x, y]
    return out + a[len(b):] + b[len(a):]


# --------------------------------------------------------------------------- grid
def test_grid_perfect_pairs_large_ns_values():
    h = stream("helios", range(100))
    t = stream("triton", range(100), bias=TRI_BIAS, rng=random.Random(2))
    pairs, s = feed(GridPairer(P), interleave(h, t))
    assert len(pairs) == 100
    assert all(p.helios.grid_k == p.triton.grid_k == p.grid_k for p in pairs)
    assert all(abs(p.offset_start_us - 17.7) < 6 for p in pairs)
    assert s["dropped"] == {"helios": 0, "triton": 0}
    assert s["window_k"] == {"helios": [0, 99], "triton": [0, 99]}
    assert s["frames_expected"] == {"helios": 100, "triton": 100}


def test_grid_drops_counted_exactly_and_unpaired_noted():
    h = stream("helios", range(50))
    t = stream("triton", [k for k in range(50) if k not in (3, 4, 20)], bias=TRI_BIAS)
    g = GridPairer(P, horizon=5)
    pairs, s = feed(g, interleave(h, t))
    assert len(pairs) == 47
    assert s["dropped"] == {"helios": 0, "triton": 3}
    lonely = [f for f in g.frames if f.camera == "helios" and not f.paired]
    assert sorted(f.grid_k for f in lonely) == [3, 4, 20]
    assert all("no_partner" in f.note for f in lonely)


def test_grid_out_of_order_arrival_between_cameras():
    h = stream("helios", range(60))
    t = stream("triton", range(60), bias=TRI_BIAS)
    # Triton frames arrive in bursts of 8, ~8 periods behind the Helios ones.
    order = []
    for i in range(0, 60, 8):
        order += h[i:i + 8]
        if i >= 8:
            order += t[i - 8:i]
    order += t[56:]                       # the last burst
    pairs, s = feed(GridPairer(P, horizon=20), order)
    assert len(pairs) == 60 and s["dropped"] == {"helios": 0, "triton": 0}


def test_grid_late_start_counts_as_drops():
    """Review item 3: a camera that starts late (or dies early) must show up as drops."""
    h = stream("helios", range(40))
    t = stream("triton", range(5, 40), bias=TRI_BIAS)       # triton stream started later
    pairs, s = feed(GridPairer(P, horizon=10), interleave(h, t))
    assert len(pairs) == 35
    assert s["dropped"] == {"helios": 0, "triton": 5}
    assert s["window_source"] == "union of received frames"
    # with a PTP-latched window that starts at k=5, the first 5 Helios frames are edges
    g = GridPairer(P, horizon=10)
    for f in interleave(h, t):
        g.add(f)
    win = g.k_window(BASE + 5 * P - P // 3, BASE + 39 * P + P // 3)
    _, s2 = g.finalize(k_window=win)
    assert s2["dropped"] == {"helios": 0, "triton": 0}
    assert s2["edge_frames"] == {"helios": 5, "triton": 0}
    # first frame of the TRITON arriving first must work too (learned anchor)
    pairs2, s2 = feed(GridPairer(P), interleave(t, h))
    assert len(pairs2) == 35


def test_grid_incomplete_not_paired_and_not_a_drop():
    h = stream("helios", range(30))
    t = stream("triton", range(30), bias=TRI_BIAS)
    for i in (7, 8):
        t[i].incomplete = True
    g = GridPairer(P)
    pairs, s = feed(g, interleave(h, t))
    assert len(pairs) == 28
    assert all(not p.triton.incomplete for p in pairs)
    assert s["dropped"] == {"helios": 0, "triton": 0}
    assert [f.grid_k for f in g.frames if f.incomplete] == [7, 8]


def test_grid_known_anchor_and_lags_centre_alignment():
    lag = 6_850_000                                        # Triton TriggerDelay 6.85 ms
    t0 = BASE + 12_345
    h = stream("helios", range(20), t0=t0, bias=23_400)
    t = stream("triton", range(20), t0=t0, bias=41_100, lag=lag)
    g = GridPairer(P, lags_ns={"helios": 0, "triton": lag}, anchor_ns=t0)
    pairs, _ = feed(g, interleave(h, t))
    assert len(pairs) == 20 and [p.grid_k for p in pairs] == list(range(20))
    assert all(abs(p.offset_start_us - (lag / 1000 + 17.7)) < 6 for p in pairs)
    res = [f.grid_residual_ns for f in g.frames if f.camera == "helios"]
    assert abs(sum(res) / len(res) - 23_400) < 1000      # residual = trigger latency


def test_grid_only_helios_delayed_by_period_minus_lag():
    """ptpsync with PTPSyncOffset only on the Helios: Helios at (P - d) + kP, Triton at kP."""
    d = 6_850_000
    h = stream("helios", range(30), t0=BASE + (P - d))
    t = stream("triton", range(1, 31), t0=BASE, bias=TRI_BIAS)
    g = GridPairer(P, lags_ns={"helios": 0, "triton": d})
    pairs, s = feed(g, interleave(h, t))
    assert len(pairs) == 30
    assert all(abs(p.offset_start_us - (d / 1000 + 17.7)) < 6 for p in pairs)
    assert s["dropped"] == {"helios": 0, "triton": 0}


def test_grid_non_integer_period_15fps():
    period = 1e9 / 15
    h = stream("helios", range(300), period=period)
    t = stream("triton", range(300), period=period, bias=TRI_BIAS)
    pairs, s = feed(GridPairer(period), interleave(h, t))
    assert len(pairs) == 300 and s["dropped"]["triton"] == 0


def test_grid_bounded_memory_when_one_camera_dies():
    h = stream("helios", range(500))
    t = stream("triton", range(10), bias=TRI_BIAS)
    g = GridPairer(P, horizon=15)
    for f in interleave(h, t):
        g.add(f)
        assert len(g.pending) <= 16
    _, s = g.finalize()
    assert s["dropped"] == {"helios": 0, "triton": 490}            # dead camera = drops
    assert s["frames_expected"]["triton"] == 500 and s["frames_received"]["triton"] == 10


def test_grid_off_grid_and_duplicate_rejected():
    h = stream("helios", range(10))
    t = stream("triton", range(10), bias=TRI_BIAS)
    t.insert(5, frame("triton", 99, BASE + 4 * P + P // 2 - 1000))   # half a period off
    t.insert(7, frame("triton", 98, BASE + 6 * P + 5000))             # second frame for k=6
    g = GridPairer(P)
    pairs, s = feed(g, h + t)
    assert len(pairs) == 10
    assert s["off_grid"]["triton"] == 1 and s["duplicates"]["triton"] == 1


# --------------------------------------------------------------------------- nearest
def test_nearest_freerun_ptp_phase_and_drift():
    rng = random.Random(3)
    phase = 31_000_000
    h = [frame("helios", i + 1, BASE + i * P + rng.gauss(0, 2000)) for i in range(100)]
    # Triton slightly faster (period 99.99 ms) and 31 ms later; frames 40-42 lost.
    t = [frame("triton", i + 1, BASE + phase + i * 99_990_000) for i in range(100)
         if i not in (40, 41, 42)]
    n = NearestPairer(P, key="ts_ns")
    pairs, _ = feed(n, interleave(h, t))
    assert len(pairs) == 97
    assert all(abs(p.triton.ts_ns - p.helios.ts_ns) <= P / 2 for p in pairs)
    assert all(p.grid_k is None for p in pairs)
    assert n.unpaired == {"helios": 3, "triton": 0}


def test_nearest_mutual_nearest_choice():
    # helios at 0 and 60 ms, triton at 55 ms: triton belongs to the helios frame at 60 ms
    h = [frame("helios", 1, BASE), frame("helios", 2, BASE + 60_000_000),
         frame("helios", 3, BASE + 160_000_000)]
    t = [frame("triton", 1, BASE + 55_000_000), frame("triton", 2, BASE + 158_000_000)]
    pairs, _ = feed(NearestPairer(P), [h[0], t[0], h[1], t[1], h[2]])
    assert [(p.helios.frame_id, p.triton.frame_id) for p in pairs] == [(2, 1), (3, 2)]


def test_nearest_host_time_legacy_freerun():
    h = [frame("helios", i, ts=5_000_000_000 + i * P, host=i * P + 51_000_000) for i in range(30)]
    t = [frame("triton", i, ts=9_999_000_000 + i * P, host=i * P + 70_000_000) for i in range(30)]
    n = NearestPairer(P, key="host_ns", clock_domain="independent")
    pairs, _ = feed(n, interleave(h, t))
    assert len(pairs) == 30
    assert pairs[0].offset_start_us is None and pairs[0].offset_center_model_us is None
    assert abs(pairs[0].host_delta_ms - 19.0) < 1e-6
    row = pairs[0].csv_row()
    assert tuple(row) == PAIR_COLUMNS and row["offset_start_us"] == ""
    assert tuple(n.frames[0].csv_row()) == FRAME_COLUMNS


def test_nearest_bounded_when_one_camera_silent():
    n = NearestPairer(P, max_pending=10)
    for i in range(200):
        n.add(frame("helios", i, BASE + i * P))
        assert len(n.queues["helios"]) <= 10
    pairs, _ = n.finalize()
    assert pairs == [] and n.unpaired["helios"] == 200


def test_pair_center_model_offset():
    h = frame("helios", 1, BASE)
    t = frame("triton", 1, BASE + 6_850_000, exp=5000.0)
    g = GridPairer(P, lags_ns={"triton": 6_850_000}, anchor_ns=BASE, helios_center_us=9350)
    (p,) = g.add(h) + g.add(t)
    # triton centre = 6850 + 2500 = 9350 us after the helios start -> 0 modelled offset
    assert p.offset_center_model_us == pytest.approx(0.0)
    assert p.offset_start_us == pytest.approx(6850.0)


def test_grid_reanchors_after_ptp_time_step_learned_anchor_only():
    step = 37_000_000                       # e.g. a new PTP master: time base jumps 37 ms
    h = stream("helios", range(60))
    t = stream("triton", range(60), bias=TRI_BIAS)
    for f in h[30:] + t[30:]:
        f.ts_ns += step
    g = GridPairer(P)
    pairs, s = feed(g, interleave(h, t))
    assert len(s["reanchors"]) == 1 and s["reanchors"][0]["shift_ns"] == pytest.approx(step, abs=5000)
    assert len(pairs) >= 58                 # only the frames before the re-anchor are lost
    assert all(abs(p.offset_start_us - 17.7) < 6 for p in pairs)
    # a KNOWN anchor (action T0) is never moved: shifted frames stay off-grid, unpaired
    g2 = GridPairer(P, anchor_ns=BASE)
    pairs2, s2 = feed(g2, interleave(h, t))
    assert len(pairs2) == 30 and s2["reanchors"] == [] and s2["off_grid"]["triton"] == 30


# --------------------------------------------------------------------------- review repros
def host_stream(cam, ks, bias_ts=0, host_bias=20_000_000, step_from=None, step=0):
    """Frames whose host arrival follows the PHYSICAL exposure (k*P + delivery delay)."""
    out = []
    for k in ks:
        ts = BASE + k * P + bias_ts + (step if step_from is not None and k >= step_from else 0)
        out.append(frame(cam, k + 1, ts, host=10**12 + k * P + host_bias))
    return out


def guarded(pairs, period=P):
    g = PairGuard(period)
    return [p for p in pairs if g.check(p)]


def test_review1_ptp_step_of_one_period_is_flagged_by_guard():
    """Triton clock steps +P at k=50: grid pairing alone mis-pairs 49 frames silently."""
    h = host_stream("helios", range(100), 23_000, 20_000_000)
    t = host_stream("triton", range(100), 41_000, 21_000_000, step_from=50, step=P)
    pairs, _ = feed(GridPairer(P), interleave(h, t))
    wrong = {p.pair_index for p in pairs if p.helios.frame_id != p.triton.frame_id}
    assert len(wrong) == 49 and all(abs(p.offset_start_us - 18) < 1 for p in pairs)
    flagged = {p.pair_index for p in guarded(pairs)}
    assert flagged == wrong                                       # every bad pair, no good one


def test_review8_huge_step_new_segment_without_fake_drops():
    big = 10**15 + 37_000_000
    h = host_stream("helios", range(60), 0, 20_000_000, step_from=30, step=big)
    t = host_stream("triton", range(60), 18_000, 21_000_000, step_from=30, step=big)
    g = GridPairer(P)
    pairs, s = feed(g, interleave(h, t))
    assert s["segments"] == 2 and len(s["reanchors"]) == 1
    # The step (> 1 h) is first rejected as implausible until 5 consistent frames per camera
    # confirm the new time base; then the grid re-anchors. k stays continuous: the
    # re-anchored frame gets its true physical index (frame_id - 1).
    ra = s["reanchors"][0]
    reframe = g.frames[ra["at_frame"]]
    assert ra["k"] == reframe.frame_id - 1 == reframe.grid_k
    assert all(p.grid_k == p.helios.frame_id - 1 for p in pairs)
    assert max(s["dropped"].values()) <= 6                        # recovery frames, not 1e7
    assert len(pairs) >= 50
    assert {p.segment_id for p in pairs} == {0, 1}


def test_review4_bad_timestamps_never_paired():
    h = host_stream("helios", range(40))
    t = host_stream("triton", range(40), 18_000, 21_000_000)
    t[5].ts_ns = 0                                                # lost leader -> ts 0
    t[9].ts_ns = BASE + 5 * 3600 * 10**9                          # implausible (+5 h)
    g = GridPairer(P)
    pairs, s = feed(g, interleave(h, t))
    assert len(pairs) == 38 and s["invalid_ts"] == {"helios": 0, "triton": 2}
    assert all(p.triton.valid_ts for p in pairs)
    assert s["dropped"]["triton"] == 2                            # no usable k -> drop
    bad = [f for f in g.frames if not f.valid_ts]
    assert [f.ts_ns for f in bad] == [0, BASE + 5 * 3600 * 10**9]   # raw ts kept
    assert all("invalid_ts" in f.note for f in bad)


def test_time_base_change_accepted_after_consistent_frames():
    h = host_stream("helios", range(40), step_from=20, step=2 * 3600 * 10**9)
    g = NearestPairer(P)
    for f in h:
        g.add(f)
    assert g.invalid_ts["helios"] == 4                            # then the new base is accepted
    assert any("time_base_step_accepted" in f.note for f in g.frames)


class FakeFires:
    def __init__(self, n, t0=10**12, period=P, jitter=None):
        rng = random.Random(5)
        self.t = [t0 + k * period + (rng.randint(-3, 3) * 10**6 if jitter else 0) for k in range(n)]

    def first_fired(self):
        return 0, self.t[0]

    def nearest_fired(self, host):
        k = min(range(len(self.t)), key=lambda i: abs(self.t[i] - host))
        return k, self.t[k]


def test_trigger_pairer_latency_longer_than_period():
    fires = FakeFires(60, jitter=True)
    lat = {"helios": 72_000_000, "triton": 81_000_000}             # > P/2, close to P
    frames = {c: [frame(c, k + 1, BASE + k * P, host=fires.t[k] + lat[c]) for k in range(60)
                  if not (c == "triton" and k in (10, 11))] for c in lat}
    g = TriggerPairer(P, fires=fires, clock_domain="independent")
    pairs, s = feed(g, interleave(frames["helios"], frames["triton"]))
    assert len(pairs) == 58 and all(p.helios.frame_id == p.grid_k + 1 for p in pairs)
    _, s = g.finalize(expected_k=range(60))
    assert s["dropped"] == {"helios": 0, "triton": 2}
    assert s["delivery_latency_ms"] == {"helios": 72.0, "triton": 81.0}
