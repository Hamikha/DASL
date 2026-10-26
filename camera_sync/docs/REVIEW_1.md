# REVIEW_1: independent review of camera_sync (lucid_sync + sync_test + probe + mock)

Reviewer: independent; I wrote none of this code. I changed no code files.
Paths are relative to `scratchpad/work/camera_sync/` unless stated otherwise.

## What I ran

- `pytest -q` with LUCID_MOCK=1: **86 passed in 87 s**.
- `sync_test.py live --method ptpsync --fps 10 --seconds 20 --out scratchpad/review_run`: 201 pairs, 10.000 fps, offset_start 17.5 µs median, p95|.| 19.7 µs, 0 drops, PTP Master/Slave after 3.0 s, no missing nodes, no failed writes.
  - The numbers are internally consistent. They are also **circular**. The mock's trigger latencies (23.4 / 41.1 µs) were chosen to reproduce LUCID's ~18 µs, and the mock's PTP is perfect white noise once locked. So the run proves the plumbing works. It says nothing about real sync quality (see item 20).
  - The run's "drift −1.554 µs/min" comes from 20 s of white noise. It is meaningless (item 12).
- `live --method action --fps 15 --seconds 10` with `missed_trigger_prob=0.02`: 146 pairs, 149 fired, 0 skipped, min margin 21 ms.
  - The minimum margin was already 29 ms below the 50 ms lead on Linux (item 13).
- Scratch experiments (`scratchpad/review_scratch/exp1.py` and inline scripts) for items 1, 2, 4 and 11. Their results are quoted in those items.

---

## Issues

### 1. MAJOR: a step of one camera's PTP clock by about n·P pairs frames n periods apart, silently, with perfect-looking offsets
`lucid_sync/pairing.py:218-276` (GridPairer.add / grid_index); `engine.py:238-239` (nothing watches PTP during the run)

**Scenario.** In PTPSync the Triton (slave) schedules exposures on its OWN PTP clock. If that clock steps by n·P + δ with |δ| < P/4, the following happens:
- Its exposures stay physically on the shared grid (shifted by δ).
- Its timestamps move by n periods.
- GridPairer gives those frames k+n and pairs them with Helios frame k+n.

Causes include lock loss and re-acquisition, a servo step after "Slave" is reported, or a master change.

**Reproduced.** I added a +P step to the Triton timestamps at k=50 of 100:
- 49 of 99 pairs combine physically different instants (100 ms apart).
- Every one of them shows `offset_start_us = 18.0`.
- off_grid = 0, reanchors = 0.
- Only `host_delta_ms = −99` reveals the error.

For a large random step, about 50 % of step sizes land inside ±P/4 of a multiple of P. The same thing happens if the step comes during the servo's first seconds (item 2).

**Fix:**
- Add a consistency check that does not depend on the PTP clock. Per pair, `host_delta_ms` and `(triton_frame_id − helios_frame_id)` must stay within a band around their running median, for example |Δhost − median| < P/2. Flag pairs outside the band, exclude them from the stats, and count them in run.json.
- Poll `PtpStatus` (and `PtpOffsetFromMaster` on the slave) of both cameras at about 1 Hz during the run, from the main loop at `engine.py:238`. Record it in run.json. Mark pairs captured while either camera was not Master/Slave.

### 2. MAJOR: "PTP locked" means only the status strings; the servo is never allowed to settle
`methods.py:297-312`, `316-317`

- `enable_ptp` stops at the first poll where the statuses read {Master, Slave}.
- `PtpOffsetFromMaster` is sampled 5 times (1 s total) but **never gated**. Capture starts immediately.
- Research C §1 says explicitly: "Let the servo settle for about 10–60 s before recording", and Basler notes the offset is not monotonic.
- The DESIGN's clock_domain = "ptp" claim is made once, at configure time, and never re-checked. So the question asked is answered only partly. The claim IS made only after both cameras report Master/Slave (correct). It is never re-verified, and early frames may be recorded while the slave is still slewing or stepping.

**Fix:**
- After the status lock, require |PtpOffsetFromMaster| < a threshold (e.g. 1–2 µs, a setting) for N consecutive samples (e.g. 10 s), with a timeout that falls back like a lock failure.
- Keep PTP_OFFSET_SAMPLES in run.json and add the in-run monitoring from item 1.
- `PtpServoStatus` is read once (`methods.py:315`); log it over time too.

### 3. MAJOR: a camera that stops delivering frames is reported as "0 dropped"
`pairing.py:319-325` (window = overlap of both cameras); `analysis.py:801-808` (per-camera first..last k); `analysis.py:880-893` (duration from paired frames only)

**Reproduced** (synthetic capture: Helios 500 frames, Triton dies after 20). The report says:

> "Frames dropped: Helios 0, Triton 0 over 2 s (20 pairs)", Duration 1.9 s, Mean pair rate 10.000 fps.

- run.json also has `dropped_triton = 0`.
- The 480 missing Triton frames appear only as `edge_frames` in run.json, which is not shown in the report.
- A 50 s run in which the colour camera died after 2 s reads as a flawless 2 s run.
- A camera that starts late or dies early is never counted.

**Fix:**
- Define the drop window from the scheduler's or stream's start/stop: for ptpsync, all k between the first and last frame of EITHER camera; for action, the fired k.
- Report `edge_frames`, per-camera `received`, `seconds_actual` vs pair duration, and a "pair yield = pairs / (seconds × fps)" row in the summary table.
- Make the verdict state the yield.

### 4. MAJOR: one bad timestamp (e.g. 0 on an incomplete buffer) makes the analysis allocate about 13 GB and crash
`analysis.py:876-878` (`t0` = nanmin over ALL frames, including incomplete ones); `analysis.py:840` (`np.arange(-shift, t_end, width)`)

**Reproduced.** I added one row `triton,999,ts_ns=0,incomplete=1` to the sync_frames.csv of a capture with epoch-scale PTP timestamps (~1.7e18 ns). Result:

> `_ArrayMemoryError: Unable to allocate 12.7 GiB` in `_fps_windows`

Without a memory limit the process was OOM-killed. Whether the real SDK returns 0 or garbage in `timestamp_ns` for an incomplete buffer (lost leader) is unknown. Incomplete buffers are expected on this rig.

**Fix:**
- Exclude incomplete frames (and timestamps outside the run's plausible range) when choosing t0 and the time axis.
- Cap the window count (e.g. refuse if t_end > 10 × seconds_setting).

### 5. MAJOR: the report overstates sync quality for a thesis
`analysis.py:1114-1121` (verdict), `1378` (MEASURED_NOTE), `1515` (compare headline)

- **(a) Wrong name for the metric.** The verdict calls the timestamp difference "the measured **start-of-exposure** offset". MEASURED_NOTE says it "shows that the cameras' exposure starts agree". Research A §(c) says what the timestamp marks is **NOT CONFIRMED** (start, end or readout). It is a timestamp offset.
- **(b) Circular measurement.** Under PTPSync and action, both timestamps come from the same schedule on the same clock. The near-constant ~18 µs mostly measures firmware bookkeeping. It is not an independent check of when light was collected. The report should say this plainly, beyond the existing "needs a scope" sentence.
- **(c) Headline contradicts the motion-relevant number.** The headline says "well below the 1 ms motion criterion". In the same run the modelled exposure-centre offset is **−8.83 ms**, which fails that criterion. The default ALIGNMENT is "start", so the shipped default is out of tolerance on the motion-relevant metric. The verdict only mentions this later, as a "Note".
- **(d) Before/after factor compares unlike quantities.** `compare()` divides a host-based phase estimate (BEFORE) by the AFTER timestamp offset and prints "improved by a factor of N" (typically about 1000×). That ratio compares different quantities, and it ignores the ~9 ms centre offset.

**Fix:**
- Rename to "timestamp offset (Triton − Helios)".
- Make the headline verdict use max(|start offset|, |modelled centre offset|), or state both with equal weight.
- In `compare`, also give the factor for the modelled centre offset, or drop the factor.

### 6. MAJOR: camera state is never restored, which can contaminate the "before" (legacy) baseline
`engine.py:167-179` (close); the methods module

The cameras keep volatile settings until power-off (the code's own docstring, `methods.py:13-15`, says so). After a run they are left in one of these states:
- `AcquisitionStartMode=PTPSync` + PtpEnable;
- `TriggerMode=On / Action0`;
- chunk mode, throughput limits and the exposure cap.

If the user then runs the original capture.py to record the BEFORE baseline without power-cycling:
- it either gets **no frames** (trigger mode), or
- it is **actually PTP-synchronised** (PTPSync), which silently invalidates the before/after comparison.

This does not break the user's "no persistent user sets" rule, but it is a real experimental hazard.

**Fix:**
- `NodeReport.writes` already stores old values. On close(), restore them in reverse order: PTPSync → Normal, TriggerMode → old, and so on. Make it a setting (default on), and log what was restored.
- Also record in run.json and the report whether the legacy run was preceded by a power cycle.

### 7. MAJOR: the fallback does not verify that the previous method's mode was actually cleared
`methods.py:570-571`, `798-801` (non-required writes); `nodes.py:372-382`

- `setup_action` and `setup_freerun` write `AcquisitionStartMode="Normal"` and `TriggerMode="Off"` as **non-required** writes.
- The entry `"Normal"` is **not in A_lucid_capabilities.md**. It is guessed; the mock has it.
- If the entry does not exist, or the write fails, NodeKeeper logs a failed write and the method still "succeeds":
  - **"action"**: the camera may stay in PTPSync from an earlier ptpsync attempt or run. [PTPKB]: "with PTP Sync you lose the ability to trigger". The PTPSync grid (multiples of P) coincides with the scheduler grid (T0 is a multiple of P, `methods.py:710`), so the pairs look perfect and are labelled "action".
  - **"freerun_ptp" / "freerun"**: the same mislabel, or TriggerMode stays On → zero frames.

**Fix:**
- When `AcquisitionStartMode` exists and currently reads `PTPSync`, the switch away from it must be required. Use `try_set_first(["Normal", ...])` (pick from the actual entries), then read it back.
- Make `TriggerMode Off` required in freerun.
- After `start_stream`, read back `AcquisitionStartMode` and `TriggerMode` on both cameras and store them in run.json.

### 8. MINOR: the re-anchor docstring is false, and a large PTP time step hangs `finalize` or the analysis
`pairing.py:188-191` ("k numbering stays continuous"); `pairing.py:278-287`; `stats.py:466`; `analysis.py:804-806`

- `_reanchor` keeps the k computed from the jumped timestamp, so k jumps by step/P.
- **Reproduced:** a 1e15 ns step on both cameras gave reanchor k = 10,000,031 and `dropped = 10,000,001` per camera. finalize took 0.9 s.
- A step on the scale of the PTP epoch (~1.7e18 ns, e.g. one clock moving from a since-boot time base to TAI) makes `count_missing` loop about 1.7e10 times and `np.arange` in the analysis allocate about 136 GB.

**Fix:**
- In `_reanchor`, keep k continuous: set k = last k of that camera + round(Δhost/P), using host time.
- Cap k-range loops, and refuse or flag when hi − lo is much larger than seconds × fps.

### 9. MINOR: engine and analysis disagree on what a "drop" is, so run.json and the report give different numbers for the same run
- engine: `pairing.py:310` counts incomplete frames as present; for action, only fired k are expected (`engine.py:265`).
- analysis: `analysis.py:801-806` treats incomplete frames as drops, counts skipped_late action commands as drops, and treats off_grid frames as present (they keep `grid_k` in sync_frames.csv, `pairing.py:242`).

The report also never shows the action scheduler's `skipped_late` / `error` counts (`analysis.py` does not read `sync.action`).

**Fix:** use one definition (preferably the engine's, with item 3's window), and print skipped and incomplete counts separately.

### 10. MINOR: the modelled centre offset silently uses the exposure CAP when chunk data is unavailable
`engine.py:346-349`; `pairing.py:100-108`; `engine.py:437`

When `ChunkExposureTime` is not available (its ChunkSelector entry "ExposureTime" is NC), every Triton frame gets exposure = 8000 µs ("cap_upper_bound"). Under auto exposure in a bright scene (e.g. 1 ms) this makes the MODELLED centre offset wrong by up to 3.5 ms. Only a per-frame note records it; run.json `offset_center_model_us` and the report do not say so.

**Fix:** carry `exposure_source` into the stats and the report row; refuse the modelled centre offset when the source is `cap_upper_bound`.

### 11. MINOR: stale per-selector trigger state and TriggerDelay are not reset for ptpsync
`methods.py:474-491`

- ptpsync turns TriggerMode Off only for `FrameStart`.
- It does not reset `TriggerDelay` (left at e.g. 5350 µs by a centre-aligned action run) or other selectors (`FrameBurstStart`, `AcquisitionStart`).
- The mock ignores TriggerDelay in PTPSync. Real firmware implements PTPSync with internal action commands, so whether TriggerDelay applies is unknown. Any shift would show in the measured offset, so it is not silent, but the result would not be what the run was configured for.

**Fix:** set `TriggerDelay=0` (when it exists) and set TriggerMode Off for every TriggerSelector entry.

### 12. MINOR: drift is reported with no uncertainty and is often meaningless
`analysis.py:916-928`, `1126-1128`

- The live 20 s run reports "−1.554 µs/min".
- The dead-Triton capture (2 s of pairs) reports "−4.181 µs/min".
- Both come from white noise with σ ≈ 0.3–1.2 µs, and both are printed as MEASURED to 3 decimals.

**Fix:**
- Report the slope's standard error and 95 % CI, plus the window length.
- Suppress the slope, or mark it "not significant", when |slope| < 2·SE or the duration is under about 2 min.
- Prefer a robust estimate (Theil–Sen) for a heavy-tailed PTP residual.

### 13. MINOR: action scheduler timing on Windows, and a dead scheduler goes unnoticed
`methods.py:727-734`, `758-761`; `engine.py:234-239`

- `_wait_until_host` uses `Event.wait(timeout)`. On Windows, CPython lock and condition timeouts have the default ~15.6 ms timer granularity. (`time.sleep` in 3.11+ uses a high-resolution waitable timer; `Event.wait` does not.)
- With lead 50 ms and min_margin 12.5 ms this usually holds. But the mock run on Linux already saw a 29 ms wake-up delay (min margin 21 ms), because acquisition threads hold the GIL.
- If ActionQueueSize cannot be read, lead is clamped to P/2: 33 ms at 15 fps, with a margin of 8.3 ms. Then Windows granularity plus GIL contention gives `skipped_late` bursts.
- After 20 consecutive errors the thread silently ends. The engine never checks `is_alive()`, so the run continues with no triggers. Those frames are not counted as drops, because they were never "fired".

**Fix:**
- Coarse-wait with `Event.wait` until about 20 ms before, then `time.sleep()` in small steps. Do not call `timeBeginPeriod`, because the user forbids OS changes.
- Store the margin distribution (p1/p50), not only the minimum.
- In the main loop, check that the scheduler is alive. If it died, stop the run and record why.

### 14. MINOR: acquisition-thread robustness gaps
`engine.py:312-313`, `284-293`, `268-269`

- An exception in `self._pairer.add` (any bug) kills that daemon thread silently. The camera is no longer drained, and the main loop does not notice.
- A non-timeout `get_buffer` error (e.g. a disconnected camera) logs a WARNING every 10 ms indefinitely, which floods the log.
- Timeout detection is `isinstance(TimeoutError) or "timeout" in str(exc)`. The real exception type is unknown (the mock raises a TimeoutError subclass). A misclassification only shifts the timeouts/errors counters, but it is untested.
- `writer_thread.join()` has no timeout, and the sentinel `put(None)` blocks while the queue is full. A hung disk blocks shutdown forever. A second Ctrl+C during `_shutdown` (sleep, joins) escapes, and no stats are written, contrary to the log message at `engine.py:242`.

**Fix:**
- Wrap the pairer call; count errors.
- Back off and rate-limit error logs; stop the run after N consecutive errors.
- Add join timeouts.
- Shield `_shutdown` from a second KeyboardInterrupt, or write partial stats first.

### 15. MINOR: PTP role writes can be ineffective when PTP is already enabled
`methods.py:289-292`

`PtpSlaveOnly` is written while `PtpEnable` may already be True from an earlier session. The value-equal skip in `nodes.py:388` means PtpEnable is not re-written, so negotiation never restarts. Whether a SlaveOnly change takes effect without re-enabling is unknown. The lock check accepts either camera as master, so this is not wrong, but the "Helios = master" intent may silently not hold.

**Fix:** if PtpSlaveOnly actually changed while PTP was on, toggle PtpEnable off/on and record that. Log the master that was reached (already done, `rep.master`).

### 16. MINOR: tests never exercise epoch-scale PTP values through the engine, and the mock never loses lock
`mock/README_MOCK.md` "Clocks"; the live run shows `anchor_ns = 253400022816` (≈ 253 s)

- The mock's PTP time is seconds since mock start. Only the pairing and analysis unit tests use ~1.76e18 ns.
- The mock's PTP has no convergence phase, no steps and no lock loss. So items 1, 2 and 8 are structurally untestable today.

**Fix:** add a mock knob for the PTP epoch (e.g. TAI ns), plus knobs for "slave step at t = X by Y ns" and "lock lost for Z s". Add engine-level tests for items 1, 3 and 4.

### 17. NIT: the safety guard is a deny-list
`nodes.py:184-190`

- Today the code writes no IP, UserSet or firmware node; I checked every `set`/`execute` call in methods.py and engine.py. Nothing touches OS or NIC settings: no subprocess, os.system, netsh or ethtool. The probe is read-only.
- But the deny-list misses names such as `GevCurrentIPAddress`, `GevCurrentSubnetMask`, `GevCurrentDefaultGateway`, `GevDeviceForceIP*` (TL device map), `GevIPConfigurationStatus` and `DeviceRegistersStreamingStart`.
- `ActionCommandTargetIP` is a host TL setting, not a camera IP, which is fine.

**Fix:** an allow-list of the ~45 node names the code is meant to write would make the "never" guarantee checkable.

### 18. NIT: write order of the trigger nodes
`methods.py:576-578`

The code writes TriggerMode=On before TriggerSource=Action0. LUCID's order in A §(b)(iii) is Selector → Source → Mode, and SFNC also recommends choosing the source before arming. It is harmless here, because it happens before start_stream. Match LUCID's order anyway, for traceability.

### 19. NIT: readability and misleading text
- `sync_test.py:68-72`: the leftover "LEAD: please re-check this against the final ... engine API (written in parallel)" note.
- `methods.py:13-15`: "each one sets EVERY node it relies on" is not true (item 11).
- `pairing.py:188-191`: "k numbering stays continuous" is false (item 8).
- `analysis.py:153-165`: `device_offset_us` converts to float BEFORE subtracting. It is safe only because the callers pass already-relative values. This is a trap, given the module's own float64 warning at `analysis.py:187-191`. Subtract as int64 first.
- run.json carries `settings` twice (`sync.settings` and top-level `settings`, `sync_test.py:86`).
- `verify_ptpsync_rate` (`methods.py:500-513`) computes `ok`, but nothing uses it. A failed check should at least be in the report's notes.
- The `_ClockMap` latch midpoint (`methods.py:696-699`): the latch happens on the write, so about the first quarter of the round trip, not the midpoint. This adds a ≤ roundtrip/4 bias, which is irrelevant at a 50 ms lead. Document it.
- `analysis.py` is 1563 lines. About 300 of them are legacy column-name guessing (`_tokens` / `_has` substring heuristics). For the thesis, requiring `--map` for legacy folders would be easier to defend than heuristics.
- DESIGN.md lists `capture_sync.py` (the real capture with a writer). It does not exist yet. The writer path is covered only by a test writer.

### 20. Mock bias: what transfers to real hardware and what does not

**Transfers** (pure logic, independent of the SDK):
- GridPairer / NearestPairer maths with exact integer and Fraction arithmetic;
- stats;
- CSV/JSON schema;
- NodeKeeper's treatment of missing nodes (both None and raise);
- fallback ordering and reason logging;
- the thread layout (pixels copied before requeue, writer never blocks acquisition);
- the analysis's refusal to subtract device timestamps across independent clocks.

**Does NOT transfer** (tests pass because the mock implements it this way):
1. The PTPSync grid at absolute multiples of 1/PTPSyncFrameRate (+ PTPSyncOffset), identical on both cameras. The Helios-only `(P − t_d) mod P` trick and the "write offset 0" logic depend on it.
2. Timestamp = exposure start, with latencies tuned to LUCID's 23/41 µs. **Every offset number from the mock is an input, not a result.**
3. PTP that is instantly perfect after "Slave" (white 1 µs noise), and never steps or loses lock.
4. `AcquisitionStartMode` entry "Normal", the ChunkSelector entry "ExposureTime", ExposureAutoLimitAuto/Upper/Lower names, `system.tl_system_nodemap`, ActionQueueSize = 4, and "late actions fire immediately".
5. Which nodes are locked while streaming or conditionally writable.
6. The get_buffer exception type (TimeoutError).
7. Bandwidth, incomplete-frame rates and switch-overflow behaviour.
8. Timing and GIL behaviour of the acquisition threads. The mock's frame generator is itself Python threads competing for the GIL, so its scheduler margins do not predict either Windows or the real C SDK.

---

## Can only be verified on real hardware
1. What `buffer.timestamp_ns` marks on each camera (exposure start, end or readout), and the true Helios↔Triton exposure alignment. Measure with an LED/scope or `ExposureActive` strobes. Also the Helios light-collection window and centroid (`HELIOS_CENTER_US = 9350` is a calculation).
2. The PTP servo behaviour through the unmanaged switch: time from "Slave" to a stable `PtpOffsetFromMaster`, the steady-state residual, and any steps or relocks over a 30–60 min run (log at 1 Hz).
3. The PTPSync grid origin: are exposures at PTP multiples of 1/PTPSyncFrameRate on both cameras? Does `PTPSyncOffset` exist on the TRI032S, with what units/range? Is TriggerDelay applied in PTPSync? Is AcquisitionFrameRate writable after AcquisitionStartMode=PTPSync? Does it read back PTPSyncFrameRate after start?
4. The `AcquisitionStartMode` entries (is "Normal" the name?), `ChunkSelector` entries, the exposure-limit node names, and `ActionQueueSize` on both cameras. Run `probe_cameras.py` first.
5. The Python accessor for the TL system nodemap, and whether action commands fire from arena_api Python. Also the minimum lead time, the behaviour of a late action (fire or drop), and the real scheduler margins on the Windows box under acquisition load.
6. The exception type and message of a `get_buffer` timeout. What `timestamp_ns`, `frame_id` and `size_filled` contain for incomplete buffers.
7. Thread safety of concurrent nodemap access (scheduler latching `PtpDataSetLatch` on the Helios) while another thread is in `get_buffer` on the same device.
8. Whether `frame_id` advances for frames lost on the host, which affects the freerun drop definition.
9. Whether 48/60 MB/s throughput limits really give zero incomplete frames at 10 and 15 fps on the actual switch and NIC, and the real AcquisitionFrameRate max per camera.
10. Whether a camera left in PTPSync or trigger mode by this code breaks or alters the original capture.py (item 6). Do one run, then the legacy script, without a power cycle.
