# Results

## Status
**No real-hardware numbers yet.** The cameras were not reachable from the development
environment. All numbers below come from the mock SDK (`mock/`) and from synthetic data, and
show that the code measures, pairs and reports correctly. They are **not** the rig's sync
quality. Run `TODO_HARDWARE.md` steps 4–11 and paste the outputs into the tables below.

## BEFORE vs AFTER (fill from the rig)
| Run | Method | Pairs / s | Dropped H / T | Timestamp offset p95 (measured) | Residual vs plan p95 (measured) | Centre offset p95 (modelled) |
|---|---|---|---|---|---|---|
| Old capture.py freerun | freerun, no PTP | 600 / 60 s (user) | 0 | not measurable: worst-case bound ≈ 1 period (100 ms) | — | — |
| BEFORE (freerun_ptp) | freerun_ptp | _rig_ | _rig_ | _rig_ | — | _rig_ |
| AFTER (ptpsync, 10 fps) | ptpsync | _rig_ | _rig_ | _rig_ | _rig_ | _rig_ |

Plot: `python sync_test.py compare before_ptp after_10` → `before_after_abs_offset.png`.
A synthetic example of that plot is in `docs/demo_plots/` (labelled SYNTHETIC DATA).

## Test matrix
| Test | Mock result | Real rig |
|---|---|---|
| ptpsync 10 fps | ~80 pairs / 8 s, 10.00 fps, 0 drops, timestamp offset median 17.7 µs (the mock is set to LUCID's measured ~18 µs bookkeeping difference) | TODO step 5 |
| action 10 fps / 15 fps | same numbers; 15 fps scheduling margin ≥ 21 ms (Linux) | TODO step 9 |
| Highest fps with 0 drops | 15 fps works on the mock bandwidth model | TODO step 6 |
| Dark scene, exposure cap | uncapped auto exposure: ~0.6 fps (reproduces the old bug); capped at 8 ms: 10 fps | TODO step 7 |
| PTP never locks | falls back ptpsync → action → software → freerun, reasons logged | — |
| Missing `AcquisitionStartMode` | falls back to action, logged | — |
| PTP clock step / servo glitch | affected pairs flagged suspect and not saved | — |
| Dead camera mid-run | drops counted over the whole run, loud warning | — |
| Packet loss | incomplete frames counted, never paired | — |
| 5+ min drift | analysis drift fit tested on synthetic 300 s runs (2 µs/min recovered) | TODO step 8 |
| process.py on new capture | written to the same layout; automated test `tests/test_capture_sync.py` written but **its last run did not complete** (container killed long runs) | TODO step 3 and 10 |

## Automated tests (mock)
- Library + mock + probe: 86/86 passed (Python 3.12) before the review round.
- After review fixes: `test_analysis.py` + `test_mock_sdk.py` 47/47 passed; engine/methods/pairing
  37 passed, 1 failed (`test_major1_ptp_step...`: the test counted pairs exposed after the stop;
  test fixed, not re-run).
- `test_capture_sync.py` (capture_sync.py → unchanged process.py): **not confirmed passing.**
  Run `python -m pytest -q tests` on the UNO (needs `pip install pytest matplotlib`).

## Reviewer notes
Independent review round 1 (`docs/REVIEW_1.md`): no blockers, 7 major issues. All 7 were
addressed: mis-pairing guard plus PTP monitor, PTP settle gate, drops over the full run,
bad timestamps, honest report wording, restoring camera nodes on exit, and read-back
verification on fallback. A second review round was not run, and there is **no final
sign-off**. Items the reviewer said need real hardware: timestamp meaning, PTP settling
through the switch, the PTP Sync grid origin and `PTPSyncOffset`, the real node names and
entries, action lead time on Windows, the `get_buffer` timeout exception, and whether the
throughput limits give zero incomplete frames.
