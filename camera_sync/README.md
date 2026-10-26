# camera_sync — synchronised Helios2+ / Triton capture

New capture script, sync measurement tool, read-only camera probe and a mock SDK for the
Helios2+ (HTP003S-001) + Triton (TRI032S-C) rig. `capture.py` and `process.py` are not modified;
`capture_sync.py` writes the same run folder layout, so `process.py` works unchanged.

**Status:** developed and tested against a mock SDK only. Nothing has been run on the real
cameras yet. Do the steps in `TODO_HARDWARE.md` before quoting any number in the thesis.

## Files
| File | What it is |
|---|---|
| `capture_sync.py` | The capture script. Settings at the top. Writes `run_*/raw/*.npy`, `frames.csv`, `run.json`, `orientation.yml` exactly like `capture.py`, plus extra sync columns and `sync_frames.csv`. |
| `sync_test.py` | `live` (run the cameras N s and measure), `analyze <run>` (works on old capture.py runs too), `compare <before> <after>`, `demo`. Writes `sync_report.md` and plots. |
| `probe_cameras.py` | Read-only dump of every node we use, with value, range and enum entries. Run this first on the rig. |
| `lucid_sync/` | Shared code: node access (`nodes.py`), sync methods and fallback (`methods.py`, `ptp.py`, `triggers.py`), pairing (`pairing.py`), the capture engine (`engine.py`), stats and the report (`analysis.py`), the Helios timing model (`timing.py`). |
| `mock/arena_api/` | Fake `arena_api` for testing without cameras (`LUCID_MOCK=1`). See `mock/README_MOCK.md` for the assumptions it makes. |
| `tests/` | pytest, all against the mock. |

## How to run (Windows, `myenv`)
1. Copy the `camera_sync` folder into `C:\dasl17\scripts\`. Install extras once: `pip install matplotlib`
   (OpenCV is already used by process.py).
2. `python probe_cameras.py --out probe_rig --all` — read-only; send me / check `probe_rig/`.
3. `python sync_test.py live --method freerun_ptp --seconds 300 --out before_ptp` — the honest
   BEFORE number (old free-run behaviour, measured on one PTP clock).
4. `python sync_test.py live --method ptpsync --seconds 300 --out after` — the AFTER number.
5. `python sync_test.py compare before_ptp after` — before/after table and plot.
6. Record for real: `python capture_sync.py`, then `python process.py <run folder>` as before.
7. Re-analyse an old run: `python sync_test.py analyze C:\Users\<you>\captures\run_...`.
Try any step without the cameras first: `set LUCID_MOCK=1`.

## The sync method and why
**Chosen: IEEE 1588 PTP + LUCID "PTP Sync" acquisition mode.** Both cameras run PTP so their
clocks agree to about a microsecond (Helios is master, Triton slave). In PTP Sync mode each
camera's firmware starts an exposure at fixed times on that shared clock (every 100 ms at
10 fps). No PC timing is involved, so Windows scheduling or a busy CPU cannot add jitter.
LUCID documents this mode for both camera families and published a Helios2 + Triton example
with ~20 µs between the two timestamps [HT].

**Alignment.** A Helios depth frame is not an instant: in Distance5000mmMultiFreq + Exp250Us it
is 8 short sub-exposures spread over roughly 12–13 ms (model, `lucid_sync/timing.py`). By
default (`ALIGNMENT="center"`) the Triton exposure is shifted so its middle lands on the middle
of the Helios window (modelled at ~6.4 ms after the Helios frame start). Using `PTPSyncOffset`
needs Helios firmware ≥ 1.16.0.0; without it the script falls back to start alignment and says so.

**Fallback chain** (automatic, every reason logged in `run.json` → `sync.fallback_chain`):
`ptpsync` → `action` (PTP + host-scheduled action commands, Triton `TriggerDelay`) → `software`
(software trigger like capture.py, PTP timestamps if PTP works) → `freerun_ptp` → `freerun`.

**Triton exposure cap.** Auto exposure stays on but is capped (`EXPOSURE_CAP_US = 8000`), so a
dark scene can never stretch a frame beyond the 100 ms period (the old 0.6 fps bug). If the
cap nodes are missing, the script uses a fixed exposure at the cap and logs it.

**Bandwidth.** Per-camera `DeviceLinkThroughputLimit` (Helios 40 MB/s, Triton 60 MB/s, as in
capture.py). Both cameras now expose at the same instant, so without the limits their bursts
would overflow a small switch buffer.

## Options compared
| Method | Expected sync error | Max fps (1 GbE, ABCY16) | Extra hardware | Risk | Effort |
|---|---|---|---|---|---|
| Freerun, no PTP (old) | up to 1 frame period (100 ms @10 fps), unmeasurable | ~15 | none | none | done |
| Freerun + PTP timestamps | ≤ ½ period after nearest-timestamp pairing, but measured | ~15 | none | low | low |
| Software trigger (old "trigger") | ~0.1–few ms; was 0.6 fps until exposure capped | ~10–15 | none | host jitter | low |
| **PTP + PTP Sync (chosen)** | ~µs clock agreement; LUCID measured ~20 µs between timestamps | ~15 | none | PTP lock through switch, firmware support | medium |
| PTP + scheduled action commands (fallback) | same as PTP Sync, plus host lead-time risk | ~15 | none | Python access to action nodes unconfirmed | medium |
| Hardware trigger cable | < 1 µs electrical; Helios strobe behaviour unconfirmed | ~15 | GPIO-M8 cables | wiring, multi-pulse strobe | high |

Higher than ~15 fps is not possible on one 1 GbE link with Coord3D_ABCY16 (84 MB/s at 15 fps,
112 MB/s at 20 fps). Sync better than ~1 ms brings no visible gain: the Helios frame itself
spans ~12 ms, which smears depth during fast rotation more than the residual sync error does.

## Known limits
- Node names marked NOT CONFIRMED (e.g. `ExposureAutoLimitAuto`, `PTPSyncOffset` on the Triton,
  the Python name `system.tl_system_nodemap`) are only used after an existence check, with a fallback.
- What each camera's timestamp marks (exposure start/end) is not documented by LUCID. The
  measured timestamp offset proves clock lock and scheduling precision, not exposure overlap.
- The Helios exposure-centre timing is a model; verify it with a scope or the pole test
  (`TODO_HARDWARE.md`). The "modelled centre offset" in reports depends on it.
- Old capture.py freerun runs can only be given a worst-case bound (~1 period): their Triton
  frame was usually already waiting in the buffer, so even its arrival time is unknown.
- Mock results show the code works, not how good the real sync is.

## Sources
- [HT] LUCID App Note *Helios2 and Triton Sync*: https://support.thinklucid.com/app-note-helios2-and-triton-sync/
- [SAC] LUCID *Multi Camera Synchronization using PTP and Scheduled Action Commands*: https://support.thinklucid.com/app-note-multi-camera-synchronization-using-ptp-and-scheduled-action-commands/
- [BW] LUCID *Bandwidth Sharing in Multi-Camera Systems* (PTP Sync code): https://support.thinklucid.com/app-note-bandwidth-sharing-in-multi-camera-systems/
- [PTPKB] LUCID KB *Precision Time Protocol*: https://support.thinklucid.com/knowledgebase/precision-time-protocol-ptp/
- [MH] LUCID KB *Using Multiple Helios Cameras* (Helios timing, PTPSyncOffset firmware): https://support.thinklucid.com/knowledgebase/using-multiple-helios-2-cameras-simultaneously/
- [GPIO] LUCID *Using GPIO on LUCID Cameras*: https://support.thinklucid.com/app-note-using-gpio-on-lucid-cameras/
- [AE] LUCID KB *Auto Exposure and Auto Gain*: https://support.thinklucid.com/knowledgebase/using-auto-exposure-and-auto-gain-on-lucid-cameras/
- [TRM-T] LUCID Triton Technical Manual v1.0.0: https://cdn.graftek.com/system/files/15519/original/Triton_Camera_Technical_Manual_v1.0.0.pdf
- Product pages: Helios2+ https://thinklucid.com/product/helios2plus-time-of-flight-tof-ip67-3d-camera/ , Triton 3.2 MP https://thinklucid.com/product/triton-32-mp-imx265/
- Allied Vision PTP app note: https://ftp.stemmer-imaging.com/webdavs/docmanager/66823-Allied-Vision---PTP-IEEE1588-Application-Note.pdf
- Basler blaze synchronous free run (IMX556 sub-frames): https://docs.baslerweb.com/synchronous-free-run-(blaze)
- Literature (TUM RGB-D, Azure Kinect, Newer College, Hilti, BotanicGarden, OpenLORIS): see `docs/research_B_literature.md`.
Full research notes: `docs/research_*.md`.
