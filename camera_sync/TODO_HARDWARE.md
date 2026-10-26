# To do on the rig, in order

Nothing below changes IP settings, user sets, firmware or Windows network settings. Anything
that would is marked **ASK FIRST** and is not done by the scripts.

1. **Power check.** The Helios2+ needs an 802.3at (PoE+) injector. Confirm the link speed is
   gigabit: `probe_rig/` (step 2) shows `DeviceLinkSpeed` = 125000000 on both cameras.
2. **Probe (read-only):** `python probe_cameras.py --out probe_rig --all`. Check:
   - `DeviceFirmwareVersion` of both cameras (Helios ≥ 1.16.0.0 needed for `PTPSyncOffset`).
   - `AcquisitionStartMode` lists `PTPSync` on both; `PTPSyncFrameRate` exists on both.
   - Triton: `ExposureAutoLimitAuto`, `ExposureAutoUpperLimit` exist; `PTPSyncOffset` exists?
   - `system.tl_system_nodemap` and `ActionCommand*` nodes exist (action fallback).
   - If any is missing the scripts fall back automatically; the probe tells you which path you will get.
3. **Mock dry run on the UNO:** `set LUCID_MOCK=1` then `python capture_sync.py` (SECONDS small)
   and `python process.py <run>`. Proves the Python environment, then `set LUCID_MOCK=`.
4. **BEFORE:** `python sync_test.py live --method freerun_ptp --seconds 300 --out before_ptp`.
5. **AFTER, 10 fps:** `python sync_test.py live --method ptpsync --seconds 300 --out after_10`.
   Check `after_10/sync_report.md`: method used = ptpsync, PTP settled = True, 0 dropped,
   residual p95 in the tens of µs.
6. **Highest fps with 0 drops:** repeat with `--fps 12`, `--fps 15`. Stop at the first rate with
   drops or incomplete frames. Expect 15 to be the limit.
7. **Dim light / exposure cap:** cover the Triton lens partly (or indoors at night), run
   `--method ptpsync --seconds 60`: fps must stay at the target; `run.json` shows exposure = cap.
8. **Drift:** `--seconds 1800` once (30 min). Report must say "no drift detectable".
9. **Fallback:** `--method action --seconds 120`, and `--method software --seconds 120`.
10. **Pipeline:** `python capture_sync.py` (60 s, hand-held, rotate the rig), then
    `python process.py <run>`; open the PLY, compare edges against an old freerun run.
11. **Compare:** `python sync_test.py compare before_ptp after_10` → paste into RESULTS.md.
12. **Validate the Helios timing model (optional but recommended for the thesis):**
    - Pole test: rotate the rig slowly clockwise, then counter-clockwise, past a vertical pole at
      3–5 m. Sync error flips the colour/depth edge shift with direction; calibration error does not.
    - Or scope: LUCID GPIO cable (GPIO-M8-1M-A), Triton `LineSource=ExposureActive` on Line1/3,
      look at both strobes. Helios strobe behaviour in multi-freq mode is not documented.
    Then set `HELIOS_CENTER_US` in capture_sync.py to the measured value.
13. **ASK FIRST / not needed now:** jumbo frames (NIC "Jumbo Packet 9014") would lower CPU load
    and give headroom at 15 fps. That is a Windows NIC setting, so only with your approval.
14. Send me `probe_rig/`, `before_ptp/`, `after_10/` and the compare output; I'll fill in RESULTS.md.
