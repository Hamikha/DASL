# Researcher A: LUCID sync capabilities for Helios2+ (HTP003S-001) and Triton 3.2 MP (TRI032S-C)

Research date: 2026-10-26. Every node name below is quoted from an official LUCID source, with the source tagged in brackets. Anything that could not be confirmed is marked **NOT CONFIRMED**.

## Sources and access notes

| Tag | Source | Date / version |
|---|---|---|
| [HT] | App Note: *Helios2 and Triton Sync*, https://support.thinklucid.com/app-note-helios2-and-triton-sync/ . Includes `Cpp_HLTRGB_3_Overlay_PTP_SAC_v7.cpp` (https://dce9ugryut4ao.cloudfront.net/Cpp_HLTRGB_3_Overlay_PTP_SAC_v7.cpp), a sample-output screenshot, and ArenaView screenshots of an HLT003S-001 | Published 2021-10-07, modified 2023-09-19, code (c) 2021 |
| [SAC] | App Note: *Multi Camera Synchronization using PTP and Scheduled Action Commands*, https://support.thinklucid.com/app-note-multi-camera-synchronization-using-ptp-and-scheduled-action-commands/ | Published 2020-05-05, modified 2025-07-25 |
| [BW] | App Note: *Bandwidth Sharing in Multi-Camera Systems* (contains PTPSync code), https://support.thinklucid.com/app-note-bandwidth-sharing-in-multi-camera-systems/ | Published 2022-04-08, modified 2025-07-25 |
| [GPIO] | App Note: *Using GPIO on LUCID Cameras*, https://support.thinklucid.com/app-note-using-gpio-on-lucid-cameras/ | Published 2020-05-05, modified 2026-09-29 |
| [PTPKB] | KB: *Precision Time Protocol (PTP)*, https://support.thinklucid.com/knowledgebase/precision-time-protocol-ptp/ | 2023-09-19, updated 2023-11-17 |
| [MH] | KB: *Using Multiple Helios Cameras Simultaneously*, https://support.thinklucid.com/knowledgebase/using-multiple-helios-2-cameras-simultaneously/ | 2024-06-25, updated 2025-03-07 |
| [AE] | KB: *Using Auto Exposure and Auto Gain on LUCID Cameras*, https://support.thinklucid.com/knowledgebase/using-auto-exposure-and-auto-gain-on-lucid-cameras/ | Updated 2026-05-26 |
| [FPS] | KB: *Tips for Reaching Maximum Frame Rate*, https://support.thinklucid.com/knowledgebase/tips-for-reaching-maximum-frame-rate/ | Updated 2025-09-24 |
| [RDMA] | *20-Camera GigE Vision RDMA Streaming ... PTP Sync, eTrigger*, https://support.thinklucid.com/20-10gige-camera-gpudirect-rdma-streaming-guide/ | 2026-02-08, modified 2026-10-02 (Triton10 only) |
| [TRM-T] | *LUCID Triton Camera Technical Manual V1.0.0*, revised May 1 2019. Official LUCID PDF, publicly hosted by distributor Graftek: https://cdn.graftek.com/system/files/15519/original/Triton_Camera_Technical_Manual_v1.0.0.pdf . This is an old revision; newer firmware adds nodes such as `AcquisitionStartMode`. | 2019-05-01 |
| [P-HTP] | Product page, Helios2+: https://thinklucid.com/product/helios2plus-time-of-flight-tof-ip67-3d-camera/ | Retrieved 2026-10-26 |
| [P-TRI] | Product page, Triton 3.2 MP: https://thinklucid.com/product/triton-32-mp-imx265/ | Retrieved 2026-10-26 |
| [ROS2] | LUCID's own GitHub repo, https://github.com/lucidvisionlabs/arena_camera_ros2 . The `ros2_ws/DEPRECATED/.../arena_camera_node/*.py` files are LUCID-written arena_api Python | HEAD on 2026-10-26 |
| [DOCKB] | KB: *How to access documentation on Windows/Linux*, https://support.thinklucid.com/knowledgebase/how-to-access-documentations-on-windows-linux/ | Updated 2025-07-31 |

**Access limits:**
- The current per-model Technical Reference Manuals (`support.thinklucid.com/camera-manuals/HTP003S/` and `.../TRI032S/`) are login-gated. So is the Arena SDK documentation, which holds the Python API reference and the py_* example list.
- I did not read the gated TRMs. Archived copies are not used.
- A free LUCID support login gives the authoritative per-model node list.
- [DOCKB] says the camera serves its own firmware feature documentation: "input the camera's IP address into the browser to access firmware documentation". This is the most reliable reference for the exact firmware on the bench.

---

## (a) Feature support table

Legend: **C** = CONFIRMED (with source), **NC** = NOT CONFIRMED, **NS** = NOT SUPPORTED.

| Feature | Helios2+ (HTP003S) | Triton (TRI032S-C) | Exact node names / enum strings (as written in source) | Firmware notes | Sources |
|---|---|---|---|---|---|
| PTP IEEE 1588 enable / role | **C**. Product spec says "Synchronization: Software trigger, hardware trigger, PTP (IEEE 1588)" [P-HTP]. The PTP code in [HT] runs on a Helios2 HLT003S, and its device filter accepts "HLT" or "HTP" (`//either Helios2 or Helios2+ can be used`). | **C** [P-TRI][SAC][HT][TRM-T] | `PtpEnable` (bool), `PtpSlaveOnly` (bool), `PtpStatus` (string read; values compared to `"Master"` and `"Slave"`; anything else = negotiation not complete) [SAC][HT][BW] | Triton FW **>= 1.4**, Arena SDK **>= 1.0.9.1** for PTP+SAC [SAC]. Helios2+ minimum PTP firmware is **NC**. | [SAC] [HT] [BW] [PTPKB] |
| PTP diagnostics | **C** for display names on HLT003S (Helios2, the base model, not 2+): "PTP Enable, PTP Status, PTP Slave Only, PTP Clock Accuracy, PTP Servo Status, PTP Timestamp, PTP Timestamp Latch, PTP Timestamp Latch Value, PTP Offset From Master" [HT screenshot]. Node names on Helios2+ are **NC** (inferred from the Triton manual, see next column). | **C** [TRM-T] | `PtpClockAccuracy`, `PtpServoStatus`, `PtpDataSet`, `PtpDataSetLatch` (command), `PtpDataSetLatchValue` (ns), `PtpOffsetFromMaster` ("Returns the latched offset from the PTP master clock in nanoseconds. If PtpOffsetFromMaster returns -1 then the synchronization has not started."), `PtpClockID`, `PtpParentClockID`, `PtpGrandmasterClockID` [TRM-T]. `PtpDataSetLatch` and `PtpDataSetLatchValue` are also used **on the Helios** in [HT] code. | (none) | [TRM-T] [HT] |
| Master/slave selection | **C** | **C** | Best Master Clock algorithm: "When PTP is enabled on LUCID cameras, they are able to organize themselves into one master clock and one or more slave clocks using PTP's Best Master Clock (BMC) algorithm. You can also set which cameras will be slaves". Delay mechanism: "LUCID cameras support the End-to-End method ... but not the Peer-to-Peer method." [PTPKB]. In [HT] the Helios is master (`PtpSlaveOnly=false`) and the Triton is slave (`PtpSlaveOnly=true`). | (none) | [PTPKB] [HT] |
| Time to PTP lock | **C** (order of magnitude) | **C** | "Depending on the initial PTP state of each camera, it can take about 40 seconds for all devices to autonegotiate" [BW code comment]. The [SAC] sample log shows roughly 5 to 46 one-second polls per camera. In the [HT] log, HLT becomes Master and TRI becomes Slave after about 1 poll each. | (none) | [BW] [SAC] [HT] |
| Device timestamp latch (non-PTP counter) | **NC** | **C** [TRM-T] | `Timestamp`, `TimestampReset`, `TimestampLatch`, `TimestampLatchValue` [TRM-T]. LUCID's PTP examples use `PtpDataSetLatch` / `PtpDataSetLatchValue` instead. | (none) | [TRM-T] |
| PTPSync acquisition mode | **C**. ArenaView screenshots of HLT003S-001 show "Acquisition Start Mode: PTP Sync" and "PTP Sync Frame Rate" (range shown 0.100 to 30.000; set to 10 Hz, and Acquisition Frame Rate then reads 10 Hz) [HT]. Helios2+ (HTP) has the `PTPSyncOffset` node from FW **1.16.0.0** [MH], which implies PTP Sync exists on HTP. | **C**. [HT] says "Set the Acquisition Start Mode node to PTP Sync on both cameras". | `AcquisitionStartMode` = `"PTPSync"`; `PTPSyncFrameRate` (float); `AcquisitionFrameRate` (set to max so it does not cap) [BW]. `PTPSyncOffset` (ns delay) [MH]. LUCID: PTPSync "schedules 4 advance action commands in the firmware itself" [BW]; "with PTP Sync, you lose the ability to trigger cameras" [PTPKB]. | [BW] needs "Arena SDK 1.0.31.8 or newer" and "Firmware version with PTPSync support" (no number given). `PTPSyncOffset` firmware: "HLT003S: 1.21.0.00 and higher; HTP: 1.16.0.0 and higher; HTR 1.6.0.0; HTW 1.12.0.0; HTN 1.1.0.0" [MH]. Triton minimum PTPSync firmware and `PTPSyncOffset` on Triton are **NC**. | [HT] [BW] [MH] [PTPKB] |
| Scheduled action commands, device side | **C**. [HT] code applies these nodes to the Helios. | **C** [TRM-T][SAC][HT] | `TriggerSelector`=`"FrameStart"`, `TriggerSource`=`"Action0"`, `TriggerMode`=`"On"`, `ActionUnconditionalMode`=`"On"`, `ActionSelector`=0, `ActionDeviceKey`, `ActionGroupKey`, `ActionGroupMask` [SAC][HT]. Also `ActionQueueSize` ("size of the scheduled action commands queue") [TRM-T]. | Triton FW >= 1.4 [SAC]. Helios2+ minimum is **NC** (works on HLT003S per [HT]). | [SAC] [HT] [TRM-T] |
| Scheduled action commands, host/system side (Arena GenTL System nodemap) | n/a | n/a | `ActionCommandDeviceKey`, `ActionCommandGroupKey`, `ActionCommandGroupMask`, `ActionCommandTargetIP` (example uses `0xFFFFFFFF` broadcast), `ActionCommandExecuteTime` (PTP ns), `ActionCommandFireCommand` (command). C++ accesses these via `pSystem->GetTLSystemNodeMap()` [SAC][HT]. The Python accessor name (expected `system.tl_system_nodemap`) is **NC** from public sources. | Arena SDK >= 1.0.9.1 [SAC] | [SAC] [HT] |
| Official example for SAC / PTP | C++: modified `Cpp_HLTRGB_3_Overlay` published as `Cpp_HLTRGB_3_Overlay_PTP_SAC_v7.cpp` (Helios master + Triton slave) [HT]. SDK HLT+RGB examples: `Cpp_HLTRGB_1_Calibration`, `Cpp_HLTRGB_2_Orientation`, `Cpp_HLTRGB_3_Overlay` [HT]. Chroma: `Cpp_HeliosChroma_Overlay`, `Cpp_HeliosChroma_Overlay_SDK`. | same | Python example names (py_*) for SAC/PTP/HLTRGB are **NC**: the SDK docs are login-gated. "py_helios_heat_map" is **NC**. | (none) | [HT] |
| TransferControl (staggered transfer) | **NC** on Helios2+ | **C** [TRM-T][SAC] | `TransferControlMode`=`"UserControlled"`, `TransferOperationMode`=`"Continuous"`, `TransferStart`, `TransferStop` (+ `TransferQueueMaxBlockCount`, `TransferQueueCurrentBlockCount`, ...) [SAC][TRM-T]. Why [SAC] uses it: "If all the connected cameras attempt to transfer images to the host at the same time, the result could be loss of packets during transfer or torn images." The two-camera [HT] example does **not** use it. | (none) | [SAC] [TRM-T] |
| Hardware trigger input | **C**. Product spec lists "hardware trigger" [P-HTP]; [HT] says hardware triggering works for HLT and TRI. Exact LineN entries accepted by `TriggerSource` on HTP are **NC**. | **C** | `LineSelector`, `LineMode`=`"Input"`, `TriggerSelector`=`"FrameStart"`, `TriggerSource`=`"Line0"` / `"Line2"`, `TriggerActivation`=`"RisingEdge"` / `"FallingEdge"` / `"AnyEdge"` (level modes break packet interleaving), `TriggerMode`=`"On"` [GPIO][BW]. Burst: `TriggerSelector`=`FrameBurstStart` + "Acquisition Burst Frame Count" [GPIO] (`AcquisitionBurstFrameCount` [TRM-T]). | (none) | [GPIO] [BW] [HT] [TRM-T] |
| TriggerDelay | **C** (Helios2 family) [MH]: `TriggerDelay` with Python `device.nodemap.nodes['TriggerDelay'].value = delay_value` | **C** [TRM-T]: "TriggerDelay: Specifies the delay in microseconds (us)" | `TriggerDelay` | (none) | [MH] [TRM-T] |
| TriggerOverlap / TriggerLatency | **NC** | **C** node names [TRM-T]: `TriggerOverlap`, `TriggerLatency` ("Enables low latency trigger mode"). Entry "One Line" is shown as a display value, measured on PHX050S-PC [GPIO]; not confirmed for TRI032S. | `TriggerOverlap`, `TriggerLatency`, `TriggerArmed` | (none) | [TRM-T] [GPIO] |
| Strobe output | **C** that the M8 GPIO exists with the same pinout [GPIO]. Whether `LineSource`=`ExposureActive` exists on Helios2+, and what it marks on a ToF multi-integration frame, is **NC**. | **C** | `LineSelector`, `LineMode`=`"Output"`, `LineSource`=`"ExposureActive"` [GPIO]; `LineInverter`, `LineStatus`, `VoltageExternalEnable` (Line4 2.5 V out) [TRM-T][GPIO] | (none) | [GPIO] [TRM-T] |
| GPIO electrical | **C** (same 8-pin M8 table applies to "Helios2/Helios2+") [GPIO]; opto input 2.5 to 24 V [P-HTP] | **C** [GPIO][TRM-T][P-TRI] | Pin 1 VAUX, 2 Line2 non-isolated bidir, 3 VDD 2.5 V out (Line4), 4 Line3 non-isolated bidir, 5 GND, 6 OPTO GND, 7 OPTO OUT Line1, 8 OPTO IN Line0 [GPIO] | (none) | [GPIO] [TRM-T] |
| Software trigger | **C** [P-HTP] | **C** | `TriggerSource`=`"Software"`, `TriggerSoftware` (execute), `TriggerArmed` [ROS2][TRM-T] | (none) | [ROS2] [TRM-T] |
| eTrigger (camera-originated action command) | **NC** | **NC**. Documented only for Triton10 in [RDMA]. | `ActionMode` (`Send`/`Receive`), `ActionSendTriggerSource`, `ActionPTPTimestampMode`=`Relative`, `ActionSendPTPTimestamp` [RDMA]. Check on device before relying on it. | (none) | [RDMA] |
| Frame-rate control | **C** that `AcquisitionFrameRate` / `AcquisitionFrameRateEnable` exist on HLT003S (screenshot); max 30 fps in Distance5000mm [P-HTP] | **C** [TRM-T]: `AcquisitionFrameRate` ("TriggerMode must be off for this parameter to take effect"), `AcquisitionFrameRateEnable` | (see left) | (none) | [HT] [P-HTP] [TRM-T] |
| Auto exposure limits | n/a (Helios2+ exposure is discrete `ExposureTimeSelector`) | Display names **C**: "Exposure Auto Lower Limit/Exposure Auto Upper Limit" [AE]. Exact node spellings (`ExposureAutoLowerLimit`, `ExposureAutoUpperLimit`, `ExposureAutoLimitAuto`) are **NC** and absent from the 2019 [TRM-T]. | `ExposureAuto` (`"Continuous"` / `"Off"` / `"Once"`), `ExposureTime` (us), `TargetBrightness`, `ExposureAutoAlgorithm`, `ExposureAutoDamping` [TRM-T][AE][SAC] | Limits were added after 2019; firmware version is **NC** | [AE] [TRM-T] |
| Bandwidth / packet nodes | **NC** per node on Helios2+ (expected to be standard) | **C** [TRM-T] | `DeviceLinkThroughputLimitMode`, `DeviceLinkThroughputLimit` (bytes/s), `DeviceLinkThroughputReserve` (%), `DeviceLinkSpeed`, `DeviceStreamChannelPacketSize`, `GevSCPSPacketSize`, `GevSCPD` ("delay (in GEV timestamp counter unit)"), `GevSCFTD` [TRM-T][BW][FPS]. Stream TL: `StreamAutoNegotiatePacketSize`, `StreamPacketResendEnable`, `StreamBufferHandlingMode` (`"NewestOnly"` / `"OldestFirst"`) [HT][BW] | (none) | [TRM-T] [BW] [FPS] |
| Chunk data | **C**: "Image, CRC, timestamp, line status, pixel format, width/height, offset X/Y, pixel dynamic range min/max" (no exposure-time chunk listed) [P-HTP] | **C**: "Image, CRC, histogram stats, timestamp, line status, sequencer set, ..., gain, black level, exposure time" [P-TRI] | `ChunkModeActive`, `ChunkSelector`, `ChunkEnable`, `ChunkExposureTime`, `ChunkCRC`, `ChunkGain` ... [TRM-T]. `ChunkSelector` entry strings (e.g. Timestamp) and a `ChunkTimestamp` node are **NC** (not in the 2019 list). | (none) | [P-HTP] [P-TRI] [TRM-T] |
| Python buffer timestamp / frame id | **C** | **C** | arena_api: `buffer.timestamp_ns`, `buffer.frame_id`, `buffer.width`, `buffer.height`, `buffer.pixel_format`. Node access: `device.nodemap['Name'].value = ...`, `device.nodemap['Name'].execute()`, `device.tl_stream_nodemap['StreamBufferHandlingMode'].value` [ROS2]. C++ equivalents: `GetTimestamp()`, `GetTimestampNs()`, `GetFrameId()` [HT][KB buffer-count]. | (none) | [ROS2] [HT] |
| Firmware / serial read | **C** in practice (`DeviceSerialNumber` is used generically in [BW] code) | **C** | `DeviceFirmwareVersion`, `DeviceSerialNumber`, `DeviceModelName` [TRM-T][BW] | Firmware update page: `http://{camera-ip}/firmware-update.html` [TRM-T] | [TRM-T] [BW] |
| Helios2+ ToF modes | **C** | n/a | `Scan3dOperatingMode` entries in [HT]: `Distance8333mmMultiFreq`, `Distance6000mmSingleFreq`, `Distance5000mmMultiFreq`, `Distance4000mmSingleFreq`, `Distance3000mmSingleFreq`, `Distance1250mmSingleFreq` ("single-frequency operating modes have faster image capture"). `ExposureTimeSelector`: `Exp1000Us`, `Exp250Us`, `Exp62_5Us`. `PixelFormat` = `Coord3D_ABCY16`. Max fps for the 5000 mm mode (120+90 MHz): **30 FPS** [P-HTP]. | (none) | [HT] [P-HTP] |
| Power | PoE+ (IEEE 802.3at) or 18 to 24 V via GPIO [P-HTP] | PoE (802.3af) or 12 to 24 V [P-TRI] | **The injector must be 802.3at for the Helios2+.** | (none) | [P-HTP] [P-TRI] |

---

## (b) Recommended node-write sequences (names quoted exactly; Python shown as arena_api style per [ROS2])

Notation: `nm = device.nodemap`, `sn = device.tl_stream_nodemap`, `tl = <Arena system TL nodemap>`. The C++ equivalent of `tl` is `pSystem->GetTLSystemNodeMap()`; the Python attribute name is **NC**, expected `system.tl_system_nodemap`.

### (i) PTP enable + wait for lock (pattern from [HT], Helios = master)
1. Helios: `nm['PtpSlaveOnly'].value = False`, then `nm['PtpEnable'].value = True`.
2. Poll Helios `nm['PtpStatus'].value` every ~1 s until it equals `"Master"`.
3. Triton: `nm['PtpSlaveOnly'].value = True`, then `nm['PtpEnable'].value = True`.
4. Poll Triton `PtpStatus` until it equals `"Slave"`. [BW] logic: done when exactly one `"Master"` exists and all others are `"Slave"`; any other state means negotiation is not complete. Allow about 40 s or more [BW].
5. Optional lock quality check on the slave: execute `PtpDataSetLatch`, then read `PtpOffsetFromMaster` (ns; `-1` = sync not started) [TRM-T]. LUCID gives no numeric "locked" threshold (**NC**). Logging the offset until it is stable and small is a reasonable criterion.
6. Read PTP time: `nm['PtpDataSetLatch'].execute()`, then `nm['PtpDataSetLatchValue'].value` (ns) [SAC][HT].

Network notes: both cameras must be on the same L2 network (a switch, or bridged NICs) [HT][PTPKB]. The PTP delay mechanism is E2E only [PTPKB]. On managed switches, IGMP snooping broke PTP in [RDMA]; an unmanaged switch normally floods multicast, but this is **NC** for this setup.

### (ii) PTPSync free-run at N fps (both cameras) [BW][HT]
Per camera, before `start_stream`:
1. `ExposureAuto` = `"Off"` and fixed `ExposureTime` (Triton). [BW] says "In order to get synchronized images, the exposure time must be synchronized as well". Alternatively keep auto exposure but cap it so it stays well below 1/N.
2. `PtpEnable` = True (+ `PtpSlaveOnly` as in (i)).
3. `sn['StreamAutoNegotiatePacketSize'] = True`, `sn['StreamPacketResendEnable'] = True`.
4. `AcquisitionMode` = `"Continuous"`.
5. `AcquisitionStartMode` = `"PTPSync"`.
6. `sn['StreamBufferHandlingMode'] = "NewestOnly"` (as in [BW]).
7. Optional packet interleave: `GevSCPD` / `GevSCFTD`. Formula in [BW]: for 1 GbE with 9014 B packets and 2 cameras, about 80000 ns each; camera 2 `GevSCFTD` about 80000.
8. `AcquisitionFrameRate` = its max. [BW]: "AcquisitionFrameRate node won't be controlling the actual frame rate but it might cap the max achievable frame rate".
9. `PTPSyncFrameRate` = N (for example 10.0). [BW]: "should always be less than the value set in AcquisitionFrameRate".
10. Wait for PTP Master/Slave (step (i) 4), then `start_stream()` on both.
11. Verify that `AcquisitionFrameRate` now reads N [HT step 5].
12. Optional phase shift: `PTPSyncOffset` in ns. Helios2+ FW >= 1.16.0.0 [MH]; Triton **NC**.

Ambiguity: [HT] says "Use the PTP Sync Frame Rate node on one camera to control the synchronized frame rate", but the [BW] code writes `PTPSyncFrameRate` on every camera. Set it identically on both.

### (iii) Scheduled action command, both cameras [HT][SAC]
Device side (both cameras), after PTP lock:
1. `TriggerSelector` = `"FrameStart"`, `TriggerSource` = `"Action0"`, `TriggerMode` = `"On"`.
2. `ActionUnconditionalMode` = `"On"`, `ActionSelector` = 0, `ActionDeviceKey` = K, `ActionGroupKey` = G, `ActionGroupMask` = M.
   - LUCID's code writes the device key variable into `ActionGroupMask`. It works only because all values are 1.
3. Optional (Triton, documented in [SAC] for many cameras): `TransferControlMode` = `"UserControlled"`, `TransferOperationMode` = `"Continuous"`, execute `TransferStop`. Later, per frame: `TransferStart`, get the image, `TransferStop`. Helios2+ support for these nodes is **NC**. For only 2 cameras at about 56 MB/s total, [HT] omits this.

Host side (Arena TL System nodemap):

4. `ActionCommandDeviceKey` = K, `ActionCommandGroupKey` = G, `ActionCommandGroupMask` = M, `ActionCommandTargetIP` = `0xFFFFFFFF` (broadcast).
5. `start_stream()` on both cameras.

Per frame:

6. On the master, execute `PtpDataSetLatch` and read `PtpDataSetLatchValue` as T_now.
7. `tl['ActionCommandExecuteTime'] = T_now + lead`, then `tl['ActionCommandFireCommand'].execute()`.
8. Get one buffer from each camera.

Lead time: LUCID's examples use 1 to 2 s ("Round up to the nearest second" + `g_action_delta_time = 1`). The minimum safe lead time for these cameras is **NC**. For 10 fps, schedule a train of future times on a fixed grid, at most `ActionQueueSize` ahead [TRM-T]. Or use PTPSync, which "schedules 4 advance action commands in the firmware itself" [BW].

LUCID's measured result with HLT003S + TRI032S-C [HT screenshot]: HLT timestamp = action time + 22.5 to 24.4 us, and TRI = action time + 40.0 to 42.3 us. The text says the ~20 us gap "is due to the firmware of these camera families marking the timestamp at slightly different points."

### (iv) Hardware trigger
**(a) External pulse to both cameras** (LUCID's first option in [HT]):
- Per camera: `LineSelector` = `"Line0"` (opto in, pin 8, ref pin 6) or `"Line2"` (non-isolated, pin 2, ref pin 5), `LineMode` = `"Input"`.
- Then `TriggerSelector` = `"FrameStart"`, `TriggerSource` = `"Line0"` (or `"Line2"`), `TriggerActivation` = `"RisingEdge"`, `TriggerMode` = `"On"` [GPIO][BW].
- Optional `TriggerDelay` (us) to align phases [MH][TRM-T].
- Over-trigger behavior: "If the input trigger frequency is greater than the sensor framerate, the trigger is discarded, not delayed" [GPIO].
- Optional: `PtpEnable` = True so both timestamps share one clock. [BW]: "The only benefit of turning on PTP is synchronization of internal clocks... If you want synchronized timestamps for your images then enable PTP."

**(b) Helios strobe to Triton trigger** (LUCID's second option in [HT]; [GPIO] "Synchronous Capture - Master and Slave", demonstrated on Phoenix):
- Master (Helios): `LineSelector` = `"Line4"`, `VoltageExternalEnable` = True (2.5 V VDD on pin 3) to pull up the output [GPIO].
- Master: `LineSelector` = `"Line3"` (or Line2), `LineMode` = `"Output"`, `LineSource` = `"ExposureActive"` [GPIO].
- Slave (Triton): trigger configuration as in (a), on `"Line2"` or `"Line3"` (non-isolated input: "Min Input High 1 V, Min Input Low 0.8 V") or `"Line0"` (opto: 2.5 to 24 V) [TRM-T][P-TRI].
- Common GND: pin 5 to pin 5.

Cables: LUCID `GPIO-M8-1M-A` / `GPIO-M8-5M-A` (M8 8-pin, Amphenol), right-angle variants, and breakout `GPIO-M8-BRK-KIT001` [P-HTP][P-TRI]. Older text names "GPIO-M8" [GPIO].

**Can the Helios2+ strobe drive the Triton trigger directly? NOT CONFIRMED for this pair.** The pinout is the same and LUCID shows the master-strobe-to-slave-trigger method in general (Phoenix). But nothing confirms that `ExposureActive` exists on HTP003S, or whether it pulses once or several times per ToF frame. [MH] describes 4 integrations per frame in single-frequency modes. **Check it with a scope.** Non-isolated outputs need a pull-up ("External Resistor" column) [TRM-T]; output polarity is set with `LineInverter`.

Trigger latency reference only: PHX032S (same IMX265 sensor, FW 1.5.0.0) measured 35.0 to 50.6 us latency with 15.6 us jitter [GPIO]. TRI032S values are **NC**.

---

## (c) What the image timestamps mean under PTP

- **Units and timebase.**
  - With PTP enabled, the buffer timestamp is in the PTP clock domain, in ns. LUCID compares `GetTimestamp()` directly against `ActionCommandExecuteTime` (PTP ns) for both HLT and TRI and prints "ns offset" [HT][SAC].
  - [PTPKB]: "timestamp produced by the cameras will share a common clock reference".
  - In Python, use `buffer.timestamp_ns` [ROS2].
- **Which event is stamped: NOT CONFIRMED** (no public LUCID text says exposure start, end or readout).
- **Empirical evidence from LUCID [HT]:**
  - With an action command at time T, HLT003S timestamps land at about T + 23 us and TRI032S-C at about T + 41 us.
  - [SAC] shows PHX016S at about T + 25 to 34 us.
  - LUCID attributes the HLT/TRI difference to the families "marking the timestamp at slightly different points". So the ~18 us inter-camera timestamp offset is at least partly a bookkeeping offset, not necessarily an exposure offset.
- **Helios ToF caveat.** A Helios frame is not an instant. [MH], for single-frequency modes at 1000 us: 4 integrations of about 0.98 ms each, each followed by a 1.55 ms readout, so "T(total) = (0.98 + 1.55) x 4 = 10.12 ms".
  - The micro-frame structure of `Distance5000mmMultiFreq` (two frequencies) is **NC**. It is likely longer.
  - Sub-ms sync can therefore only mean aligned start, or a chosen alignment point. Consider `TriggerDelay` / `PTPSyncOffset` on the Triton to place its exposure inside the Helios integration window.
- **Event timestamps (Triton).** `EventExposureStartTimestamp` and `EventExposureEndTimestamp` exist [TRM-T]. Enabling them gives an unambiguous exposure-start timestamp to compare against the buffer timestamp. They are **NC** on Helios2+.

---

## (d) Open questions to answer on the real cameras

Read the nodes in ArenaView or via `http://<camera-ip>/` firmware docs [DOCKB].

1. `DeviceFirmwareVersion` of each camera.
   - Is HTP >= 1.16.0.0 (needed for `PTPSyncOffset`)?
   - Which Triton firmware adds `AcquisitionStartMode`=`PTPSync`, `PTPSyncFrameRate` and auto-exposure limits?
2. On the HTP003S, list the enum entries of `AcquisitionStartMode`, `TriggerSource` (Action0? Line0/2/3? Software?), `TriggerSelector`, `TriggerActivation`, `LineSource` (ExposureActive?), `LineSelector`, and `ChunkSelector`. Check that `TriggerOverlap`, `TriggerLatency`, `TransferControlMode` and `PTPSyncOffset` exist.
3. On the Triton, check the exact names and units of the auto-exposure limit nodes (display "Exposure Auto Lower/Upper Limit"; "ExposureAutoLimitAuto" is **NC**), and whether `PTPSyncOffset` exists.
4. Python accessor for the Arena system TL nodemap (expected `system.tl_system_nodemap`; check with `dir(system)`), and whether it exposes `ActionCommand*` identically to C++.
5. What exactly the buffer timestamp marks on each camera. Measure with `ExposureActive` strobes from both cameras on a scope (Triton is confirmed; Helios is **NC**), or compare against Triton `EventExposureStartTimestamp`.
6. Helios `ExposureActive` behavior in `Distance5000mmMultiFreq` + `Exp1000Us`: one pulse or several per frame, and the total integration span.
7. Minimum action-command lead time and `ActionQueueSize` on both cameras. Does an action whose time has passed fire immediately or get dropped? LUCID does not document this.
8. PTP lock quality through the unmanaged switch: log `PtpOffsetFromMaster` and `PtpServoStatus` over time. Decide whether the Helios or the Triton should be master (the BMCA default, versus LUCID's HLT-master example).
9. The PoE injector rating (Helios2+ needs 802.3at).
10. Whether the Triton's `ExposureAuto`=`Continuous` (as in [HT]) is acceptable, or whether a fixed or capped exposure is needed so a 10 fps trigger is never over-run.
