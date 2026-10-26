# Mock `arena_api` (LUCID Helios2+ HTP003S-001 + Triton TRI032S-C)

A pure-Python + numpy stand-in for LUCID's `arena_api`. Capture and sync scripts can run
end to end without cameras. It models the **timing**: PTP, PTPSync, scheduled action
commands, trigger latency, exposure drag, link bandwidth and buffer loss. The image
content is synthetic and costs no CPU per frame.

**This is not the real SDK.** Scripts should log this at startup:

```python
import arena_api
if getattr(arena_api, "__mock__", False):
    print("RUNNING ON MOCK SDK", arena_api.__version__)   # "mock-0.1.0"
```

## Activate

Put `camera_sync/mock` first on `sys.path`, either with `PYTHONPATH=camera_sync/mock python capture_sync.py`
or with `sys.path.insert(0, ".../camera_sync/mock")` before `import arena_api`. The
capture scripts' `LUCID_MOCK=1` switch does this insert. The mock itself does not read
`LUCID_MOCK`.

Tests: `python -m pytest camera_sync/tests/test_mock_sdk.py -q` (about 17 s).

## API surface (same names as the real SDK)

| Object | Implemented |
|---|---|
| `arena_api.system.system` | `device_infos` (dicts: model, serial, ip, mac, vendor, subnetmask, defaultgateway, name, dhcp, presistentip, lla, version), `create_device(device_infos=None)`, `destroy_device(device=None)`, `tl_system_nodemap`, `DEVICE_INFOS_TIMEOUT_MILLISEC` |
| `Device` | `nodemap`, `tl_stream_nodemap`, `tl_device_nodemap`, `tl_interface_nodemap` (empty), `start_stream(number_of_buffers=10)`, `stop_stream()`, `get_buffer(timeout=ms, number_of_buffers=None)`, `requeue_buffer(buf or list)` |
| `Buffer` | `pdata` (ctypes `POINTER(c_ubyte)` to real memory of full payload size), `data` (read-only `memoryview`; the real SDK returns a list), `width`, `height`, `pixel_format` (`enums.PixelFormat`), `bits_per_pixel`, `size_filled`, `payload_size`, `buffer_size`, `timestamp_ns` (= `timestamp`), `frame_id`, `is_incomplete`, `has_chunkdata`, `has_imagedata`, `get_chunk(name or [names])` |
| `arena_api.buffer.BufferFactory` | `copy(buf)`, `destroy(copy)` (`convert` raises NotImplementedError) |
| `arena_api.enums` | `PixelFormat` (Mono8, Mono16, BayerRG8, BayerRG16, RGB8, Coord3D_C16, Coord3D_ABC16, Coord3D_ABCY16), `InterfaceType`, `AccessMode` |
| Node map | `nm['X']` raises KeyError if missing; `nm.get_node('X')` returns None if missing; `nm.get_node([..])` returns a dict with None for missing names; `nm.nodes`, `in` |
| Node | `.name .value .is_readable .is_writable .access_mode .min .max .inc .unit .enumentry_names .execute() .interface_type` |

Errors are built-in types, so catch the built-ins:

| Situation | Exception |
|---|---|
| Out of range, bad increment, invalid enum entry, wrong type | `ValueError` |
| Write to a read-only node, or to a node locked while streaming | `RuntimeError` |
| `get_buffer` timeout | `TimeoutError` |
| `nm['Missing']` | `KeyError` |
| Use of a buffer after `requeue_buffer` | `RuntimeError` |

Mock-only helpers (not in the real SDK; never call them from production code paths):
- `arena_api.mock_reset(**knobs)` power-cycles both cameras with defaults + env + knobs.
- `arena_api.mock_config()` returns the live config object.
- `device.mock_stats()` returns counters: frames_exposed, delivered, lost_no_buffer, incomplete, busy_drops, missed_triggers, actions_received, actions_late, actions_dropped_late, actions_dropped_queue_full, actions_ignored_not_armed, switch_overflows, ptpsync_skipped_fps_cap, software_trigger_ignored.
- `buffer.mock_info` gives the true model times of the frame.
- `system.mock_hardware_trigger("Line0")` simulates an electrical edge on a line of both cameras.
- `system.mock_world` exposes the simulation object.

## Nodes

Node names come from the research reports A and C. Names or entries that are not in those
reports are marked `# NAME NOT CONFIRMED` in `_sim.py` and in the table below (NC).

**Both cameras**
- DeviceVendorName (NC), DeviceModelName, DeviceSerialNumber, DeviceFirmwareVersion, DeviceLinkSpeed (125,000,000 B/s)
- Width, Height, PixelFormat
- AcquisitionMode (Continuous, SingleFrame NC), AcquisitionFrameRateEnable, AcquisitionFrameRate, AcquisitionStartMode (Normal, LowLatency NC, PTPSync), PTPSyncFrameRate, PTPSyncOffset (Helios by default)
- TriggerSelector (FrameStart, FrameBurstStart), TriggerMode, TriggerSource (Software, Line0, Line2, Line3, Action0), TriggerActivation, TriggerDelay (us), TriggerSoftware, TriggerArmed
- ActionUnconditionalMode, ActionSelector, ActionDeviceKey, ActionGroupKey, ActionGroupMask, ActionQueueSize
- PtpEnable, PtpSlaveOnly, PtpStatus, PtpServoStatus (entries Unknown/Locked NC), PtpOffsetFromMaster, PtpDataSetLatch, PtpDataSetLatchValue
- DeviceLinkThroughputLimitMode, DeviceLinkThroughputLimit, DeviceLinkThroughputReserve, GevSCPD, GevSCFTD, GevSCPSPacketSize
- ChunkModeActive, ChunkSelector (entries NC), ChunkEnable, ChunkTimestamp (NC)
- LineSelector (Line0..Line4), LineMode, LineSource (Off, ExposureActive)

**Triton only**
- ExposureAuto, ExposureTime (30.25 us to 10 s)
- ExposureAutoLimitAuto (NC; can be removed with a knob), ExposureAutoLowerLimit, ExposureAutoUpperLimit
- GainAuto (NC), Gain (NC)
- TargetBrightness, ChunkExposureTime, ChunkGain

**Helios only**
- Scan3dOperatingMode (6 entries), ExposureTimeSelector (Exp1000Us, Exp250Us, Exp62_5Us)
- Scan3dCoordinateSelector (CoordinateA/B/C, NC), Scan3dCoordinateScale, Scan3dCoordinateOffset

**Stream TL nodemap**
- StreamAutoNegotiatePacketSize, StreamPacketResendEnable, StreamBufferHandlingMode (OldestFirst, OldestFirstOverwrite, NewestOnly)
- StreamDeliveredFrameCount, StreamLostFrameCount, StreamIncompleteFrameCount (all NC)

**System TL nodemap** (`system.tl_system_nodemap`; this Python accessor name is NC on the real SDK)
- ActionCommandDeviceKey, ActionCommandGroupKey, ActionCommandGroupMask, ActionCommandTargetIP, ActionCommandExecuteTime, ActionCommandFireCommand

**TL device nodemap**
- DeviceModelName, DeviceSerialNumber, GevDeviceIPAddress (NC), GevDeviceSubnetMask (NC), GevDeviceMACAddress (NC)

**Access rules modelled**

Locked while streaming:
- Width, Height, PixelFormat
- AcquisitionMode, AcquisitionStartMode
- TriggerMode
- GevSCPSPacketSize
- ChunkModeActive
- Scan3dOperatingMode
- StreamAutoNegotiatePacketSize, StreamBufferHandlingMode

Conditionally writable:
- AcquisitionFrameRate only when AcquisitionFrameRateEnable is True.
- DeviceLinkThroughputLimit only when DeviceLinkThroughputLimitMode is On.
- ExposureTime only when ExposureAuto is Off. Gain only when GainAuto is Off.
- ExposureAutoLower/UpperLimit only when ExposureAutoLimitAuto is Off.
- LineMode only for Line2 and Line3.

Read-only: device info, link speed, PTP status and latch values, ActionQueueSize, and the Helios Width and Height.

**Selectors:**
- TriggerMode, Source, Activation and Delay are stored per TriggerSelector entry.
- ChunkEnable is stored per ChunkSelector entry.
- LineMode and LineSource are stored per LineSelector entry.
- ActionGroupKey and Mask are stored per ActionSelector entry.
- Scan3dCoordinateScale and Offset are stored per Scan3dCoordinateSelector entry.

## What is modelled

**Clocks.**
- True time is `time.monotonic_ns()`.
- Each camera's own clock is `boot_offset + (t - t0) * (1 + ppm*1e-6)`. The boot offset is random, 50 to 5000 s. Drift is Helios +12 ppm and Triton −18 ppm.
- Without PTP, timestamps are on the camera's own clock, so the two cameras' clocks cannot be compared.

**PTP.**
- Roles:
  - With PtpEnable on, the master is the lowest-serial camera among those with `PtpSlaveOnly=False`. This stands in for BMCA. The Helios serial is the lower one.
  - If every camera is slave-only, all stay `Listening`.
- Status sequence:
  - `Disabled` while PtpEnable is off.
  - `Initializing` for 0.2 s after enable.
  - Then `Listening`, and `Uncalibrated` for the last 30 % of the lock time on a slave.
  - Then `Master` / `Slave` after `ptp_lock_s`. The lock time is counted from the later of the camera's own enable and the master's.
- A locked slave's clock is the master's clock plus a Gaussian residual (σ = `ptp_jitter_ns_std`, 1 µs). Before lock, the slave keeps its own clock.
- `PtpDataSetLatch` latches the current clock into `PtpDataSetLatchValue`. It also sets `PtpOffsetFromMaster`: −1 before sync, 0 on the master, residual on a slave.

**Frame generation** (one background thread per streaming camera; all event times are model times, so timestamps are exact):
- **Freerun** (`TriggerMode Off`, `AcquisitionStartMode Normal/LowLatency`):
  - Frames run on the camera's current clock with a random phase.
  - Period = `max(1/fps, exposure + readout)`, so a long Triton exposure drags the frame rate.
  - fps = AcquisitionFrameRate if Enable is True, else the current max.
- **PTPSync:**
  - Requires PTP state Master or Slave. Without lock the camera produces **no frames** (it keeps polling, and frames start once it locks).
  - Exposure starts at PTP times `PTPSyncOffset + k * 1e9/PTPSyncFrameRate`, plus the trigger latency below. The grid is the same on both cameras, so frames align.
  - The AcquisitionFrameRate set before `start_stream` caps the rate: grid points closer together than 1/cap are skipped.
  - After start, `AcquisitionFrameRate` reads `PTPSyncFrameRate`. This stays after stop.
  - Trigger settings are ignored, and TriggerDelay is not applied.
- **Action command** (`TriggerSource Action0`, `TriggerMode On`):
  - `ActionCommandFireCommand` delivers to every visible camera.
  - A camera accepts it when:
    - TargetIP is the broadcast address or equals the camera's IP;
    - DeviceKey and GroupKey match;
    - `GroupMask & ActionGroupMask != 0`;
    - the camera is streaming in trigger mode with source Action0.
  - Exposure starts at the action time on the camera's PTP clock + TriggerDelay + latency.
  - `ExecuteTime == 0` executes on receipt.
  - An ExecuteTime that has already passed on arrival (arrival = 100 µs network latency) either fires immediately or is dropped, per `late_action_behavior`.
  - More than `ActionQueueSize` pending scheduled actions: the extra ones are dropped.
- **Software trigger** (`TriggerSoftware.execute()`): now + latency + TriggerDelay. The trigger is ignored, and counted, if the camera is not in software trigger mode.
- **Line trigger:** `system.mock_hardware_trigger(line)`, if the camera's source is that line and the line is an Input.
- **Busy:**
  - A trigger whose exposure would start before the previous frame's exposure + readout has finished is dropped (`busy_drops`). LUCID: "the trigger is discarded, not delayed".
  - The `missed_trigger_prob` fault drops triggers and PTPSync grid events at random.

**Timestamps.**
- `timestamp_ns` = the camera's clock (PTP once locked) at the event selected by `timestamp_event`, plus `timestamp_extra_offset_us`.
- The default event is exposure start.
- Default trigger latency is 23.4 µs (Helios) and 41.1 µs (Triton). This reproduces LUCID's measurement: HLT timestamps at action time + 22.5 to 24.4 µs, TRI at + 40.0 to 42.3 µs, a difference of about 18 µs. Freerun frames have no trigger latency.

**Sensor windows.**
- Triton busy = exposure + 18 ms readout.
- Helios window = n_sub × (0.98 × selector exposure + 1.55 ms), with n_sub = 8 for MultiFreq modes and 4 for SingleFreq modes.
  - Exp1000Us + MultiFreq: 20.24 ms window, light span 18.69 ms.
  - Exp250Us: 14.36 ms window.
  - Exp62_5Us: 12.9 ms window.
- Helios max fps is 30. Triton sensor max is 55.6 fps.
- `AcquisitionFrameRate.max` = min(sensor, 1/window for the Helios or for the Triton with manual exposure, link payload rate × (1 − reserve) / frame bytes).
  - Triton, no limit, 9000 B packets: 35.4 fps. This matches LUCID's listed 35.4 fps.
  - Triton, 60 MB/s limit: 17.0 fps.

**Transfer and delivery.**
- Transfer starts at readout end + GevSCFTD, and transfers are serialised per camera.
- Packet efficiency = (pkt − 48)/(pkt + 38).
- Wire rate:
  - With `DeviceLinkThroughputLimitMode On`: min(limit, 125 MB/s).
  - Otherwise: set by `GevSCPD` (ns of inter-packet delay).
- `StreamAutoNegotiatePacketSize=True` sets GevSCPSPacketSize to `host_mtu` (9000) at `start_stream`.
- Delivery = transfer end + 300 µs host latency, plus any `host_stall` window.
- **Shared-link congestion:** while two transfers overlap, the switch's PC port drains at line rate. If the excess (sum of rates − capacity) × overlap is more than `switch_buffer_bytes` (192 KiB), the later-registered frame is incomplete with probability `switch_overflow_incomplete_prob` (0.9).
  - With the recommended limits (48 + 60 MB/s) this never triggers.
  - With unthrottled cameras exposing together (PTPSync or action), it almost always does.
- `packet_loss_prob` adds random incomplete buffers. An incomplete buffer has `is_incomplete=True` and `size_filled` between 30 % and 97 % of the payload.

**Buffers.**
- `start_stream(N)` allocates N real buffers. A frame needs a free buffer when it is delivered:
  - `OldestFirst`: no free buffer means the new frame is lost.
  - `OldestFirstOverwrite`: the oldest waiting frame is overwritten.
  - `NewestOnly`: only the newest frame is kept waiting.
- Lost frames leave gaps in `frame_id`. `frame_id` starts at 1 at each `start_stream` and increments for every exposed frame.
- `get_buffer` returns a fresh view. After `requeue_buffer` that view raises, which catches use-after-requeue bugs. Copy the data first.

**Triton auto exposure.**
- The scene wants exposure `scene_exposure_us[scene] × TargetBrightness/70`: bright 1 ms, dim 15 ms, dark 1.65 s.
- `ExposureAuto Continuous` gives clamp(wanted, lower, upper):
  - With `ExposureAutoLimitAuto Continuous` (the default) the limits are the full range, 30.25 µs to 10 s. A dark scene therefore gives 1.668 s per frame, about **0.6 fps**: the user's bug.
  - Switching `ExposureAutoLimitAuto Off` keeps the last auto limits (the full range). You must then write `ExposureAutoUpperLimit`, e.g. 20,000 µs, which restores 10 fps.
- `Once` applies one frame and then turns itself Off.
- Gain auto makes up the deficit: 20·log10(wanted/actual), 0 to 48 dB.
- `ChunkExposureTime` and `ChunkGain` carry per-frame values when chunks are on.

**Helios data.**
- `Coord3D_ABCY16` contains a plane about 3 m away (Z rises from 2.8 to 3.2 m down the image).
- X and Y come from a 67° × 51° pinhole model. Y is intensity with radial fall-off. Corners are invalid (C = 0, Y = 0).
- Raw values follow `mm = raw × 0.25 + offset`, with offset −8192 mm for A and B and 0 for C.
- `Coord3D_C16`, `Coord3D_ABC16` and `Mono16` (intensity) are also available.
- ABCY16 has no confidence channel. Invalid pixels are marked with 0.

**Small frames.**
- `LUCID_MOCK_SMALL=1` (or `small_frames=True`) divides Width and Height by 4: Helios 160×120, Triton 512×384.
- By default the bandwidth model still uses full-size byte counts (`small_frames_keep_link_timing`), so congestion and timing behave like the real rig.

## Knobs (`arena_api/_mock_config.py`)

Order of application:
1. Defaults.
2. The environment: `LUCID_MOCK_SMALL=1`, and `LUCID_MOCK_FAULTS='{"knob": value}'` as JSON.
3. `arena_api.mock_reset(**knobs)`.

Live knobs can also be changed via `arena_api.mock_config().knob = v`. These are the probabilities, `scene` and `late_action_behavior`. Per-camera dict knobs merge, so `{"drift_ppm": {"triton": -30}}` keeps the Helios value.

| Knob | Default | Meaning |
|---|---|---|
| `seed` | None | RNG seed (offsets, jitter, faults) |
| `small_frames` | False | /4 resolution (env `LUCID_MOCK_SMALL=1`) |
| `small_frames_keep_link_timing` | True | Bandwidth model uses full-size bytes |
| `device_missing` | None | `"helios"` / `"triton"`: hide that camera |
| `drift_ppm` | helios 12, triton −18 | Oscillator error |
| `initial_offset_s_range` | (50, 5000) | Random clock value at mock start (s) |
| `ptp_lock_s` | 3.0 | Enable → Master/Slave time |
| `ptp_init_s` | 0.2 | Duration of "Initializing" |
| `ptp_never_locks` | False | Slaves stay "Listening" (master still Master) |
| `ptp_jitter_ns_std` | 1000 | Slave clock residual, 1 σ |
| `trigger_latency_us` | helios 23.4, triton 41.1 | Trigger event → exposure start |
| `trigger_jitter_ns_std` | 500 | Latency jitter, 1 σ |
| `timestamp_event` | "exposure_start" | Or "exposure_end", "readout_end" |
| `timestamp_extra_offset_us` | 0, 0 | Extra constant per camera |
| `action_queue_size` | 4, 4 | ActionQueueSize value |
| `action_network_latency_us` | 100 | Host → camera delivery of an action command |
| `late_action_behavior` | "fire_immediately" | Or "drop" |
| `triton_readout_ms` | 18.0 | IMX265 readout |
| `triton_sensor_max_fps` | 55.6 | |
| `helios_max_fps` | 30.0 | |
| `helios_readout_ms` | 1.55 | Per sub-frame |
| `helios_integration_ratio` | 0.98 | Real integration / selector value |
| `scene` | "bright" | "bright", "dim" or "dark" |
| `scene_exposure_us` | 1000 / 15000 / 1650000 | Exposure auto wants per scene |
| `triton_has_exposure_auto_limit_auto` | True | Include the ExposureAutoLimitAuto node |
| `triton_has_ptpsync_offset` | False | Include PTPSyncOffset on the Triton |
| `helios_has_ptpsync_offset` | True | Include PTPSyncOffset on the Helios |
| `link_speed_Bps` | 125,000,000 | DeviceLinkSpeed |
| `host_mtu` | 9000 | Result of packet-size auto-negotiation |
| `switch_buffer_bytes` | 196,608 | Shared switch buffer |
| `switch_overflow_incomplete_prob` | 0.9 | P(incomplete) when the buffer overflows |
| `packet_loss_prob` | 0.0 | Extra random incomplete buffers |
| `missed_trigger_prob` | 0.0 | Random lost triggers and PTPSync grid events |
| `drop_nodes` | {} | e.g. `{"triton": ["PTPSyncOffset", "AcquisitionStartMode"]}`; a plain list applies to both cameras (`drop_node` also accepted) |
| `host_stall` | None | `{"camera": "triton", "after_s": 5, "duration_s": 2}` delays deliveries |

## Assumptions in the mock that are NOT verified on real hardware

Each item gives the number used and where it came from.

1. **Which event the timestamp marks.** The mock stamps exposure start. Trigger→exposure latency is 23.4 µs on the Helios and 41.1 µs on the Triton. Only the sums (timestamp − action time) come from LUCID's HLT003S/TRI032S-C measurement (22.5–24.4 µs and 40.0–42.3 µs) [A §(iii), HT app note]. The split between real latency and a bookkeeping offset is unknown, so it is a knob.
2. **PTP lock time** is 3 s by default. LUCID says it can take "about 40 seconds" [BW], and its HT sample locked in about one 1 s poll. The `Uncalibrated` phase, the 0.2 s `Initializing`, and timing the lock from the later enable are inventions.
3. **PTP residual** is 1 µs σ, white noise [C §1: 0.1–2 µs expected; Allied Vision < ±1 µs]. A real servo residual is correlated, not white.
4. **BMCA** is modelled as lowest serial wins. Real BMCA compares clock quality, then clock identity (MAC-based).
5. **`PtpServoStatus` entries** Unknown/Locked, and `PtpOffsetFromMaster` on the master = 0, are invented. The −1 before sync comes from the Triton TRM.
6. **PTPSync grid.**
   - The mock assumes exposures land on PTP times that are exact multiples of 1/PTPSyncFrameRate, plus PTPSyncOffset. LUCID only says the firmware schedules action commands itself, 4 in advance [BW].
   - It assumes the same trigger latency applies, and that TriggerDelay does not.
   - No frames before lock is the mock's choice.
   - The `AcquisitionFrameRate` cap rule is interpreted from [BW]. The read-back of PTPSyncFrameRate comes from [HT].
7. **PTPSyncOffset range** is 0 to 999,999,999 ns. It is absent on the Triton by default, because it is not confirmed there [MH].
8. **Late action commands** fire immediately by default, and **ActionQueueSize = 4**. LUCID does not document either [A open question 7]. The 100 µs network latency is an estimate.
9. **Triggers while busy are dropped.** Busy means exposure + readout of the previous frame, following [GPIO] "discarded, not delayed". Real IMX265 overlap (TriggerOverlap) may accept earlier triggers.
10. **Triton readout is 18 ms** (1/55.6 fps, IMX265 sensor maximum, FRAMOS) [C §5]. The frame period is exposure + readout, which is the no-overlap case. A real free-running Triton may overlap exposure with readout.
11. **Helios timing.** The sub-frame count (8 for MultiFreq, 4 for SingleFreq) and the 0.98 integration ratio for the 250 and 62.5 µs selectors are derived [C §4, from MH and Basler blaze]. Max 30 fps is used for all modes; only Distance5000mmMultiFreq is sourced [P-HTP].
12. **Bandwidth.**
    - The packet overhead of 48 B (IP + UDP + GVSP extended) plus 38 B (Ethernet) comes from [C §2].
    - DeviceLinkThroughputLimit is treated as a wire rate, and the reserve is applied only to the max-fps calculation. How Limit and Reserve interact is undocumented [C §2].
    - GevSCPD is in ns, per [BW], although the TRM calls it timestamp-counter units.
    - The default `DeviceLinkThroughputLimitMode` is Off, Limit is 125 MB/s, Reserve is 10 % [C: LUCID default 10 %], and the default packet size is 1500.
    - The 300 µs host latency is an estimate.
13. **Switch overflow** uses a 192 KiB shared buffer (Netgear GS108 / TP-Link TL-SG108) [C §2] and P(incomplete | overflow) = 0.9, which is invented because resends can rescue some frames. Only the later-registered of two overlapping frames is marked. Throughput is not reduced by congestion.
14. **Auto exposure.**
    - The scene exposures (1 ms, 15 ms, 1.65 s) are invented. 1.65 s was chosen to reproduce the user's ~0.6 fps symptom.
    - The `ExposureAutoLimitAuto` name and its default Continuous are NOT CONFIRMED [A §(a)].
    - Limit nodes are read-only while it is Continuous, and switching it Off keeps the full range. These follow GenICam convention and are assumptions.
    - Convergence is instant: there is no damping.
    - TargetBrightness defaults to 70. Gain max is 48 dB. ExposureAuto and GainAuto default to Continuous. All unverified.
15. **Access rules.**
    - TriggerMode is locked while streaming, as the design requires. Real firmware may allow it.
    - AcquisitionFrameRate is writable only when its Enable is True, and DeviceLinkThroughputLimit only when its Mode is On. These are GenICam convention, unverified.
    - `AcquisitionFrameRateEnable` defaults to False.
    - TriggerSource defaults to Line0.
16. **Scan3d conversion** is 0.25 mm/unit for A, B and C, with offsets A, B = −8192 mm and C = 0, which gives a symmetric ±8.19 m X/Y range. The design brief suggested "offset −8192×0.25", i.e. −2048 mm, which would make the X range asymmetric. Both are unverified, so read the real nodes per coordinate. The `CoordinateA/B/C` entry names are NC.
17. **Clock drift** is +12 / −18 ppm, and the random clock offset is 50–5000 s. These are illustrative values within the ±100 ppm IEEE 802.3 bound [C §1].
18. **`frame_id`** resets to 1 at each `start_stream`, and the counter also advances for frames that are later lost on the host. Unverified.
19. **ChunkSelector entries** (Image, CRC, Timestamp, ExposureTime, Gain) and `ChunkTimestamp` are NC.
20. **Action commands while not streaming** are ignored. `ActionUnconditionalMode` has no effect in the mock. `ActionDeviceKey` is readable here, but SFNC defines it as write-only.
21. **`device_infos` 'version' key** and the other extra keys are NC. The serials, IPs and MACs are invented. The MAC prefix 1c:0f:af is assumed to be LUCID's.

## Known differences from the real SDK

- `Buffer.data` is a memoryview, not a list. Use `pdata` with `np.ctypeslib.as_array` either way.
- The exception classes differ. Catch the built-ins listed above.
- Node values persist across `create_device` / `destroy_device` within a process, as on a powered camera. `mock_reset()` power-cycles both cameras.
- Not modelled: resends, TransferControl, burst triggers, event channel (EventExposureStart), UserSets, LUT/gamma, image conversion, LowLatency timing.
- Timing is real-time. Under heavy host load, get_buffer *delivery* is late, but timestamps stay exact model values.
