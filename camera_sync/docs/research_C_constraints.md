# Researcher C: constraints report (Helios2+ HTP003S-001 + Triton TRI032S-C on one GigE link)

Labels used below: **[src N]** means the number comes from source N. **[calc]** means I worked it out from sourced inputs. **[bench]** means I measured it on this sandbox (Xeon at 2.1 GHz, AVX-512, OpenCV 5.0.0, numpy 2.5). **[est]** means engineering judgement that is not sourced.

## TL;DR

| Topic | Answer |
|---|---|
| Hardware checked | TRI032S-C uses a **Sony IMX265 global-shutter** sensor (2048x1536, 3.45 um, 12-bit). It is listed at **35.4 fps with a 10 % throughput reserve, or 38 fps with 0 %**, and has a 128 MB frame buffer [src 1]. Helios2+ uses the **Sony DepthSense IMX556PLR** [src 2,3]. |
| Helios2+ fps in 5000 mm multi-frequency (120 + 90 MHz) | **30 fps maximum** [src 2,3]. So the Helios does **not** cap the pair at 15 fps. The link does. |
| Highest pair rate on one 1 GbE link with Helios in Coord3D_ABCY16 | **15 fps.** That is 84 MB/s of payload, 68 % of line rate [calc]. 20 fps needs 112 MB/s (91 % of the wire with jumbo frames), which leaves no room for resends, so it is not realistic. |
| To go faster | Switch Helios to Coord3D_C16 (2 B/px) or CY16 (4 B/px) and rebuild XYZ on the host from intrinsics. With C16 the pair needs 75 MB/s at 20 fps and 94 MB/s at 25 fps [calc]. |
| Throughput limit (works for both 10 and 15 fps) | **Triton DeviceLinkThroughputLimit = 60,000,000 B/s, Helios = 48,000,000 B/s** (108 MB/s total). Per-frame transfer is about 52 ms for Triton and 51 ms for Helios, or about 58 ms if a 10 % reserve is also applied. That fits the 66.7 ms period at 15 fps [calc]. A throttle is **mandatory**: two cameras bursting at line rate overflow a 128–192 KB switch buffer in 1.0–1.5 ms [calc, src 9,10]. |
| PTP through an unmanaged switch | Expect **sub-µs to a few µs** clock agreement. Frame starts from a scheduled action command land within about **10–30 µs** of each other [src 4,6,7]. The camera-to-camera PTP path does **not** go through the congested PC-facing egress queue. |
| Without PTP | Each 1 GbE oscillator is allowed ±100 ppm (IEEE 802.3) [src 11]. Typical relative drift of 10–50 ppm gives **3–15 ms per 5 min**, and the worst case is 60 ms. Free-running cameras with no sync have an arbitrary phase of up to ±½ frame period (±50 ms at 10 fps). |
| ToF frame duration (multi-frequency, Exp1000Us) | About **8 sub-frames x (0.98 ms integration + 1.55 ms readout) = 20.2 ms**. Light is collected from **0 to about 18.7 ms**, centred at about **9.35 ms** after the Helios frame start [calc from src 5 and 8]. This internal skew is 1000x larger than the PTP error, so **PTP plus action commands is more than sufficient. A trigger cable buys nothing.** |
| Triton trigger delay | **TriggerDelay ≈ 9350 − E/2 µs.** For E = 5 ms that is **6850 µs**. For E = 8 ms it is **5350 µs** [calc]. Verify once with an oscilloscope. |
| Triton exposure cap | 20 ms is technically fine at 10 and 15 fps. For a hand-held rig, cap **ExposureAutoUpperLimit at about 5000–8000 µs** and let gain take up the rest [calc/est]. |
| Disk and CPU | Raw writes take about 2–3 ms per frame [bench], are GIL-free and are trivial. PNG costs about **116 ms per Triton frame and 76 ms per Helios frame at level 1** on this box [bench], so the pair needs about 1.9 cores at 10 fps. **Do not encode PNG in the capture loop.** Watch for QLC SSDs: the Samsung 870 QVO 1 TB drops to **80 MB/s** once its cache is full (up to 42 GB of SLC cache) [src 13]. That is below the 84 MB/s needed at 15 fps. |

---

## 1. PTP through an unmanaged (non-PTP-aware) switch

### What vendors state

* **LUCID.** Cameras implement IEEE 1588-2008 using **End-to-End delay measurement only, not Peer-to-Peer**. They pick a master with the Best Master Clock algorithm and must share a network ("via a network bridge or a switch"). LUCID notes that "when you use a switch, bandwidth sharing may be required" and that with PTP Sync "you lose the ability to trigger cameras" [src 4]. The Helios2–Triton app note says "PTP assumes that the cameras will be connected to a switch" and reports **"a difference of around 20 microseconds between the Helios and Triton timestamps"**, caused by the two firmware families stamping at slightly different points [src 5]. LUCID publishes **no** managed vs unmanaged guidance and no accuracy figure.
* **LUCID measured data.** Seven PHX016S cameras behind a Netgear MS510TXPP were driven by a scheduled action command, with ExposureTime 500. Their frame timestamps landed **24.8–33.9 µs after the scheduled time, a spread of 9.15 µs** [src 6; spread is my calc].
* **Allied Vision.** Slaves count as synchronised when they are "within 2 µs of the master". In a 69-hour test of 4 cameras on a switch, the **maximum slave offset was < ±1 µs**. A scope comparison of strobe outputs showed **about 125 ns** between two cameras [src 7].
* **Basler.** Precision "depends to a large extent on your network hardware and setup". Basler recommends PTP-enabled switches for best results and says lock "may take a few seconds or minutes". It also notes the offset does not decrease monotonically, so you should watch the maximum over a time window. Peer-to-peer measurement needs every switch to be PTP-aware [src 8a]. Basler's blaze ToF free-run sync adds a **250 µs safety delay** "to compensate for jitter" [src 8b].

### Traffic reasoning for your topology [calc/est]

* PTP event messages (Sync, Delay_Req, Delay_Resp) are small, about 86 bytes on the wire. PTPv2 sends them to the multicast group 224.0.1.129, so an unmanaged switch floods them to every port.
* The path that matters is master camera port → switch → slave camera port, and back. On the **camera-facing egress ports** the only other traffic is GVCP from the PC, resend requests, broadcast/multicast noise and the other camera's PTP. Those queues are almost always empty.
* The heavy GVSP stream queues on the **PC-facing egress port**. A copy of each flooded PTP packet also queues there, but that copy is irrelevant because the PC is not a PTP participant.
* **So store-and-forward queueing on the PC port does not touch the camera-to-camera PTP path.** The worst-case extra delay on a camera port is one maximum-size frame already being sent: 1518 B is 12 µs. That is rare, it is random rather than a bias, and the PTP servo filters it out.
* Store-and-forward latency for 86-byte frames (about 0.7 µs of serialisation plus a few µs of fabric time) is roughly **symmetric** for Sync and Delay_Req. The E2E calculation cancels it, leaving residual asymmetry well under 1 µs.
* **Failure modes to watch for:**
  * **Unknown-unicast flooding.** If the switch forgets the PC's MAC address (aging is typically about 300 s), GVSP unicast floods to the camera ports. Camera-port queues then fill with jumbo frames and PTP jitter rises to tens of µs. The PC sends GVCP heartbeats at least every few seconds, so in practice the MAC stays learned.
  * **Shared buffer exhaustion.** Cheap switches share one buffer pool. If the PC port overflows (see §2), the switch can also drop PTP packets. That gives missing samples rather than bias, but it can delay lock. Throttling the streams (§2) prevents it.
  * **Camera-internal transmit queueing.** PTP packets share the camera's transmit MAC with its own image stream. GigE Vision cameras normally timestamp PTP in hardware at the MAC or PHY, which makes this harmless. LUCID does not document this. The measured µs-level results above suggest it is handled.
* **Expectation:**
  * **Clock offset:** about 0.1–2 µs at steady state.
  * **Frame-start alignment with an action command:** about 10–30 µs, including LUCID's roughly 20 µs Helios vs Triton timestamp-point difference.
  * **Lock:** master and slave negotiation takes a few seconds. LUCID's sample polls PtpStatus once a second until it reads Master or Slave [src 5]. Let the servo settle for about 10–60 s before recording [est, consistent with src 8a].
  * **Verify** by comparing chunk timestamps of action-triggered frames from both cameras. The constant ~20 µs offset is expected.

### PoE injector

A midspan PoE injector is a passive layer-1 device: it adds DC on the pairs and does no buffering. Its added delay is cable propagation only, about 5 ns/m, which is negligible and symmetric [est]. Two real risks:

1. The Helios2+ **needs PoE+ (802.3at)**. LUCID: "requires a IEEE 802.3at (PoE+) compliant interface card or injector" [src 2]. The Triton runs on 802.3af [src 1].
2. Some cheap injectors are **10/100 only**. Check that DeviceLinkSpeed reads **125,000,000 B/s** on both cameras [src 12].

### Drift without PTP

IEEE 802.3 nodes need a reference clock within **±100 ppm** [src 11]. That allows up to 200 ppm between two devices in the worst case. Typical crystals sit at 10–50 ppm [est].

| Relative error | Drift per second | Drift per 5 min |
|---|---|---|
| 10 ppm | 10 µs | 3 ms |
| 20 ppm | 20 µs | 6 ms |
| 50 ppm | 50 µs | 15 ms |
| 100 ppm | 100 µs | 30 ms |

All rows are [calc]. Two free-running cameras at a nominal 10 fps also have an arbitrary starting phase of up to ±50 ms that walks slowly over time. **Use PTP.**

### Which PTP mode

* **Scheduled action commands.** The Helios is master and the Triton is slave. Both cameras use TriggerSource = Action0 and the same action time [src 5,6]. This keeps your TriggerDelay control. **Recommended.**
* **LUCID PTPSync.** Firmware schedules action commands itself: "4 advance action commands" [src 10b]. You must set PTPSyncFrameRate lower than AcquisitionFrameRate [src 10b], and you lose triggering. You can shift one camera with **PTPSyncOffset in ns**, which requires Helios2+ (HTP) firmware ≥ 1.16.0.0 [src 5b]. This is also a valid choice: set PTPSyncOffset on the Triton to the delay in §4.
* **Hardware trigger cable.** No benefit here, because the ToF frame itself spans about 19 ms (§4).

## 2. Bandwidth

### Inputs and assumptions [calc]

* Triton frame: 2048x1536x1 B = **3,145,728 B**.
* Helios frame: 640x480x8 B = **2,457,600 B**.
* Per-packet overhead: IP 20 + UDP 8 + GVSP 20 B (extended-ID mode, which LUCID uses [src 12b]). Ethernet adds 38 B per packet (header 14, FCS 4, preamble 8, inter-frame gap 12).
* With packet size 9000, each packet carries 8952 B of payload and the stream is **98.9 %** efficient.
* With packet size 1500, each packet carries 1452 B of payload and the stream is **94.4 %** efficient.
* Line rate is 125,000,000 B/s [src 12]. Maximum payload is therefore **123.6 MB/s with jumbo frames** and **118.0 MB/s with 1500-byte packets**.
* With LUCID's default 10 % DeviceLinkThroughputReserve held back for resends [src 12b], about **111 MB/s** is safely usable.

### Rates [calc]

| fps | Triton MB/s | Helios ABCY16 MB/s | Sum payload | Wire, jumbo 9000 | Wire, 1500 | % of 125 MB/s (jumbo / 1500) | Verdict |
|---|---|---|---|---|---|---|---|
| 10 | 31.5 | 24.6 | **56.0** | 56.7 | 59.4 | 45 % / 47 % | Comfortable |
| 15 | 47.2 | 36.9 | **84.0** | 85.0 | 89.1 | 68 % / 71 % | **OK, the recommended maximum** |
| 20 | 62.9 | 49.2 | **112.1** | 113.3 | 118.7 | 91 % / 95 % | Marginal, eats the resend reserve, expect drops |
| 25 | 78.6 | 61.4 | 140.1 | 141.7 | 148.4 | 113 % / 119 % | Impossible |
| 30 | 94.4 | 73.7 | 168.1 | 170.0 | 178.1 | 136 % / 142 % | Impossible |

**Pair total in MB/s for other Helios pixel formats, with Triton added [calc]:**

| Helios format | 10 fps | 15 fps | 20 fps | 25 fps | 30 fps |
|---|---|---|---|---|---|
| C16 (2 B/px) | 37.6 | 56.4 | 75.2 | 94.0 | 112.8 |
| CY16 (4 B/px) | 43.7 | 65.6 | 87.5 | 109.4 | — |
| ABC16 (6 B/px) | 49.9 | 74.8 | 99.8 | — | — |

C16 and CY16 are HTP-only formats [src 3]. LUCID provides intrinsics, so you can recompute XYZ on the host [src 3].

**Helios2+ maximum fps in the 5000 mm mode** is **30 fps** in the normal-mode table, using 120 + 90 MHz [src 2]. The datasheet also lists 30 fps as the normal-mode figure [src 3]. HDR modes run slower (10 fps or less), and high-speed modes cannot use 5000 mm. The Helios is therefore **not** the 15 fps limiter. The shared 1 GbE link is.

### Simultaneous bursts and switch buffers

* Typical unmanaged gigabit switch buffers are **shared** across all ports:
  * Netgear GS105: **128 KB**
  * Netgear GS108v4: **192 KB**
  * Netgear GS116: 256 KB [src 9]
  * TP-Link TL-SG108: **1.5 Mbit = 192 KB** [src 9b]
* An unthrottled LUCID camera sends a whole frame at line rate. That takes **25.5 ms for a Triton frame and 19.9 ms for a Helios frame** [calc].
* If both start together, the PC-facing port receives 2 Gb/s and drains 1 Gb/s. The excess of 125 MB/s fills **128 KB in 1.0 ms**, **192 KB in 1.5 ms** and 1 MB in 8 ms [calc].
* **Result: guaranteed packet loss**, followed by resend storms. Balluff and IDS describe the same mechanism and fix it with inter-packet delay or a throughput limit [src 14].
* **So a throughput limit or packet delay is required**, even at 10 fps.
* With a per-camera limit, each camera sends one packet and then waits. The egress queue then never holds more than a few packets (about 9–27 KB), which fits easily in a 128 KB buffer [calc].

### Recommended per-camera settings [calc]

Set **DeviceLinkThroughputLimitMode = On** on both cameras. The firmware then derives GevSCPD from the limit [src 14].

| Camera | DeviceLinkThroughputLimit | Transfer time per frame | With 10 % reserve on top | Fits 15 fps (66.7 ms)? | Fits 10 fps (100 ms)? |
|---|---|---|---|---|---|
| Triton TRI032S-C | **60,000,000 B/s** | 52.4 ms | 58.3 ms | yes | yes |
| Helios2+ ABCY16 | **48,000,000 B/s** | 51.2 ms | 56.9 ms | yes | yes |
| Sum | 108 MB/s, about 87 % of wire | | | | |

* **10-fps-only, gentler alternative:** Triton **40,000,000** and Helios **32,000,000** B/s, 72 MB/s in total. Transfer times are 78.6 ms and 76.8 ms, which is under 100 ms and leaves about 40 % of the link for resends.
* After setting a limit, read back the maximum AcquisitionFrameRate (or LinkAcquisitionFrameRateLimit) and confirm it is at or above your target. LUCID's interaction between Limit and Reserve is not documented.
* **Packet-delay alternative** from LUCID's bandwidth-sharing formula with 2 cameras and 9014-byte packets [src 10]:
  * GevSCPD = 80,000 ns on both cameras.
  * GevSCFTD = 0 ns on camera 1 and 80,000 ns on camera 2.
  * Each camera then gets about 59 MB/s [calc]. That is nearly the same as the limits above.
  * Use **one** mechanism, not both stacked.

### Host-side settings

* **NIC:** Jumbo Packet = **9014 B**, Receive Buffers = **4096** (Windows advanced adapter properties) [src 12b]. LUCID also lists Interrupt Moderation Enabled/Adaptive and Energy-Efficient Ethernet Disabled [src 12c]. On Linux use `ip link set <if> mtu 9000` and `ethtool -G <if> rx 4096`, and raise `net.core.rmem_max` and `rmem_default` to tens of MB [est, standard GigE Vision practice].
* **Switch:** must support 9 K jumbo frames. GS108 and TL-SG108 do [src 9,9b].
* **Arena stream node map** (`device.tl_stream_nodemap`):
  * `StreamAutoNegotiatePacketSize = True`. Packet size should then read 9000 [src 12].
  * `StreamPacketResendEnable = True` [src 12c].
  * `StreamBufferHandlingMode = OldestFirst`, so no frame is silently replaced.
* **Resends:** keep DeviceLinkThroughputReserve at **10 %**, not 0 % (0 % disables resends [src 12]). If incomplete frames appear, raise PacketResendWindowFrameCount from 4 to **8** [src 12b].
* **Buffers:** `device.start_stream(100)`, which LUCID recommends against timeouts [src 12b]. 100 buffers use about 315 MB for Triton plus 246 MB for Helios, which 8 GB RAM easily holds [calc].
* **Triton frame buffer:** 128 MB [src 1] absorbs about 40 frames, so throttled transfer and resends will not overflow the camera.

## 3. Disk and CPU (dual-core i5-7300U, SATA SSD, Python 3.12)

### Data rate and volume [calc]

| fps | MB/s (pair, raw) | GB per hour | 1 TB lasts |
|---|---|---|---|
| 10 | 56.0 | 202 | about 5.0 h |
| 15 | 84.0 | 302 | about 3.3 h |
| 20 | 112.1 | 404 | about 2.5 h |

* A SATA SSD at 400–530 MB/s handles these rates **only while its write cache lasts**.
* Example: the **Samsung 870 QVO 1 TB** (QLC) is rated 530 MB/s while writes hit its cache. Its SLC cache holds **up to 42 GB**, after which it sustains only **80 MB/s** [src 13].
  * At 15 fps (84 MB/s) the cache fills after about 8 min. **Recording then falls behind.**
  * At 10 fps (56 MB/s) the drive keeps up indefinitely.
* **Action:** identify the SSD model in the UNO-2484G (it takes 2.5" SATA and mSATA [src 15]). A TLC drive with DRAM is strongly preferred for 15 fps.

### Per-frame cost [bench on Xeon @2.1 GHz, single thread]

For the i5-7300U (Kaby Lake, 2.6 GHz base and 3.5 GHz turbo, fanless chassis) assume about 1–1.5x these times [est]. Sizes use synthetic textured images with sensor noise, so real images may compress somewhat better.

| Operation | Triton 2048x1536 u8 (3.15 MB) | Helios 640x480x4 u16 (2.46 MB) |
|---|---|---|
| copy (`tobytes`) | 0.28 ms | 0.32 ms |
| `np.save` to page cache | 2.4 ms | 2.1 ms |
| `ndarray.tofile` (raw .bin) | 2.1 ms | 2.8 ms |
| `cv2.imencode` PNG level 0 (no compression) | 36 ms, ratio 1.00 | 23 ms, ratio 1.00 |
| PNG level 1 (OpenCV default [src 16]) | **116 ms**, ratio 1.31 | **76 ms**, ratio 1.64 |
| PNG level 3 | 126 ms, ratio 1.31 | 111 ms, ratio 1.69 |
| PNG level 6 | 173 ms, ratio 1.29 | 291 ms, ratio 1.74 |
| PNG level 9 | 164 ms, ratio 1.29 | 545 ms, ratio 1.75 |

### CPU budget per second of recording [calc]

* **Raw writes:** about 5 ms per pair. That is about **5 % of one core at 10 fps and 8 % at 15 fps**, and the job is SSD-bound.
* **PNG level 1:** about 192 ms per pair. That is **1.9 cores at 10 fps and 2.9 cores at 15 fps**.
  * This is **impossible** on a 2-core / 4-thread i5 that must also handle about 84 MB/s of GigE packet reception (Arena receive threads and NIC interrupts).
  * It only saves 25–40 % of disk space.
* **PNG level 0:** costs 0.6–0.9 core for zero compression. Pointless.

### GIL

* `ndarray.tofile` writes directly via `fileno()`, equivalent to `file.write(a.tobytes())` [src 17]. Python file writes release the GIL during the syscall.
* I confirmed that `cv2.imencode` also releases the GIL [bench]. A pure-Python thread kept 90 % of its speed during encoding, and two encode threads gave a 1.5x speedup.
* Background threads therefore work for both. The CPU-time problem above still makes PNG unusable live.

### Recommended writer design

1. **One acquisition thread per camera.**
   * Call `buf = device.get_buffer(timeout)`.
   * Immediately copy the data: `np.ctypeslib.as_array(buf.pdata, ...)` followed by `.copy()`, or `BufferFactory.copy`. This costs about 0.3 ms.
   * Read `buf.timestamp_ns` and `frame_id`.
   * Call `device.requeue_buffer(buf)` right away.
   * Push `(ts, id, ndarray)` to a **bounded `queue.Queue(maxsize≈60)`**, about 2–4 s of frames, roughly 340 MB at most.
2. **One writer thread per camera.**
   * Append raw bytes to one large pre-opened file per camera per session (`f.write(arr)` or `arr.tofile(f)`).
   * Log `(frame_id, ts, offset)` to a small CSV or index file.
   * One big file avoids per-file create and metadata cost. On Windows it also avoids a Defender scan per file; add an AV exclusion for the folder anyway.
   * Optionally use `os.posix_fadvise` or `f.flush()` every N frames.
3. **Monitoring.** If `queue.full()`, count and log drops; never block the acquisition thread. Also monitor queue depth.
4. **Do PNG or other compression offline,** or in a low-priority process after capture. If storage is tight, the cheapest real saving is the **Helios pixel-format change** (ABCY16 to C16 or CY16 cuts Helios bytes 2–4x at zero CPU cost). Fast lz4 or zstd-1 compressors are an untested middle ground [est].

## 4. Helios2/2+ ToF timing and Triton alignment

### Sourced facts

* From LUCID's multi-Helios knowledge base, which covers the normal (non-high-speed) modes at 1000 µs exposure [src 5c]:
  * Integration is **0.98 ms** for a 1000 µs ExposureTime.
  * Readout is **1.55 ms**, "always constant independent of operating mode on all Helios 2 variants".
  * Reset is negligible.
  * The sequence repeats 4 times, so **T = (0.98 + 1.55) x 4 = 10.12 ms** in the single-frequency example.
* LUCID's HDR table counts "Depth Frames (x2 in multi-freq)" [src 2]. This confirms that multi-frequency modes double the number of phase captures.
* Basler's blaze, which uses the same IMX556 sensor, states "each frame is composed of eight subframes (4 subframes if Fast Mode is enabled)". The illumination is switched off and the sensor is reset after each one [src 8b].
* Exposure presets are 62.5, 250 and 1000 µs [src 2,3].

### Derived window for Distance5000mmMultiFreq [calc, 8 sub-frames]

| ExposureTimeSelector | Sub-frame (integration + readout) | Total frame (8x) | Light-collection span | Centre of light |
|---|---|---|---|---|
| Exp1000Us | 0.98 + 1.55 = 2.53 ms | **20.24 ms** | 0 → **18.69 ms** | **≈ 9.35 ms** |
| Exp250Us | about 0.245 + 1.55 = 1.80 ms | about 14.4 ms | 0 → about 12.8 ms | ≈ 6.4 ms |
| Exp62_5Us | about 0.061 + 1.55 = 1.61 ms | about 12.9 ms | 0 → about 11.3 ms | ≈ 5.7 ms |

The 250 µs and 62.5 µs integration times assume the same 98 % ratio [est]. At 30 fps the 20.24 ms frame fits inside 33.3 ms, which is consistent with the 30 fps datasheet figure.

### Where the timestamp lands [inference]

* In LUCID's action-command test with 500 µs exposure, timestamps fall **25–34 µs after the scheduled trigger time** [src 6]. That is far less than the exposure length, so the LUCID timestamp marks **frame or exposure start**, not end.
* The Helios timestamp is about 20 µs off the Triton's [src 5]. It therefore also sits at the **start of the ToF sequence**, at the first sub-frame, not mid-frame or end of frame.
* LUCID does not document this explicitly. **Verify** once by routing ExposureActive (or a strobe) from both cameras to GPIO outputs and measuring on an oscilloscope. Repeat if firmware changes.

### Timing diagram (Exp1000Us, 5000 mm multi-frequency, both cameras on the same action time t0) [calc]

```
t (ms) 0    2.5   5.1   7.6  10.1  12.7  15.2  17.7  20.2
Helios [I]rr[I]rr[I]rr[I]rr[I]rr[I]rr[I]rr[I]rr|   I = 0.98 ms laser integration
        f1φ0  f1φ1  f1φ2  f1φ3  f2φ0  f2φ1  f2φ2  f2φ3    r = 1.55 ms readout
       ^ts (≈t0+~25µs)          ^ centre of light ≈ 9.35 ms        light ends ≈ 18.7 ms
Triton           |<---- E ---->|                         (centred on 9.35 ms)
                 ^ t0 + TriggerDelay,  TriggerDelay = 9350 µs − E/2
```

The phase and frequency order inside the Helios frame is illustrative. LUCID does not publish it.

### Recommended Triton TriggerDelay [calc]

| Triton exposure E | TriggerDelay |
|---|---|
| 2 ms | 8350 µs |
| 5 ms | **6850 µs** |
| 8 ms | **5350 µs** |
| 10 ms | 4350 µs |
| 18.7 ms (matches the whole ToF span) | 0 µs |
| 20 ms | −650 µs. Delay the Helios by 650 µs instead. |

* If you use PTPSync instead of action commands, apply the same value as PTPSyncOffset in ns on the Triton.
* Correct for the measured ~20 µs timestamp offset if you want, though it is irrelevant at this scale.

### Internal skew and motion of the ToF frame [calc]

* The depth frame integrates over **about 18.7 ms**. Each phase pair is 2.5 ms apart, and the two frequency sets are about 10 ms apart.
* Hand-held motion in that window:

  | Motion | Over the ToF window |
  |---|---|
  | Translation at 0.5 m/s | 9 mm |
  | Translation at 1 m/s | 19 mm |
  | Rotation at 10 °/s | 0.19°, about **1.8 Helios pixels** (IFOV = 67°/640 = 0.105°/px [src 2]) |
  | Rotation at 30 °/s | 0.56°, about **5.3 Helios pixels** |

* With an iToF sensor this does not show up as blur. It shows up as **depth errors and flying pixels at edges**, because the phase images are misaligned.
* **Conclusion:** camera-to-camera sync better than about 1 ms is irrelevant next to a 19 ms ToF acquisition. PTP gives µs-level sync, which is ample. Centring the Triton exposure on the ToF window (above) matters more than the sync method.
* To shrink the ToF window, use Exp250Us. That gives about a 12.8 ms span, but at a cost in range and precision. LUCID's 5000 mm precision figures used 1000 µs beyond 0.5 m [src 2].

## 5. Triton exposure

* **Range:** 30.25 µs to 10 s (normal mode) [src 1].
* **Maximum at a given fps:** LUCID says exposure is capped by frame rate, for example 1/30 s = 33.3 ms at 30 fps [src 18]. So the cap is about **100 ms at 10 fps and about 66 ms at 15 fps** in free-run with overlapped exposure and readout.
* **Conservative triggered case (no overlap):** the IMX265 sensor maximum is 55.6 fps, so readout is about 18 ms [src 19 for the sensor, calc for readout]. That leaves about **82 ms at 10 fps and about 48 ms at 15 fps** [calc].
* Image transfer does not block the next exposure, thanks to the 128 MB on-camera buffer [src 1].
* **ExposureAutoUpperLimit = 20,000 µs at 10 fps (or 15 fps) is therefore no timing problem.**

### The real issue is motion blur [calc]

* Blur in pixels ≈ ω·E / IFOV.
* With an 8 mm lens on 3.45 µm pixels, IFOV = 0.43 mrad, or 0.0247°/px. This lens is an assumption; scale by focal length.

| Hand-held rotation | E = 20 ms | E = 8 ms | E = 5 ms |
|---|---|---|---|
| 5 °/s | 4 px | 1.6 px | 1 px |
| 10 °/s | **8 px** | 3.2 px | **2 px** |
| 30 °/s | 24 px | 10 px | 6 px |

### Recommendation [est]

* **Hand-held under forest canopy:** ExposureAutoUpperLimit = **8000 µs**, or **5000 µs** if mostly moving. Let GainAuto rise to about 18–24 dB (the IMX265 handles gain well), with TargetBrightness slightly lowered.
* **Stationary or tripod segments:** allow up to 20 ms.
* **Keep the Triton exposure centred on the Helios window** using the TriggerDelay table in §4.
* Because auto-exposure changes E, either fix E per session or recompute TriggerDelay. A 2–3 ms mis-centre is harmless relative to the 18.7 ms ToF span.

---

## Sources

1. LUCID Triton 3.2 MP (TRI032S) product page (IMX265, global shutter, 35.4 fps / 38 fps, exposure 30.25 µs–10 s, 128 MB buffer, PoE 802.3af, PTP): https://thinklucid.com/de/?p=64131 . Reseller confirmation: https://www.edmundoptics.com/p/lucid-vision-labs-tritontrade-tri032s-cc-sony-imx265-32mp-color-camera/41819/
2. LUCID Helios2+ product page (IMX556PLR, normal-mode table: 5000 mm = 120 + 90 MHz, 30 fps; HDR table "x2 in multi-freq"; exposure 62.5/250/1000 µs; PoE+ requirement; 67°x51°): https://thinklucid.com/?p=87923
3. LUCID Helios2 Family datasheet PDF (HTP003S-001: 30 fps Normal, 10 HDR, 103 High-Speed; pixel formats; intrinsics available): https://www.1stvision.com/cameras/Lucid-Vision-Labs/dataman/Helios2-Family-datasheet.pdf
4. LUCID KB "Precision Time Protocol (PTP)" (E2E only, BMC, switch, bandwidth sharing, PTP Sync disables triggering): https://support.thinklucid.com/?p=8070
5. LUCID app note "Helios2 and Triton Synchronization" (~20 µs timestamp difference, PTP assumes a switch, Helios as master, Action0, PTPSync steps): https://support.thinklucid.com/app-note-helios2-and-triton-sync/
   - 5b/5c. LUCID KB "Using Multiple Helios Cameras Simultaneously" (0.98 ms integration, 1.55 ms readout, (0.98+1.55)x4 = 10.12 ms, PTPSyncOffset in ns, HTP firmware ≥ 1.16.0.0): https://support.thinklucid.com/knowledgebase/using-multiple-helios-2-cameras-simultaneously/
6. LUCID app note "Multi-Camera Synchronization using PTP and Scheduled Action Commands" (7-camera timestamps 2074000024760–2074000033912 ns, ExposureTime 500): https://support.thinklucid.com/?p=4184
7. Allied Vision "IEEE 1588 PTP Application Note" (within 2 µs; 69 h test < ±1 µs; scope ~125 ns; 125 MB per GigE link): https://ftp.stemmer-imaging.com/webdavs/docmanager/66823-Allied-Vision---PTP-IEEE1588-Application-Note.pdf
8. Basler:
   - 8a. Precision Time Protocol (precision depends on network hardware; few seconds to minutes to sync; offset in ns ticks): https://docs.baslerweb.com/precision-time-protocol
   - 8b. blaze Synchronous Free Run (8 sub-frames, 4 in Fast Mode; 250 µs jitter safety delay): https://docs.baslerweb.com/synchronous-free-run-(blaze)
9. Netgear GS105/GS108/GS116 datasheet (128/192/256 KB buffer): https://www.alldatasheet.net/html-pdf/2162255/NETGEAR/GS108/1034/3/GS108.html . Product page: https://www.netgear.com/in/business/wired/switches/unmanaged/gs108/
   - 9b. TP-Link TL-SG108 spec listing (1.5 Mbit buffer, 16 KB jumbo): https://powertec.com.au/tp-link-TL-SG108-8-port-10-100-1000mbps-desktop-switch
10. LUCID app note "Bandwidth Sharing in Multi-Camera Systems" (GevSCPD/GevSCFTD formula, 9014 B packet = 72,112 ns, 80,000 ns packet delay example):
    - 10b. Same note: "PTPSync ... schedules 4 advance action commands", PTPSyncFrameRate < AcquisitionFrameRate.
    - https://support.thinklucid.com/app-note-bandwidth-sharing-in-multi-camera-systems/
11. TI application note SNLA290 (IEEE 802.3 nodes need ±100 ppm reference clock): https://www.ti.com/lit/pdf/snla290 . IEEE 802.3 RTPGE slide (±100 ppm standard for 1000BASE-T): https://www.ieee802.org/3/RTPGE/public/july12/zinner_02_0712.pdf
12. LUCID KB "Tips for reaching maximum frame rate" (DeviceLinkSpeed 125,000,000 Bps; StreamAutoNegotiatePacketSize; reserve 0 % disables resend): https://support.thinklucid.com/knowledgebase/tips-for-reaching-maximum-frame-rate/
    - 12b. LUCID KB "Troubleshooting timeouts caused due to incomplete images (Windows)" (Jumbo 9014, Receive Buffers 4096, reserve default 10 %, PacketResendWindowFrameCount 4 → 8, StartStream(100)): https://support.thinklucid.com/knowledgebase/troubleshooting-timeouts-caused-due-to-incomplete-images-windows/
    - 12c. LUCID KB "Testing multiple camera system performance" (PacketSize 9000, GevSCPD 80, StreamPacketResendEnable True, ExtendedID On, NIC settings): https://support.thinklucid.com/knowledgebase/testing-multiple-camera-system-performance/
13. Samsung 870 QVO datasheet Rev 1.1 (530 MB/s burst; after TurboWrite 80 MB/s for 1 TB; cache up to 42 GB): https://download.semiconductor.samsung.com/resources/data-sheet/Samsung_SSD_870_QVO_Data_Sheet_Rev1.1_10129514052310.pdf
14. Balluff GigE bandwidth quickstart (switch buffer overrun with simultaneous senders): https://assets.balluff.com/documents/DRF_957327_AA_000/mvBC_subsection_quickstart_bandwidth.html . IDS DeviceLinkThroughputLimit (firmware derives GevSCPD; sum ≤ adapter capacity): https://www.1stvision.com/cameras/IDS/IDS-manuals/en/device-link-throughput-limit.html
15. Advantech UNO-2484G (i5-7300U option, 8 GB, 4x GbE): https://buy.advantech.com/Compact-Computers/Embedded-Computers-Automation-Series-UNO-i-Series-Processor/UNO-2484G-9S52/system-21936.htm
16. OpenCV imgcodecs flags (IMWRITE_PNG_COMPRESSION 0–9, default 1 = best speed; the page blocked automated fetch, so this is from documentation knowledge): https://docs.opencv.org/4.x/d8/d6a/group__imgcodecs__flags.html
17. NumPy `ndarray.tofile` (raw binary, equivalent to `file.write(a.tobytes())`, writes via `fileno`): https://numpy.org/doc/stable/reference/generated/numpy.ndarray.tofile.html
18. LUCID KB "How to increase maximum allowable exposure time" (exposure capped by frame rate, 1/30 s example): https://support.thinklucid.com/knowledgebase/how-to-increase-maximum-allowable-exposure-time/
19. FRAMOS IMX265 sensor page (55.6 fps, global shutter, 3.45 µm): https://framos.com/products/sensors/area-sensors/imx265lqr-c-18256/

Benchmark scripts:
- /tmp/claude-0/-home-user-DSA-in-cpp/b1250959-7d60-5709-8e98-b6e6b4d860f9/scratchpad/bench_png.py
- /tmp/claude-0/-home-user-DSA-in-cpp/b1250959-7d60-5709-8e98-b6e6b4d860f9/scratchpad/bench_gil.py
- /tmp/claude-0/-home-user-DSA-in-cpp/b1250959-7d60-5709-8e98-b6e6b4d860f9/scratchpad/bw.py

## Open items to verify on the real hardware

1. Helios and Triton timestamp points, and the true 8-sub-frame ToF window: measure with an oscilloscope on GPIO ExposureActive or strobe outputs.
2. How LUCID combines DeviceLinkThroughputLimit with DeviceLinkThroughputReserve: read back the maximum frame rate after setting.
3. The SSD model in the UNO-2484G and its sustained write rate after cache.
4. The PoE injector really links at 1 Gb/s: DeviceLinkSpeed should read 125,000,000.
