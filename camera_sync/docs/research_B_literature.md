# Researcher B: Literature on depth/color camera synchronization

Scope: how published RGB-D and multi-sensor rigs synchronize their cameras, the sync error each method class achieves, how sync error turns into misregistration, what counts as acceptable, how to validate it, and the pitfalls. Every number has a source in Section 6 [n]. Numbers marked **(derived)** are my own calculations from the cited inputs.

## 1. Sync methods and typical error

| Method class | Example rigs / datasets | Typical sync error (pair offset) | Source |
|---|---|---|---|
| (a) Free-run on independent clocks + nearest-timestamp matching | **TUM RGB-D** (Kinect v1, handheld): the docs say color and depth are captured "in an un-synchronized way". `associate.py` pairs frames by timestamp with `max_difference` = 0.02 s. **OpenLORIS-Scene**: data from different devices are aligned in software by minimizing ATE; the standard deviation of the fitted offsets is 1.7 to 7.4 ms. **RealSense, no HW sync**: frames from different cameras "never misalign by more than 1 frame". | Up to half a frame period after correct matching (**our rig at 10 fps: ≤ 50 ms, mean ≈ 25 ms (derived)**). Up to a full period (100 ms) with "latest-frame" pairing. Crystal drift makes the phase walk: "a few ms over 10s of minutes" [4]; 58 ms/h ≈ 16 ppm [6]. Timestamps from two clocks are not comparable without an offset estimate. | [1][2][4][6][8] |
| (b) Software trigger from host (one command per camera) | Generic GigE Vision | Host-to-camera trigger latency 100 to 500 µs, depending on host software and camera [10]. Two commands sent one after the other, through a non-RT OS, give skew of roughly 0.1 to a few ms **(derived)**. Allied Vision found a host-software PTP master on Windows/Linux gave "very poor synchronization plagued by significant jitter" [11]. | [10][11] |
| (c) PTP (IEEE 1588) clock sync, free-run or PTP-aligned acquisition | **Newer College** (Ouster↔host PTP, "sub-microsecond"). **Hilti 2021** (PTP between modules, observed < 1 ms via gyro correlation). **LUCID "PTP Sync" mode** (frame-rate alignment). | PTP-aware switches (transparent or boundary clock) and HW timestamping reach ~100 ns or better [12][13]. On a plain switch, accuracy is limited by packet-delay variation: a PTP packet queued behind one 1500 B frame waits 12 µs, behind a 9000 B jumbo frame 72 µs at 1 GbE **(derived from [12])**. Expect roughly 1 to tens of µs on a lightly loaded single switch, worse under image bursts **(estimate; measure on the rig)**. | [6][7][12][13] |
| (d) PTP + scheduled action commands | **LUCID Helios2 + Triton app note**: Helios is PTP master, an action command is scheduled ~1 s ahead. **LUCID 7-camera test.** **Allied Vision PTP note.** | LUCID Helios2↔Triton: "around 20 microseconds" timestamp offset, which LUCID attributes to the two firmwares latching at slightly different points [14]. 7 PHX016S cameras: 9.2 µs spread of frame timestamps **(derived from the logged values)** [15]. Allied Vision: slaves within ±1 µs of the master over 69 h, 125 ns on an oscilloscope; scheduled triggers within 2 µs [11]. | [11][14][15] |
| (e) Hardware trigger line (or master strobe → slave trigger) | **Azure Kinect** (internal depth↔color plus a 3.5 mm sync cable between devices). **RealSense D4xx** (sync pin 5). **ETH3D SLAM / BAD SLAM** (synchronized global-shutter rig). **BotanicGarden** (MCU triggers cameras, LiDAR PPS). **Hilti-Oxford** (FPGA-synced cameras and IMU). | Azure Kinect: color images across devices aligned "within 100 µs of the center of exposure". Depth vs color default is center-of-exposure alignment, adjustable with `depth_delay_off_color_usec`. BotanicGarden: "sub-µs level triggering consistency", checked on a scope. ETH3D: color and depth "recorded at exactly the same points in time". Hilti: camera↔IMU < 1 ms, timestamps at mid-exposure. | [3][3b][4][5][9][7][17] |

Other notes:

- **ToF + RGB with LUCID hardware.** The only primary source I found is LUCID's own app notes. They give three sync options: (1) hardware trigger, or Helios strobe → Triton trigger; (2) PTP + scheduled action commands, about 20 µs offset; (3) PTP Sync mode [14]. Coloring is done by projecting the Helios XYZ points into the Triton image with an OpenCV `solvePnP`/`projectPoints` calibration [14][16]. I found no peer-reviewed paper that reports a measured Helios↔Triton sync error. This is a gap the thesis can fill.
- **Forestry and natural-environment rigs.** BotanicGarden is the best documented: an STM32 sends 1 Hz PPS to the LiDARs and a 40 Hz trigger to the global-shutter cameras. Each image timestamp is the trigger stamp plus half the exposure. A PTP network ties in the other hosts [9]. Other forestry datasets are weaker. Viking Hill uses PTP time-stamping on its Ouster LiDAR but gives no camera-LiDAR sync figure [18].

## 2. Motion error from sync error

**Formulas.** A time offset dt during rotation at rate ω gives Δθ = ω·dt.

- Lateral error at range R: e = R·tan(Δθ) ≈ R·ω·dt.
- Color-image shift near the image center: Δu ≈ f_px·tan(Δθ) ≈ f_px·ω·dt.
- For translation at speed v perpendicular to the view direction: e = v·dt, and Δu ≈ f_px·v·dt / Z. Pure translation does not depend on range in metres.

**Assumptions.**

- Triton TRI032S: Sony IMX265 global shutter, 2048×1536 pixels, 3.45 µm pixels, 7.07×5.3 mm active area [19][20].
- Assumed lens: **6 mm C-mount**. This gives f = 6 / 0.00345 = **1739 px** and HFOV = 2·atan(1024/1739) = **61°**. A 5 mm lens would give f ≈ 1450 px and HFOV ≈ 70°; scale the pixel column by f/1739.
- Helios2+: 640×480 pixels, 67°×51° FOV [21], so f_depth ≈ 320 / tan(33.5°) ≈ **483 px**.

**Handheld angular rates from the literature:**

- TUM RGB-D handheld sequences average 8.9 to 50.1 °/s (fr1/desk 23.3, fr1/room 29.9, fr1/360 41.6, fr1/rpy 50.1) [2].
- Newer College "dynamic spinning" reaches 2.5 rad/s ≈ 143 °/s [6].
- Hilti-Oxford "shaking" reaches up to 3.5 rad/s ≈ 200 °/s [17].
- So 30, 60 and 90 °/s cover calm, normal and brisk handheld turning.

**Table** (rows are rotation only). All rows are **(derived)**.

| dt | ω (°/s) | Δθ (°) | Lateral @ 5 m | Lateral @ 10 m | Shift in Triton px (f=1739) | Shift in Helios px (f=483) |
|---|---|---|---|---|---|---|
| 100 ms | 30 / 60 / 90 | 3.0 / 6.0 / 9.0 | 262 / 526 / 792 mm | 524 / 1051 / 1584 mm | 91 / 183 / 276 | 25 / 51 / 77 |
| 50 ms | 30 / 60 / 90 | 1.5 / 3.0 / 4.5 | 131 / 262 / 394 mm | 262 / 524 / 787 mm | 46 / 91 / 137 | 13 / 25 / 38 |
| 10 ms | 30 / 60 / 90 | 0.3 / 0.6 / 0.9 | 26 / 52 / 79 mm | 52 / 105 / 157 mm | 9 / 18 / 27 | 2.5 / 5.1 / 7.6 |
| 1 ms | 30 / 60 / 90 | 0.03 / 0.06 / 0.09 | 2.6 / 5.2 / 7.9 mm | 5.2 / 10.5 / 15.7 mm | 0.9 / 1.8 / 2.7 | 0.25 / 0.5 / 0.76 |
| 0.1 ms | 30 / 60 / 90 | 0.003 / 0.006 / 0.009 | 0.3 / 0.5 / 0.8 mm | 0.5 / 1.0 / 1.6 mm | 0.1 / 0.2 / 0.3 | 0.03 / 0.05 / 0.08 |

**Translation.** Walking at 1 m/s with dt = 50 ms gives a 50 mm offset, which is 17 Triton pixels at Z = 5 m. With dt = 1 ms it gives 1 mm, or 0.35 px **(derived)**. Rotation dominates for a handheld rig.

**Reading the table.** Today's free-run pairing (≤ 50 to 100 ms) misplaces color by 13 to 79 cm at 5 m. That is larger than most stem diameters, which explains the doubled edges.

The tolerance for 1 Triton pixel is dt < 1/(f·ω): 1.1 ms at 30 °/s and 0.37 ms at 90 °/s. For 1 Helios pixel, the size of one colored point's footprint, it is 4.0 ms and 1.3 ms **(derived)**. So **≤ 1 ms makes sync error sub-depth-pixel, and ≤ 0.1 ms makes it negligible**. Note that this affects only colorization: DBH and positions come from depth alone, unless color drives segmentation.

## 3. What reviewers accept and how to report it

**No universal threshold.** I found no standard "acceptable sync" number. Published practice falls into three tiers:

1. **Datasets that claim "synchronized"** use hardware triggering or PTP and report µs-level figures:
   - Azure Kinect: 100 µs [3]
   - Allied Vision: < 1 µs [11]
   - BotanicGarden: sub-µs [9]
   - LUCID: about 20 µs [14]
2. **"< 1 ms" is the common claim for cross-module alignment.** Hilti 2021 [7] and the Hilti camera↔IMU alignment both use it.
3. **Software alignment at ms level is presented as a limitation.** OpenLORIS calls 1.7 to 7.4 ms "acceptable for our scenarios, yet better synchronization methods can be discussed" [8].

BAD SLAM showed that worst-case asynchronous RGB-D pairs (color halfway between depth frames) "significantly degrade" results. They argue the problem is "better addressed in hardware" [5].

**Justified criterion for the thesis.** I recommend two criteria together:

- **Motion criterion:** require ω_max·dt·f < 1 pixel of the coarser sensor (Helios). This gives roughly 1 ms at 90 °/s (Section 2).
- **Exposure criterion:** require sync error to be much smaller than the shorter of the Triton exposure and the ToF integration window (~10 ms, see Section 5). Once that holds, intra-frame blur dominates, not sync.

**How good papers report sync:**

- **Statistics on a common clock.** Give the distribution of per-pair offsets |t_depth − t_rgb|, where both timestamps are mid-exposure and in the PTP domain. Report mean, SD, median, 95th percentile and max, the number of pairs N, and the duration. Allied Vision gives a histogram over 69 h, max < ±1 µs [11].
- **Clock-sync health.** Log PTP offset-from-master over time and report drift. Newer College reports 58 ms/h between unsynced clocks [6].
- **Independent physical validation.** The timestamps can only show that the clocks agree. A physical test shows that the exposures agree:
  - oscilloscope on the strobe outputs: 125 ns [11]
  - high-speed LED panel at "several hundred Hz" with exposure < 3 ms [4]
  - flash + rolling-shutter line timing: ≤ 250 µs [22]
  - gyro cross-correlation: < 1 ms [7]

## 4. Validation methods for this rig

1. **Electrical, best resolution (~1 µs).**
   - Set each camera's GPIO output to ExposureActive (or a strobe with ExposureStart as source). Check that the Helios2 exposes this on its GPIO.
   - Record both lines on an oscilloscope, logic analyzer, or MCU timer-capture.
   - This measures the true exposure-edge offset and its jitter over thousands of frames (Allied Vision did the same to get 125 ns [11]). It also catches the about 20 µs firmware latch difference [14] and any constant delay.
   - Report the offset between exposure centers. For the Helios, use the center of its multi-phase window.
2. **End-to-end geometric, resolves ~0.1 to 0.5 ms.**
   - Mount the rig on a tripod head or turntable. Rotate it at a known, IMU- or encoder-measured ω, around 60 to 90 °/s, in front of vertical poles or trunks at 3 to 10 m.
   - Measure the horizontal offset Δu between the depth edge (projected into the Triton image) and the color edge. Then dt = Δu / (f·ω).
   - Rotate **clockwise and counter-clockwise**. Sync error flips sign with direction; calibration error does not. So dt = (Δu_cw − Δu_ccw) / (2·f·ω).
   - With 0.5 px edge localization at 90 °/s, resolution is about 0.2 ms **(derived)**. This directly measures the quantity the thesis cares about.
3. **Optical counter (Triton only, or a coarse check).**
   - An MCU-driven LED bar or binary counter stepping every 100 µs to 1 ms, with a short Triton exposure, resolves about one step. RealSense uses an LED panel the same way [4].
   - A phone stopwatch on a screen is limited by the display refresh (8 to 17 ms at 60 to 120 Hz) and is only useful for checking whether frames are 0 or 1 apart. Visible LEDs are likely invisible to or rejected by the Helios, which has an 850 nm VCSEL and active-illumination demodulation. So this method works for Triton↔host or Triton↔reference checks, not for Helios↔Triton.
   - Alternative: a rotating arm or disk seen by both cameras. Compare the arm angle in depth vs color: dt = Δφ / ω_arm. Keep ω_arm low enough that the ~10 ms ToF window does not smear the arm (resolution about 1 ms).

## 5. Pitfalls

- **Two clocks.** Free-running devices have separate tick counters with arbitrary offsets and 10s of ppm drift [4][6]. Subtracting raw timestamps is meaningless unless both are on one PTP clock, and you must check `PtpStatus` = Master/Slave and the offset before recording [11]. LUCID's PTP Sync mode disables triggering [14]. Cameras on separate NICs must be bridged so PTP can see them [14].
- **Host receive time vs exposure time.**
  - A Triton frame takes about 25 ms (Bayer8) or 75 ms (RGB8) to cross 1 GbE. A Helios ABCY16 frame takes about 20 ms. Both share the switch uplink **(derived)**.
  - GigE delivery jitter can reach 2.7 ms worst case even with hardware triggering [10].
  - Never pair frames by host arrival time. Use device timestamps on a common clock, and limit bandwidth so that triggered bursts do not collide.
- **Which instant the timestamp marks.** Start of exposure, mid-exposure and readout differ by vendor. Azure Kinect aligns centers of exposure [3b]. BotanicGarden adds half the exposure to the trigger stamp [9]. Hilti stamps the mid-exposure pulse [7].
  - With a common trigger, Triton exposure E_rgb starts together with Helios integration. Their centers then differ by (T_ToF − E_rgb)/2, about 4 ms for a 10 ms ToF window and a 2 ms RGB exposure **(derived)**.
  - Fix: delay the Triton trigger by that amount, which is what `depth_delay_off_color_usec` does on Azure Kinect [3b].
  - Use **fixed** exposure. Auto-exposure moves the center and breaks timing [3].
- **Shutter type.** The Triton TRI032S IMX265 is global shutter (verified [19][20]). Rolling shutter adds a per-row time offset (Kinect v1: about 26 ms color, 30.5 ms depth [5]). The RealSense D435 RGB is rolling shutter and is not HW-synced to depth [23]. The Helios2 IMX556 is a global ToF pixel array.
- **ToF frame spans a time window.** Helios2 single-frequency mode at 1000 µs runs 4 × (0.98 ms integration + 1.55 ms readout) = **10.12 ms** per depth frame [24]. Multi-frequency modes (5000/8333 mm) need more sub-frames, so the window is longer (estimate about 2×; verify in the Helios2 technical reference). Azure Kinect NFOV is 12.8 ms [3].
  - At 60 °/s, a 10 ms window smears depth edges by about 5 Helios px (derived). The phases see different geometry, which produces flying pixels. Sync better than about 1 ms is therefore below the ToF's own motion floor.
  - Shorter exposure (250 / 62.5 µs) or the Helios2+ high-speed single-phase mode (103 fps) reduces this [21].
- **Network.** A non-PTP switch plus image bursts gives PTP packet-delay variation, and jumbo frames make it worse. Give PTP QoS priority, or use a PTP-aware (transparent-clock) switch [12]. A host-software PTP master is a poor choice; let the Helios or a camera be master, as LUCID's note does [11][14].

## 6. Sources

1. TUM RGB-D tools (associate.py, unsynchronized Kinect, max_difference 0.02 s): https://cvg.cit.tum.de/data/datasets/rgbd-dataset/tools
2. TUM RGB-D sequence statistics (average angular and translational velocity): https://webshare.cvg.cit.tum.de/g/rgbd/dataset/
3. Azure Kinect multi-device sync (100 µs, 160 µs offset, 12.8 ms exposure, manual exposure): https://learn.microsoft.com/en-us/previous-versions/azure/kinect-dk/multi-camera-sync
   - 3b. Azure Kinect capture synchronization (center-of-exposure, depth_delay_off_color_usec): https://learn.microsoft.com/en-us/azure/kinect-dk/capture-device-synchronization
4. Intel RealSense multi-camera white paper (HW sync, ≤ 1 frame unsynced, drift, LED panel validation): https://simplecore.intel.com/realsensehub/wp-content/uploads/sites/63/Multiple_Camera_WhitePaper04.pdf
5. Schöps et al., BAD SLAM / ETH3D SLAM benchmark, CVPR 2019: https://openaccess.thecvf.com/content_CVPR_2019/papers/Schops_BAD_SLAM_Bundle_Adjusted_Direct_RGB-D_SLAM_CVPR_2019_paper.pdf
6. Ramezani et al., Newer College dataset, IROS 2020 (PTP, 58 ms/h drift, 2.5 rad/s): https://arxiv.org/abs/2003.05691
7. Helmberger et al., Hilti SLAM Challenge dataset, RA-L 2022 (PTP, < 1 ms, mid-exposure stamps): https://arxiv.org/abs/2109.11316
8. Shi et al., OpenLORIS-Scene, ICRA 2020 (software sync, offset SD 1.7 to 7.4 ms): https://arxiv.org/abs/1911.05603
9. Liu et al., BotanicGarden dataset, RA-L 2024 (MCU trigger + PPS + PTP, sub-µs, mid-exposure stamps): https://arxiv.org/abs/2306.14137
10. E. Carey (DALSA), "Exposing jitter and latency myths in Camera Link and GigE Vision systems", Vision Systems Design 2011: https://www.vision-systems.com/home/article/16737772/system-performance-exposing-jitter-and-latency-myths-in-camera-link-and-gige-vision-systems
11. Allied Vision, "IEEE 1588 PTP with Prosilica GT/GC, Manta, Mako G" (2 µs, ±1 µs over 69 h, 125 ns scope, host-PTP jitter): https://www.alliedvision.com/assets/support/Camera-Documentation/Allied-Vision/FAQs-Application-Notes/PTP_IEEE1588_with_Prosilica_GT_GC_Manta.pdf
12. Calnex, "Time Transfer in Industrial Ethernet Applications" (packet-delay variation, jumbo frames, PTP-aware switches sub-100 ns): https://calnexsolutions.atlassian.net/wiki/download/attachments/37683369/Time+Transfer+in+Industrial+Ethernet+Applications.pdf?version=1
13. Basler, Precision Time Protocol (precision depends on network; PTP-enabled switches recommended): https://docs.baslerweb.com/precision-time-protocol
14. LUCID, App Note "Helios2 and Triton Sync" (HW trigger / PTP + action / PTP Sync; about 20 µs): https://support.thinklucid.com/app-note-helios2-and-triton-sync/
15. LUCID, "Multi Camera Synchronization using PTP and Scheduled Action Commands": https://thinklucid.com/multi-camera-synchronization-using-ptp-and-scheduled-action-commands/
16. LUCID, App Note "Add RGB Color to Helios 3D Point Cloud": https://support.thinklucid.com/app-note-helios-3d-point-cloud-with-rgb-color/
17. Zhang et al., Hilti-Oxford dataset, RA-L 2022 (HW sync, up to 3.5 rad/s): https://arxiv.org/abs/2208.09825
18. Viking Hill forest dataset, 2026: https://arxiv.org/abs/2606.19154
19. LUCID Triton IMX265 announcement (global shutter): https://www.automate.org/vision/news/lucid-expands-its-gige-vision-triton-camera-series-with-sony-imx265-imx250mzr-and-imx250myr-sensors
20. Edmund Optics TRI032S-CC listing (2048×1536, 3.45 µm, 7.07×5.3 mm): https://www.edmundoptics.com/p/lucid-vision-labs-tritontrade-tri032s-cc-sony-imx265-32mp-color-camera/41819/
21. Helios2 family datasheet (67°×51°, exposures, 30/103 fps): https://www.1stvision.com/cameras/Lucid-Vision-Labs/dataman/Helios2-Family-datasheet.pdf
22. Akhmetyanov et al., "Sub-millisecond Video Synchronization of Multiple Android Smartphones" (≤ 250 µs, flash validation): https://arxiv.org/abs/2107.00987
23. librealsense issue on D435/D455 RGB not HW-synced to depth: https://github.com/IntelRealSense/librealsense/issues/11669
24. LUCID KB, "Using multiple Helios2 cameras simultaneously" (4 × (0.98 + 1.55) = 10.12 ms): https://support.thinklucid.com/knowledgebase/using-multiple-helios-2-cameras-simultaneously/
