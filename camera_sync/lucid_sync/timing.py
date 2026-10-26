"""Timing model of one Helios2/2+ depth frame - an UNVERIFIED model, used only for the
"modelled" exposure-centre numbers and for centre alignment of the Triton.

Sources:
  [MH]  LUCID KB "Using Multiple Helios Cameras Simultaneously": at 1000 us exposure each
        integration is 0.98 ms, followed by a 1.55 ms readout that is "always constant
        independent of operating mode", repeated 4 times: T = (0.98 + 1.55) x 4 = 10.12 ms
        (single-frequency modes).
  [C §4] research/C_constraints.md: multi-frequency modes use 8 sub-frames (two modulation
        frequencies x 4 phases); confirmed for the same Sony IMX556 sensor by Basler's blaze
        documentation ("each frame is composed of eight subframes").
  The 0.98 integration ratio for the 250 us and 62.5 us selectors is an assumption [C §4].

Model: the frame starts with the first integration (the buffer timestamp is assumed to mark
this start - also unverified, A §(c)). Sub-frame i (0-based) integrates from
i*(int + readout) to i*(int + readout) + int. The centre of light collection is the mean of
the sub-frame centres: (n - 1)/2 * (int + readout) + int/2.

Examples: Distance5000mmMultiFreq + Exp1000Us -> 8 sub-frames, centre ~9345 us;
          Distance5000mmMultiFreq + Exp250Us  -> centre ~6405 us;
          Distance6000mmSingleFreq + Exp250Us -> 4 sub-frames, centre ~2815 us.
Verify on the rig with an oscilloscope (ExposureActive strobe / LED) before trusting it.
"""
from __future__ import annotations

NOMINAL_EXPOSURE_US = {"Exp1000Us": 1000.0, "Exp250Us": 250.0, "Exp62_5Us": 62.5}
INTEGRATION_RATIO = 0.98        # [MH]: 1000 us selector -> 0.98 ms integration
READOUT_US = 1550.0             # [MH]: constant in all modes


def helios_light_window(mode: str, exposure_selector: str) -> dict:
    """Modelled light-collection window of one Helios frame (all values in us).

    Returns dict(n_subframes, integration_us, readout_us, light_end_us, center_us,
    window_us, model). Raises ValueError for an unknown exposure selector or a mode
    name that is neither single- nor multi-frequency.
    """
    if exposure_selector not in NOMINAL_EXPOSURE_US:
        raise ValueError(f"unknown ExposureTimeSelector {exposure_selector!r}")
    if "MultiFreq" in mode:
        n = 8
    elif "SingleFreq" in mode:
        n = 4
    else:
        raise ValueError(f"cannot tell single/multi frequency from mode {mode!r}")
    integ = INTEGRATION_RATIO * NOMINAL_EXPOSURE_US[exposure_selector]
    step = integ + READOUT_US
    return dict(n_subframes=n, integration_us=integ, readout_us=READOUT_US,
                light_end_us=(n - 1) * step + integ,
                center_us=(n - 1) / 2.0 * step + integ / 2.0,
                window_us=n * step,
                model="unverified: [MH] 0.98*exposure + 1.55 ms x n, n=8 multi-freq [C §4]")
