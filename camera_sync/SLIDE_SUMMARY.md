Camera synchronisation (Helios2+ ToF + Triton RGB). The two cameras previously free-ran on
independent clocks, so a depth/colour pair could be up to one frame (100 ms at 10 fps) apart,
causing colour smear on a moving hand-held rig. I replaced this with IEEE 1588 PTP clock
synchronisation and LUCID's PTP Sync mode, in which both cameras' firmware starts exposures
on the same shared time grid, with the Triton exposure centred on the Helios depth frame.
Automatic fallbacks (action commands, software trigger, free-run) are built in. LUCID reports
~20 µs between Helios and Triton timestamps with this approach (vs. up to 100 ms before), and
my capture now records both cameras' timestamps on the shared clock so every pair's offset is
measured, at 10 fps with the Triton exposure capped so it can no longer reduce the frame rate.
The pipeline and measurement tool are tested in simulation; on-rig verification is the next step.
