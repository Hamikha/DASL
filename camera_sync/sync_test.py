"""Measure and PROVE camera synchronisation (Helios2+ ToF + Triton colour).

    python sync_test.py analyze <capture_folder> [--out DIR] [--map helios_ts=COL,...,units=ns]
    python sync_test.py compare <before_folder> <after_folder> [--out DIR]
    python sync_test.py live    [--method ptpsync] [--fps 10] [--seconds 300] [--out DIR]
    python sync_test.py demo    [--out DIR]          # synthetic before/after, to eyeball the plots

Each analysis writes sync_report.md, sync_report.json and PNG plots into the output folder.
Every number is labelled MEASURED (shared PTP clock), MODELLED, HOST-BASED or PER-CAMERA;
see lucid_sync/analysis.py for the definitions.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# ---------------------------------------------------------------- settings (defaults)
DEFAULT_METHOD = "ptpsync"     # live: sync method to request (fallback chain is the engine's job)
DEFAULT_FPS = 10.0             # live: pair rate [Hz]
DEFAULT_SECONDS = 300          # live: 5 min = 3000 pairs at 10 fps -> stable p95 / drift numbers
DEFAULT_ANALYSIS_SUBDIR = "sync_analysis"   # analyze: output goes to <capture>/sync_analysis
DEFAULT_COMPARE_OUT = "before_after"        # compare: output folder (relative to cwd)
DEFAULT_DEMO_OUT = "sync_demo"              # demo: output folder (relative to cwd)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lucid_sync import analysis  # noqa: E402


def _print_summary(result: dict, files: dict) -> None:
    print()
    for row in analysis.headline_rows(result):
        print(f"  HEADLINE {row[0]}: median {row[1]}, p95 {row[2]}, max {row[3]}  [{row[6]}]")
    print()
    print(result["verdict"])
    print()
    for metric, value, basis in analysis.summary_rows(result):
        print(f"  {metric:<58} {value:<28} [{basis}]")
    for w in result["meta"]["warnings"]:
        print(f"  WARNING: {w}")
    for n in result["meta"]["notes"]:
        print(f"  note: {n}")
    print(f"\nReport: {files['md']}\nNumbers: {files['json']}")
    for p in files.get("plots", []):
        print(f"Plot:   {p}")


def cmd_analyze(args) -> dict:
    result, files = analysis.analyze_folder(args.folder, args.out, mapping=args.map,
                                            clock_domain=args.clock_domain, fps=args.fps,
                                            plots=not args.no_plots)
    _print_summary(result, files)
    return result


def cmd_compare(args) -> None:
    out = args.out or DEFAULT_COMPARE_OUT
    res = {}
    for tag, folder, mapping in (("before", args.before, args.map_before), ("after", args.after, args.map_after)):
        res[tag], files = analysis.analyze_folder(folder, os.path.join(out, tag), mapping=mapping,
                                                  plots=not args.no_plots)
        print(f"{tag.upper()}: {files['md']}")
    files = analysis.compare(res["before"], res["after"], out, plots=not args.no_plots)
    print(f"Before/after: {files['md']}")
    if files.get("png"):
        print(f"Plot:         {files['png']}")


def run_live(args) -> str:
    """Capture without saving images (writer=None), write the sync CSVs + run.json, then analyse.

    Engine contract used: CaptureEngine(SyncSettings, writer=None) as a context manager,
    .configure(), .run(seconds) -> CaptureResult with .write_sync_csvs(pairs_csv, frames_csv) and
    .run_info (= the run.json "sync" object, which already contains the settings).
    LUCID_MOCK=1 runs it on the mock SDK.
    """
    import json                                        # noqa: PLC0415
    from lucid_sync.engine import CaptureEngine        # lazy: needs arena_api (or the mock)
    from lucid_sync.settings import SyncSettings

    out = args.out or time.strftime("sync_test_%Y%m%d_%H%M%S")
    os.makedirs(out, exist_ok=True)
    settings = SyncSettings(SYNC_METHOD=args.method, FRAME_RATE=args.fps, SECONDS=args.seconds)
    with CaptureEngine(settings, writer=None) as engine:          # writer=None: no images saved
        engine.configure()
        result = engine.run(seconds=args.seconds)
    result.write_sync_csvs(os.path.join(out, "frames.csv"), os.path.join(out, "sync_frames.csv"))
    with open(os.path.join(out, "run.json"), "w", encoding="utf-8") as f:
        json.dump({"sync": result.run_info}, f, indent=2, default=str)
    args.folder, args.out, args.map, args.clock_domain = out, None, None, None
    cmd_analyze(args)
    return out


def cmd_demo(args) -> None:
    """SYNTHETIC demo, two like-for-like comparisons into <out>/analysis_*:
      legacy:  original capture.py freerun (independent clocks, host ESTIMATE) -> ptpsync centre-aligned
      ptp:     freerun_ptp (MEASURED, random phase)                         -> ptpsync centre-aligned
    """
    from lucid_sync.synthetic import make_capture_py_capture, make_ptp_capture

    out = args.out or DEFAULT_DEMO_OUT
    caps = {"before_legacy": os.path.join(out, "before_legacy_capture_py"),
            "before_ptp": os.path.join(out, "before_freerun_ptp"),
            "after": os.path.join(out, "after_ptpsync_center")}
    make_capture_py_capture(caps["before_legacy"], sync_mode="freerun", n_pairs=3000)
    make_ptp_capture(caps["before_ptp"], method="freerun_ptp", helios_drop_k=(), triton_drop_k=())
    make_ptp_capture(caps["after"], method="ptpsync", alignment="center")
    print("SYNTHETIC captures written: " + ", ".join(caps.values()))
    for tag in ("legacy", "ptp"):
        args.before, args.after = caps[f"before_{tag}"], caps["after"]
        args.map_before = args.map_after = None
        args.out = os.path.join(out, f"analysis_{tag}")
        cmd_compare(args)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="analyse one capture folder (new, legacy or sync_frames-only)")
    a.add_argument("folder")
    a.add_argument("--out", help=f"output folder (default <folder>/{DEFAULT_ANALYSIS_SUBDIR})")
    a.add_argument("--map", action="append",
                   help="legacy column map, e.g. helios_ts=depth_ts,triton_ts=rgb_ts,"
                        "helios_host=depth_recv,triton_host=rgb_recv,units=ns "
                        "(keys: " + ", ".join(analysis.MAP_KEYS) + ")")
    a.add_argument("--clock-domain", choices=["ptp", "independent"],
                   help="override the clock domain (only if you KNOW both cameras were on PTP)")
    a.add_argument("--fps", type=float, help="nominal frame rate if run.json lacks it")
    a.add_argument("--no-plots", action="store_true")

    c = sub.add_parser("compare", help="before/after report of two capture folders")
    c.add_argument("before")
    c.add_argument("after")
    c.add_argument("--out", help=f"output folder (default ./{DEFAULT_COMPARE_OUT})")
    c.add_argument("--map-before", action="append", help="--map for the BEFORE folder")
    c.add_argument("--map-after", action="append", help="--map for the AFTER folder")
    c.add_argument("--no-plots", action="store_true")

    lv = sub.add_parser("live", help="capture now (no images saved) and analyse")
    lv.add_argument("--method", default=DEFAULT_METHOD,
                    choices=["ptpsync", "action", "freerun_ptp", "freerun"])
    lv.add_argument("--fps", type=float, default=DEFAULT_FPS)
    lv.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    lv.add_argument("--out", help="output folder (default ./sync_test_<timestamp>)")
    lv.add_argument("--no-plots", action="store_true")

    d = sub.add_parser("demo", help="SYNTHETIC before/after example to eyeball the plots")
    d.add_argument("--out", help=f"output folder (default ./{DEFAULT_DEMO_OUT})")
    d.add_argument("--no-plots", action="store_true")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.cmd == "analyze":
            cmd_analyze(args)
        elif args.cmd == "compare":
            cmd_compare(args)
        elif args.cmd == "live":
            run_live(args)
        elif args.cmd == "demo":
            cmd_demo(args)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
