"""
process.py - turn a capture.py run into coloured point clouds and a
TUM RGB-D dataset for SLAM.

    python process.py C:\\Users\\you\\captures\\run_20260924_101500
    python process.py <run> --no-ply            TUM dataset only
    python process.py <run> --every 2           every second frame
    python process.py <run> --max-range 4000    drop points beyond 4 m
    python process.py <run> --calib other.yml   use a different calibration

Writes inside the run folder:
    ply/frame_000000.ply        coloured points, mm, Helios coordinates
    tum/rgb/000000.png          colour image
    tum/depth/000000.png        16-bit depth in mm, aligned pixel for pixel
    tum/rgb.txt, depth.txt,     standard TUM index files
    tum/associations.txt
    tum/camera.yaml             intrinsics of the RGB-D images
    process_report.json         what was done and how well

The RGB-D images are the Helios's own pixel grid, one point per pixel with
its depth and the colour sampled from the Triton, then undistorted into a
clean pinhole camera using a lens model fitted from the Helios data itself.
Depth and colour are aligned by construction and the images are dense.
"""

import argparse
import csv
import glob
import json
import os
import sys
import time

import cv2
import numpy as np


# A point is treated as hidden from the Triton (and left uncoloured) when it
# lies further behind the nearest surface in its small image cell than
# max(OCCLUSION_TOLERANCE_MM, OCCLUSION_TOLERANCE_FRAC * depth). The depth
# term keeps noisy or steeply angled surfaces from being flagged falsely.
# Tuned on real Helios2+ data: 0.2% false rejects on a flat surface while
# still catching 98.7% of truly hidden points in a synthetic test.
OCCLUSION_TOLERANCE_MM = 50.0
OCCLUSION_TOLERANCE_FRAC = 0.05
OCCLUSION_CELL_PX = 4
# Points whose undistorted direction lies this far beyond the Triton image
# border are never projected. Stops the 8-term distortion model folding
# far-off points back into the picture.
FOV_MARGIN = 1.10
# Frames (spread across the run) and points used to fit the Helios lens.
LENS_FIT_FRAMES = 8
LENS_FIT_POINTS = 90000
# The centre offset can only be told apart from focal length when the fit
# frames cover a range of depths. Below this ratio of far to near depth it
# is fixed at zero rather than guessed.
LENS_OFFSET_MIN_DEPTH_RATIO = 1.3
DEPTH_SCALE = 1000.0          # depth PNG units per metre (1 unit = 1 mm)

# Point cleaning, tuned on real Helios2+ data. Depth noise tracks the
# returned infrared signal closely: lowest-signal points carry 50-70 mm of
# noise, strong ones 4-6 mm.
#   MIN_SIGNAL   drops only the weakest returns (about 7% of a typical indoor
#                frame). Signal scales with exposure and gain, so it is kept
#                deliberately low; the spike filter does the adaptive work.
#   SPIKE_*      removes any point further than max(SPIKE_MM, SPIKE_FRAC*z)
#                from the median of its 5x5 neighbourhood. That catches flying
#                pixels at depth edges and isolated noise spikes alike, and
#                is independent of exposure settings.
MIN_SIGNAL = 300
SPIKE_MM = 40.0
SPIKE_FRAC = 0.05


def log(message):
    print(f"  {message}", flush=True)


# ============================================================================
# LOADING
# ============================================================================

def load_run(run_dir):
    with open(os.path.join(run_dir, "run.json")) as f:
        info = json.load(f)
    for key in ("helios", "triton", "files"):
        if key not in info:
            raise SystemExit(f"run.json has no '{key}'. Was capture interrupted "
                             f"before recording started?")
    return info


def load_calibration(path):
    if not path or not os.path.exists(path):
        raise SystemExit(f"Calibration not found: {path}. Pass one with --calib.")
    fs = cv2.FileStorage(path, cv2.FileStorage_READ)
    calib = {
        "K": fs.getNode("cameraMatrix").mat(),
        "D": fs.getNode("distCoeffs").mat(),
        "rvec": fs.getNode("rotationVector").mat(),
        "tvec": fs.getNode("translationVector").mat(),
    }
    fs.release()
    for name, value in calib.items():
        if value is None:
            raise SystemExit(f"{path} is missing {name}")
    calib["K"] = calib["K"].astype(np.float64).reshape(3, 3)
    calib["D"] = calib["D"].astype(np.float64).reshape(1, -1)
    calib["rvec"] = calib["rvec"].astype(np.float64).reshape(3, 1)
    calib["tvec"] = calib["tvec"].astype(np.float64).reshape(3, 1)
    calib["R"], _ = cv2.Rodrigues(calib["rvec"])
    return calib


def load_timestamps(run_dir):
    stamps = {}
    path = os.path.join(run_dir, "frames.csv")
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    stamps[int(row["frame"])] = float(row["host_time"])
                except (KeyError, ValueError):
                    pass
    return stamps


def list_pairs(run_dir):
    pairs = []
    for helios in sorted(glob.glob(os.path.join(run_dir, "raw", "helios_*.npy"))):
        index = int(os.path.basename(helios)[7:13])
        triton = os.path.join(run_dir, "raw", f"triton_{index:06d}.npy")
        if os.path.exists(triton):
            pairs.append((index, helios, triton))
    return pairs


# ============================================================================
# PER FRAME
# ============================================================================

def decode_helios(raw, scale, offsets, max_range_mm):
    """uint16 (H, W, 4) -> xyz in mm (H, W, 3), validity mask, signal (H, W).

    Only removes pixels the camera marked invalid (and, optionally, points
    beyond max_range_mm). Cleaning is a separate step, see clean_mask.
    """
    abc = raw[..., :3]
    invalid = (abc == 0xFFFF).any(axis=-1) | (abc == 0).all(axis=-1)
    xyz = abc.astype(np.float32) * np.float32(scale)
    xyz += np.asarray(offsets, dtype=np.float32)
    valid = ~invalid & (xyz[..., 2] > 0)
    if max_range_mm:
        valid &= np.linalg.norm(xyz, axis=-1) <= max_range_mm
    return xyz, valid, raw[..., 3]


def clean_mask(xyz, valid, signal, min_signal, spike_mm, spike_frac, counts=None):
    """Valid mask with weak returns and spikes removed.

    Used for the point clouds. Deliberately NOT used for the TUM depth that
    feeds tracking: on real data the removed clutter is what ICP uses to
    locate itself, and tracking on cleaned depth roughly doubled pose error.
    """
    keep = valid.copy()
    if counts is not None:
        counts["before_cleaning"] = counts.get("before_cleaning", 0) + int(valid.sum())
    if min_signal:
        weak = keep & (signal < min_signal)
        keep &= ~weak
        if counts is not None:
            counts["weak_signal"] = counts.get("weak_signal", 0) + int(weak.sum())
    if spike_mm or spike_frac:
        z = np.where(keep, xyz[..., 2], 0).astype(np.float32)
        local = cv2.medianBlur(z, 5)
        limit = np.maximum(spike_mm, spike_frac * z)
        spike = keep & ((local == 0) | (np.abs(z - local) > limit))
        keep &= ~spike
        if counts is not None:
            counts["spike"] = counts.get("spike", 0) + int(spike.sum())
    return keep


def load_triton_rgb(path, pixel_format):
    pixels = np.load(path)
    if pixel_format == "BayerRG8":
        # GenICam BayerRG (R G / G B) is OpenCV's "BG" code. Verified on a
        # synthetic mosaic: the RG code swaps red and blue.
        return cv2.cvtColor(pixels, cv2.COLOR_BayerBG2RGB_EA)
    if pixel_format == "RGB8":
        return pixels
    raise SystemExit(f"Unsupported Triton pixel format {pixel_format}")


def colour_points(points, rgb, calib, fov_limits):
    """Sample a Triton colour for each point (Helios frame, mm).

    Returns colours (N, 3) uint8 and a mask of points that received one.
    """
    height, width = rgb.shape[:2]
    n = len(points)
    colours = np.zeros((n, 3), np.uint8)
    coloured = np.zeros(n, bool)
    if n == 0:
        return colours, coloured

    cam = points.astype(np.float64) @ calib["R"].T + calib["tvec"].reshape(1, 3)
    z = cam[:, 2]
    front = z > 1.0
    xn = np.divide(cam[:, 0], z, out=np.zeros_like(z), where=front)
    yn = np.divide(cam[:, 1], z, out=np.zeros_like(z), where=front)
    x_lo, x_hi, y_lo, y_hi = fov_limits
    candidate = front & (xn > x_lo) & (xn < x_hi) & (yn > y_lo) & (yn < y_hi)
    idx = np.flatnonzero(candidate)
    if idx.size == 0:
        return colours, coloured

    uv, _ = cv2.projectPoints(points[idx].astype(np.float64), calib["rvec"],
                              calib["tvec"], calib["K"], calib["D"])
    uv = uv.reshape(-1, 2)
    u = np.rint(uv[:, 0]).astype(np.int64)
    v = np.rint(uv[:, 1]).astype(np.int64)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    idx, u, v = idx[inside], u[inside], v[inside]
    depth = z[idx]

    # Occlusion: in each small cell of the Triton image keep only surfaces
    # near the closest one. Anything further back was hidden from the Triton.
    grid_w = width // OCCLUSION_CELL_PX + 1
    cell = (v // OCCLUSION_CELL_PX) * grid_w + (u // OCCLUSION_CELL_PX)
    nearest = np.full(grid_w * (height // OCCLUSION_CELL_PX + 1), np.inf)
    np.minimum.at(nearest, cell, depth)
    tolerance = np.maximum(OCCLUSION_TOLERANCE_MM, OCCLUSION_TOLERANCE_FRAC * depth)
    visible = depth <= nearest[cell] + tolerance

    idx, u, v = idx[visible], u[visible], v[visible]
    colours[idx] = rgb[v, u]
    coloured[idx] = True
    return colours, coloured


def fov_limits_for(calib, width, height):
    """Undistorted normalised image bounds of the Triton, with a margin."""
    corners = np.array([[[0, 0]], [[width - 1, 0]], [[0, height - 1]],
                        [[width - 1, height - 1]], [[width / 2, 0]],
                        [[width / 2, height - 1]], [[0, height / 2]],
                        [[width - 1, height / 2]]], np.float64)
    und = cv2.undistortPoints(corners, calib["K"], calib["D"]).reshape(-1, 2)
    cx_lim = np.abs(und[:, 0]).max() * FOV_MARGIN
    cy_lim = np.abs(und[:, 1]).max() * FOV_MARGIN
    # Fall back to the pinhole bounds if undistortion misbehaves at corners.
    k = calib["K"]
    pin_x = max(k[0, 2], width - k[0, 2]) / k[0, 0] * FOV_MARGIN * 1.5
    pin_y = max(k[1, 2], height - k[1, 2]) / k[1, 1] * FOV_MARGIN * 1.5
    cx_lim = min(cx_lim, pin_x) if np.isfinite(cx_lim) else pin_x
    cy_lim = min(cy_lim, pin_y) if np.isfinite(cy_lim) else pin_y
    return -cx_lim, cx_lim, -cy_lim, cy_lim


# ============================================================================
# VIRTUAL CAMERA AND OUTPUTS
# ============================================================================

def fit_helios_lens(samples, width, height):
    """Fit the Helios lens from its own data: pinhole, 5-term distortion
    (k1 k2 p1 p2 k3) and a small offset of the true lens centre from the
    origin of the reported XYZ.

    Every valid Helios pixel is an exact 3D point to pixel correspondence.
    On a Helios2+ a plain pinhole leaves ~4 px of error; distortion brings it
    to ~0.3 px; the centre offset (about 3 mm along the axis) brings it to
    ~0.08 px across frames at different depths. Levenberg-Marquardt with
    OpenCV's analytic Jacobian, so no extra packages are needed.

    Returns K, dist, offset (mm, add to XYZ to get lens-centred points), rms.
    """
    rows, cols = np.mgrid[0:height, 0:width]
    pts, pix = [], []
    for xyz, valid in samples:
        pts.append(xyz[valid])
        pix.append(np.stack([cols[valid], rows[valid]], axis=1))
    pts = np.concatenate(pts).astype(np.float64)
    pix = np.concatenate(pix).astype(np.float64)
    if len(pts) < 1000:
        raise SystemExit("Too few valid Helios points to fit the lens. "
                         "Were the fit frames empty?")
    if len(pts) > LENS_FIT_POINTS:
        pick = np.random.default_rng(0).choice(len(pts), LENS_FIT_POINTS, replace=False)
        pts, pix = pts[pick], pix[pick]

    near, far = np.percentile(pts[:, 2], [5, 95])
    fit_offset = far / max(near, 1.0) >= LENS_OFFSET_MIN_DEPTH_RATIO

    a, b = pts[:, 0] / pts[:, 2], pts[:, 1] / pts[:, 2]
    fx, cx = np.linalg.lstsq(np.stack([a, np.ones_like(a)], 1), pix[:, 0], rcond=None)[0]
    fy, cy = np.linalg.lstsq(np.stack([b, np.ones_like(b)], 1), pix[:, 1], rcond=None)[0]
    params = np.array([fx, fy, cx, cy, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
    zero = np.zeros(3)

    def evaluate(p):
        k = np.array([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1]])
        offset = p[9:12] if fit_offset else zero
        proj, jac = cv2.projectPoints(pts, zero, offset, k, p[4:9])
        # Jacobian columns: rvec(3) tvec(3) f(2) c(2) dist(5)
        j_off = jac[:, 3:6] if fit_offset else np.zeros((len(jac), 3))
        return (proj.reshape(-1, 2) - pix).ravel(), np.hstack([jac[:, 6:15], j_off])

    resid, jac = evaluate(params)
    cost = float(resid @ resid)
    damping = 1e-3
    ridge = 1e-9 * np.eye(len(params))        # keeps the unused offset columns solvable
    for _ in range(150):
        normal = jac.T @ jac + ridge
        step = np.linalg.solve(normal + damping * np.diag(np.diag(normal)), -(jac.T @ resid))
        trial = params + step
        t_resid, t_jac = evaluate(trial)
        t_cost = float(t_resid @ t_resid)
        if t_cost < cost:
            done = (cost - t_cost) < 1e-10 * cost
            params, resid, jac, cost = trial, t_resid, t_jac, t_cost
            damping = max(damping / 3.0, 1e-9)
            if done:
                break
        else:
            damping *= 5.0
            if damping > 1e8:
                break

    k = np.array([[params[0], 0, params[2]], [0, params[1], params[3]], [0, 0, 1]])
    offset = params[9:12].copy() if fit_offset else np.zeros(3)
    return k, params[4:9].copy(), offset, float(np.sqrt(cost / len(pts)))


def build_rgbd_camera(k, dist, width, height):
    """Undistorted pinhole camera for the RGB-D images and the pixel maps
    that take native Helios pixels into it. alpha=0 crops to the region
    where every output pixel has a real source pixel."""
    new_k, _ = cv2.getOptimalNewCameraMatrix(k, dist, (width, height), 0, (width, height))
    map_x, map_y = cv2.initUndistortRectifyMap(k, dist, None, new_k,
                                               (width, height), cv2.CV_32FC1)
    intr = {"fx": float(new_k[0, 0]), "fy": float(new_k[1, 1]),
            "cx": float(new_k[0, 2]), "cy": float(new_k[1, 2]),
            "width": width, "height": height}
    return intr, (map_x, map_y)


def render_rgbd(xyz, valid, colours, maps, offset):
    """Native Helios grid (one point per pixel, no holes) resampled through
    the undistortion maps. Nearest neighbour, so depth values are never
    blended across edges into depths that do not exist."""
    height, width = valid.shape
    depth_native = np.zeros((height, width), np.float32)
    depth_native[valid] = xyz[..., 2][valid] + offset[2]   # depth from the true lens centre
    colour_native = np.zeros((height, width, 3), np.uint8)
    colour_native[valid] = colours
    map_x, map_y = maps
    depth = cv2.remap(depth_native, map_x, map_y, cv2.INTER_NEAREST,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    colour = cv2.remap(colour_native, map_x, map_y, cv2.INTER_NEAREST,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    depth_units = np.where(depth > 0, np.rint(depth * DEPTH_SCALE / 1000.0), 0)
    return np.clip(depth_units, 0, 65535).astype(np.uint16), colour


def write_ply(path, points, colours, signal):
    """Binary PLY: x y z (mm), colour, and the Helios signal strength as a
    scalar ('intensity') so later steps can filter harder without redoing
    this run. CloudCompare shows it as a scalar field; Open3D ignores it."""
    n = len(points)
    vertex = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                                ("intensity", "<u2")])
    vertex["x"], vertex["y"], vertex["z"] = points[:, 0], points[:, 1], points[:, 2]
    vertex["red"], vertex["green"], vertex["blue"] = colours[:, 0], colours[:, 1], colours[:, 2]
    vertex["intensity"] = signal
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "property ushort intensity\n"
              "end_header\n").encode("ascii")
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(header)
        f.write(vertex.tobytes())
    os.replace(tmp, path)


def write_camera_yaml(path, intr):
    text = (
        "# Virtual pinhole camera at the Helios. Depth PNG units: "
        f"{DEPTH_SCALE:.0f} per metre (1 = 1 mm).\n"
        f"width: {intr['width']}\nheight: {intr['height']}\n"
        f"fx: {intr['fx']:.6f}\nfy: {intr['fy']:.6f}\n"
        f"cx: {intr['cx']:.6f}\ncy: {intr['cy']:.6f}\n"
        f"depth_scale: {DEPTH_SCALE:.1f}\n"
        "distortion: [0, 0, 0, 0, 0]   # images are already undistorted\n"
        f"# Helios lens fit rms {intr['lens_fit_rms_px']:.3f} px\n"
        f"# Camera centre = Helios XYZ + {intr['helios_lens']['centre_offset_mm']} mm. "
        "Subtract this offset from back-projected points to return to Helios XYZ.\n"
    )
    with open(path, "w") as f:
        f.write(text)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Process a capture.py run.")
    parser.add_argument("run", help="run folder made by capture.py")
    parser.add_argument("--calib", help="calibration yml (default: the run's copy)")
    parser.add_argument("--every", type=int, default=1, help="use every Nth frame")
    parser.add_argument("--max-range", type=float, default=0.0,
                        help="drop points further than this, mm (0 = keep all)")
    parser.add_argument("--min-signal", type=int, default=MIN_SIGNAL,
                        help=f"drop points with weaker infrared return (default {MIN_SIGNAL}, 0 = off)")
    parser.add_argument("--spike-mm", type=float, default=SPIKE_MM,
                        help=f"spike filter floor in mm (default {SPIKE_MM:.0f}, 0 = filter off)")
    parser.add_argument("--tum-clean", action="store_true",
                        help="also clean the TUM depth (default keeps it whole for tracking)")
    parser.add_argument("--no-ply", action="store_true", help="skip per-frame PLYs")
    parser.add_argument("--no-tum", action="store_true", help="skip the TUM dataset")
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run)
    if not os.path.isdir(run_dir):
        raise SystemExit(f"Not a folder: {run_dir}")
    if args.every < 1:
        raise SystemExit("--every must be 1 or more")

    info = load_run(run_dir)
    helios, triton = info["helios"], info["triton"]
    calib_path = args.calib or os.path.join(run_dir, "orientation.yml")
    calib = load_calibration(calib_path)
    stamps = load_timestamps(run_dir)
    pairs = list_pairs(run_dir)[::args.every]
    if not pairs:
        raise SystemExit("No complete frame pairs in raw/.")

    tw, th = int(triton["width"]), int(triton["height"])
    k = calib["K"]
    if not (0.25 * tw < k[0, 2] < 0.75 * tw and 0.25 * th < k[1, 2] < 0.75 * th):
        raise SystemExit(f"Calibration centre ({k[0, 2]:.0f}, {k[1, 2]:.0f}) does not "
                         f"fit a {tw} x {th} Triton image. Wrong calibration file?")
    fov = fov_limits_for(calib, tw, th)
    scale, offsets = helios["scale"], helios["offsets"]
    spike_frac = SPIKE_FRAC if args.spike_mm > 0 else 0.0
    clean_counts = {}

    print(f"\nprocess.py  {os.path.basename(run_dir)}\n")
    log(f"{len(pairs)} frame pairs, Triton {triton['pixel_format']} {tw} x {th}")
    log(f"Calibration: {calib_path}")

    ply_dir = os.path.join(run_dir, "ply")
    tum_dir = os.path.join(run_dir, "tum")
    if not args.no_ply:
        os.makedirs(ply_dir, exist_ok=True)

    intr = maps = None
    lens_offset = np.zeros(3)
    if not args.no_tum:
        os.makedirs(os.path.join(tum_dir, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(tum_dir, "depth"), exist_ok=True)
        samples = []
        spread = np.unique(np.linspace(0, len(pairs) - 1, min(LENS_FIT_FRAMES, len(pairs))).astype(int))
        for _, hpath, _ in (pairs[i] for i in spread):
            xyz, valid, signal = decode_helios(np.load(hpath), scale, offsets, args.max_range)
            samples.append((xyz, clean_mask(xyz, valid, signal, args.min_signal,
                                            args.spike_mm, spike_frac)))
        hw, hh = int(helios["width"]), int(helios["height"])
        lens_k, lens_d, lens_offset, lens_rms = fit_helios_lens(samples, hw, hh)
        intr, maps = build_rgbd_camera(lens_k, lens_d, hw, hh)
        intr["lens_fit_rms_px"] = lens_rms
        intr["helios_lens"] = {"K": lens_k.tolist(), "dist": lens_d.tolist(),
                               "centre_offset_mm": lens_offset.tolist()}
        write_camera_yaml(os.path.join(tum_dir, "camera.yaml"), intr)
        offset_note = (f"centre offset {lens_offset[2]:+.1f} mm" if lens_offset.any()
                       else "centre offset fixed at 0, depth range too narrow to fit")
        log(f"Helios lens fit {lens_rms:.2f} px (k1 {lens_d[0]:+.3f}, {offset_note})")
        if lens_rms > 1.0:
            log("WARNING: lens fit above 1 px. Check the first frames are not "
                "mostly empty or noisy.")
        log(f"RGB-D camera fx {intr['fx']:.1f} fy {intr['fy']:.1f} "
            f"cx {intr['cx']:.1f} cy {intr['cy']:.1f} (undistorted)")

    rgb_lines, depth_lines, assoc_lines = [], [], []
    stats = {"frames": 0, "points": 0, "coloured": 0, "skipped": []}
    t0 = time.perf_counter()
    nominal_period = 1.0 / float(info.get("frame_rate_target") or 10.0)

    for n, (index, hpath, tpath) in enumerate(pairs, 1):
        try:
            xyz, valid, signal = decode_helios(np.load(hpath), scale, offsets, args.max_range)
            clean = clean_mask(xyz, valid, signal, args.min_signal, args.spike_mm,
                               spike_frac, clean_counts)
            rgb = load_triton_rgb(tpath, triton["pixel_format"])
            if rgb.shape[:2] != (th, tw):
                raise ValueError(f"Triton image {rgb.shape[:2]} is not {th} x {tw}")
        except Exception as exc:
            stats["skipped"].append({"frame": index, "reason": str(exc)})
            continue

        # Colour every valid point once. The map (PLY) keeps the cleaned subset;
        # the TUM depth keeps all valid points for tracking.
        points = xyz[valid]
        colours, coloured = colour_points(points, rgb, calib, fov)
        in_map = clean[valid]

        if not args.no_ply:
            write_ply(os.path.join(ply_dir, f"frame_{index:06d}.ply"), points[in_map],
                      colours[in_map], signal[valid][in_map])

        if intr is not None:
            stamp = stamps.get(index, index * nominal_period)
            name = f"{index:06d}.png"
            tum_valid = clean if args.tum_clean else valid
            tum_colours = colours[in_map] if args.tum_clean else colours
            depth_img, colour_img = render_rgbd(xyz, tum_valid, tum_colours, maps, lens_offset)
            cv2.imwrite(os.path.join(tum_dir, "depth", name), depth_img)
            cv2.imwrite(os.path.join(tum_dir, "rgb", name),
                        cv2.cvtColor(colour_img, cv2.COLOR_RGB2BGR))
            rgb_lines.append(f"{stamp:.6f} rgb/{name}")
            depth_lines.append(f"{stamp:.6f} depth/{name}")
            assoc_lines.append(f"{stamp:.6f} rgb/{name} {stamp:.6f} depth/{name}")

        stats["frames"] += 1
        stats["points"] += int(in_map.sum())
        stats["coloured"] += int(coloured[in_map].sum())
        if n % 10 == 0 or n == len(pairs):
            rate = n / (time.perf_counter() - t0)
            print(f"\r  {n}/{len(pairs)} frames  {rate:.1f} frames/s   ", end="", flush=True)
    print()

    if intr is not None:
        header = "# timestamp filename\n"
        for fname, lines in (("rgb.txt", rgb_lines), ("depth.txt", depth_lines)):
            with open(os.path.join(tum_dir, fname), "w") as f:
                f.write(header + "\n".join(lines) + "\n")
        with open(os.path.join(tum_dir, "associations.txt"), "w") as f:
            f.write("\n".join(assoc_lines) + "\n")

    frames = max(stats["frames"], 1)
    report = {
        "run": os.path.basename(run_dir),
        "calibration": calib_path,
        "frames_processed": stats["frames"],
        "frames_skipped": stats["skipped"],
        "mean_points_per_frame": round(stats["points"] / frames),
        "coloured_fraction": round(stats["coloured"] / max(stats["points"], 1), 3),
        "rgbd_camera": intr,
        "cleaning": {
            "weak_signal_removed": round(clean_counts.get("weak_signal", 0) /
                                         max(clean_counts.get("before_cleaning", 1), 1), 4),
            "spikes_removed": round(clean_counts.get("spike", 0) /
                                    max(clean_counts.get("before_cleaning", 1), 1), 4),
        },
        "options": {"every": args.every, "max_range_mm": args.max_range,
                    "min_signal": args.min_signal, "spike_mm": args.spike_mm,
                    "ply": not args.no_ply, "tum": not args.no_tum},
    }
    with open(os.path.join(run_dir, "process_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    before = max(clean_counts.get("before_cleaning", 1), 1)
    log(f"Cleaning removed {clean_counts.get('weak_signal', 0) / before * 100:.1f}% weak "
        f"returns, {clean_counts.get('spike', 0) / before * 100:.1f}% spikes")
    log(f"Processed {stats['frames']} frames, "
        f"{report['mean_points_per_frame']:,} points each, "
        f"{report['coloured_fraction'] * 100:.0f}% coloured")
    if stats["skipped"]:
        log(f"Skipped {len(stats['skipped'])} unreadable frames (see process_report.json)")
    if not args.no_ply:
        log(f"PLY:  {ply_dir}")
    if intr is not None:
        log(f"TUM:  {tum_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())