"""
Compute disparity maps with StereoSGBM + WLS filter, convert to metric depth.
Reads calibration from calibration/stereo_calib.npz
Saves colourised depth maps to results/classical/

Improvements over plain SGBM:
  - CLAHE pre-processing on rectified images
  - Right matcher for left-right consistency check
  - WLS (Weighted Least Squares) filter: edge-aware smoothing + hole filling
"""
import cv2
import numpy as np
from pathlib import Path

CALIB_FILE = Path("calibration/stereo_calib.npz")
LEFT_DIR   = Path("frames/scene/left")
RIGHT_DIR  = Path("frames/scene/right")
OUT_DIR    = Path("results/classical")

BASELINE = 0.17  # metres

# SGBM parameters
NUM_DISPARITIES = 128   # must be divisible by 16
BLOCK_SIZE      = 7     # smaller = more detail, noisier
P1 = 8  * 3 * BLOCK_SIZE ** 2
P2 = 32 * 3 * BLOCK_SIZE ** 2

# WLS filter parameters
WLS_LAMBDA = 8000   # smoothness strength (higher = smoother)
WLS_SIGMA  = 1.5    # edge sensitivity (lower = sharper edges preserved)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def load_calib():
    d = np.load(CALIB_FILE)
    return d["map1_l"], d["map2_l"], d["map1_r"], d["map2_r"], d["P1"]


def depth_colormap(depth_m, max_depth=5.0):
    depth_clipped = np.clip(depth_m, 0, max_depth)
    norm = (depth_clipped / max_depth * 255).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_JET)


def main():
    if not CALIB_FILE.exists():
        print("[!] Calibration file not found. Run 02_calibrate.py first.")
        return

    map1_l, map2_l, map1_r, map2_r, P1_mat = load_calib()
    focal = float(P1_mat[0, 0])
    print(f"Focal: {focal:.1f} px  Baseline: {BASELINE} m")

    # Left matcher
    matcher_left = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=NUM_DISPARITIES,
        blockSize=BLOCK_SIZE,
        P1=P1, P2=P2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    # Right matcher (mirror of left — needed for WLS)
    matcher_right = cv2.ximgproc.createRightMatcher(matcher_left)

    # WLS filter
    wls = cv2.ximgproc.createDisparityWLSFilter(matcher_left)
    wls.setLambda(WLS_LAMBDA)
    wls.setSigmaColor(WLS_SIGMA)

    left_paths  = sorted(LEFT_DIR.glob("*.png"))
    right_paths = sorted(RIGHT_DIR.glob("*.png"))

    if not left_paths:
        print("[!] No scene frames. Run 01_extract_frames.py first.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Processing {len(left_paths)} frames...")

    for pl, pr in zip(left_paths, right_paths):
        img_l = cv2.imread(str(pl))
        img_r = cv2.imread(str(pr))

        # Rectify
        rect_l = cv2.remap(img_l, map1_l, map2_l, cv2.INTER_LINEAR)
        rect_r = cv2.remap(img_r, map1_r, map2_r, cv2.INTER_LINEAR)

        # CLAHE on grayscale before matching
        gray_l = clahe.apply(cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY))
        gray_r = clahe.apply(cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY))

        # Compute both disparities
        disp_left  = matcher_left.compute(gray_l, gray_r)
        disp_right = matcher_right.compute(gray_r, gray_l)

        # WLS filter (uses right disparity for consistency check)
        disp_filtered = wls.filter(disp_left, rect_l, None, disp_right)

        # Fixed-point → float
        disp_f = disp_filtered.astype(np.float32) / 16.0

        # Depth Z = f * B / d
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = np.where(disp_f > 1.0, focal * BASELINE / disp_f, 0)

        color = depth_colormap(depth)
        stem = pl.stem
        cv2.imwrite(str(OUT_DIR / f"{stem}_depth.png"), color)
        np.save(str(OUT_DIR / f"{stem}_depth_raw.npy"), depth)

    print(f"Done. Saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
