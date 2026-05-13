"""
Compute disparity maps with StereoSGBM and convert to metric depth.
Reads calibration from calibration/stereo_calib.npz
Saves colourised depth maps to results/classical/
"""
import cv2
import numpy as np
from pathlib import Path

CALIB_FILE = Path("calibration/stereo_calib.npz")
LEFT_DIR = Path("frames/scene/left")
RIGHT_DIR = Path("frames/scene/right")
OUT_DIR = Path("results/classical")

BASELINE = 0.17  # metres — known hardware baseline

# SGBM parameters (tunable)
NUM_DISPARITIES = 128   # must be divisible by 16
BLOCK_SIZE = 11
P1 = 8 * 3 * BLOCK_SIZE ** 2
P2 = 32 * 3 * BLOCK_SIZE ** 2


def load_calib():
    data = np.load(CALIB_FILE)
    return (
        data["map1_l"], data["map2_l"],
        data["map1_r"], data["map2_r"],
        data["P1"],
    )


def depth_colormap(depth_m, max_depth=5.0):
    depth_clipped = np.clip(depth_m, 0, max_depth)
    norm = (depth_clipped / max_depth * 255).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_JET)


def main():
    if not CALIB_FILE.exists():
        print("[!] Calibration file not found. Run 02_calibrate.py first.")
        return

    map1_l, map2_l, map1_r, map2_r, P1_mat = load_calib()
    focal = float(P1_mat[0, 0])  # pixels
    print(f"Focal length: {focal:.1f} px,  Baseline: {BASELINE} m")

    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=NUM_DISPARITIES,
        blockSize=BLOCK_SIZE,
        P1=P1,
        P2=P2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    left_paths = sorted(LEFT_DIR.glob("*.png"))
    right_paths = sorted(RIGHT_DIR.glob("*.png"))

    if not left_paths:
        print("[!] No scene frames found. Run 01_extract_frames.py first.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Processing {len(left_paths)} frames...")

    for pl, pr in zip(left_paths, right_paths):
        img_l = cv2.imread(str(pl))
        img_r = cv2.imread(str(pr))

        # Rectify
        rect_l = cv2.remap(img_l, map1_l, map2_l, cv2.INTER_LINEAR)
        rect_r = cv2.remap(img_r, map1_r, map2_r, cv2.INTER_LINEAR)

        # Disparity (SGBM returns fixed-point ×16)
        disp = stereo.compute(
            cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY),
        ).astype(np.float32) / 16.0

        # Depth Z = f * B / d
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = np.where(disp > 0, focal * BASELINE / disp, 0)

        color = depth_colormap(depth)

        stem = pl.stem
        cv2.imwrite(str(OUT_DIR / f"{stem}_depth.png"), color)
        np.save(str(OUT_DIR / f"{stem}_depth_raw.npy"), depth)

    print(f"Saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
