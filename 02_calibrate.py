"""
Stereo camera calibration using chessboard images.
Board: 10x7 cells = 9x6 inner corners, square size 35 mm.
Saves calibration data to calibration/stereo_calib.npz

Speed strategy:
  1. Rank all frames by Laplacian sharpness (fast).
  2. Run findChessboardCorners only on top-N sharpest frames.
  3. Use CALIB_CB_FAST_CHECK to skip clearly-wrong frames instantly.
"""
import cv2
import numpy as np
from pathlib import Path

SQUARE_SIZE  = 0.035   # metres (35 mm)
BOARD_SIZE   = (9, 6)  # inner corners (cols, rows)
TOP_N_FRAMES  = 200    # consider only the N sharpest frames per camera
MIN_PAIRS     = 10     # minimum valid pairs needed for calibration
MAX_CALIB_PAIRS = 50   # subsample before calibrateCamera (50 is plenty)

FRAMES_LEFT  = Path("frames/chess/left")
FRAMES_RIGHT = Path("frames/chess/right")
OUT_DIR      = Path("calibration")

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def sharpness(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def enhance(gray):
    sharp = clahe.apply(gray)
    blur  = cv2.GaussianBlur(sharp, (0, 0), 3)
    return cv2.addWeighted(sharp, 1.5, blur, -0.5, 0)


def detect(gray):
    img   = enhance(gray)
    fast  = cv2.CALIB_CB_FAST_CHECK
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE

    # Quick rejection
    found, corners = cv2.findChessboardCorners(img, BOARD_SIZE, fast)
    if not found:
        return False, None

    # Full detection
    found, corners = cv2.findChessboardCorners(img, BOARD_SIZE, flags)
    if found:
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
    return found, corners


def rank_by_sharpness(paths, top_n):
    scored = []
    for p in paths:
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g is not None:
            scored.append((sharpness(g), p))
    scored.sort(reverse=True)
    selected = [p for _, p in scored[:top_n]]
    print(f"  Sharpness range (top {top_n}): "
          f"{scored[0][0]:.1f} … {scored[min(top_n,len(scored))-1][0]:.1f}")
    return selected


def collect_points(left_paths, right_paths):
    cols, rows = BOARD_SIZE
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * SQUARE_SIZE

    left_map  = {p.name: p for p in left_paths}
    right_map = {p.name: p for p in right_paths}
    common    = sorted(set(left_map) & set(right_map))

    obj_pts, ipl, ipr = [], [], []
    img_size = None

    print(f"  Running detection on {len(common)} matched frame pairs...")
    for name in common:
        gl = cv2.imread(str(left_map[name]),  cv2.IMREAD_GRAYSCALE)
        gr = cv2.imread(str(right_map[name]), cv2.IMREAD_GRAYSCALE)
        fl, cl = detect(gl)
        fr, cr = detect(gr)
        if fl and fr:
            obj_pts.append(objp)
            ipl.append(cl)
            ipr.append(cr)
            if img_size is None:
                img_size = (gl.shape[1], gl.shape[0])

    print(f"  Valid pairs with board: {len(obj_pts)}")

    # Subsample evenly to keep calibrateCamera fast
    if len(obj_pts) > MAX_CALIB_PAIRS:
        step = len(obj_pts) // MAX_CALIB_PAIRS
        obj_pts = obj_pts[::step][:MAX_CALIB_PAIRS]
        ipl     = ipl[::step][:MAX_CALIB_PAIRS]
        ipr     = ipr[::step][:MAX_CALIB_PAIRS]
        print(f"  Subsampled to {len(obj_pts)} pairs for calibration")

    return obj_pts, ipl, ipr, img_size


def main():
    left_all  = sorted(FRAMES_LEFT.glob("*.png"))
    right_all = sorted(FRAMES_RIGHT.glob("*.png"))

    if not left_all:
        print("[!] No frames found. Run 01_extract_frames.py first.")
        return

    print(f"Total frames: {len(left_all)} left / {len(right_all)} right")
    print(f"Board: {BOARD_SIZE}, square {int(SQUARE_SIZE*1000)} mm\n")

    print(f"Ranking by sharpness, keeping top {TOP_N_FRAMES}...")
    left_sharp  = rank_by_sharpness(left_all,  TOP_N_FRAMES)
    right_sharp = rank_by_sharpness(right_all, TOP_N_FRAMES)

    obj_pts, ipl, ipr, img_size = collect_points(left_sharp, right_sharp)

    if len(obj_pts) < MIN_PAIRS:
        print(f"\n[!] Only {len(obj_pts)} valid pairs — need {MIN_PAIRS}.")
        print("    Re-run 01_extract_frames.py with CHESS_STEP=1, then retry.")
        return

    print("\nCalibrating individual cameras...")
    ret_l, K_l, d_l, _, _ = cv2.calibrateCamera(obj_pts, ipl, img_size, None, None)
    ret_r, K_r, d_r, _, _ = cv2.calibrateCamera(obj_pts, ipr, img_size, None, None)
    print(f"  Left  error: {ret_l:.4f} px")
    print(f"  Right error: {ret_r:.4f} px")

    print("Stereo calibration...")
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    ret_s, K_l, d_l, K_r, d_r, R, T, E, F = cv2.stereoCalibrate(
        obj_pts, ipl, ipr, K_l, d_l, K_r, d_r, img_size,
        flags=cv2.CALIB_FIX_INTRINSIC, criteria=crit,
    )
    print(f"  Stereo error: {ret_s:.4f} px")

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K_l, d_l, K_r, d_r, img_size, R, T, alpha=0
    )
    map1_l, map2_l = cv2.initUndistortRectifyMap(K_l, d_l, R1, P1, img_size, cv2.CV_32FC1)
    map1_r, map2_r = cv2.initUndistortRectifyMap(K_r, d_r, R2, P2, img_size, cv2.CV_32FC1)

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "stereo_calib.npz"
    np.savez(out_path,
             K_l=K_l, d_l=d_l, K_r=K_r, d_r=d_r,
             R=R, T=T, E=E, F=F,
             R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
             map1_l=map1_l, map2_l=map2_l,
             map1_r=map1_r, map2_r=map2_r,
             img_size=np.array(img_size))
    print(f"\nSaved: {out_path.resolve()}")
    print(f"Focal length : {P1[0,0]:.1f} px")
    print(f"Baseline     : {abs(float(T.ravel()[0])):.4f} m  (expected ~0.17)")
    if ret_s > 1.5:
        print("[!] High reprojection error — calibration quality may be low.")


if __name__ == "__main__":
    main()
