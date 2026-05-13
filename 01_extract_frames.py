"""
Extract left/right frames from side-by-side stereo videos.
  chess/ videos  -> frames/chess/left/  frames/chess/right/
  strereo/ video -> frames/scene/left/  frames/scene/right/
"""
import cv2
import os
from pathlib import Path

CHESS_DIR = "chess"
SCENE_VIDEO = "strereo/1954522024.mp4"
OUT_BASE = Path("frames")
CHESS_STEP = 3   # for calibration: lower = more frames (use 1 if calibration fails)
SCENE_STEP = 1   # for depth: keep every frame


def split_sbs(frame):
    w = frame.shape[1] // 2
    return frame[:, :w], frame[:, w:]


def extract(video_path, out_left, out_right, step=CHESS_STEP):
    out_left.mkdir(parents=True, exist_ok=True)
    out_right.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[!] Cannot open {video_path}")
        return 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    saved = 0
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            left, right = split_sbs(frame)
            name = f"{saved:04d}.png"
            cv2.imwrite(str(out_left / name), left)
            cv2.imwrite(str(out_right / name), right)
            saved += 1
        idx += 1

    cap.release()
    print(f"  {video_path}: {saved} frames saved (total {total})")
    return saved


def main():
    # Calibration videos (chessboard)
    chess_videos = sorted(Path(CHESS_DIR).glob("*.mp4"))
    print(f"Chess videos: {len(chess_videos)}")
    for vid in chess_videos:
        extract(
            vid,
            OUT_BASE / "chess" / "left",
            OUT_BASE / "chess" / "right",
            step=CHESS_STEP,
        )

    # Scene video
    print(f"\nScene video: {SCENE_VIDEO}")
    extract(
        Path(SCENE_VIDEO),
        OUT_BASE / "scene" / "left",
        OUT_BASE / "scene" / "right",
        step=SCENE_STEP,
    )

    print("\nDone. Frames saved to:", OUT_BASE.resolve())


if __name__ == "__main__":
    main()
