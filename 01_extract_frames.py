"""
Шаг 1: Извлечь кадры из стерео-видео формата Side-By-Side (SBS).
В SBS-видео левый и правый кадр склеены горизонтально в одно изображение —
скрипт разрезает каждый кадр пополам и сохраняет отдельно.

Выходные папки:
  frames/chess/left  + frames/chess/right  — кадры для калибровки (шахматная доска)
  frames/scene/left  + frames/scene/right  — кадры сцены для карт глубины
"""
import cv2
import os
from pathlib import Path

CHESS_DIR   = "chess"                    # папка с SBS-видео шахматной доски
SCENE_VIDEO = "strereo/1954522024.mp4"  # SBS-видео сцены (одно видео)
OUT_BASE    = Path("frames")

CHESS_STEP = 3  # брать каждый N-й кадр для калибровки (меньше = больше кадров, точнее калибровка)
SCENE_STEP = 1  # брать каждый кадр сцены (1 = максимальное разрешение по времени)


def split_sbs(frame):
    """Разрезает SBS-кадр пополам по горизонтали → (левый, правый)."""
    w = frame.shape[1] // 2
    return frame[:, :w], frame[:, w:]


def extract(video_path, out_left, out_right, step=CHESS_STEP):
    """
    Читает видео, разрезает каждый N-й кадр на левый/правый и сохраняет как PNG.
    Возвращает количество сохранённых пар кадров.
    """
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
            name = f"{saved:04d}.png"  # нумерация с нулями: 0000.png, 0001.png ...
            cv2.imwrite(str(out_left / name), left)
            cv2.imwrite(str(out_right / name), right)
            saved += 1
        idx += 1

    cap.release()
    print(f"  {video_path}: {saved} frames saved (total {total})")
    return saved


def main():
    # Извлекаем кадры из всех видео шахматной доски (их может быть несколько)
    chess_videos = sorted(Path(CHESS_DIR).glob("*.mp4"))
    print(f"Chess videos: {len(chess_videos)}")
    for vid in chess_videos:
        extract(vid,
                OUT_BASE / "chess" / "left",
                OUT_BASE / "chess" / "right",
                step=CHESS_STEP)

    # Извлекаем кадры из видео сцены
    print(f"\nScene video: {SCENE_VIDEO}")
    extract(Path(SCENE_VIDEO),
            OUT_BASE / "scene" / "left",
            OUT_BASE / "scene" / "right",
            step=SCENE_STEP)

    print("\nDone. Frames saved to:", OUT_BASE.resolve())


if __name__ == "__main__":
    main()
