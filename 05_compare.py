"""
Side-by-side comparison of classical (SGBM) vs neural (MiDaS) depth maps.
Saves grid images to results/comparison/ and shows the first one.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

LEFT_DIR = Path("frames/scene/left")
CLASSICAL_DIR = Path("results/classical")
NEURAL_DIR = Path("results/neural")
OUT_DIR = Path("results/comparison")


def resize_to(img, target_h, target_w):
    return cv2.resize(img, (target_w, target_h))


def make_collage(rgb, classical, neural, label_h=30):
    h, w = rgb.shape[:2]
    classical = resize_to(classical, h, w)
    neural = resize_to(neural, h, w)

    def add_label(img, text):
        out = img.copy()
        cv2.putText(out, text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return out

    row = np.hstack([
        add_label(rgb, "Original (left)"),
        add_label(classical, "Classical SGBM"),
        add_label(neural, "Neural MiDaS"),
    ])
    return row


def main():
    left_paths = sorted(LEFT_DIR.glob("*.png"))
    if not left_paths:
        print("[!] No scene frames. Run 01_extract_frames.py first.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    first_collage = None
    missing = []

    for pl in left_paths:
        stem = pl.stem
        cl_path = CLASSICAL_DIR / f"{stem}_depth.png"
        nn_path = NEURAL_DIR / f"{stem}_neural.png"

        if not cl_path.exists() or not nn_path.exists():
            missing.append(stem)
            continue

        rgb = cv2.imread(str(pl))
        classical = cv2.imread(str(cl_path))
        neural = cv2.imread(str(nn_path))

        collage = make_collage(rgb, classical, neural)
        out_path = OUT_DIR / f"{stem}_compare.png"
        cv2.imwrite(str(out_path), collage)

        if first_collage is None:
            first_collage = collage

    if missing:
        print(f"[!] Skipped {len(missing)} frames (missing classical or neural results)")

    saved = len(left_paths) - len(missing)
    print(f"Saved {saved} comparison images to: {OUT_DIR.resolve()}")

    if first_collage is not None:
        rgb_display = cv2.cvtColor(first_collage, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(18, 5))
        plt.imshow(rgb_display)
        plt.axis("off")
        plt.title("Original  |  Classical SGBM depth  |  Neural MiDaS depth")
        plt.tight_layout()
        plt.savefig(str(OUT_DIR / "first_frame_comparison.png"), dpi=150)
        plt.show()
    else:
        print("[!] No collages created. Check that steps 03 and 04 completed.")


if __name__ == "__main__":
    main()
