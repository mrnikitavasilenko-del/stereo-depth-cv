"""
Шаг 5: Собираем коллажи для сравнения двух методов.
Для каждого кадра создаём изображение вида:
  [ Оригинал (левый кадр) | Классический SGBM | Нейросетевой MiDaS ]

Результаты сохраняются в results/comparison/
Первый коллаж показывается на экране через matplotlib.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

LEFT_DIR      = Path("frames/scene/left")
CLASSICAL_DIR = Path("results/classical")   # выход шага 3
NEURAL_DIR    = Path("results/neural")      # выход шага 4
OUT_DIR       = Path("results/comparison")


def resize_to(img, target_h, target_w):
    """Масштабируем изображение до заданных размеров."""
    return cv2.resize(img, (target_w, target_h))


def make_collage(rgb, classical, neural):
    """
    Склеиваем три изображения горизонтально в одну картинку.
    Добавляем текстовую подпись в нижний левый угол каждой панели.
    """
    h, w = rgb.shape[:2]
    # Приводим все три изображения к одному размеру
    classical = resize_to(classical, h, w)
    neural    = resize_to(neural, h, w)

    def add_label(img, text):
        out = img.copy()
        cv2.putText(out, text, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return out

    # hstack — горизонтальная склейка трёх панелей
    row = np.hstack([
        add_label(rgb,       "Original (left)"),
        add_label(classical, "Classical SGBM"),
        add_label(neural,    "Neural MiDaS"),
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
        # Ищем соответствующие карты глубины из шагов 3 и 4
        cl_path = CLASSICAL_DIR / f"{stem}_depth.png"
        nn_path = NEURAL_DIR    / f"{stem}_neural.png"

        # Если какой-то метод не обработал этот кадр — пропускаем
        # (нейросеть обрабатывает каждый 5-й кадр, поэтому пропусков много)
        if not cl_path.exists() or not nn_path.exists():
            missing.append(stem)
            continue

        rgb       = cv2.imread(str(pl))
        classical = cv2.imread(str(cl_path))
        neural    = cv2.imread(str(nn_path))

        collage = make_collage(rgb, classical, neural)
        out_path = OUT_DIR / f"{stem}_compare.png"
        cv2.imwrite(str(out_path), collage)

        if first_collage is None:
            first_collage = collage  # запоминаем первый для показа на экране

    if missing:
        print(f"[!] Skipped {len(missing)} frames (missing classical or neural results)")

    saved = len(left_paths) - len(missing)
    print(f"Saved {saved} comparison images to: {OUT_DIR.resolve()}")

    # Показываем первый коллаж через matplotlib (красивое окно с заголовком)
    if first_collage is not None:
        rgb_display = cv2.cvtColor(first_collage, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(18, 5))
        plt.imshow(rgb_display)
        plt.axis("off")
        plt.title("Original  |  Classical SGBM depth  |  Neural MiDaS depth")
        plt.tight_layout()
        # Сохраняем превью первого кадра отдельным файлом
        plt.savefig(str(OUT_DIR / "first_frame_comparison.png"), dpi=150)
        plt.show()
    else:
        print("[!] No collages created. Check that steps 03 and 04 completed.")


if __name__ == "__main__":
    main()
