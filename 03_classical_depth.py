"""
Шаг 3: Классическая карта глубины через SGBM + WLS-фильтр.
Читает калибровку из calibration/stereo_calib.npz
Сохраняет карты глубины в results/classical/

Пайплайн:
  1. Ректификация — устраняем дисторсию и выравниваем строки левого/правого кадра
  2. CLAHE — усиливаем контраст перед поиском диспаритета
  3. SGBM — ищем диспаритет (смещение пикселей между левым и правым кадром)
  4. WLS-фильтр — сглаживаем диспаритет с учётом краёв, заполняем дыры
  5. Глубина: Z = f * B / d  (фокус * база / диспаритет)
  6. EMA — временное сглаживание между соседними кадрами
"""
import cv2
import numpy as np
from pathlib import Path

CALIB_FILE = Path("calibration/stereo_calib.npz")
LEFT_DIR   = Path("frames/scene/left")
RIGHT_DIR  = Path("frames/scene/right")
OUT_DIR    = Path("results/classical")

BASELINE = 0.17  # расстояние между камерами в метрах (17 см)

# Экспоненциальное скользящее среднее: 0 = полное сглаживание, 1 = без сглаживания
EMA_ALPHA = 0.4

# Параметры SGBM — чем больше NUM_DISPARITIES, тем дальше видит (но медленнее)
NUM_DISPARITIES = 128   # должно делиться на 16
BLOCK_SIZE      = 7     # меньше = больше деталей, больше шума
P1 = 8  * 3 * BLOCK_SIZE ** 2   # штраф за малое изменение диспаритета
P2 = 32 * 3 * BLOCK_SIZE ** 2   # штраф за большое изменение диспаритета

# WLS: lambda — сила сглаживания, sigma — чувствительность к краям
WLS_LAMBDA = 8000
WLS_SIGMA  = 1.5

# CLAHE — адаптивная эквализация гистограммы (улучшает контраст локально)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def load_calib():
    """Загружаем карты ректификации и матрицу проекции из файла калибровки."""
    d = np.load(CALIB_FILE)
    return d["map1_l"], d["map2_l"], d["map1_r"], d["map2_r"], d["P1"]


def depth_colormap(depth_m, max_depth=5.0):
    """
    Переводим метрическую глубину в цветную карту JET:
    красный = близко, синий = далеко. Обрезаем по max_depth метров.
    """
    depth_clipped = np.clip(depth_m, 0, max_depth)
    norm = (depth_clipped / max_depth * 255).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_JET)


def main():
    if not CALIB_FILE.exists():
        print("[!] Calibration file not found. Run 02_calibrate.py first.")
        return

    map1_l, map2_l, map1_r, map2_r, P1_mat = load_calib()
    focal = float(P1_mat[0, 0])  # фокусное расстояние в пикселях
    print(f"Focal: {focal:.1f} px  Baseline: {BASELINE} m")

    # Левый матчер ищет диспаритет слева направо
    matcher_left = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=NUM_DISPARITIES,
        blockSize=BLOCK_SIZE,
        P1=P1, P2=P2,
        disp12MaxDiff=1,        # максимальная разница при перекрёстной проверке
        uniquenessRatio=10,     # фильтр неоднозначных совпадений
        speckleWindowSize=100,  # размер окна фильтрации артефактов
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    # Правый матчер — зеркальный, нужен для WLS-фильтра (проверка согласованности)
    matcher_right = cv2.ximgproc.createRightMatcher(matcher_left)

    # WLS-фильтр: сглаживает диспаритет там где текстура слабая,
    # но сохраняет чёткие края
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

        # Ректификация: убираем дисторсию и выравниваем строки по вертикали
        # После ректификации одинаковые точки сцены лежат на одной горизонтали
        rect_l = cv2.remap(img_l, map1_l, map2_l, cv2.INTER_LINEAR)
        rect_r = cv2.remap(img_r, map1_r, map2_r, cv2.INTER_LINEAR)

        # CLAHE улучшает поиск соответствий в тёмных и засвеченных областях
        gray_l = clahe.apply(cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY))
        gray_r = clahe.apply(cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY))

        # Вычисляем диспаритет в обоих направлениях
        disp_left  = matcher_left.compute(gray_l, gray_r)
        disp_right = matcher_right.compute(gray_r, gray_l)

        # WLS объединяет оба диспаритета для лучшего результата
        disp_filtered = wls.filter(disp_left, rect_l, None, disp_right)

        # SGBM возвращает фиксированную точку (×16) → переводим во float
        disp_f = disp_filtered.astype(np.float32) / 16.0

        # Формула глубины: Z = f * B / d
        # Где d ≤ 1 — невалидный пиксель (нет совпадения), ставим 0
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = np.where(disp_f > 1.0, focal * BASELINE / disp_f, 0)

        color = depth_colormap(depth)
        stem = pl.stem
        cv2.imwrite(str(OUT_DIR / f"{stem}_depth.png"), color)
        np.save(str(OUT_DIR / f"{stem}_depth_raw.npy"), depth)  # сырые данные для шага 5

    print(f"Done. Saved to: {OUT_DIR.resolve()}")
    temporal_smooth()


def temporal_smooth():
    """
    Второй проход: экспоненциальное сглаживание между соседними кадрами.
    Убирает мерцание карты глубины от кадра к кадру.
    EMA: smoothed[t] = alpha * depth[t] + (1-alpha) * smoothed[t-1]
    """
    raw_files = sorted(OUT_DIR.glob("*_depth_raw.npy"))
    if not raw_files:
        return

    print(f"\nTemporal smoothing (EMA alpha={EMA_ALPHA}) over {len(raw_files)} frames...")
    prev = None
    for path in raw_files:
        depth = np.load(str(path))
        if prev is None:
            smoothed = depth
        else:
            # Смешиваем только там где оба кадра имеют валидную глубину
            valid = (depth > 0) & (prev > 0)
            smoothed = np.where(valid,
                                EMA_ALPHA * depth + (1 - EMA_ALPHA) * prev,
                                depth)
        prev = smoothed

        # Перезаписываем цветную PNG сглаженной версией
        color_path = str(path).replace("_depth_raw.npy", "_depth.png")
        cv2.imwrite(color_path, depth_colormap(smoothed))

    print("Temporal smoothing done.")


if __name__ == "__main__":
    main()
