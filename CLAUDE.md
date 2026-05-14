# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Stereo 3D computer vision project using a Raspberry Pi dual-camera setup. The directory contains captured stereo video samples (side-by-side format) and notes on camera operation.

## Hardware Setup

- **Baseline (B):** 17 cm between cameras
- **Working distance (Z):** 210–310 cm
- **Calibration target:** 35 mm chessboard squares

## Key Commands

### Remote access
```bash
xhost +
ssh -X user@10.5.20.232
```

### Record stereo video (side-by-side)
```bash
raspivid -3d sbs -fps 60 -3dswap -rot 180 -md 6 -t 20000 -w $(( 640 * 2 )) -h 480 -n -o - \
  | sudo ffmpeg -r 60 -i - -y -vcodec copy "$(date '+%H%M%S%Y').mp4"
```

### Stream/translate raw YUV stereo frames
```bash
raspividyuv -3d sbs -fps 3 -3dswap -rot 180 -md 4 -t 0 -w $(( 1296 * 2 )) -h 972 -n -o - | ./a.out
```
`a.out` is the compiled processing binary that consumes the raw YUV pipe.

## Depth Reconstruction Pipeline

Five scripts run in order:

| Script | Purpose | Runtime |
|--------|---------|---------|
| `01_extract_frames.py` | Split SBS video → `frames/chess/` and `frames/scene/` | ~1 min |
| `02_calibrate.py` | Stereo calibration from chessboard frames | ~2 min |
| `03_classical_depth.py` | SGBM + WLS filter disparity → metric depth maps | ~5 min |
| `04_neural_depth.py` | MiDaS monocular depth (every 5th frame on CPU) | ~8 min |
| `05_compare.py` | Side-by-side collages: original / SGBM / MiDaS | ~1 min |

```powershell
pip install -r requirements.txt
python 01_extract_frames.py
python 02_calibrate.py
python 03_classical_depth.py
python 04_neural_depth.py
python 05_compare.py
```

### Calibration results (`calibration/stereo_calib.npz`)
- Reprojection error: **0.38 px** (excellent)
- Focal length: **792 px**
- Baseline measured: **0.1708 m** (matches hardware 0.17 m)
- Board: 9×6 inner corners, 35 mm squares, 50 frames used

### Classical depth improvements (v2)
`03_classical_depth.py` uses CLAHE pre-processing + WLS filter (Weighted Least Squares):
- Computes left→right and right→left disparities, cross-checks them
- Edge-aware smoothing via WLS (λ=8000, σ=1.5) fills holes and reduces noise
- Tune `WLS_LAMBDA` (smoothness) and `WLS_SIGMA` (edge sharpness) in the script

### Known constraints
- Python 3.14 has no CUDA PyTorch build → MiDaS runs on CPU with `MiDaS_small`
- With a Python 3.11/3.12 + CUDA environment, change `MODEL_TYPE = "DPT_Large"` in `04_neural_depth.py` for higher quality

## Directory Contents

- `chess/` — stereo SBS calibration videos (chessboard)
- `strereo/` — stereo SBS scene video for depth reconstruction
- `calibration/` — saved calibration data (`stereo_calib.npz`)
- `results/classical/` — SGBM depth maps (metric, COLORMAP_JET)
- `results/neural/` — MiDaS depth maps (relative, COLORMAP_MAGMA)
- `results/comparison/` — side-by-side collages
- `Lecture_7_3D(1).pptx` — course lecture slides on 3D vision
