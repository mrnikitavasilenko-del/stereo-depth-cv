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

## Directory Contents

- `chess/` — stereo video recordings of the chessboard calibration target
- `strereo/` — stereo video recordings of the scene
- `Lecture_7_3D(1).pptx` — course lecture slides on 3D vision
