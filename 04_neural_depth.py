"""
Monocular depth estimation using MiDaS.
Input:  frames/scene/left/*.png
Output: results/neural/  (colourised depth maps + raw .npy)

Model selection:
  - If CUDA available: DPT_Large  (best quality)
  - If CPU only:       MiDaS_small (fast enough for FRAME_STEP subsampling)
"""
import cv2
import numpy as np
import torch
import torch.hub
from pathlib import Path

# Auto-trust all hub repos to avoid interactive prompts
_orig_check = torch.hub._check_repo_is_trusted
torch.hub._check_repo_is_trusted = lambda *a, **kw: None

LEFT_DIR = Path("frames/scene/left")
OUT_DIR  = Path("results/neural")

FRAME_STEP_CPU = 5   # on CPU process every N-th frame to keep runtime reasonable


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        model_type = "DPT_Large"
        transform_key = "dpt_transform"
    else:
        model_type = "MiDaS_small"
        transform_key = "small_transform"

    print(f"Device : {device}  ({torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'})")
    print(f"Model  : {model_type}")

    model     = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms",  trust_repo=True)
    transform  = getattr(transforms, transform_key)

    model.to(device).eval()
    return model, transform, device, model_type


def depth_colormap(depth):
    norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return cv2.applyColorMap(norm, cv2.COLORMAP_MAGMA)


def main():
    left_paths = sorted(LEFT_DIR.glob("*.png"))
    if not left_paths:
        print("[!] No scene frames found. Run 01_extract_frames.py first.")
        return

    print(f"Loading MiDaS...")
    model, transform, device, model_type = load_model()

    step = 1 if device.type == "cuda" else FRAME_STEP_CPU
    selected = left_paths[::step]
    print(f"Processing {len(selected)} frames (step={step} of {len(left_paths)} total)...")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for i, path in enumerate(selected):
            img_bgr = cv2.imread(str(path))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            input_tensor = transform(img_rgb).to(device)
            prediction   = model(input_tensor)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

            depth = prediction.cpu().numpy()
            color = depth_colormap(depth)

            stem = path.stem
            cv2.imwrite(str(OUT_DIR / f"{stem}_neural.png"), color)
            np.save(str(OUT_DIR / f"{stem}_neural_raw.npy"), depth)

            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(selected)} done")

    print(f"Saved {len(selected)} depth maps to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
