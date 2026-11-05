"""Vision system 1: monocular depth estimation with Depth Anything V2 (Large).

For every image we save
  <id>_depth.png   colour-coded relative depth (near = red/yellow, far = blue)
  <id>_relief.png  the predicted depth rendered as a lit 3D relief (what 3D shape the model "sees")
and for all images of a set
  depth_raw.npz    predicted relative inverse depth (float16, 256x256) for the quantitative analysis
"""
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from common import base_args, colorize, load_rgb, sets

MODEL = "depth-anything/Depth-Anything-V2-Large-hf"


def relief(disp, light=(-0.5, 0.6, 0.62)):
    """Shade the predicted surface (disparity treated as height) with a directional light."""
    d = (disp - disp.min()) / (np.ptp(disp) + 1e-9) * disp.shape[0] * 0.25
    gy, gx = np.gradient(d)
    n = np.dstack([-gx, gy, np.ones_like(d)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    l = np.array(light) / np.linalg.norm(light)
    shade = np.clip(n @ l, 0, 1) ** 0.8
    return (shade * 255).astype(np.uint8)


def main():
    a = base_args(__doc__).parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained(MODEL)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL, torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
    for name, files, out in sets(a):
        raw = {}
        t0 = time.time()
        for f in files:
            img = load_rgb(f)
            inputs = proc(images=img, return_tensors="pt").to(dev)
            with torch.no_grad():
                pred = model(pixel_values=inputs["pixel_values"].to(model.dtype)).predicted_depth  # (1,h,w) relative inverse depth
            pred = pred.float()[None]
            full = F.interpolate(pred, size=(512, 512), mode="bicubic", align_corners=False)[0, 0].cpu().numpy()
            small = F.interpolate(pred, size=(256, 256), mode="area")[0, 0].cpu().numpy()
            raw[f.stem] = small.astype(np.float16)
            Image.fromarray(colorize(full)).save(out / f"{f.stem}_depth.png", optimize=True)
            Image.fromarray(relief(full)).save(out / f"{f.stem}_relief.png", optimize=True)
        np.savez_compressed(out / "depth_raw.npz", **raw)
        print(f"{name}: {len(files)} images in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
