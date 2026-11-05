"""Shared helpers for running the vision systems on the Original and Mooney image sets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def list_images(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in EXT)


def load_rgb(path, size=None) -> Image.Image:
    """Mooney images are single-channel 0/255 PNGs - every system receives them as 3-channel RGB."""
    img = Image.open(path).convert("RGB")
    if size:
        img = img.resize((size, size), Image.LANCZOS) if isinstance(size, int) else img.resize(size, Image.LANCZOS)
    return img


def base_args(desc):
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--original", required=True, help="folder with the original images")
    ap.add_argument("--mooney", required=True, help="folder with the Mooney images")
    ap.add_argument("--out", required=True, help="result folder (gets Original/ and Mooney/ sub-folders)")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N images (debug)")
    return ap


def sets(a):
    """Yield (set_name, list_of_paths, output_folder) for the two image sets."""
    for name, folder in (("Original", a.original), ("Mooney", a.mooney)):
        files = list_images(folder)
        if a.limit:
            files = files[: a.limit]
        out = Path(a.out) / name
        out.mkdir(parents=True, exist_ok=True)
        yield name, files, out


def save_json(obj, path):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))


def colorize(x, cmap="Spectral_r", mask=None, vmin=None, vmax=None):
    import matplotlib
    cm = matplotlib.colormaps[cmap]
    v = x[mask] if mask is not None and mask.any() else x
    lo = np.percentile(v, 1) if vmin is None else vmin
    hi = np.percentile(v, 99) if vmax is None else vmax
    y = np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)
    rgb = (cm(y)[..., :3] * 255).astype(np.uint8)
    if mask is not None:
        rgb[~mask] = 0
    return rgb
