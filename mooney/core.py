"""Core Mooney-image algorithm.

A Mooney image is a two-tone (black/white) image of an object under strong,
directional illumination.  Following Egger et al. (Paper 1) we generate it in
three steps:

    1. convert the image to grayscale luminance,
    2. blur it with a Gaussian filter (removes texture and fine detail so that only
       the large-scale shading / shadow pattern survives),
    3. threshold it so that a fixed percentage of the pixels becomes white.

Instead of hand-tuning a threshold per image, the threshold is the luminance
percentile that leaves exactly ``white_percent`` % of the pixels above it.  This is
the closed form of the binary search over thresholds described in the paper.

The two user controls are therefore
    * ``blur``           - strength of the Gaussian blur (sigma), and
    * ``white_percent``  - percentage of white pixels in the result.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import gaussian_filter, label

# ITU-R BT.601 luma weights (identical to PIL's "L" conversion)
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


@dataclass
class MooneyParams:
    blur: float = 1.5            # Gaussian sigma in % of the shorter image side (resolution independent)
    white_percent: float = 40.0  # percentage of pixels that become white
    invert: bool = False         # swap black/white (contrast-polarity experiment)
    min_blob: float = 0.0        # remove black/white islands smaller than this % of the image (0 = off)
    work_size: int = 0           # resize so the longer side has this many px before processing (0 = keep)
    white_reference: str = "image"  # "image": % of all pixels; "object": % of the object mask (3D meshes)

    def sigma_px(self, shape) -> float:
        return self.blur / 100.0 * min(shape[:2])


def load_image(path_or_image) -> np.ndarray:
    """Load an image (path or PIL image) as float32 RGB in [0, 1]; EXIF rotation is respected."""
    img = path_or_image if isinstance(path_or_image, Image.Image) else Image.open(path_or_image)
    img = ImageOps.exif_transpose(img).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


def to_grayscale(rgb: np.ndarray) -> np.ndarray:
    """RGB in [0,1] -> luminance in [0,1]."""
    if rgb.ndim == 2:
        return rgb.astype(np.float32)
    return rgb[..., :3] @ _LUMA


def resize_long_side(rgb: np.ndarray, long_side: int) -> np.ndarray:
    if not long_side or max(rgb.shape[:2]) == long_side:
        return rgb
    h, w = rgb.shape[:2]
    s = long_side / max(h, w)
    img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def remove_small_blobs(binary: np.ndarray, min_pixels: int) -> np.ndarray:
    """Flip connected black or white islands smaller than ``min_pixels`` (despeckling)."""
    if min_pixels <= 0:
        return binary
    out = binary.copy()
    for value in (True, False):
        lab, n = label(out == value)
        if n == 0:
            continue
        sizes = np.bincount(lab.ravel())
        small = sizes < min_pixels
        small[0] = False                       # label 0 = the other colour
        out[small[lab]] = not value
    return out


def make_mooney(image, params: MooneyParams | None = None, mask=None, **kw):
    """Turn an image into a Mooney image.

    Returns ``(mooney, info)`` where ``mooney`` is a uint8 array with values {0, 255}
    and ``info`` is a dict with the parameters used, the threshold and the achieved
    percentage of white pixels.  If ``mask`` (bool HxW) is given and
    ``white_reference == "object"``, the percentage refers to the pixels inside the mask.
    """
    p = params or MooneyParams(**kw)
    rgb = load_image(image) if not isinstance(image, np.ndarray) else image
    rgb = resize_long_side(rgb, p.work_size)

    gray = to_grayscale(rgb)                                   # 1) luminance
    sigma = p.sigma_px(gray.shape)
    blurred = gaussian_filter(gray, sigma) if sigma > 0 else gray   # 2) blur
    use_mask = mask is not None and p.white_reference == "object" and mask.any()
    ref = blurred[mask] if use_mask else blurred
    threshold = np.percentile(ref, 100.0 - p.white_percent)         # 3) percentile threshold
    mooney = blurred > threshold
    if use_mask:
        mooney &= mask

    mooney = remove_small_blobs(mooney, int(p.min_blob / 100.0 * mooney.size))
    if p.invert:
        mooney = ~mooney

    info = asdict(p) | dict(sigma_px=round(float(sigma), 3), threshold=round(float(threshold), 5),
                            white_actual=round(100.0 * float(mooney.mean()), 3),
                            white_in_object=round(100.0 * float(mooney[mask].mean()), 3) if use_mask else None,
                            height=mooney.shape[0], width=mooney.shape[1])
    return (mooney.astype(np.uint8) * 255), info


def blurred_gray(image, params: MooneyParams) -> np.ndarray:
    """Intermediate result (grayscale after blur), used by the GUI for visualisation."""
    rgb = resize_long_side(load_image(image) if not isinstance(image, np.ndarray) else image, params.work_size)
    gray = to_grayscale(rgb)
    s = params.sigma_px(gray.shape)
    return gaussian_filter(gray, s) if s > 0 else gray
