"""Shared settings and helpers for Project Part 2 (flyvis: the connectome-constrained fly visual system model).

Protocol (chosen from the timing experiment in simulate.py, see Results/comparison_1_timing.png):
    * pre-stimulus: flyvis default initial state = steady state after 1 s of uniform grey (0.5)
    * every image is shown as a static video (the same frame repeated) for T_STIM = 2 s;
      all 65 cell types are within 2 % of their final state after <= 1.1 s, at 2 s within 0.1 %
    * transients are read out T_TRANSIENT = 120 ms after an image appears (median peak of the
      onset and of the switch transient)
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DT = 0.01                  # integration time step in s (as in flyvis tutorial 7)
T_STIM = 2.0               # presentation time per image in s
T_TRANSIENT = 0.12         # read-out time after an image appears, in s
N_STIM = int(round(T_STIM / DT))
I_TRANSIENT = int(round(T_TRANSIENT / DT)) - 1   # frame index: frame i is (i + 1) * DT after onset
I_STEADY = N_STIM - 1
MODEL = "flow/0000/000"    # best task-performing model of the pretrained ensemble (as in the tutorials)
EXTENT, KERNEL = 15, 13    # BoxEye: 721 receptors, 13 x 13 px boxes (flyvis tutorial 7 defaults)
LUMA = np.array([0.299, 0.587, 0.114], np.float32)   # ITU-R BT.601, same as the Part-1 Mooney tool


def selection(path=ROOT / "work" / "selection.json"):
    """The 10 image pairs selected by hand in Part 1."""
    return json.load(open(path))["top10"]


def load_gray(path):
    """RGB photo -> luminance in [0, 1] (flyvis photoreceptors take one grey value per ommatidium)."""
    return np.asarray(Image.open(path).convert("RGB"), np.float32) @ LUMA / 255.0


def load_mooney(path):
    return np.asarray(Image.open(path).convert("L"), np.float32) / 255.0


def load_pairs(data=ROOT / "data", ids=None):
    """Returns ids, grey-scale originals (n, H, W) and Mooney images (n, H, W)."""
    ids = ids or selection()
    gray = np.stack([load_gray(data / "Original" / f"{k}.jpg") for k in ids])
    moon = np.stack([load_mooney(data / "Mooney" / f"{k}.png") for k in ids])
    return ids, gray, moon


def render(images):
    """Cartesian images (n, H, W) -> photoreceptor input on the hexagonal eye (n, 1, 1, 721).

    BoxEye places 721 receptors 13 px apart (31 columns across) and averages a 13 x 13 px box around
    each one. For our 512 px face crops it samples the central 391 px, so the face fills the eye.
    """
    import torch
    from flyvis.datasets.rendering import BoxEye
    eye = BoxEye(extent=EXTENT, kernel_size=KERNEL)
    return eye(torch.tensor(np.asarray(images, np.float32)[:, None])).float()


def receptor_pixels():
    """(721, 2) pixel coordinates (y, x) of the receptors in a 512 x 512 image."""
    from flyvis.datasets.rendering import BoxEye
    return BoxEye(extent=EXTENT, kernel_size=KERNEL).receptor_centers.numpy() + 256


def face_masks(ids, results=ROOT / "results" / "Result3" / "Original"):
    """Receptors inside the face box found by InsightFace on the original (Part 1, Result3)."""
    c = receptor_pixels()
    out = []
    for k in ids:
        x0, y0, x1, y1 = json.load(open(results / f"{k}.json"))["detections"][0]["bbox"]
        out.append((c[:, 1] >= x0) & (c[:, 1] <= x1) & (c[:, 0] >= y0) & (c[:, 0] <= y1))
    return np.array(out)


def load_network(model=MODEL):
    import flyvis
    return flyvis.NetworkView(flyvis.results_dir / model).init_network()


def cell_groups(connectome):
    """Cell types in network order, their region (retina / intermediate / output) and node indices."""
    nodes = connectome.nodes.to_df()
    layout = dict(connectome.layout[:].astype(str))
    types = list(connectome.unique_cell_types[:].astype(str))
    idx = {t: np.flatnonzero(nodes.type.values == t) for t in types}
    return types, layout, idx, nodes
