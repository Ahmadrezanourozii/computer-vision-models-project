"""Quality control and ground-truth labelling of the generated portraits.

For every candidate image we measure (independently of the prompt):
  * face detection (InsightFace SCRFD) - exactly one clear face is required,
  * head yaw (InsightFace pose)        - frontal / turned toward image left / right,
  * light direction                     - from the shading gradient over the face:
        L(x, y) ~ a + b*x + c*y  fitted on the face ellipse;  (b, -c) points to the light
        (classic gradient-based light-source estimation, cf. Pentland 1982),
  * lighting contrast                  - how dramatic the lighting is (std/mean on the face).
"""
from __future__ import annotations

import numpy as np

DIRS = {"right": 0.0, "top": 90.0, "left": 180.0, "bottom": -90.0}


def face_ellipse_mask(shape, bbox, shrink=0.9):
    h, w = shape[:2]
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    rx, ry = (x2 - x1) / 2 * shrink, (y2 - y1) / 2 * shrink
    yy, xx = np.mgrid[0:h, 0:w]
    return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0


def light_from_shading(gray, mask):
    """Fit a plane to the luminance inside ``mask``; return angle (deg, 0=right, 90=top) and strength."""
    yy, xx = np.nonzero(mask)
    L = gray[yy, xx].astype(np.float64)
    sx = (xx - xx.mean()) / (np.ptp(xx) + 1e-9)          # normalised coordinates, x right
    sy = -(yy - yy.mean()) / (np.ptp(yy) + 1e-9)         # y up
    A = np.stack([np.ones_like(sx), sx, sy], 1)
    coef, *_ = np.linalg.lstsq(A, L, rcond=None)
    b, c = coef[1], coef[2]
    angle = float(np.degrees(np.arctan2(c, b)))
    strength = float(np.hypot(b, c) / (L.mean() + 1e-6))
    contrast = float(L.std() / (L.mean() + 1e-6))
    return angle, strength, contrast


def nearest_direction(angle):
    diffs = {k: abs((angle - v + 180) % 360 - 180) for k, v in DIRS.items()}
    k = min(diffs, key=diffs.get)
    return k, diffs[k]


def yaw_label(yaw, thr=12.0):
    """InsightFace yaw: the sign convention is calibrated in qc_select.py on known examples."""
    if yaw is None:
        return "unknown"
    if abs(yaw) < thr:
        return "frontal"
    return "turned toward image right" if yaw > 0 else "turned toward image left"


def analyse(app, img_bgr):
    """Run detection + lighting analysis on one BGR image; returns a dict of measurements."""
    faces = sorted(app.get(img_bgr), key=lambda f: -f.det_score)
    out = dict(n_faces=sum(f.det_score >= 0.5 for f in faces), det_score=float(faces[0].det_score) if faces else 0.0)
    if not faces:
        return out
    f = faces[0]
    gray = img_bgr[..., ::-1].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32) / 255.0
    mask = face_ellipse_mask(gray.shape, f.bbox)
    angle, strength, contrast = light_from_shading(gray, mask)
    d, err = nearest_direction(angle)
    pitch, yaw, roll = (f.pose.tolist() if f.get("pose") is not None else (None, None, None))
    out.update(bbox=[float(v) for v in f.bbox], face_frac=float(mask.mean()), light_angle=angle, light_strength=strength,
               light_dir=d, light_dir_err=err, face_contrast=contrast, yaw=yaw, pitch=pitch, roll=roll,
               sex=f.get("sex"), age=int(f.age) if f.get("age") is not None else None)
    return out
