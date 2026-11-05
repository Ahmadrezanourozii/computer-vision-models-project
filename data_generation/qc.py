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


def disk(shape, c, r):
    h, w = shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    return (xx - c[0]) ** 2 + (yy - c[1]) ** 2 <= r * r


def light_from_landmarks(gray, kps):
    """Second, anatomical light-direction estimate from 5 key points (eyes, nose, mouth corners).

    Mean luminance of cheek / forehead / chin disks -> left-right and top-bottom contrasts.
    Returns (direction, lr, tb) with lr > 0 = image-left side brighter, tb > 0 = forehead brighter.
    """
    k = np.asarray(kps, float)
    eyes = k[np.argsort(k[:2, 0])]                        # image-left eye first
    mouth = k[3:5][np.argsort(k[3:5, 0])]
    d = np.linalg.norm(eyes[1] - eyes[0]) + 1e-6           # inter-ocular distance (px)
    centre = eyes.mean(0)
    up = centre - (mouth.mean(0) - centre) / (np.linalg.norm(mouth.mean(0) - centre) + 1e-6) * d * 0.6
    chin = mouth.mean(0) + (mouth.mean(0) - centre) / (np.linalg.norm(mouth.mean(0) - centre) + 1e-6) * d * 0.45
    cheek_l = (eyes[0] + mouth[0]) / 2 + (eyes[0] - eyes[1]) / d * d * 0.15
    cheek_r = (eyes[1] + mouth[1]) / 2 + (eyes[1] - eyes[0]) / d * d * 0.15
    m = lambda c, r: float(gray[disk(gray.shape, c, r)].mean())
    Ll, Lr, Lt, Lb = m(cheek_l, 0.22 * d), m(cheek_r, 0.22 * d), m(up, 0.28 * d), m(chin, 0.22 * d)
    lr = (Ll - Lr) / (Ll + Lr + 1e-6)
    tb = (Lt - Lb) / (Lt + Lb + 1e-6)
    if abs(lr) >= abs(tb):
        direction = "left" if lr > 0 else "right"
    else:
        direction = "top" if tb > 0 else "bottom"
    return direction, lr, tb


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


def border_brightness(gray, frac=0.03):
    """Mean luminance of a thin frame around the image (generation artefacts such as white bars)."""
    h, w = gray.shape
    b = max(2, int(frac * min(h, w)))
    band = np.concatenate([gray[:b].ravel(), gray[-b:].ravel(), gray[:, :b].ravel(), gray[:, -b:].ravel()])
    return float(band.mean())


def analyse(app, img_bgr):
    """Run detection + lighting analysis on one BGR image; returns a dict of measurements."""
    faces = sorted(app.get(img_bgr), key=lambda f: -f.det_score)
    g = img_bgr[..., ::-1].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32) / 255.0
    out = dict(n_faces=sum(f.det_score >= 0.5 for f in faces), det_score=float(faces[0].det_score) if faces else 0.0,
               border=border_brightness(g))
    if not faces:
        return out
    f = faces[0]
    gray = img_bgr[..., ::-1].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32) / 255.0
    mask = face_ellipse_mask(gray.shape, f.bbox)
    angle, strength, contrast = light_from_shading(gray, mask)
    d, err = nearest_direction(angle)
    pitch, yaw, roll = (f.pose.tolist() if f.get("pose") is not None else (None, None, None))
    lm_dir, lr, tb = light_from_landmarks(gray, f.kps)
    out.update(bbox=[float(v) for v in f.bbox], face_frac=float(mask.mean()), light_angle=angle, light_strength=strength,
               light_dir=d, light_dir_err=err, light_dir_landmarks=lm_dir, lm_lr=lr, lm_tb=tb,
               face_contrast=contrast, yaw=yaw, pitch=pitch, roll=roll,
               est_sex=f.get("sex"), est_age=int(f.age) if f.get("age") is not None else None)
    return out
