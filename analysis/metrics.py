"""Quantitative comparison of vision-system outputs on Original vs. Mooney images.

All metrics are computed inside a face mask (ellipse inside the face box detected on the
ORIGINAL image), so the large black background does not dominate the numbers.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import pearsonr, spearmanr

# ----------------------------------------------------------------------------- helpers
def ellipse_mask(bbox, img_size, out=256, shrink=0.9):
    s = out / img_size
    x1, y1, x2, y2 = [v * s for v in bbox]
    yy, xx = np.mgrid[0:out, 0:out]
    cx, cy, rx, ry = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) / 2 * shrink, (y2 - y1) / 2 * shrink
    return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1


def load_gray(path, size=256):
    return np.asarray(Image.open(path).convert("L").resize((size, size), Image.BILINEAR), np.float32) / 255.0


def normals_from_depth(d):
    gy, gx = np.gradient(d.astype(np.float64))
    n = np.dstack([-gx, gy, np.ones_like(d) * (1.0 / d.shape[0])])
    return n / np.linalg.norm(n, axis=2, keepdims=True)


def angular_error(n1, n2, mask):
    dot = np.clip((n1 * n2).sum(-1), -1, 1)
    return np.degrees(np.arccos(dot))[mask]


def zscore(x, m):
    v = x[m]
    return (x - v.mean()) / (v.std() + 1e-9)


# ----------------------------------------------------------------------------- system 1: depth
def depth_metrics(res_dir, meta, mooney_dir):
    O = np.load(Path(res_dir) / "Original" / "depth_raw.npz")
    M = np.load(Path(res_dir) / "Mooney" / "depth_raw.npz")
    rows = []
    for _, r in meta.iterrows():
        k = r["id"]
        if k not in O or k not in M:
            continue
        m = ellipse_mask(r["bbox"], r["img_size"])
        do, dm = O[k].astype(np.float64), M[k].astype(np.float64)
        moon = load_gray(Path(mooney_dir) / f"{k}.png")
        # scale/shift alignment of the Mooney depth to the original depth (relative depth has no scale)
        A = np.stack([dm[m], np.ones(m.sum())], 1)
        (s, t), *_ = np.linalg.lstsq(A, do[m], rcond=None)
        dm_al = s * dm + t
        rel = np.abs(dm_al[m] - do[m]).mean() / (np.abs(do[m] - do[m].mean()).mean() + 1e-9)
        ang = angular_error(normals_from_depth(zscore(do, m)), normals_from_depth(zscore(dm_al, m)), m)
        r_dm_moon = pearsonr(dm[m], moon[m])[0] if moon[m].std() > 0 else np.nan
        r_do_moon = pearsonr(do[m], moon[m])[0] if moon[m].std() > 0 else np.nan
        r_om = pearsonr(do[m], dm[m])[0]
        # partial correlation: does Mooney depth follow the white/black pattern beyond the true shape?
        X = np.stack([do[m], np.ones(m.sum())], 1)
        res_dm = dm[m] - X @ np.linalg.lstsq(X, dm[m], rcond=None)[0]
        res_mo = moon[m] - X @ np.linalg.lstsq(X, moon[m], rcond=None)[0]
        partial = pearsonr(res_dm, res_mo)[0] if res_mo.std() > 0 else np.nan
        rows.append(dict(id=k, light=r["light"], depth_r=r_om, depth_spearman=spearmanr(do[m], dm[m])[0],
                         depth_relerr=rel, normal_err_med=float(np.median(ang)), r_mooneydepth_vs_pattern=r_dm_moon,
                         r_origdepth_vs_pattern=r_do_moon, partial_r_pattern=partial))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- system 2: VLM
FACE_WORDS = re.compile(r"\b(face|faces|person|people|man|woman|men|women|boy|girl|portrait|head|human|guy|lady|"
                        r"profile|someone|individual|self-portrait|selfie)\b", re.I)


def norm_ans(x):
    return (str(x).strip().lower() if x is not None else "missing")


def vlm_table(res_dir):
    out = {}
    for s in ("Original", "Mooney"):
        rows = []
        for f in sorted((Path(res_dir) / s).glob("*.json")):
            d = json.load(open(f))
            st = d.get("structured") or {}
            rows.append(dict(id=Path(d["image"]).stem, description=d["description"],
                             mentions_face=bool(FACE_WORDS.search(d["description"])),
                             **{k: norm_ans(st.get(k)) for k in ("contains_face", "gender", "age_group", "face_orientation",
                                                                  "light_direction", "black_regions_are")},
                             face_confidence=pd.to_numeric(st.get("face_confidence"), errors="coerce")))
        out[s] = pd.DataFrame(rows).set_index("id")
    return out


LIGHT_GT = {"left": "from the left", "right": "from the right", "top": "from above", "bottom": "from below"}


def vlm_metrics(tabs, meta):
    gt = meta.set_index("id")[["light", "gender", "age", "pose_label"]].add_prefix("gt_")
    res = {}
    for s, t in tabs.items():
        t = t.join(gt, how="inner")
        res[s] = dict(
            n=len(t),
            mentions_face=t.mentions_face.mean(),
            says_face=(t.contains_face == "yes").mean(),
            face_conf=t.face_confidence.mean(),
            gender_acc=(t.gender == t.gt_gender).mean(),
            age_acc=(t.age_group == t.gt_age).mean(),
            light_acc=(t.light_direction == t.gt_light.map(LIGHT_GT)).mean(),
            orientation_acc=(t.face_orientation == t.gt_pose_label).mean(),
            black_is_shadow=(t.black_regions_are == "mostly shadows").mean(),
        )
    return pd.DataFrame(res)


# ----------------------------------------------------------------------------- system 3: faces
def face_metrics(res_dir, meta):
    base = Path(res_dir)
    so = pd.read_csv(base / "Original" / "summary.csv").set_index("image")
    sm = pd.read_csv(base / "Mooney" / "summary.csv").set_index("image")
    eo = dict(np.load(base / "Original" / "embeddings.npz"))
    em = {k: v for k, v in np.load(base / "Mooney" / "embeddings.npz").items()
          if k in sm.index and bool(sm.loc[k, "detected"])}          # only real detections (score >= 0.5)
    oracle_path = base / "Mooney" / "embeddings_oracle_aligned.npz"
    eor = dict(np.load(oracle_path)) if oracle_path.exists() else {}
    ids = [k for k in meta["id"] if k in eo]
    G = np.stack([eo[k] for k in ids])                       # gallery: originals

    def ident(embs):
        ranks, sims = [], []
        for i, k in enumerate(ids):
            if k not in embs:
                ranks.append(np.nan)
                sims.append(np.nan)
                continue
            c = G @ embs[k]
            sims.append(c[i])
            ranks.append(int((c > c[i]).sum()) + 1)
        return np.array(ranks, float), np.array(sims, float)

    r_det, s_det = ident(em)
    r_or, s_or = ident(eor)
    imp = (G @ G.T)[~np.eye(len(ids), dtype=bool)]
    return dict(det_rate_orig=so.detected.mean(), det_rate_mooney=sm.detected.mean(),
                score_orig=so.best_score.mean(), score_mooney=sm.best_score.mean(),
                rank1_detected=np.nanmean(r_det == 1) if np.isfinite(r_det).any() else np.nan,
                rank1_detected_over_all=np.nansum(r_det == 1) / len(ids),
                rank1_oracle=np.nanmean(r_or == 1) if np.isfinite(r_or).any() else np.nan,
                mean_rank_oracle=np.nanmean(r_or), cos_same_oracle=np.nanmean(s_or), cos_same_detected=np.nanmean(s_det),
                cos_impostor_orig=float(imp.mean())), (r_det, s_det, r_or, s_or, ids)


# ----------------------------------------------------------------------------- system 4: Marigold
def marigold_metrics(res_dir, meta, mooney_dir, eps=1e-3):
    O = np.load(Path(res_dir) / "Original" / "marigold_raw.npz")
    M = np.load(Path(res_dir) / "Mooney" / "marigold_raw.npz")
    rows = []
    for _, r in meta.iterrows():
        k = r["id"]
        if f"{k}__normals" not in O or f"{k}__normals" not in M:
            continue
        m = ellipse_mask(r["bbox"], r["img_size"])
        no = O[f"{k}__normals"].astype(np.float64)
        nm = M[f"{k}__normals"].astype(np.float64)
        no /= np.linalg.norm(no, axis=2, keepdims=True) + 1e-9
        nm /= np.linalg.norm(nm, axis=2, keepdims=True) + 1e-9
        moon = load_gray(Path(mooney_dir) / f"{k}.png") > 0.5
        W, B = m & moon, m & ~moon
        row = dict(id=k, light=r["light"], normal_err_med=float(np.median(angular_error(no, nm, m))))
        if W.sum() > 20 and B.sum() > 20:
            for s, P in (("orig", O), ("mooney", M)):
                A = P[f"{k}__albedo"].astype(np.float64).mean(-1)
                S = P[f"{k}__shading"].astype(np.float64)
                S = S.mean(-1) if S.ndim == 3 else S
                dS = np.log(S[W] + eps).mean() - np.log(S[B] + eps).mean()
                dA = np.log(A[W] + eps).mean() - np.log(A[B] + eps).mean()
                row[f"shadow_index_{s}"] = dS / (abs(dS) + abs(dA) + 1e-9)
                row[f"dlogS_{s}"], row[f"dlogA_{s}"] = dS, dA
        rows.append(row)
    return pd.DataFrame(rows)
