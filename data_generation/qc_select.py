"""Quality control: pick one portrait per design cell and write the final Original/ set.

    python data_generation/qc_select.py --candidates work/candidates --out data --report work/qc

For every candidate: InsightFace detection (exactly one clear face), measured light direction
(shading gradient, see qc.py) must match the designed direction, and among valid candidates the
one with the most dramatic lighting (largest normalised shading gradient) wins.
Writes data/Original/face_XXX.jpg (shuffled order) and data/metadata.csv (ground-truth labels).
"""
import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from qc import analyse, yaw_label  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", nargs="+", required=True, help="folders with cXXX_k.png + meta.json")
    ap.add_argument("--out", default="data")
    ap.add_argument("--report", default="work/qc")
    ap.add_argument("--max-angle-err", type=float, default=50.0)
    ap.add_argument("--min-det", type=float, default=0.6)
    ap.add_argument("--exclude", nargs="*", default=[], help="candidate files rejected by visual inspection")
    ap.add_argument("--reuse", action="store_true", help="reuse measurements from REPORT/candidates_qc.csv")
    a = ap.parse_args()

    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.3)

    meta = []
    for folder in a.candidates:
        meta += [dict(m, folder=folder) for m in json.load(open(Path(folder) / "meta.json"))]
    cache = {}
    cache_file = Path(a.report) / "candidates_qc.csv"
    if a.reuse and cache_file.exists():
        old = pd.read_csv(cache_file)
        meas = [c for c in old.columns if c not in meta[0] and c not in ("folder", "angle_err", "valid", "landmarks_agree")]
        for _, rr in old.iterrows():
            d = {c: rr[c] for c in meas}
            if isinstance(d.get("bbox"), str):
                d["bbox"] = json.loads(d["bbox"])
            cache[rr["file"]] = d
    rows = []
    for m in meta:
        if m["file"] in cache:
            rows.append(dict(m, **cache[m["file"]]))
            continue
        img = cv2.imread(str(Path(m["folder"]) / m["file"]))
        r = analyse(app, img)
        rows.append(dict(m, **r, img_size=img.shape[0]))
        print(f"{m['file']:12s} {m['light']:6s} det={r.get('det_score', 0):.2f} n={r.get('n_faces')} "
              f"meas={r.get('light_dir', '-'):6s} ({r.get('light_angle', float('nan')):5.0f}) "
              f"lm={r.get('light_dir_landmarks', '-'):6s} str={r.get('light_strength', 0):.2f} yaw={r.get('yaw') or 0:+.0f}",
              flush=True)
    df = pd.DataFrame(rows)
    target = {"right": 0, "top": 90, "left": 180, "bottom": -90}
    df["angle_err"] = [abs((ang - target[l] + 180) % 360 - 180) if pd.notna(ang) else 999
                       for ang, l in zip(df.get("light_angle", np.nan), df["light"])]
    def landmark_ok(r):
        """Anatomical check of the designed axis: cheeks for left/right, forehead vs chin for top/bottom.
        The chin test is skipped for beards/moustaches (dark albedo there, independent of the light)."""
        if pd.isna(r.get("lm_lr")):
            return False
        if r["light"] in ("left", "right") and abs(r.get("yaw") or 0) > 20:
            return True          # cheek disks are unreliable on clearly turned heads: rely on the gradient check
        if r["light"] == "left":
            return r["lm_lr"] > 0.05
        if r["light"] == "right":
            return r["lm_lr"] < -0.05
        if r["light"] == "top":
            return r["lm_tb"] > 0.05
        beard = any(w in str(r["extra"]) for w in ("beard", "mustache", "stubble"))
        return beard or r["lm_tb"] < -0.05
    df["landmarks_agree"] = df.apply(landmark_ok, axis=1)
    df["valid"] = (df.n_faces == 1) & (df.det_score >= a.min_det) & (df.angle_err <= a.max_angle_err) & \
                  df.landmarks_agree & (df.border < 0.12) & (~df.file.isin(a.exclude))
    Path(a.report).mkdir(parents=True, exist_ok=True)
    rep = df.drop(columns=["prompt"]).copy()
    rep["bbox"] = rep["bbox"].apply(lambda b: json.dumps(list(b)) if isinstance(b, (list, tuple, np.ndarray)) else b)
    rep.to_csv(Path(a.report) / "candidates_qc.csv", index=False)

    chosen = (df[df.valid].sort_values("light_strength", ascending=False).groupby("cell").head(1)
              .sort_values("cell"))
    missing = sorted(set(df.cell) - set(chosen.cell))
    print(f"\nvalid candidates: {df.valid.sum()}/{len(df)}  cells covered: {len(chosen)}/{df.cell.nunique()}")
    if missing:
        print("cells without a valid candidate (regenerate):", " ".join(missing))
    json.dump(missing, open(Path(a.report) / "missing_cells.json", "w"))

    out = Path(a.out)
    (out / "Original").mkdir(parents=True, exist_ok=True)
    order = list(chosen.index)
    random.Random(2026).shuffle(order)
    final = []
    for i, idx in enumerate(order, 1):
        r = chosen.loc[idx]
        fid = f"face_{i:03d}"
        Image.open(Path(r.folder) / r.file).convert("RGB").save(out / "Original" / f"{fid}.jpg", quality=95)
        final.append(dict(id=fid, cell=r.cell, source=r.file, seed=r.seed, light=r.light, gender=r.gender,
                          age=r.age, people=r.people, pose=r.pose, hair=r.hair, extra=r.extra,
                          yaw=r.yaw, pose_label=yaw_label(r.yaw), light_angle_measured=r.light_angle,
                          light_dir_landmarks=r.light_dir_landmarks, lm_lr=r.lm_lr, lm_tb=r.lm_tb,
                          light_strength=r.light_strength, face_contrast=r.face_contrast, det_score=r.det_score,
                          bbox=json.dumps([round(v, 1) for v in r.bbox]), img_size=int(r.img_size), prompt=r.prompt))
    pd.DataFrame(final).to_csv(out / "metadata.csv", index=False)
    print(f"wrote {len(final)} images -> {out / 'Original'}  +  {out / 'metadata.csv'}")


if __name__ == "__main__":
    main()
