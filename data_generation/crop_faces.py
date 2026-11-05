"""Normalise head size: square crop around the detected face, as Paper 1 normalised head size.

    python data_generation/crop_faces.py --data data --sources work/final_g0/candidates work/final_g1/candidates \
        work/final_regen/candidates --factor 1.7 --size 512

The selected 768 px portraits (metadata 'source') are cropped to a square of side factor * max(face box)
centred on the face and resized to SIZE px.  The face is then re-detected on the crop so that the
metadata (bbox, yaw, key-point light check) refers to the delivered image.
"""
import argparse
import json
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
    ap.add_argument("--data", default="data")
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--factor", type=float, default=1.7)
    ap.add_argument("--size", type=int, default=512)
    a = ap.parse_args()
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.3)

    data = Path(a.data)
    meta = pd.read_csv(data / "metadata.csv")
    rows = []
    for _, r in meta.iterrows():
        src = next(Path(s) / r["source"] for s in a.sources if (Path(s) / r["source"]).exists())
        im = Image.open(src).convert("RGB")
        W = im.width
        x1, y1, x2, y2 = json.loads(r["bbox"])
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        s = min(W, a.factor * max(x2 - x1, y2 - y1))
        left = int(np.clip(cx - s / 2, 0, W - s))
        top = int(np.clip(cy - s / 2, 0, W - s))
        crop = im.crop((left, top, left + int(s), top + int(s))).resize((a.size, a.size), Image.LANCZOS)
        crop.save(data / "Original" / f"{r['id']}.jpg", quality=95)
        m = analyse(app, cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2BGR))
        r = r.copy()
        r["crop_box_768"] = json.dumps([left, top, int(s)])
        r["bbox"] = json.dumps([round(v, 1) for v in m["bbox"]])
        r["img_size"] = a.size
        r["det_score"], r["yaw"], r["pose_label"] = m["det_score"], m["yaw"], yaw_label(m["yaw"])
        r["light_angle_measured"], r["light_dir_landmarks"] = m["light_angle"], m["light_dir_landmarks"]
        r["lm_lr"], r["lm_tb"], r["light_strength"] = m["lm_lr"], m["lm_tb"], m["light_strength"]
        rows.append(r)
        print(r["id"], r["light"], f"det={m['det_score']:.2f}", f"crop={int(s)}px", flush=True)
    pd.DataFrame(rows).to_csv(data / "metadata.csv", index=False)
    print("cropped", len(rows), "images ->", data / "Original")


if __name__ == "__main__":
    main()
