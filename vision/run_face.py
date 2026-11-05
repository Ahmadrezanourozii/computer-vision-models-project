"""Vision system 3: face detection, landmarks, attributes and identity with InsightFace (buffalo_l).

buffalo_l = SCRFD-10G face detector + 2D-106/3D-68 landmark networks + gender/age network +
ArcFace ResNet-50 (WebFace600K) identity embedding.  This is the "DCNN trained for faces"
account of Paper 1 (there: VGG-Face).

Per image:  <id>_faces.jpg (annotated)  and  <id>.json (all detections, best face first)
Per set:    embeddings.npz (ArcFace embedding of the best detection), summary.csv
For Mooney images we additionally compute an "oracle-aligned" ArcFace embedding, i.e. the
Mooney image is aligned with the 5 key points detected in the corresponding ORIGINAL image.
This separates "the detector does not find the face" from "the recogniser cannot identify it".
"""
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.utils import face_align

from common import base_args, save_json, sets

LOW_THRESH = 0.05      # record weak detections too; the analysis applies the default 0.5


def annotate(img, faces, size=512):
    vis = img.copy()
    s = size / max(img.shape[:2])
    vis = cv2.resize(vis, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    for k, f in enumerate(faces):
        col = (0, 220, 0) if f.det_score >= 0.5 else (0, 160, 255)
        x1, y1, x2, y2 = (f.bbox * s).astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
        for p in (f.landmark_2d_106 * s).astype(int) if f.get("landmark_2d_106") is not None else []:
            cv2.circle(vis, tuple(p), 1, (0, 0, 255), -1)
        label = f"{f.det_score:.2f} {'MF'[int(f.sex == 'F')] if hasattr(f, 'sex') else ''}{int(f.age) if f.get('age') is not None else ''}"
        cv2.putText(vis, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    if not faces:
        cv2.putText(vis, "no face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return vis


def main():
    ap = base_args(__doc__)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    providers = ["CPUExecutionProvider"] if a.cpu else ["CUDAExecutionProvider", "CPUExecutionProvider"]
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=-1 if a.cpu else 0, det_size=(640, 640), det_thresh=LOW_THRESH)
    rec = app.models["recognition"]

    orig_kps = {}
    for name, files, out in sets(a):
        emb, oracle, rows = {}, {}, []
        for f in files:
            img = cv2.imread(str(f))                       # BGR, 3 channels (Mooney -> replicated)
            faces = sorted(app.get(img), key=lambda x: -x.det_score)
            dets = []
            for x in faces:
                dets.append(dict(det_score=float(x.det_score), bbox=x.bbox.tolist(), kps=x.kps.tolist(),
                                 landmark_2d_106=x.landmark_2d_106.tolist() if x.get("landmark_2d_106") is not None else None,
                                 pose_pitch_yaw_roll=x.pose.tolist() if x.get("pose") is not None else None,
                                 sex=(x.sex if x.get("gender") is not None else None),
                                 age=int(x.age) if x.get("age") is not None else None))
            if faces:
                emb[f.stem] = faces[0].normed_embedding.astype(np.float32)
            if name == "Original" and faces:
                orig_kps[f.stem] = faces[0].kps
            if name == "Mooney" and f.stem in orig_kps:          # oracle alignment with the original's key points
                crop = face_align.norm_crop(img, landmark=orig_kps[f.stem], image_size=112)
                v = rec.get_feat(crop).flatten()
                oracle[f.stem] = (v / np.linalg.norm(v)).astype(np.float32)
            save_json(dict(image=f.name, n_detections=len(dets), detections=dets), out / f"{f.stem}.json")
            cv2.imwrite(str(out / f"{f.stem}_faces.jpg"), annotate(img, faces), [cv2.IMWRITE_JPEG_QUALITY, 88])
            best = dets[0] if dets else {}
            rows.append(dict(image=f.stem, n_detections=len(dets), best_score=best.get("det_score", 0.0),
                             detected=bool(best and best["det_score"] >= 0.5), sex=best.get("sex"), age=best.get("age"),
                             yaw=(best.get("pose_pitch_yaw_roll") or [None] * 3)[1]))
        np.savez_compressed(out / "embeddings.npz", **emb)
        if oracle:
            np.savez_compressed(out / "embeddings_oracle_aligned.npz", **oracle)
        with open(out / "summary.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(name, "detected (>=0.5):", sum(r["detected"] for r in rows), "/", len(rows), flush=True)


if __name__ == "__main__":
    main()
