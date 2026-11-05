"""Global choice of the two Mooney controls (blur, % white) for the whole image set.

    python analysis/mooney_sweep.py --data data --out work/sweep

For every (blur, white) pair all originals are converted and two objective proxies of a good
Mooney face are measured inside the face ellipse (detected on the original):
  * face_white : share of white pixels inside the face   (balanced two-tone face: ~0.35-0.65)
  * blobs      : number of separate white blobs inside the face (few, large blobs; no speckle)
The recommended pair maximises the share of images that satisfy both criteria.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import label

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mooney.core import MooneyParams, make_mooney, load_image  # noqa: E402
from metrics import ellipse_mask  # noqa: E402

BLURS = [0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
WHITES = [12, 15, 18, 21, 24, 28]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="work/sweep")
    ap.add_argument("--size", type=int, default=384, help="working resolution for the sweep")
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(data / "metadata.csv")
    meta["bbox"] = meta["bbox"].apply(json.loads)
    imgs = {r.id: load_image(data / "Original" / f"{r.id}.jpg") for r in meta.itertuples()}
    rows = []
    for b in BLURS:
        for w in WHITES:
            p = MooneyParams(blur=b, white_percent=w, work_size=a.size)
            for r in meta.itertuples():
                m, _ = make_mooney(imgs[r.id], p)
                mask = ellipse_mask(r.bbox, r.img_size, out=m.shape[0])
                face = (m > 0) & mask
                lab, n = label(face)
                sizes = np.bincount(lab.ravel())[1:]
                n_big = int((sizes >= 0.002 * m.size).sum())
                rows.append(dict(blur=b, white=w, id=r.id, light=r.light, face_white=face.sum() / mask.sum(), blobs=n_big))
    df = pd.DataFrame(rows)
    df["good"] = df.face_white.between(0.35, 0.65) & (df.blobs <= 6)
    agg = df.groupby(["blur", "white"]).agg(good=("good", "mean"), face_white=("face_white", "median"),
                                            blobs=("blobs", "median")).reset_index()
    agg.to_csv(out / "sweep_summary.csv", index=False)
    best = agg.sort_values(["good", "blobs"], ascending=[False, True]).iloc[0]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), dpi=130)
    for ax, col, title, cmap in zip(axs, ["good", "face_white", "blobs"],
                                    ["share of 'good' Mooneys", "median white share in face", "median # blobs in face"],
                                    ["YlOrBr", "Greys", "Blues"]):
        piv = agg.pivot(index="blur", columns="white", values=col)
        im = ax.imshow(piv.values, cmap=cmap, aspect="auto", origin="lower")
        ax.set_xticks(range(len(WHITES)), [f"{x}%" for x in WHITES])
        ax.set_yticks(range(len(BLURS)), [f"{x}%" for x in BLURS])
        ax.set_xlabel("white pixels")
        ax.set_ylabel("blur σ (% of image side)")
        ax.set_title(title, fontsize=10)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                ax.text(j, i, f"{piv.values[i, j]:.2f}" if col != "blobs" else f"{piv.values[i, j]:.0f}", ha="center",
                        va="center", fontsize=7, color="black" if col != "face_white" or piv.values[i, j] > .4 else "white")
        fig.colorbar(im, ax=ax, fraction=0.046)
    bi, bj = BLURS.index(best.blur), WHITES.index(best.white)
    for ax in axs:
        ax.add_patch(plt.Rectangle((bj - .5, bi - .5), 1, 1, fill=False, ec="crimson", lw=2))
    fig.tight_layout()
    fig.savefig(out / "sweep_heatmaps.png")
    rec = dict(blur=float(best.blur), white_percent=float(best.white), good_share=float(best.good))
    json.dump(rec, open(out / "recommended.json", "w"), indent=1)
    print(agg.sort_values("good", ascending=False).head(8).to_string(index=False))
    print("recommended:", rec)


if __name__ == "__main__":
    main()
