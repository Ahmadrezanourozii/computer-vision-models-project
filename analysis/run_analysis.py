"""Compute all metrics + example panels for the slides.

    python analysis/run_analysis.py --data data --results results --out work/analysis

Outputs: summary.json (numbers for the native PowerPoint charts), per-image CSV tables and
example panels (PNG) for every vision system.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from metrics import depth_metrics, face_metrics, marigold_metrics, vlm_metrics, vlm_table

LIGHTS = ["left", "right", "top", "bottom"]


def font(size):
    for n in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            pass
    return ImageFont.load_default()


def tile(path, size=300):
    return Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)


def panel(rows, col_titles, row_titles, out, size=300, gap=8):
    """rows: list of lists of image paths -> labelled grid PNG."""
    head, left = 44, 150 if row_titles else 0
    W = left + len(col_titles) * (size + gap)
    H = head + len(rows) * (size + gap)
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    f = font(22)
    for j, t in enumerate(col_titles):
        tw = d.textlength(t, font=f)
        d.text((left + j * (size + gap) + (size - tw) / 2, 10), t, fill=(30, 30, 30), font=f)
    for i, r in enumerate(rows):
        if row_titles:
            d.text((8, head + i * (size + gap) + size / 2 - 12), row_titles[i], fill=(30, 30, 30), font=f)
        for j, p in enumerate(r):
            im.paste(tile(p, size), (left + j * (size + gap), head + i * (size + gap)))
    im.save(out)
    return out


def by_light(df, col):
    return {l: float(df.loc[df.light == l, col].mean()) for l in LIGHTS if (df.light == l).any()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with Original/, Mooney/, metadata.csv")
    ap.add_argument("--results", required=True, help="folder with Result1..Result4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--examples", nargs="*", default=None, help="image ids used in example panels")
    a = ap.parse_args()
    data, res, out = Path(a.data), Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(data / "metadata.csv")
    meta["bbox"] = meta["bbox"].apply(lambda s: json.loads(s) if isinstance(s, str) else s)
    ex = a.examples or list(meta.groupby("light").head(1)["id"])
    S = dict(n=len(meta), examples=ex)

    # ---------------- system 1: depth
    if (res / "Result1").exists():
        dm = depth_metrics(res / "Result1", meta, data / "Mooney")
        dm.to_csv(out / "depth_per_image.csv", index=False)
        S["depth"] = dict(r_mean=dm.depth_r.mean(), r_median=dm.depth_r.median(), relerr=dm.depth_relerr.mean(),
                          normal_err=dm.normal_err_med.mean(), partial_pattern=dm.partial_r_pattern.mean(),
                          r_pattern_mooney=dm.r_mooneydepth_vs_pattern.abs().mean(),
                          r_pattern_orig=dm.r_origdepth_vs_pattern.abs().mean(),
                          by_light_r=by_light(dm, "depth_r"), by_light_partial=by_light(dm, "partial_r_pattern"),
                          hist_r=np.histogram(dm.depth_r, bins=10, range=(-1, 1))[0].tolist())
        rows = [[data / "Original" / f"{k}.jpg", res / "Result1/Original" / f"{k}_depth.png",
                 res / "Result1/Original" / f"{k}_relief.png", data / "Mooney" / f"{k}.png",
                 res / "Result1/Mooney" / f"{k}_depth.png", res / "Result1/Mooney" / f"{k}_relief.png"] for k in ex[:3]]
        panel(rows, ["Original", "Depth", "3D relief", "Mooney", "Depth", "3D relief"], None, out / "depth_examples.png")

    # ---------------- system 2: VLM
    if (res / "Result2").exists():
        tabs = vlm_table(res / "Result2")
        vm = vlm_metrics(tabs, meta)
        vm.to_csv(out / "vlm_metrics.csv")
        for s, t in tabs.items():
            t.to_csv(out / f"vlm_answers_{s}.csv")
        mo = tabs["Mooney"].join(meta.set_index("id")[["light"]])
        S["vlm"] = dict(table=vm.round(4).to_dict(),
                        mooney_says_face_by_light={l: float((mo[mo.light == l].contains_face == "yes").mean()) for l in LIGHTS
                                                   if (mo.light == l).any()},
                        mooney_mentions_face_by_light={l: float(mo[mo.light == l].mentions_face.mean()) for l in LIGHTS
                                                       if (mo.light == l).any()},
                        mooney_content_top=tabs["Mooney"].description.str.extract(
                            r"(abstract|silhouette|face|portrait|ink|map|cloud|animal|figure|shape)", expand=False)
                        .value_counts().head(8).to_dict(),
                        examples={k: dict(orig=tabs["Original"].loc[k, "description"], mooney=tabs["Mooney"].loc[k, "description"])
                                  for k in ex if k in tabs["Mooney"].index})

    # ---------------- system 3: faces
    if (res / "Result3").exists():
        fm, (r_det, s_det, r_or, s_or, ids) = face_metrics(res / "Result3", meta)
        so = pd.read_csv(res / "Result3/Original/summary.csv").set_index("image")
        sm = pd.read_csv(res / "Result3/Mooney/summary.csv").set_index("image")
        cmc = [float(np.nanmean(r_or <= k)) for k in range(1, 21)]
        lab = meta.set_index("id")
        sex_ok_o = (so.loc[lab.index, "sex"].map({"M": "male", "F": "female"}) == lab.gender)
        S["faces"] = dict(fm, cmc_oracle=cmc, chance_rank1=1.0 / len(ids),
                          det_by_light={l: float(sm.loc[lab.index[lab.light == l], "detected"].mean()) for l in LIGHTS},
                          score_hist_orig=np.histogram(so.best_score, bins=10, range=(0, 1))[0].tolist(),
                          score_hist_mooney=np.histogram(sm.best_score, bins=10, range=(0, 1))[0].tolist(),
                          gender_acc_orig=float(sex_ok_o.mean()),
                          rank1_oracle_by_light={l: float(np.nanmean(r_or[np.array([lab.loc[k, 'light'] == l for k in ids])] == 1))
                                                 for l in LIGHTS})
        rows = [[res / "Result3/Original" / f"{k}_faces.jpg", res / "Result3/Mooney" / f"{k}_faces.jpg"] for k in ex[:4]]
        panel([sum(rows[i:i + 2], []) for i in range(0, len(rows), 2)],
              ["Original", "Mooney", "Original", "Mooney"], None, out / "face_examples.png")

    # ---------------- system 4: Marigold
    if (res / "Result4").exists():
        mg = marigold_metrics(res / "Result4", meta, data / "Mooney")
        mg.to_csv(out / "marigold_per_image.csv", index=False)
        S["marigold"] = dict(normal_err=mg.normal_err_med.mean(), shadow_index_orig=mg.shadow_index_orig.mean(),
                             shadow_index_mooney=mg.shadow_index_mooney.mean(),
                             by_light_shadow_mooney=by_light(mg, "shadow_index_mooney"),
                             by_light_shadow_orig=by_light(mg, "shadow_index_orig"))
        rows = [[data / "Original" / f"{k}.jpg", res / "Result4/Original" / f"{k}_normals.jpg",
                 res / "Result4/Original" / f"{k}_albedo.jpg", res / "Result4/Original" / f"{k}_shading.jpg",
                 data / "Mooney" / f"{k}.png", res / "Result4/Mooney" / f"{k}_normals.jpg",
                 res / "Result4/Mooney" / f"{k}_albedo.jpg", res / "Result4/Mooney" / f"{k}_shading.jpg"] for k in ex[:2]]
        panel(rows, ["Original", "Normals", "Albedo", "Shading", "Mooney", "Normals", "Albedo", "Shading"], None,
              out / "marigold_examples.png", size=240)

    json.dump(S, open(out / "summary.json", "w"), indent=1, default=float)
    print(json.dumps(S, indent=1, default=float)[:3000])


if __name__ == "__main__":
    main()
