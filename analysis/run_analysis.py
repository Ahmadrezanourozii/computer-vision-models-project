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


def font(size, bold=True):
    for n in (("DejaVuSans-Bold.ttf", "DejaVuSans.ttf") if bold else ("DejaVuSans.ttf",)):
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


def wrap(d, text, font_, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if d.textlength(t, font=font_) <= width:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def vlm_panel(rows, out, size=230, text_w=760):
    """rows: list of (image_path, label, text) -> image with thumbnail + wrapped description per row."""
    f_lab, f_txt = font(22), font(21, bold=False)
    tmp = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    blocks = [(p, lab, wrap(tmp, txt, f_txt, text_w)[:6]) for p, lab, txt in rows]
    H = sum(max(size, 40 + 28 * len(l)) + 18 for *_, l in blocks)
    im = Image.new("RGB", (size + 30 + text_w, H), "white")
    d = ImageDraw.Draw(im)
    y = 0
    for p, lab, lines in blocks:
        im.paste(tile(p, size), (0, y))
        d.text((size + 24, y + 2), lab, fill=(185, 120, 26) if lab.startswith("Mooney") else (40, 40, 40), font=f_lab)
        for i, ln in enumerate(lines):
            d.text((size + 24, y + 36 + 28 * i), ln, fill=(45, 47, 51), font=f_txt)
        y += max(size, 40 + 28 * len(lines)) + 18
    im.save(out)
    return out


def first_sentences(t, n=2):
    import re
    parts = re.split(r"(?<=[.!?])\s+", t.strip())
    return " ".join(parts[:n])


def by_light(df, col):
    return {l: float(df.loc[df.light == l, col].mean()) for l in LIGHTS if (df.light == l).any()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with Original/, Mooney/, metadata.csv")
    ap.add_argument("--results", required=True, help="folder with Result1..Result4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--examples", nargs="*", default=None, help="image ids used in example panels")
    ap.add_argument("--ratings", default=None, help="JSON with blind human ratings {'blind': {id: 1-5}}")
    ap.add_argument("--ratings-v1", default=None, help="ratings of the first Mooney version (optional)")
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
                          by_light_pattern_orig={l: float(dm.loc[dm.light == l, "r_origdepth_vs_pattern"].abs().mean()) for l in LIGHTS},
                          by_light_pattern_mooney={l: float(dm.loc[dm.light == l, "r_mooneydepth_vs_pattern"].abs().mean()) for l in LIGHTS},
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
        vrows = []
        for k in ex[:2]:
            vrows += [(data / "Original" / f"{k}.jpg", "Original - Qwen3-VL:", first_sentences(tabs["Original"].loc[k, "description"])),
                      (data / "Mooney" / f"{k}.png", "Mooney - Qwen3-VL:", first_sentences(tabs["Mooney"].loc[k, "description"]))]
        vlm_panel(vrows, out / "vlm_examples.png")

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
                 res / "Result4/Mooney" / f"{k}_albedo.jpg", res / "Result4/Mooney" / f"{k}_shading.jpg"] for k in ex[:2] + ex[3:4]]
        panel(rows, ["Original", "Normals", "Albedo", "Shading", "Mooney", "Normals", "Albedo", "Shading"], None,
              out / "marigold_examples.png", size=240)

    # ---------------- human (blind ratings) vs. machines, per image (Paper 1 style trial-level correlation)
    if a.ratings:
        from scipy.stats import spearmanr
        hum = pd.Series(json.load(open(a.ratings))["blind"], dtype=float)
        lab = meta.set_index("id")
        H = dict(n=int(hum.size), mean_by_light={l: float(hum[lab.loc[hum.index, "light"] == l].mean()) for l in LIGHTS},
                 face_share_by_light={l: float((hum[lab.loc[hum.index, "light"] == l] >= 4).mean()) for l in LIGHTS},
                 face_share=float((hum >= 4).mean()), seen_share=float((hum >= 3).mean()))
        if a.ratings_v1:
            h1 = pd.Series(json.load(open(a.ratings_v1))["blind"], dtype=float)
            H["v1_mean_by_light"] = {l: float(h1[lab.loc[h1.index, "light"] == l].mean()) for l in LIGHTS}
            H["v1_face_share"] = float((h1 >= 4).mean())
        scores = pd.DataFrame(index=hum.index)
        if "vlm" in S:
            tm = vlm_table(res / "Result2")["Mooney"]
            scores["VLM face confidence"] = tm["face_confidence"]
            scores["VLM says 'face'"] = (tm["contains_face"] == "yes").astype(float)
        if "faces" in S:
            sm = pd.read_csv(res / "Result3/Mooney/summary.csv").set_index("image")
            scores["SCRFD detection score"] = sm["best_score"]
            sim = pd.Series(dict(zip(ids, s_or)))
            scores["ArcFace similarity to original"] = sim
        if "depth" in S:
            scores["Depth agreement r"] = dm.set_index("id")["depth_r"]
        if "marigold" in S:
            scores["Shadow index"] = mg.set_index("id")["shadow_index_mooney"]
        rng = np.random.default_rng(0)
        corr = {}
        for c in scores.columns:
            x = pd.concat([hum, scores[c]], axis=1).dropna().values
            if len(x) < 10 or np.std(x[:, 1]) == 0:
                corr[c] = dict(rho=None, lo=None, hi=None, n=int(len(x)))
                continue
            rho = spearmanr(x[:, 0], x[:, 1])[0]
            boots = []
            for _ in range(2000):
                i = rng.integers(0, len(x), len(x))
                if np.std(x[i, 1]) > 0 and np.std(x[i, 0]) > 0:
                    boots.append(spearmanr(x[i, 0], x[i, 1])[0])
            corr[c] = dict(rho=float(rho), lo=float(np.percentile(boots, 2.5)), hi=float(np.percentile(boots, 97.5)), n=int(len(x)))
        H["corr_with_human"] = corr
        pd.concat([hum.rename("human_rating"), lab["light"], scores], axis=1).to_csv(out / "human_vs_machine_per_image.csv")
        S["human"] = H

    json.dump(S, open(out / "summary.json", "w"), indent=1, default=float)
    print(json.dumps(S, indent=1, default=float)[:3000])


if __name__ == "__main__":
    main()
