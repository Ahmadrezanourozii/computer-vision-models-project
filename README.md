# Mooney Faces vs. State-of-the-Art Vision Systems

Computational Visual Perception (CompVP), FAU Erlangen-Nürnberg, Project Part 1
Ahmadreza Nourozi (M.Sc. Artificial Intelligence, 23726011)

## The Mooney tool

A Mooney image is produced in three steps (as in Egger et al., *Inverse Rendering Best Explains
Face Perception Under Extreme Illuminations*):

1. convert the image to grayscale luminance,
2. Gaussian blur, **strength = sigma in % of the shorter image side** (resolution independent),
3. threshold at the luminance percentile that makes exactly **X % of the pixels white**.

```
pip install -r requirements.txt
```

### GUI

```
python -m mooney.gui          # opens http://127.0.0.1:7860
```

Tabs: *Image → Mooney* (live sliders for blur and % white, intermediate steps and histogram),
*3D mesh → Mooney* (mesh rendered under one directional light with cast shadows, then converted),
*Batch* (whole folders, CSV log, contact sheet, zip download), *Select top 10* (click the best
Mooney images, export `Subset/Original` + `Subset/Mooney`).

### CLI

```
# batch conversion of a folder (100+ images), parallel, with a CSV log and a contact sheet
python -m mooney convert data/Original -o data/Mooney --blur 1.5 --white 20 --contact-sheet sheet.jpg

# effect of the two controls on one image
python -m mooney sweep face.jpg --blur 0.5 1 2 4 --white 10 20 30 -o sweep.jpg

# 3D mesh input: render with a light from the upper right, then convert
python -m mooney mesh head.obj -o head_mooney.png --azimuth 70 --elevation 25 --white 40 --white-of object
```

Other options: `--invert` (contrast polarity), `--min-blob` (despeckling), `--size` (working
resolution), `--params-csv` (per-image overrides), `--workers` (parallel processes).

## Repository layout

| folder | content |
|---|---|
| `mooney/` | the Mooney tool (core algorithm, CLI, GUI, 3D mesh renderer) |
| `data_generation/` | stimulus design, portrait generation and quality control |
| `vision/` | the four vision systems (depth, VLM, face analysis, inverse rendering) |
| `analysis/` | metrics and figures comparing original vs. Mooney results |
| `slides/` | builds the presentation |
| `kaggle/` | GPU jobs (Kaggle kernels) |

## Part 1 pipeline (as submitted)

1. `data_generation/design.py` – balanced design: 4 light directions × 25, gender, age, 6 backgrounds, pose.
2. `data_generation/make_gen_kernel.py` – Z-Image-Turbo portraits on Kaggle (2 candidates per cell, 204 in total).
3. `data_generation/qc_select.py` – QC: one face (SCRFD), light direction verified by shading gradient **and**
   facial key-point regions, border artefacts; best candidate per cell -> 100 originals + `data/metadata.csv`.
4. `data_generation/crop_faces.py` – face-centred square crops (1.7× face box, 512 px), as head size was normalised in Paper 1.
5. `analysis/mooney_sweep.py` – global choice of the two controls (first choice: σ = 1.5 %, 15 % white). After the lecturer's feedback
   (too much blur: age, beard and glasses were lost) reduced to **blur σ = 0.7 % of the image side, 10 % white**.
6. `python -m mooney convert data/Original -o data/Mooney --blur 0.7 --white 10` – the 100 Mooney images.
7. `tools/make_review_page.py` – blind 1–5 face rating + manual top-10 pick (done by the author).
8. `vision/run_*.py` via `kaggle/make_vision_kernel.py` – the four vision systems (Result1–Result4).
9. `analysis/run_analysis.py` – metrics, example panels, human-vs-machine correlations.
10. `slides/prepare_deck.py` + `slides/build_deck.js` – the presentation; `tools/make_submission.py` – the zip.

## Main results (100 originals vs. 100 Mooneys)

| system | original | Mooney |
|---|---|---|
| SCRFD face detection | 100 % | 0 % |
| Qwen3-VL "contains a face" | 100 % | 78 % (top-lit 88 %, bottom-lit 60 %) |
| Qwen3-VL "black regions are shadows" | 99 % | 0 % |
| Depth Anything V2: depth agreement with the original (r) | – | 0.40 (partial r with the white pattern 0.40) |
| ArcFace identification among 100 (aligned with the original's key points) | – | rank-1 29 % (chance 1 %) |
| Marigold: share of contrast explained as shading | 0.84 | 0.32 |
| Human observer (blind, n = 1): rating ≥ 4 | – | 25 % (right-lit 68 %, left/top-lit 16 %/16 %, bottom-lit 0 %) |
