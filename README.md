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
