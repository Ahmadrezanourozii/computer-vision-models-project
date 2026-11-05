"""Batch processing, logging and contact sheets for the Mooney tool."""
from __future__ import annotations

import csv
import glob
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .core import MooneyParams, make_mooney

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
MESH_EXT = {".obj", ".ply", ".stl", ".off", ".glb", ".gltf"}


def collect_inputs(inputs, recursive=False, exts=IMAGE_EXT):
    """Expand files / folders / glob patterns into a sorted list of files."""
    files = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            it = p.rglob("*") if recursive else p.iterdir()
            files += [f for f in it if f.suffix.lower() in exts]
        elif p.exists():
            files.append(p)
        else:
            files += [Path(f) for f in glob.glob(item, recursive=recursive) if Path(f).suffix.lower() in exts]
    return sorted(set(files))


def read_overrides(csv_path):
    """Per-image parameter overrides: CSV with a 'file' column (file name or stem) + any MooneyParams field."""
    if not csv_path:
        return {}
    fields = MooneyParams.__dataclass_fields__
    out = {}
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            key = Path(row.pop("file")).stem
            kw = {}
            for k, v in row.items():
                if k in fields and v not in (None, ""):
                    t = fields[k].type
                    kw[k] = (v.lower() in ("1", "true", "yes")) if t in (bool, "bool") else float(v) \
                        if t in (float, "float") else int(float(v)) if t in (int, "int") else v
            out[key] = kw
    return out


def _process_one(args):
    src, dst, params = args
    t = time.time()
    m, info = make_mooney(str(src), params)
    Image.fromarray(m).save(dst, optimize=True)
    return dict(file=Path(src).name, output=Path(dst).name, seconds=round(time.time() - t, 3)) | info


def convert_batch(files, out_dir, params: MooneyParams, overrides=None, workers=None, suffix="",
                  log_csv=None, progress=None):
    """Convert many images; returns a list of per-image info dicts (also written to ``log_csv``)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    overrides = overrides or {}
    jobs = [(f, out_dir / f"{Path(f).stem}{suffix}.png", replace(params, **overrides.get(Path(f).stem, {})))
            for f in files]
    rows = []
    workers = workers or min(8, os.cpu_count() or 1)
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(workers) as ex:
            for i, row in enumerate(ex.map(_process_one, jobs, chunksize=2)):
                rows.append(row)
                if progress:
                    progress(i + 1, len(jobs), row)
    else:
        for i, job in enumerate(jobs):
            rows.append(_process_one(job))
            if progress:
                progress(i + 1, len(jobs), rows[-1])
    if log_csv:
        with open(log_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return rows


def _font(size):
    for name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "arialbd.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def contact_sheet(files, out_path, cols=10, thumb=200, labels=True, pad=4, bg=(128, 128, 128)):
    """Grid of thumbnails with index labels (used for review and the manual top-10 selection)."""
    files = list(files)
    rows = int(np.ceil(len(files) / cols))
    sheet = Image.new("RGB", (cols * (thumb + pad) + pad, rows * (thumb + pad) + pad), bg)
    draw = ImageDraw.Draw(sheet)
    font = _font(max(12, thumb // 9))
    for i, f in enumerate(files):
        im = Image.open(f).convert("RGB")
        im.thumbnail((thumb, thumb))
        x, y = pad + (i % cols) * (thumb + pad), pad + (i // cols) * (thumb + pad)
        sheet.paste(im, (x + (thumb - im.width) // 2, y + (thumb - im.height) // 2))
        if labels:
            txt = labels[i] if isinstance(labels, (list, tuple)) else Path(f).stem
            tb = draw.textbbox((0, 0), txt, font=font)
            draw.rectangle([x, y, x + tb[2] - tb[0] + 6, y + tb[3] - tb[1] + 6], fill=(220, 30, 30))
            draw.text((x + 3, y + 1), txt, fill=(255, 255, 255), font=font)
    sheet.save(out_path, quality=92)
    return out_path


def parameter_sweep(image, blurs, whites, out_path, thumb=256, invert=False):
    """Grid showing the effect of the two controls (rows: blur, columns: % white)."""
    font = _font(max(12, thumb // 12))
    head = 34
    W = len(whites) * thumb + head * 3
    H = len(blurs) * thumb + head
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for j, wp in enumerate(whites):
        d.text((head * 3 + j * thumb + thumb // 2 - 30, 8), f"{wp:g}% white", fill=(0, 0, 0), font=font)
    for i, b in enumerate(blurs):
        d.text((6, head + i * thumb + thumb // 2 - 10), f"blur {b:g}%", fill=(0, 0, 0), font=font)
        for j, wp in enumerate(whites):
            m, _ = make_mooney(image, MooneyParams(blur=b, white_percent=wp, invert=invert))
            im = Image.fromarray(m).convert("RGB").resize((thumb - 4, thumb - 4), Image.LANCZOS)
            sheet.paste(im, (head * 3 + j * thumb + 2, head + i * thumb + 2))
    sheet.save(out_path, quality=92)
    return out_path
