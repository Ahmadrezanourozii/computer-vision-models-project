"""Command-line interface of the Mooney tool.

Examples
--------
Batch-convert a folder (100+ images) with blur sigma = 1.5 % of the image size and 40 % white pixels::

    python -m mooney convert data/Original -o data/Mooney --blur 1.5 --white 40

Explore the two controls on one image::

    python -m mooney sweep face.jpg --blur 0.5 1 2 4 --white 25 40 55 -o sweep.jpg

Render a 3D mesh under a single directional light and convert it::

    python -m mooney mesh head.obj -o head_mooney.png --azimuth 70 --elevation 25 --blur 1 --white 35
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .batch import MESH_EXT, collect_inputs, contact_sheet, convert_batch, parameter_sweep, read_overrides
from .core import MooneyParams


def _add_params(p):
    g = p.add_argument_group("Mooney controls")
    g.add_argument("--blur", type=float, default=1.5,
                   help="strength of the Gaussian blur: sigma in %% of the shorter image side (default 1.5)")
    g.add_argument("--white", type=float, default=40.0,
                   help="percentage of white pixels in the Mooney image (default 40)")
    g.add_argument("--invert", action="store_true", help="invert contrast polarity (white <-> black)")
    g.add_argument("--min-blob", type=float, default=0.0,
                   help="remove black/white islands smaller than this %% of the image (default 0 = off)")
    g.add_argument("--size", type=int, default=0,
                   help="resize so that the longer side has SIZE pixels before processing (0 = keep)")


def _params(a) -> MooneyParams:
    return MooneyParams(blur=a.blur, white_percent=a.white, invert=a.invert, min_blob=a.min_blob,
                        work_size=a.size)


def cmd_convert(a):
    files = collect_inputs(a.inputs, recursive=a.recursive)
    if not files:
        sys.exit("no input images found")
    out = Path(a.output)
    t0 = time.time()

    def progress(i, n, row):
        print(f"[{i:4d}/{n}] {row['file']:<40s} sigma={row['sigma_px']:6.2f}px  "
              f"thr={row['threshold']:.3f}  white={row['white_actual']:5.1f}%", flush=True)

    rows = convert_batch(files, out, _params(a), overrides=read_overrides(a.params_csv), workers=a.workers,
                         suffix=a.suffix, log_csv=a.log or out / "mooney_log.csv", progress=progress)
    print(f"converted {len(rows)} images in {time.time() - t0:.1f}s -> {out}")
    if a.contact_sheet:
        contact_sheet(sorted(out.glob(f"*{a.suffix}.png")), a.contact_sheet, cols=a.cols)
        print("contact sheet ->", a.contact_sheet)


def cmd_sweep(a):
    parameter_sweep(a.image, a.blur, a.white, a.output, invert=a.invert)
    print("sweep grid ->", a.output)


def cmd_sheet(a):
    files = collect_inputs(a.inputs)
    contact_sheet(files, a.output, cols=a.cols, thumb=a.thumb)
    print(f"contact sheet with {len(files)} images ->", a.output)


def cmd_mesh(a):
    from .mesh import mesh_to_mooney          # optional dependency (trimesh)
    files = collect_inputs(a.inputs, exts=MESH_EXT)
    out = Path(a.output)
    single = len(files) == 1 and out.suffix.lower() == ".png"
    if not single:
        out.mkdir(parents=True, exist_ok=True)
    for f in files:
        dst = out if single else out / f"{f.stem}.png"
        params = _params(a)
        params.white_reference = a.white_of
        mesh_to_mooney(f, dst, params, azimuth=a.azimuth, elevation=a.elevation, yaw=a.yaw,
                       pitch=a.pitch, ambient=a.ambient, resolution=a.resolution, zoom=a.zoom,
                       center=tuple(a.center), save_render=not a.no_render)
        print("rendered + converted", f.name, "->", dst)


def build_parser():
    ap = argparse.ArgumentParser(prog="python -m mooney", description="Generate Mooney images from images or 3D meshes.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("convert", help="convert images (files, folders or glob patterns) - batch capable")
    c.add_argument("inputs", nargs="+")
    c.add_argument("-o", "--output", required=True, help="output folder")
    c.add_argument("-r", "--recursive", action="store_true")
    c.add_argument("--suffix", default="", help="suffix appended to output file names")
    c.add_argument("--params-csv", help="CSV with per-image overrides (columns: file, blur, white_percent, ...)")
    c.add_argument("--workers", type=int, default=None, help="parallel worker processes")
    c.add_argument("--log", help="CSV log file (default: OUTPUT/mooney_log.csv)")
    c.add_argument("--contact-sheet", help="also write a contact sheet of all results to this file")
    c.add_argument("--cols", type=int, default=10)
    _add_params(c)
    c.set_defaults(func=cmd_convert)

    s = sub.add_parser("sweep", help="grid of results for several blur / white-percentage values")
    s.add_argument("image")
    s.add_argument("--blur", type=float, nargs="+", default=[0.5, 1, 2, 4])
    s.add_argument("--white", type=float, nargs="+", default=[25, 40, 55])
    s.add_argument("--invert", action="store_true")
    s.add_argument("-o", "--output", default="sweep.jpg")
    s.set_defaults(func=cmd_sweep)

    h = sub.add_parser("sheet", help="contact sheet (thumbnail grid with labels)")
    h.add_argument("inputs", nargs="+")
    h.add_argument("-o", "--output", default="contact_sheet.jpg")
    h.add_argument("--cols", type=int, default=10)
    h.add_argument("--thumb", type=int, default=200)
    h.set_defaults(func=cmd_sheet)

    m = sub.add_parser("mesh", help="render 3D mesh(es) under a directional light, then convert")
    m.add_argument("inputs", nargs="+")
    m.add_argument("-o", "--output", required=True, help="output .png (single mesh) or folder")
    m.add_argument("--azimuth", type=float, default=60.0, help="light azimuth in degrees (0 = from camera, +90 = from the right)")
    m.add_argument("--elevation", type=float, default=20.0, help="light elevation in degrees (+ = from above)")
    m.add_argument("--yaw", type=float, default=0.0, help="object rotation around the vertical axis (degrees)")
    m.add_argument("--pitch", type=float, default=0.0, help="object rotation around the horizontal axis (degrees)")
    m.add_argument("--ambient", type=float, default=0.02, help="ambient light level (0..1)")
    m.add_argument("--resolution", type=int, default=768)
    m.add_argument("--zoom", type=float, default=1.0, help="camera zoom (1 = whole mesh)")
    m.add_argument("--center", type=float, nargs=2, default=[0.0, 0.0], metavar=("X", "Y"),
                   help="view centre in normalised mesh units (e.g. 0 0.2 to frame the head of a bust)")
    m.add_argument("--white-of", choices=["image", "object"], default="object",
                   help="percentage of white pixels relative to the whole image or to the object only")
    m.add_argument("--no-render", action="store_true", help="do not save the shaded rendering next to the Mooney")
    _add_params(m)
    m.set_defaults(func=cmd_mesh)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
