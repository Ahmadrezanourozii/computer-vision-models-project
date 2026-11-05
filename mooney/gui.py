"""Graphical user interface of the Mooney tool (runs in the browser via Gradio).

Start with::

    python -m mooney.gui            # then open http://127.0.0.1:7860

Tabs
  1. Image -> Mooney   : live preview with the two controls (blur strength, % white pixels)
  2. 3D mesh -> Mooney : render a mesh under a single directional light, then convert
  3. Batch             : convert a whole folder (100+ images) with a CSV log + contact sheet
  4. Select top 10     : click the 10 best Mooney images and export the Subset folders
"""
from __future__ import annotations

import io
import shutil
import tempfile
import time
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .batch import IMAGE_EXT, collect_inputs, contact_sheet, convert_batch
from .core import MooneyParams, blurred_gray, make_mooney, to_grayscale

BLUR_HELP = "Gaussian sigma in % of the shorter image side"
WHITE_HELP = "share of pixels that become white"


# ----------------------------------------------------------------------------- tab 1
def _histogram(blurred, threshold, white):
    fig, ax = plt.subplots(figsize=(5, 2.4), dpi=110)
    ax.hist(blurred.ravel(), bins=128, range=(0, 1), color="#888", log=True)
    ax.axvline(threshold, color="crimson", lw=2, label=f"threshold = {threshold:.3f}")
    ax.axvspan(threshold, 1, color="gold", alpha=0.25, label=f"white = {white:.1f}%")
    ax.set_xlabel("blurred luminance")
    ax.set_ylabel("pixels (log)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def preview(image, blur, white, invert, min_blob):
    if image is None:
        return None, None, None, "Load an image to start.", None
    rgb = image.astype(np.float32) / 255.0
    p = MooneyParams(blur=blur, white_percent=white, invert=invert, min_blob=min_blob)
    t = time.time()
    mooney, info = make_mooney(rgb, p)
    dt = time.time() - t
    gray = (to_grayscale(rgb) * 255).astype(np.uint8)
    blurred = blurred_gray(rgb, p)
    text = (f"**sigma** = {info['sigma_px']:.1f} px  |  **threshold** = {info['threshold']:.3f}  |  "
            f"**white** = {info['white_actual']:.1f}% (target {white:.0f}%)  |  "
            f"{info['width']}x{info['height']} px  |  {1000 * dt:.0f} ms")
    fig = _histogram(blurred, info["threshold"], info["white_actual"])
    return gray, (blurred * 255).astype(np.uint8), mooney, text, fig


# ----------------------------------------------------------------------------- tab 2
_mesh_cache = {}


def render_mesh(mesh_file, texture, azimuth, elevation, yaw, pitch, zoom, cy, ambient, blur, white, white_of, invert):
    if mesh_file is None:
        return None, None, None, "Upload a mesh (.obj, .ply, .stl, .glb, ...)."
    import trimesh
    from .mesh import load_mesh, render_directional

    path = mesh_file if isinstance(mesh_file, str) else mesh_file.name
    key = (path, texture if isinstance(texture, str) else None)
    if key not in _mesh_cache:
        m = load_mesh(path)
        if texture is not None and isinstance(m.visual, trimesh.visual.TextureVisuals):
            m.visual.material = trimesh.visual.material.SimpleMaterial(image=Image.open(texture))
        _mesh_cache.clear()
        _mesh_cache[key] = m
    t = time.time()
    r = render_directional(_mesh_cache[key], azimuth=azimuth, elevation=elevation, yaw=yaw, pitch=pitch,
                           zoom=zoom, center=(0.0, cy), ambient=ambient, resolution=512)
    p = MooneyParams(blur=blur, white_percent=white, invert=invert, white_reference=white_of)
    mooney, info = make_mooney(r["image"], p, mask=r["mask"])
    d = r["depth"]
    valid = np.isfinite(d) & r["mask"]
    depth_vis = np.zeros_like(d)
    if valid.any():
        depth_vis[valid] = (d[valid] - d[valid].min()) / (np.ptp(d[valid]) + 1e-9)
    depth_rgb = (plt.get_cmap("magma")(depth_vis)[..., :3] * 255 * r["mask"][..., None]).astype(np.uint8)
    obj = f" ({info['white_in_object']:.1f}% of the object)" if info.get("white_in_object") is not None else ""
    text = (f"light = ({r['light'][0]:+.2f}, {r['light'][1]:+.2f}, {r['light'][2]:+.2f})  |  "
            f"white = {info['white_actual']:.1f}% of image{obj}  |  {1000 * (time.time() - t):.0f} ms")
    return (r["image"] * 255).astype(np.uint8), depth_rgb, mooney, text


# ----------------------------------------------------------------------------- tab 3
def run_batch(in_dir, out_dir, files, blur, white, invert, min_blob, progress=gr.Progress()):
    paths = collect_inputs([in_dir]) if in_dir and Path(in_dir).is_dir() else []
    if files:
        paths += [Path(f if isinstance(f, str) else f.name) for f in files]
    if not paths:
        return None, None, None, "No images found - give a folder path or upload files."
    out = Path(out_dir) if out_dir else Path(tempfile.mkdtemp()) / "Mooney"
    p = MooneyParams(blur=blur, white_percent=white, invert=invert, min_blob=min_blob)
    t = time.time()

    def cb(i, n, row):
        progress(i / n, desc=f"{i}/{n}  {row['file']}")

    rows = convert_batch(paths, out, p, log_csv=out / "mooney_log.csv", progress=cb)
    sheet = contact_sheet(sorted(out.glob("*.png")), out / "_contact_sheet.jpg", cols=10, thumb=160)
    archive = shutil.make_archive(str(out), "zip", out)
    whites = [r["white_actual"] for r in rows]
    msg = (f"Converted **{len(rows)}** images in {time.time() - t:.1f}s -> `{out}`  |  "
           f"white: {np.mean(whites):.1f}% +- {np.std(whites):.2f}")
    return str(sheet), str(out / "mooney_log.csv"), archive, msg


# ----------------------------------------------------------------------------- tab 4
def load_for_selection(orig_dir, mooney_dir):
    mo = collect_inputs([mooney_dir]) if mooney_dir and Path(mooney_dir).is_dir() else []
    if not mo:
        return [], [], "No Mooney images found.", []
    items = [(str(f), f.stem) for f in mo]
    return items, [], "Click images to (de)select. Selected: 0 / 10", [str(f) for f in mo]


def toggle(evt: gr.SelectData, selected, files):
    name = Path(files[evt.index]).stem
    selected = list(selected)
    if name in selected:
        selected.remove(name)
    else:
        selected.append(name)
    flag = "  -  exactly 10 selected" if len(selected) == 10 else ""
    return selected, f"Selected: {len(selected)} / 10{flag}\n\n" + ", ".join(selected)


def export_subset(selected, orig_dir, mooney_dir, subset_dir):
    if not selected:
        return "Nothing selected."
    so, sm = Path(subset_dir) / "Original", Path(subset_dir) / "Mooney"
    so.mkdir(parents=True, exist_ok=True)
    sm.mkdir(parents=True, exist_ok=True)
    n = 0
    for stem in selected:
        for f in Path(orig_dir).glob(stem + ".*"):
            if f.suffix.lower() in IMAGE_EXT:
                shutil.copy2(f, so / f.name)
                n += 1
        for f in Path(mooney_dir).glob(stem + ".*"):
            if f.suffix.lower() in IMAGE_EXT:
                shutil.copy2(f, sm / f.name)
    (Path(subset_dir) / "selection.txt").write_text("\n".join(selected) + "\n")
    return f"Exported {n} originals + Mooneys to `{subset_dir}` (selection order saved in selection.txt)."


# ----------------------------------------------------------------------------- layout
def build_app(blur_default=1.5, white_default=40.0):
    with gr.Blocks(title="Mooney Image Generator") as app:
        gr.Markdown("## Mooney Image Generator\nGrayscale → Gaussian blur → threshold at a fixed percentage "
                    "of white pixels.  *CompVP project, Part 1*")
        with gr.Tab("Image → Mooney"):
            with gr.Row():
                with gr.Column(scale=1):
                    inp = gr.Image(label="Input image", type="numpy", height=320)
                    blur = gr.Slider(0, 6, value=blur_default, step=0.1, label="Blur strength σ (%)", info=BLUR_HELP)
                    white = gr.Slider(1, 99, value=white_default, step=1, label="White pixels (%)", info=WHITE_HELP)
                    min_blob = gr.Slider(0, 1, value=0, step=0.02, label="Despeckle (%)",
                                         info="remove black/white islands smaller than this % of the image")
                    invert = gr.Checkbox(label="Invert contrast polarity", value=False)
                with gr.Column(scale=2):
                    with gr.Row():
                        g_gray = gr.Image(label="1) Grayscale", height=300, interactive=False)
                        g_blur = gr.Image(label="2) Blurred", height=300, interactive=False)
                        g_moon = gr.Image(label="3) Mooney", height=300, type="numpy",
                                          interactive=False, format="png")
                    info = gr.Markdown()
                    hist = gr.Plot(label="Luminance histogram and threshold")
            ins = [inp, blur, white, invert, min_blob]
            outs = [g_gray, g_blur, g_moon, info, hist]
            inp.change(preview, ins, outs)
            for c in (blur, white, min_blob):
                c.release(preview, ins, outs)
            invert.change(preview, ins, outs)

        with gr.Tab("3D mesh → Mooney"):
            with gr.Row():
                with gr.Column(scale=1):
                    mesh_in = gr.File(label="Mesh (.obj .ply .stl .glb)", file_types=[".obj", ".ply", ".stl", ".glb", ".gltf", ".off"])
                    tex_in = gr.Image(label="Optional colour texture", type="filepath", height=120)
                    az = gr.Slider(-180, 180, value=70, step=1, label="Light azimuth (°)", info="0 = from camera, +90 = from the right")
                    el = gr.Slider(-90, 90, value=25, step=1, label="Light elevation (°)", info="+ = from above")
                    yaw = gr.Slider(-90, 90, value=0, step=1, label="Object yaw (°)")
                    pitch = gr.Slider(-45, 45, value=0, step=1, label="Object pitch (°)")
                    zoom = gr.Slider(0.5, 3, value=1.8, step=0.05, label="Zoom")
                    cy = gr.Slider(-0.5, 0.5, value=0.18, step=0.01, label="View centre (vertical)")
                    amb = gr.Slider(0, 0.3, value=0.0, step=0.01, label="Ambient light")
                    mblur = gr.Slider(0, 6, value=1.2, step=0.1, label="Blur strength σ (%)", info=BLUR_HELP)
                    mwhite = gr.Slider(1, 99, value=40, step=1, label="White pixels (%)", info=WHITE_HELP)
                    mref = gr.Radio(["object", "image"], value="object", label="White % relative to")
                    minv = gr.Checkbox(label="Invert polarity", value=False)
                with gr.Column(scale=2):
                    with gr.Row():
                        m_render = gr.Image(label="Rendering (single directional light)", height=340, interactive=False)
                        m_depth = gr.Image(label="Ground-truth depth", height=340, interactive=False)
                        m_moon = gr.Image(label="Mooney", height=340, interactive=False, format="png")
                    m_info = gr.Markdown()
            mins = [mesh_in, tex_in, az, el, yaw, pitch, zoom, cy, amb, mblur, mwhite, mref, minv]
            mouts = [m_render, m_depth, m_moon, m_info]
            mesh_in.change(render_mesh, mins, mouts)
            tex_in.change(render_mesh, mins, mouts)
            for c in (az, el, yaw, pitch, zoom, cy, amb, mblur, mwhite):
                c.release(render_mesh, mins, mouts)
            for c in (mref, minv):
                c.change(render_mesh, mins, mouts)

        with gr.Tab("Batch"):
            with gr.Row():
                with gr.Column(scale=1):
                    in_dir = gr.Textbox(label="Input folder", placeholder="e.g. data/Original")
                    files = gr.File(label="... or upload images", file_count="multiple", height=120)
                    out_dir = gr.Textbox(label="Output folder", placeholder="e.g. data/Mooney")
                    bblur = gr.Slider(0, 6, value=blur_default, step=0.1, label="Blur strength σ (%)", info=BLUR_HELP)
                    bwhite = gr.Slider(1, 99, value=white_default, step=1, label="White pixels (%)", info=WHITE_HELP)
                    bblob = gr.Slider(0, 1, value=0, step=0.02, label="Despeckle (%)")
                    binv = gr.Checkbox(label="Invert contrast polarity", value=False)
                    run = gr.Button("Run batch", variant="primary")
                    bmsg = gr.Markdown()
                    blog = gr.File(label="CSV log", height=60)
                    bzip = gr.File(label="All results (.zip)", height=60)
                with gr.Column(scale=2):
                    bsheet = gr.Image(label="Contact sheet of all results", height=620, interactive=False)
            run.click(run_batch, [in_dir, out_dir, files, bblur, bwhite, binv, bblob], [bsheet, blog, bzip, bmsg])

        with gr.Tab("Select top 10"):
            with gr.Row():
                s_orig = gr.Textbox(label="Original folder", value="data/Original")
                s_moon = gr.Textbox(label="Mooney folder", value="data/Mooney")
                s_sub = gr.Textbox(label="Subset folder (export)", value="data/Subset")
                s_load = gr.Button("Load")
            s_msg = gr.Markdown()
            gallery = gr.Gallery(label="Click to select / deselect", columns=10, height=640, allow_preview=False)
            selected = gr.State([])
            files_state = gr.State([])
            s_export = gr.Button("Export selected to Subset/Original + Subset/Mooney", variant="primary")
            s_done = gr.Markdown()
            s_load.click(load_for_selection, [s_orig, s_moon], [gallery, selected, s_msg, files_state])
            gallery.select(toggle, [selected, files_state], [selected, s_msg])
            s_export.click(export_subset, [selected, s_orig, s_moon, s_sub], s_done)
    return app


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Mooney tool GUI")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--blur", type=float, default=1.5, help="default blur strength (sigma in %% of the image side)")
    ap.add_argument("--white", type=float, default=40.0, help="default percentage of white pixels")
    a = ap.parse_args()
    build_app(a.blur, a.white).launch(server_port=a.port, share=a.share, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
