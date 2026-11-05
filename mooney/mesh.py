"""3D-mesh input for the Mooney tool.

A mesh (.obj/.ply/.stl/.glb/...) is rendered with a minimal ray tracer under a single
directional light ("extreme illumination"), with hard cast shadows and a black
background.  The rendering is then converted with exactly the same Mooney algorithm
that is used for photographs (``core.make_mooney``).

Camera: orthographic, looking along -z (x to the right, y up).
Light:  azimuth 0 deg = light from the camera, +90 deg = light from the right side of
        the image, -90 deg = from the left; elevation +90 deg = light from above.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from .core import MooneyParams, make_mooney


def light_vector(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
    return np.array([np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el)])


def load_mesh(path) -> trimesh.Trimesh:
    mesh = trimesh.load(str(path), force="mesh", process=True)
    mesh.apply_translation(-mesh.bounding_box.centroid)
    mesh.apply_scale(1.0 / mesh.extents.max())
    return mesh


def _rotation(yaw_deg, pitch_deg):
    ry = trimesh.transformations.rotation_matrix(np.radians(yaw_deg), [0, 1, 0])
    rx = trimesh.transformations.rotation_matrix(np.radians(pitch_deg), [1, 0, 0])
    return rx @ ry


def _albedo(mesh, tri, bary):
    """Per-hit albedo from UV texture or vertex colours (falls back to uniform grey)."""
    try:
        vis = mesh.visual
        if isinstance(vis, trimesh.visual.TextureVisuals) and vis.uv is not None:
            mat = vis.material
            img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
            if img is not None:
                tex = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
                uv = (vis.uv[mesh.faces[tri]] * bary[:, :, None]).sum(1) % 1.0
                h, w = tex.shape[:2]
                px = np.clip((uv[:, 0] * (w - 1)).astype(int), 0, w - 1)
                py = np.clip(((1 - uv[:, 1]) * (h - 1)).astype(int), 0, h - 1)
                return tex[py, px] ** 2.2                       # sRGB -> linear
        if isinstance(vis, trimesh.visual.ColorVisuals) and vis.kind == "vertex":
            col = vis.vertex_colors[:, :3].astype(np.float32) / 255.0
            return ((col[mesh.faces[tri]] * bary[:, :, None]).sum(1)) ** 2.2
    except Exception:
        pass
    return np.full((len(tri), 3), 0.6, dtype=np.float32)


def render_directional(mesh, azimuth=60.0, elevation=20.0, yaw=0.0, pitch=0.0, ambient=0.02,
                       resolution=768, ssaa=2, shadows=True, use_texture=True, zoom=1.0, center=(0.0, 0.0)):
    """Render ``mesh`` under one directional light.

    Returns a dict with the sRGB rendering (H,W,3 float in [0,1]) and ground-truth
    maps: depth (z towards the camera, NaN for background), normals and the mask.
    """
    mesh = mesh.copy()
    mesh.apply_transform(_rotation(yaw, pitch))
    l = light_vector(azimuth, elevation)

    n = resolution * ssaa
    half = 0.55 / zoom                                          # view window (mesh fits in [-0.5, 0.5])
    xs = np.linspace(-half, half, n)
    X, Y = np.meshgrid(xs + center[0], -xs + center[1])        # image row 0 = top
    origins = np.stack([X.ravel(), Y.ravel(), np.full(X.size, 2.0)], 1)
    dirs = np.tile([0.0, 0.0, -1.0], (origins.shape[0], 1))

    tri, ray, loc = mesh.ray.intersects_id(origins, dirs, multiple_hits=False, return_locations=True)
    bary = trimesh.triangles.points_to_barycentric(mesh.triangles[tri], loc)
    nrm = (mesh.vertex_normals[mesh.faces[tri]] * bary[:, :, None]).sum(1)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    flip = (nrm @ np.array([0, 0, 1.0])) < 0                    # make normals face the camera
    nrm[flip] *= -1

    lambert = np.clip(nrm @ l, 0.0, None)
    if shadows and lambert.any():
        lit = lambert > 0
        o = loc[lit] + nrm[lit] * 1e-3
        blocked = mesh.ray.intersects_any(o, np.tile(l, (o.shape[0], 1)))
        idx = np.flatnonzero(lit)[blocked]
        lambert[idx] = 0.0

    alb = _albedo(mesh, tri, bary) if use_texture else np.full((len(tri), 3), 0.6, np.float32)
    radiance = alb * (lambert[:, None] + ambient)

    img = np.zeros((n * n, 3), np.float32)
    img[ray] = radiance
    depth = np.full(n * n, np.nan, np.float32)
    depth[ray] = loc[:, 2]
    normals = np.zeros((n * n, 3), np.float32)
    normals[ray] = nrm
    mask = np.zeros(n * n, bool)
    mask[ray] = True

    def down(a, c):
        a = a.reshape(n, n, c) if c > 1 else a.reshape(n, n)
        if ssaa == 1:
            return a
        return a.reshape(resolution, ssaa, resolution, ssaa, *a.shape[2:]).mean((1, 3))

    srgb = np.clip(down(img, 3), 0, 1) ** (1 / 2.2)
    return dict(image=srgb, depth=down(np.nan_to_num(depth, nan=-np.inf), 1), normals=down(normals, 3),
                mask=down(mask.astype(np.float32), 1) > 0.5, light=l)


def mesh_to_mooney(path, dst, params: MooneyParams | None = None, save_render=True, **render_kw):
    """Render a mesh file and save its Mooney image (and optionally the shaded rendering)."""
    params = params or MooneyParams()
    r = render_directional(load_mesh(path), **render_kw)
    m, info = make_mooney(r["image"], params, mask=r["mask"])
    dst = Path(dst)
    Image.fromarray(m).save(dst)
    if save_render:
        Image.fromarray((r["image"] * 255).astype(np.uint8)).save(dst.with_name(dst.stem + "_render.png"))
    return m, r, info
