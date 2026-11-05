"""Vision system 4: diffusion-based inverse rendering with Marigold (v1-1).

  * prs-eth/marigold-normals-v1-1       -> surface normals
  * prs-eth/marigold-iid-lighting-v1-1  -> intrinsic decomposition: albedo, diffuse shading,
                                           non-diffuse residual  (image ~ albedo * shading + residual)

Key question for Mooney images: are the black regions explained as SHADING (shadows,
like humans perceive them) or as ALBEDO (dark paint)?
Per image: <id>_normals.jpg, <id>_albedo.jpg, <id>_shading.jpg   Per set: marigold_raw.npz
"""
import time

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import MarigoldIntrinsicsPipeline, MarigoldNormalsPipeline

from common import base_args, load_rgb, sets


def small(x, size=256):
    t = torch.from_numpy(np.ascontiguousarray(x)).float().permute(2, 0, 1)[None]
    return F.interpolate(t, size=(size, size), mode="area")[0].permute(1, 2, 0).numpy().astype(np.float16)


def main():
    ap = base_args(__doc__)
    ap.add_argument("--steps", type=int, default=None, help="denoising steps (default: model default)")
    ap.add_argument("--ensemble", type=int, default=1)
    a = ap.parse_args()
    dev = "cuda"
    kw = dict(variant="fp16", torch_dtype=torch.float16)
    p_norm = MarigoldNormalsPipeline.from_pretrained("prs-eth/marigold-normals-v1-1", **kw).to(dev)
    p_iid = MarigoldIntrinsicsPipeline.from_pretrained("prs-eth/marigold-iid-lighting-v1-1", **kw).to(dev)
    names = p_iid.target_properties["target_names"]
    print("IID targets:", names, flush=True)
    gen_kw = dict(ensemble_size=a.ensemble)
    if a.steps:
        gen_kw["num_inference_steps"] = a.steps

    for name, files, out in sets(a):
        raw = {}
        t0 = time.time()
        for f in files:
            img = load_rgb(f)
            g = torch.Generator(dev).manual_seed(0)
            n = p_norm(img, generator=g, **gen_kw)
            p_norm.image_processor.visualize_normals(n.prediction)[0].resize((512, 512)).save(out / f"{f.stem}_normals.jpg", quality=90)
            raw[f"{f.stem}__normals"] = small(n.prediction[0])

            g = torch.Generator(dev).manual_seed(0)
            r = p_iid(img, generator=g, **gen_kw)
            vis = p_iid.image_processor.visualize_intrinsics(r.prediction, p_iid.target_properties)[0]
            for k in ("albedo", "shading"):
                vis[k].resize((512, 512)).save(out / f"{f.stem}_{k}.jpg", quality=90)
            for j, k in enumerate(names):
                raw[f"{f.stem}__{k}"] = small(r.prediction[j])
        np.savez_compressed(out / "marigold_raw.npz", **raw)
        print(f"{name}: {len(files)} images in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
