"""Builds a self-contained Kaggle kernel that generates the candidate portraits of one design shard.

    python data_generation/make_gen_kernel.py --account ah22reza --shard 0 --nshards 2 -o kaggle/gen0

Generator: Z-Image-Turbo (Tongyi-MAI, Apache-2.0, 6B single-stream DiT, 8 steps).  It follows
long, explicit lighting descriptions far better than SDXL (pilot: SDXL ignored left/right/top/bottom).
The T4 has no native bf16 and the model overflows in fp16, so it runs in (emulated) bf16:
  phase 1 - all prompts are encoded once with the Qwen3 text encoder on GPU 0,
  phase 2 - the text encoder is dropped and one transformer copy per T4 renders images in parallel.
The design table (data_generation/design.py) is embedded, so the kernel needs no input dataset.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from design import build  # noqa: E402

LIGHT = {
    "left": "a single hard light from the left side of the frame (camera left, split lighting): the left half of the face "
            "is brightly lit and the right half falls into deep black shadow",
    "right": "a single hard light from the right side of the frame (camera right, split lighting): the right half of the "
             "face is brightly lit and the left half falls into deep black shadow",
    "top": "a single hard spotlight directly above the head (overhead top lighting): the top of the head, the forehead, the "
           "nose bridge and the cheekbones are bright, while the eye sockets, the area under the nose, under the lips and "
           "under the chin are in deep black shadow",
    "bottom": "a single hard light directly below the face pointing upwards (under-lighting, like a flashlight held under "
              "the chin): the underside of the chin, the jaw, the nostrils and the lower cheeks are brightly lit, while the "
              "forehead, the eye sockets and the top of the head are in deep shadow",
}
POSE = {"frontal": "facing the camera directly", "three-quarter": "head turned in a three-quarter view"}

TEMPLATE = ("Close-up low-key studio portrait photograph of a {age} {people} {noun} with {hair}{extra}, {pose}, neutral "
            "expression, wearing a plain black shirt; the head fills most of the frame. Lighting: {light}. The light source "
            "itself is outside the frame. No other light, no fill light, no rim light. Pure black background, "
            "high-contrast chiaroscuro, sharp focus, photorealistic, 85mm lens.")


def prompt(r):
    extra = f", {r['extra']}" if r["extra"] else ""
    return TEMPLATE.format(age=r["age"], people=r["people"], noun="woman" if r["gender"] == "female" else "man",
                           hair=r["hair"], extra=extra, pose=POSE[r["pose"]], light=LIGHT[r["light"]])


KERNEL = r'''
import gc, json, os, subprocess, sys, threading, time
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "diffusers>=0.42", "transformers", "accelerate"], check=False)
import torch
from diffusers import AutoencoderKL, ZImagePipeline, ZImageTransformer2DModel
MODEL = "Tongyi-MAI/Z-Image-Turbo"
ROWS = json.loads(__ROWS__)
SIZE = __SIZE__
OUT = "/kaggle/working/candidates"; os.makedirs(OUT, exist_ok=True)
ngpu = torch.cuda.device_count()
print(ngpu, torch.cuda.get_device_name(0), len(ROWS), "images", flush=True)
t0 = time.time()

# ---- phase 1: encode all prompts once
pipe = ZImagePipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
pipe.text_encoder.to("cuda:0")
emb = {}
with torch.no_grad():
    for r in ROWS:
        pe, _ = pipe.encode_prompt(prompt=r["prompt"], device="cuda:0", do_classifier_free_guidance=False)
        emb[r["file"]] = [e.to("cpu") for e in pe]
pipe.text_encoder = None
gc.collect(); torch.cuda.empty_cache()
print(f"phase 1 done {time.time()-t0:.0f}s", flush=True)

# ---- phase 2: one transformer per GPU
pipes = []
for g in range(ngpu):
    dev = f"cuda:{g}"
    if g == 0:
        pipe.transformer.to(dev); pipe.vae.to(dev); p = pipe
    else:
        tr = ZImageTransformer2DModel.from_pretrained(MODEL, subfolder="transformer", torch_dtype=torch.bfloat16).to(dev)
        vae = AutoencoderKL.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.bfloat16).to(dev)
        p = ZImagePipeline(scheduler=pipe.scheduler.__class__.from_config(pipe.scheduler.config), vae=vae,
                           text_encoder=None, tokenizer=None, transformer=tr)
    pipes.append((p, dev))
print(f"phase 2 ready {time.time()-t0:.0f}s", flush=True)
meta, lock = [], threading.Lock()

def worker(g):
    p, dev = pipes[g]
    for r in ROWS[g::ngpu]:
        path = os.path.join(OUT, r["file"])
        if os.path.exists(path):
            continue
        t = time.time()
        with torch.no_grad():
            img = p(prompt_embeds=[e.to(dev) for e in emb[r["file"]]], height=SIZE, width=SIZE, num_inference_steps=9,
                    guidance_scale=0.0, generator=torch.Generator("cpu").manual_seed(int(r["seed"]))).images[0]
        img.save(path)
        with lock:
            meta.append(dict(r, sec=round(time.time() - t, 1)))
            json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)
            print(r["file"], f"{time.time() - t:.0f}s", len(meta), flush=True)

th = [threading.Thread(target=worker, args=(g,)) for g in range(ngpu)]
[t.start() for t in th]; [t.join() for t in th]
print("done", len(meta), f"{time.time()-t0:.0f}s")
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--account", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--k", type=int, default=2, help="candidates per design cell")
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--cells", nargs="*", help="only these design cells (e.g. regeneration)")
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--slug", default=None)
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    rows = build(k=a.k)
    cells = sorted({r["cell"] for r in rows})
    mine = set(a.cells) if a.cells else set(cells[a.shard::a.nshards])
    rows = [dict(r, seed=r["seed"] + a.seed_offset, prompt=prompt(r), file=f"{r['cell']}_{r['cand'] + a.seed_offset}.png")
            for r in rows if r["cell"] in mine]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    code = KERNEL.replace("__ROWS__", repr(json.dumps(rows))).replace("__SIZE__", str(a.size))
    (out / "gen.py").write_text(code)
    slug = a.slug or f"mooney-gen-shard{a.shard}"
    json.dump({"id": f"{a.account}/{slug}", "title": slug, "code_file": "gen.py", "language": "python",
               "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
               "machine_shape": "NvidiaTeslaT4", "dataset_sources": [], "competition_sources": [],
               "kernel_sources": [], "model_sources": []}, open(out / "kernel-metadata.json", "w"), indent=2)
    print(f"{len(rows)} images ({len(mine)} cells) -> {out}")
    print("example prompt:", rows[0]["prompt"])


if __name__ == "__main__":
    main()
