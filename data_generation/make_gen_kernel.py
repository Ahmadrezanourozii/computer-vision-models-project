"""Builds a self-contained Kaggle kernel that generates the candidate portraits of one design shard.

    python data_generation/make_gen_kernel.py --account ah22reza --shard 0 --nshards 2 -o kaggle/gen0

The design table (data_generation/design.py) is embedded into the kernel script, so the kernel
needs no input dataset.  Generator: RealVisXL V5.0 (SDXL fine-tune, openrail++), fp16 on a T4.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from design import build  # noqa: E402

LIGHT = {
    "left": "split lighting from the left, right half of the face in deep shadow",
    "right": "split lighting from the right, left half of the face in deep shadow",
    "top": "hard overhead top light, dark eye sockets, shadows under nose and chin",
    "bottom": "lit from below, flashlight under the chin, horror lighting, upper face in shadow",
}
AGE = {"young adult": "young", "middle-aged": "middle-aged", "elderly": "elderly"}
POSE = {"frontal": "looking straight at the camera", "three-quarter": "three-quarter view"}
NEG = ("soft light, fill light, flat lighting, even lighting, bright background, colorful background, white background, "
       "bad anatomy, deformed, face asymmetry, eyes asymmetry, deformed eyes, deformed mouth, open mouth, text, "
       "watermark, painting, drawing, cartoon, 3d render, hands")


def prompt(r):
    who = f"{AGE[r['age']]} {r['people']} {'woman' if r['gender'] == 'female' else 'man'}"
    extra = f", {r['extra']}" if r["extra"] else ""
    return (f"low key studio portrait photo of a {who} with {r['hair']}{extra}, {POSE[r['pose']]}, {LIGHT[r['light']]}, "
            f"single hard light, pure black background, chiaroscuro, dark clothes, photorealistic")


KERNEL = r'''
import json, os, subprocess, sys, time
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "diffusers>=0.42", "accelerate"], check=False)
import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, DPMSolverMultistepScheduler
ROWS = json.loads(__ROWS__)
NEG = __NEG__
OUT = "/kaggle/working/candidates"; os.makedirs(OUT, exist_ok=True)
print(torch.cuda.device_count(), torch.cuda.get_device_name(0), len(ROWS), "images", flush=True)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

def load(dev):
    p = StableDiffusionXLPipeline.from_pretrained("SG161222/RealVisXL_V5.0", vae=vae, torch_dtype=torch.float16)
    p.scheduler = DPMSolverMultistepScheduler.from_config(p.scheduler.config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++")
    return p.to(dev)

# two T4s -> two pipelines, images alternate between them (threads release the GIL inside CUDA)
import threading
devs = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
pipes = [load(d) for d in devs]
meta, lock = [], threading.Lock()

def worker(wi):
    pipe, dev = pipes[wi], devs[wi]
    for r in ROWS[wi::len(devs)]:
        name = f"{r['cell']}_{r['cand']}.png"
        if os.path.exists(os.path.join(OUT, name)):
            continue
        t = time.time()
        img = pipe(prompt=r["prompt"], negative_prompt=NEG, num_inference_steps=30, guidance_scale=5.0, height=1024, width=1024,
                   generator=torch.Generator(dev).manual_seed(int(r["seed"]))).images[0]
        img.save(os.path.join(OUT, name))
        with lock:
            meta.append(dict(r, file=name, sec=round(time.time() - t, 1)))
            print(name, f"{time.time() - t:.1f}s", len(meta), flush=True)

th = [threading.Thread(target=worker, args=(i,)) for i in range(len(devs))]
[t.start() for t in th]; [t.join() for t in th]
json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)
print("done", len(meta))
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--account", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--k", type=int, default=2, help="candidates per design cell")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    rows = build(k=a.k)
    cells = sorted({r["cell"] for r in rows})
    mine = set(cells[a.shard::a.nshards])
    rows = [dict(r, prompt=prompt(r)) for r in rows if r["cell"] in mine]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    code = KERNEL.replace("__ROWS__", repr(json.dumps(rows))).replace("__NEG__", repr(NEG))
    (out / "gen.py").write_text(code)
    slug = f"mooney-gen-shard{a.shard}"
    json.dump({"id": f"{a.account}/{slug}", "title": slug, "code_file": "gen.py", "language": "python",
               "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
               "machine_shape": "NvidiaTeslaT4", "dataset_sources": [], "competition_sources": [],
               "kernel_sources": [], "model_sources": []}, open(out / "kernel-metadata.json", "w"), indent=2)
    print(f"{len(rows)} images ({len(mine)} cells) -> {out}")
    print("example prompt:", rows[0]["prompt"])


if __name__ == "__main__":
    main()
