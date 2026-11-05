"""Builds a self-contained Kaggle kernel that generates the candidate portraits of one design shard.

    python data_generation/make_gen_kernel.py --account ah22reza --shard 0 --nshards 2 -o kaggle/gen0

Generator: Z-Image-Turbo (Tongyi-MAI, Apache-2.0, 6B single-stream DiT, 8 steps).  It follows long,
explicit lighting descriptions far better than SDXL (pilot: SDXL ignored left/right/top/bottom).
A T4 has no native bf16 and the model overflows in fp16, so it runs in (emulated) bf16:
  1. one process encodes all prompts with the Qwen3 text encoder and saves the embeddings,
  2. one worker process per GPU (own CUDA context) loads transformer + VAE and renders its share;
     VAE decoding is tiled on the GPU with a float32 CPU fallback (the 12 GB transformer leaves ~2 GB).
The design table (data_generation/design.py) is embedded, so the kernel needs no input dataset.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from design import build  # noqa: E402

LIGHT = {
    "left": ("from the far left side of the frame (90 degrees to the camera, split lighting)",
             "Only the left half of the face is lit; the right half of the face is completely black"),
    "right": ("from the far right side of the frame (90 degrees to the camera, split lighting)",
              "Only the right half of the face is lit; the left half of the face is completely black"),
    "top": ("from directly above the head (overhead top lighting)",
            "The top of the head, the forehead, the nose bridge and the cheekbones are lit; the eye sockets, the areas "
            "under the nose and under the lips, the chin and the neck are in deep black shadow"),
    "bottom": ("from directly below the chin, pointing upwards (under-lighting, like a flashlight held under the chin)",
               "The chin, the jaw, the underside of the nose and the lower cheeks are lit from below; the forehead, the "
               "eye sockets and the top of the head are in deep black shadow"),
}
POSE = {"frontal": "facing the camera directly", "three-quarter": "head turned in a three-quarter view"}

TEMPLATE = ("Dramatic chiaroscuro portrait lit by a single hard light {light}. {effect}. Close-up studio photograph of a "
            "{age} {people} {noun} with {hair}{extra}, {pose}, neutral expression, wearing a plain black shirt, the head "
            "filling most of the frame. Pure black background, no fill light, no rim light, the light source itself is "
            "outside the frame, extreme contrast, photorealistic, sharp focus, 85mm lens.")


def prompt(r):
    extra = f", {r['extra']}" if r["extra"] else ""
    light, effect = LIGHT[r["light"]]
    return TEMPLATE.format(light=light, effect=effect, age=r["age"], people=r["people"],
                           noun="woman" if r["gender"] == "female" else "man", hair=r["hair"], extra=extra,
                           pose=POSE[r["pose"]])


WORKER = r'''
import json, os, sys, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from diffusers import AutoencoderKL, ZImagePipeline
MODEL = "Tongyi-MAI/Z-Image-Turbo"
TMP, OUT = "/tmp/zimg", "/kaggle/working/candidates"
ROWS = json.load(open(os.path.join(TMP, "rows.json")))
SIZE = int(os.environ["GEN_SIZE"])
mode = sys.argv[1]

if mode == "encode":
    pipe = ZImagePipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16, transformer=None, vae=None)
    pipe.text_encoder.to("cuda")
    emb = {}
    with torch.no_grad():
        for r in ROWS:
            pe, _ = pipe.encode_prompt(prompt=r["prompt"], device="cuda", do_classifier_free_guidance=False)
            emb[r["file"]] = [e.to("cpu") for e in pe]
    torch.save(emb, os.path.join(TMP, "emb.pt"))
    print("encoded", len(emb), flush=True)
    sys.exit(0)

g, n = int(sys.argv[2]), int(sys.argv[3])
emb = torch.load(os.path.join(TMP, "emb.pt"))
pipe = ZImagePipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16, text_encoder=None, tokenizer=None)
pipe.to("cuda")
pipe.vae.enable_tiling()
cpu_vae = AutoencoderKL.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float32)
open(os.path.join(TMP, f"ready{g}"), "w").write("1")
print(f"worker {g} ready", flush=True)

def decode(lat):
    z = lat.to(pipe.vae.dtype) / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    try:
        with torch.no_grad():
            x = pipe.vae.decode(z, return_dict=False)[0]
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        z = lat.float().cpu() / cpu_vae.config.scaling_factor + cpu_vae.config.shift_factor
        with torch.no_grad():
            x = cpu_vae.decode(z, return_dict=False)[0]
    return pipe.image_processor.postprocess(x.float().cpu(), output_type="pil")[0]

meta, fails = [], 0
for r in ROWS[g::n]:
    path = os.path.join(OUT, r["file"])
    if os.path.exists(path):
        continue
    t = time.time()
    try:
        with torch.no_grad():
            lat = pipe(prompt_embeds=[e.to("cuda") for e in emb[r["file"]]], height=SIZE, width=SIZE, num_inference_steps=9,
                       guidance_scale=0.0, output_type="latent",
                       generator=torch.Generator("cpu").manual_seed(int(r["seed"]))).images
        img = decode(lat)
    except Exception as e:
        fails += 1
        print(f"worker {g} FAILED {r['file']}: {repr(e)[:200]}", flush=True)
        if fails >= 3 and not meta:
            sys.exit(f"worker {g}: first images failed")
        continue
    img.save(path)
    meta.append(dict(r, sec=round(time.time() - t, 1)))
    json.dump(meta, open(os.path.join(TMP, f"meta{g}.json"), "w"), indent=1)
    print(f"worker {g}: {r['file']} {time.time() - t:.0f}s ({len(meta)})", flush=True)
'''

KERNEL = r'''
import json, os, subprocess, sys, time, glob
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "diffusers>=0.42", "transformers", "accelerate"], check=False)
TMP, OUT = "/tmp/zimg", "/kaggle/working/candidates"
os.makedirs(TMP, exist_ok=True); os.makedirs(OUT, exist_ok=True)
json.dump(json.loads(__ROWS__), open(os.path.join(TMP, "rows.json"), "w"))
open(os.path.join(TMP, "worker.py"), "w").write(__WORKER__)
env = dict(os.environ, GEN_SIZE="__SIZE__")
t0 = time.time()
import torch
ngpu = torch.cuda.device_count()
print("GPUs:", ngpu, flush=True)
r = subprocess.run([sys.executable, os.path.join(TMP, "worker.py"), "encode"], env=dict(env, CUDA_VISIBLE_DEVICES="0"))
if r.returncode:
    sys.exit("prompt encoding failed")
print(f"encoding done {time.time() - t0:.0f}s", flush=True)
procs = []
for g in range(ngpu):
    procs.append(subprocess.Popen([sys.executable, os.path.join(TMP, "worker.py"), "work", str(g), str(ngpu)],
                                  env=dict(env, CUDA_VISIBLE_DEVICES=str(g))))
    while not os.path.exists(os.path.join(TMP, f"ready{g}")) and procs[-1].poll() is None:
        time.sleep(5)                       # stagger the loads: one 12 GB model in RAM at a time
codes = [p.wait() for p in procs]
meta = []
for f in sorted(glob.glob(os.path.join(TMP, "meta*.json"))):
    meta += json.load(open(f))
json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)
print("worker exit codes", codes, "| images:", len(meta), f"| {time.time() - t0:.0f}s")
if not meta:
    sys.exit("no image was generated")
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
    code = (KERNEL.replace("__ROWS__", repr(json.dumps(rows))).replace("__WORKER__", repr(WORKER))
            .replace("__SIZE__", str(a.size)))
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
