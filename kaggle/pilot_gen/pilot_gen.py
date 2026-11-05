"""Pilot: can Z-Image-Turbo produce photorealistic portraits with controlled, dramatic
single-source lighting (good Mooney candidates)?  4 light directions x 6 identities."""
import json, os, subprocess, sys, time

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                "diffusers>=0.42", "transformers", "accelerate"], check=False)
try:
    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
except Exception as e:
    print("nvidia-smi unavailable:", e)

import torch
print("torch", torch.__version__, "cuda available:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
if not torch.cuda.is_available():
    raise SystemExit("NO GPU")
from diffusers import ZImagePipeline

OUT = "/kaggle/working/pilot"
os.makedirs(OUT, exist_ok=True)

identities = [
    ("f", "young adult", "East Asian woman with long straight black hair"),
    ("m", "middle-aged", "Black man with a short beard"),
    ("f", "elderly", "white woman with short grey hair and wrinkles"),
    ("m", "young adult", "South Asian man with short dark hair"),
    ("f", "middle-aged", "Middle Eastern woman with dark wavy hair tied back"),
    ("m", "elderly", "Latino man with a grey mustache"),
]
poses = ["facing the camera directly", "head turned slightly to the left in a three-quarter view",
         "facing the camera directly", "head turned slightly to the right in a three-quarter view",
         "facing the camera directly", "head turned slightly to the left in a three-quarter view"]
lights = {
    "left": "a single hard light from the left side of the frame (camera left, split lighting): "
            "the left half of the face is brightly lit and the right half falls into deep black shadow",
    "right": "a single hard light from the right side of the frame (camera right, split lighting): "
             "the right half of the face is brightly lit and the left half falls into deep black shadow",
    "top": "a single hard spotlight directly above the head (overhead top lighting): forehead, nose "
           "bridge and cheekbones are lit, with deep black shadows in the eye sockets, under the nose and under the chin",
    "bottom": "a single hard light below the face pointing upwards (under-lighting, like a flashlight held "
              "under the chin): chin, underside of the nose and cheeks are lit, the upper face is in deep shadow",
}
TEMPLATE = ("Low-key studio portrait photograph of a {age} {who}, {pose}, neutral expression, wearing a "
            "plain black shirt, head and shoulders framing. Lighting: {light}. No other light, no fill light. "
            "Pure black background, high-contrast chiaroscuro, sharp focus, photorealistic, 85mm lens.")

t0 = time.time()
cap = torch.cuda.get_device_capability(0)
dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
print("compute capability", cap, "-> dtype", dtype, "| GPUs:", torch.cuda.device_count())
pipe = ZImagePipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo", torch_dtype=dtype, low_cpu_mem_usage=False)
if torch.cuda.get_device_properties(0).total_memory / 2**30 > 22:
    pipe.to("cuda")
else:
    pipe.enable_model_cpu_offload()
# sanity check for fp16 overflow (black / NaN images)
import numpy as np
test = pipe(prompt="studio portrait photo of a man", height=512, width=512, num_inference_steps=9,
            guidance_scale=0.0, generator=torch.Generator("cuda").manual_seed(0)).images[0]
arr = np.asarray(test).astype(float)
print("sanity mean/std:", arr.mean(), arr.std())
if dtype == torch.float16 and (arr.std() < 2 or not np.isfinite(arr).all()):
    print("fp16 failed -> retry in float32 for the VAE / bf16 transformer")
    pipe.vae.to(torch.float32)
print(f"load {time.time()-t0:.1f}s")

meta = []
for i, (g, age, who) in enumerate(identities):
    for li, (lname, lphrase) in enumerate(lights.items()):
        prompt = TEMPLATE.format(age=age, who=who, pose=poses[i], light=lphrase)
        seed = 1000 + i
        t = time.time()
        img = pipe(prompt=prompt, height=1024, width=1024, num_inference_steps=9, guidance_scale=0.0,
                   generator=torch.Generator("cuda").manual_seed(seed)).images[0]
        name = f"id{i}_{lname}.png"
        img.save(os.path.join(OUT, name))
        meta.append(dict(file=name, seed=seed, gender=g, age=age, who=who, pose=poses[i], light=lname,
                         prompt=prompt, sec=round(time.time() - t, 2)))
        print(name, f"{time.time()-t:.1f}s", flush=True)
json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)
print("total", time.time() - t0)
