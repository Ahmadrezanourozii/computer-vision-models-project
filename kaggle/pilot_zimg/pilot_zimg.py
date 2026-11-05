"""Pilot B: Z-Image-Turbo in bfloat16 on 2x T4 (no native bf16) - is it fast enough?"""
import json, os, subprocess, sys, time
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "diffusers>=0.42", "transformers", "accelerate"], check=False)
import torch, numpy as np
print("torch", torch.__version__, torch.cuda.device_count(), torch.cuda.get_device_name(0), flush=True)
from diffusers import ZImagePipeline
OUT = "/kaggle/working/pilot_zimg"; os.makedirs(OUT, exist_ok=True)
t0 = time.time()
pipe = ZImagePipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()
print(f"load {time.time()-t0:.1f}s", flush=True)
TEMPLATE = ("Low-key studio portrait photograph of a {who}, {pose}, neutral expression, wearing a plain black shirt, "
            "head and shoulders framing. Lighting: {light}. No other light, no fill light. Pure black background, "
            "high-contrast chiaroscuro, sharp focus, photorealistic, 85mm lens.")
lights = {"left": "a single hard light from the left side of the frame (camera left, split lighting): the left half of the face is brightly lit and the right half falls into deep black shadow",
          "right": "a single hard light from the right side of the frame (camera right, split lighting): the right half of the face is brightly lit and the left half falls into deep black shadow",
          "top": "a single hard spotlight directly above the head (overhead top lighting): forehead, nose bridge and cheekbones are lit, with deep black shadows in the eye sockets, under the nose and under the chin",
          "bottom": "a single hard light below the face pointing upwards (under-lighting, like a flashlight held under the chin): chin, underside of the nose and cheeks are lit, the upper face is in deep shadow"}
ids = [("middle-aged Black man with a short beard", "head turned slightly in a three-quarter view"),
       ("elderly white woman with short grey hair", "facing the camera directly")]
meta = []
for i, (who, pose) in enumerate(ids):
    for lname, lp in lights.items():
        t = time.time()
        img = pipe(prompt=TEMPLATE.format(who=who, pose=pose, light=lp), height=768, width=768, num_inference_steps=9,
                   guidance_scale=0.0, generator=torch.Generator("cpu").manual_seed(3000 + i)).images[0]
        a = np.asarray(img).astype(float)
        name = f"z{i}_{lname}.png"; img.save(os.path.join(OUT, name))
        print(name, f"{time.time()-t:.1f}s mean={a.mean():.1f} std={a.std():.1f}", flush=True)
        meta.append(dict(file=name, sec=round(time.time()-t, 1)))
json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)
print("total", time.time() - t0)
