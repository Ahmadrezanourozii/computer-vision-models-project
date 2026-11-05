"""Pilot A: RealVisXL V5.0 (SDXL, fp16-safe) dramatic-lighting portraits. 6 identities x 4 light directions."""
import json, os, subprocess, sys, time
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "diffusers>=0.42", "accelerate"], check=False)
import torch
print("torch", torch.__version__, torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0))
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, DPMSolverMultistepScheduler

OUT = "/kaggle/working/pilot_sdxl"; os.makedirs(OUT, exist_ok=True)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLPipeline.from_pretrained("SG161222/RealVisXL_V5.0", vae=vae, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True,
                                                         algorithm_type="sde-dpmsolver++")
pipe.to("cuda")

ids = [("young East Asian woman with long black hair", "facing the camera"),
       ("middle-aged Black man with a short beard", "three-quarter view"),
       ("elderly white woman with short grey hair", "facing the camera"),
       ("young South Asian man", "three-quarter view"),
       ("middle-aged Middle Eastern woman, hair tied back", "facing the camera"),
       ("elderly Latino man with a grey mustache", "three-quarter view")]
lights = {"left": "split lighting from the left, right half of the face in deep shadow",
          "right": "split lighting from the right, left half of the face in deep shadow",
          "top": "hard overhead top light, dark eye sockets, shadow under nose and chin",
          "bottom": "lit from below, flashlight under the chin, horror lighting, upper face in shadow"}
NEG = ("soft light, fill light, flat lighting, bright background, colorful background, bad anatomy, deformed, "
       "face asymmetry, eyes asymmetry, deformed eyes, deformed mouth, open mouth, text, watermark, painting, cartoon")
meta = []
t0 = time.time()
for i, (who, pose) in enumerate(ids):
    for lname, lphr in lights.items():
        prompt = (f"low key studio portrait photo of a {who}, {pose}, {lphr}, single hard light, "
                  f"pure black background, chiaroscuro, high contrast, dark clothes, 85mm photo, photorealistic")
        t = time.time()
        img = pipe(prompt=prompt, negative_prompt=NEG, num_inference_steps=30, guidance_scale=5.0,
                   height=1024, width=1024, generator=torch.Generator("cuda").manual_seed(2000 + i)).images[0]
        name = f"id{i}_{lname}.png"; img.save(os.path.join(OUT, name))
        meta.append(dict(file=name, prompt=prompt, sec=round(time.time() - t, 1)))
        print(name, f"{time.time()-t:.1f}s", flush=True)
json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)
print("total", time.time() - t0)
