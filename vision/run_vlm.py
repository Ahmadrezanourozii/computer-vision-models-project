"""Vision system 2: vision-language model (Qwen3-VL-8B-Instruct) used for recognition and
"inverse rendering by language" (where is the light, what are the dark regions, ...).

Two independent conversations per image (no information leaks between them):
  A) unprimed free description   - does the model spontaneously see a face?
  B) structured questionnaire    - JSON with face / gender / age / head orientation /
                                   light direction / meaning of the black regions / confidence
Per image: <id>.json   Per set: answers.csv
"""
import csv
import json
import re
import time

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from common import base_args, load_rgb, save_json, sets

PROMPT_FREE = "Describe this image in two or three sentences."

PROMPT_STRUCT = """Answer with a single JSON object and nothing else. Use exactly these keys and one of the allowed values:
{
  "contains_face": "yes" | "no",
  "main_content": a short phrase naming what the image shows,
  "gender": "female" | "male" | "unclear",
  "age_group": "young adult" | "middle-aged" | "elderly" | "unclear",
  "face_orientation": "frontal" | "turned toward image left" | "turned toward image right" | "unclear",
  "light_direction": "from the left" | "from the right" | "from above" | "from below" | "from the front" | "unclear",
  "black_regions_are": "mostly shadows" | "mostly dark-colored surfaces or materials" | "mostly empty background" | "unclear",
  "face_confidence": an integer from 0 to 100 (how sure you are that the image shows a human face)
}
"light_direction" is where the main light source is located. Left and right always refer to the image as the viewer sees it."""

KEYS = ["contains_face", "main_content", "gender", "age_group", "face_orientation", "light_direction",
        "black_regions_are", "face_confidence"]


def parse_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    s = m.group(0)
    for attempt in (s, s.replace("'", '"')):
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            continue
    out = {}
    for k in KEYS:                                       # last resort: key-by-key regex
        mm = re.search(rf'"{k}"\s*:\s*"?([^",}}\n]+)"?', s)
        if mm:
            out[k] = mm.group(1).strip()
    return out


def main():
    ap = base_args(__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--size", type=int, default=768, help="images are resized to size x size")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    a = ap.parse_args()

    dtype = a.dtype
    if dtype == "auto":
        dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else "float16"
    model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=getattr(torch, dtype), device_map="auto").eval()
    proc = AutoProcessor.from_pretrained(a.model)
    proc.tokenizer.padding_side = "left"
    print("model", a.model, dtype, flush=True)

    def ask(images, prompt, max_new):
        msgs = [[{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": prompt}]}]
                for im in images]
        inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                          return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        return proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    for name, files, out in sets(a):
        rows = []
        t0 = time.time()
        for i in range(0, len(files), a.batch):
            chunk = files[i:i + a.batch]
            ims = [load_rgb(f, a.size) for f in chunk]
            free = ask(ims, PROMPT_FREE, 120)
            struct = ask(ims, PROMPT_STRUCT, 200)
            for f, d, s in zip(chunk, free, struct):
                parsed = parse_json(s)
                rec = dict(image=f.name, model=a.model, description=d.strip(), structured_raw=s.strip(), structured=parsed)
                save_json(rec, out / f"{f.stem}.json")
                rows.append(dict(image=f.stem, description=d.strip().replace("\n", " "),
                                 **{k: parsed.get(k) for k in KEYS}))
            print(f"{name} {i + len(chunk)}/{len(files)}  {time.time() - t0:.0f}s", flush=True)
        with open(out / "answers.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
