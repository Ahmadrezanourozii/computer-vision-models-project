"""Create a Kaggle kernel folder that runs selected vision systems on the dataset.

    python kaggle/make_vision_kernel.py --account ah22reza --dataset ah22reza/mooney-faces-data \
        --systems vlm marigold -o kaggle/vision_run

The attached Kaggle dataset must contain  Original/  Mooney/  code/vision/*.py
Results are packed into /kaggle/working/results.tar (one file -> fast download).
"""
import argparse
import json
from pathlib import Path

SYSTEMS = {
    "depth": ("Result1", "run_depth.py", ""),
    "vlm": ("Result2", "run_vlm.py", ""),
    "face": ("Result3", "run_face.py", ""),
    "marigold": ("Result4", "run_marigold.py", ""),
}

TEMPLATE = r'''
import glob, os, subprocess, sys, time, shutil, tarfile
SYSTEMS = __SYSTEMS__
EXTRA = __EXTRA__
pkgs = ["transformers>=4.57", "accelerate", "diffusers>=0.42", "qwen-vl-utils"]
if "face" in [s for s, *_ in SYSTEMS]:
    pkgs += ["insightface", "onnxruntime-gpu"]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U"] + pkgs, check=False)
data = os.path.dirname(glob.glob("/kaggle/input/**/Original", recursive=True)[0])
print("data:", data, flush=True)
code = "/kaggle/working/code"
shutil.copytree(os.path.join(data, "code", "vision"), code, dirs_exist_ok=True)
for name, folder, script, args in SYSTEMS:
    t = time.time()
    cmd = [sys.executable, os.path.join(code, script), "--original", os.path.join(data, "Original"),
           "--mooney", os.path.join(data, "Mooney"), "--out", f"/kaggle/working/{folder}"] + args.split() + EXTRA.split()
    print(">>>", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=code)
    print(f"<<< {name} exit={r.returncode} {time.time() - t:.0f}s", flush=True)
with tarfile.open("/kaggle/working/results.tar", "w") as tar:
    for name, folder, *_ in SYSTEMS:
        if os.path.isdir(f"/kaggle/working/{folder}"):
            tar.add(f"/kaggle/working/{folder}", arcname=folder)
for name, folder, *_ in SYSTEMS:
    shutil.rmtree(f"/kaggle/working/{folder}", ignore_errors=True)
shutil.rmtree(code, ignore_errors=True)
print("done")
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--account", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--systems", nargs="+", choices=list(SYSTEMS), required=True)
    ap.add_argument("--slug", default=None)
    ap.add_argument("--extra", default="", help="extra arguments passed to every script (e.g. '--limit 4')")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    systems = [(s, *SYSTEMS[s]) for s in a.systems]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.py").write_text(TEMPLATE.replace("__SYSTEMS__", repr(systems)).replace("__EXTRA__", repr(a.extra)))
    slug = a.slug or "mooney-vision-" + "-".join(a.systems)
    json.dump({"id": f"{a.account}/{slug}", "title": slug, "code_file": "run.py", "language": "python",
               "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
               "machine_shape": "NvidiaTeslaT4", "dataset_sources": [a.dataset], "competition_sources": [],
               "kernel_sources": [], "model_sources": []}, open(out / "kernel-metadata.json", "w"), indent=2)
    print("kernel ->", out, "| systems:", a.systems)


if __name__ == "__main__":
    main()
