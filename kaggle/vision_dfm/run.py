
import glob, os, subprocess, sys, time, shutil, tarfile
SYSTEMS = [('depth', 'Result1', 'run_depth.py', ''), ('face', 'Result3', 'run_face.py', ''), ('marigold', 'Result4', 'run_marigold.py', '')]
EXTRA = ''
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
