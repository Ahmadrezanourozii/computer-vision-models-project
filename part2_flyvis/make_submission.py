"""Assemble the Part-2 zip in the required structure.

    python part2_flyvis/make_submission.py --results work/part2/Results --video work/part2/video/<name>.mp4 \
        --readme part2_flyvis/README_submission.md --name Nourozi_23726011_CompVP_Part2 --outdir submission

Zip layout:
    Original/      the 10 selected original (RGB) images of Part 1
    Mooney/        the 10 corresponding Mooney images
    Results/SteadyState, Results/RGBFirst, Results/MooneyFirst   (20 visualisations each)
    Results/Comparisons/   additional comparison figures
    <video>.mp4    1-minute findings video with audio
    README.md      team members, protocol, file naming
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(C.ROOT / "work/part2/Results"))
    ap.add_argument("--video", required=True)
    ap.add_argument("--readme", default=str(C.ROOT / "part2_flyvis/README_submission.md"))
    ap.add_argument("--name", default="Nourozi_23726011_CompVP_Part2")
    ap.add_argument("--outdir", default=str(C.ROOT / "submission"))
    a = ap.parse_args()
    ids = C.selection()
    res, out = Path(a.results), Path(a.outdir)
    stage = out / a.name
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "Original").mkdir(parents=True)
    (stage / "Mooney").mkdir()
    for k in ids:
        shutil.copy2(C.ROOT / "data/Original" / f"{k}.jpg", stage / "Original" / f"{k}.jpg")
        shutil.copy2(C.ROOT / "data/Mooney" / f"{k}.png", stage / "Mooney" / f"{k}.png")
    for d in ["SteadyState", "RGBFirst", "MooneyFirst", "Comparisons"]:
        shutil.copytree(res / d, stage / "Results" / d)
    shutil.copy2(a.video, stage / Path(a.video).name)
    shutil.copy2(a.readme, stage / "README.md")

    # checks: counts, every image has its visualisations, video duration and codecs
    counts = {d: len(list((stage / "Results" / d).glob("*.png"))) for d in ["SteadyState", "RGBFirst", "MooneyFirst"]}
    assert counts == {"SteadyState": 20, "RGBFirst": 20, "MooneyFirst": 20}, counts
    for k in ids:
        for d in ["SteadyState", "RGBFirst", "MooneyFirst"]:
            assert len(list((stage / "Results" / d).glob(f"{k}_*.png"))) == 2, (d, k)
    probe = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,codec_name",
         "-of", "json", str(stage / Path(a.video).name)], capture_output=True, text=True).stdout)
    dur = float(probe["format"]["duration"])
    codecs = sorted((s["codec_type"], s["codec_name"]) for s in probe["streams"])
    assert 55 <= dur <= 60.0, dur
    assert codecs == [("audio", "aac"), ("video", "h264")], codecs

    zpath = out / f"{a.name}.zip"
    if zpath.exists():
        zpath.unlink()
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(stage).as_posix())
    md5 = hashlib.md5(zpath.read_bytes()).hexdigest()
    (out / f"{a.name}.zip.md5").write_text(f"{md5}  {zpath.name}\n")
    print(json.dumps(dict(zip=str(zpath), size_mb=round(zpath.stat().st_size / 1e6, 1), md5=md5, counts=counts,
                          video_s=round(dur, 2), codecs=codecs), indent=1))


if __name__ == "__main__":
    main()
