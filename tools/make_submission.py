"""Assemble the Part-1 submission zip in exactly the required folder structure.

    python tools/make_submission.py --data data --results results --pptx work/deck.pptx \
        --selection work/selection.json --name Nourozi_23726011_CompVP_Part1

Zip layout (as on the 'Part 1 deliverables' slide):
    Original/            100 original images
    Mooney/              100 Mooney images
    Subset/Original/     the 10 manually selected originals
    Subset/Mooney/       the 10 manually selected Mooney images
    Result1..Result4/Original/  results of vision system n for all originals
    Result1..Result4/Mooney/    results of vision system n for all Mooney images
    <name>.pptx          the presentation
Large raw arrays that are only needed for the analysis are left out (they stay in the repository).
"""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

EXCLUDE = {"marigold_raw.npz"}            # ~150 MB per set; the per-image JPEGs are the delivered results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--results", default="results")
    ap.add_argument("--pptx", required=True)
    ap.add_argument("--selection", default="work/selection.json")
    ap.add_argument("--name", default="Nourozi_23726011_CompVP_Part1")
    ap.add_argument("--outdir", default="submission")
    a = ap.parse_args()
    data, res, out = Path(a.data), Path(a.results), Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    sel = json.load(open(a.selection))["top10"]
    assert len(sel) == 10, f"selection must contain 10 images, has {len(sel)}"
    zpath = out / f"{a.name}.zip"
    counts = {}
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        def add(src, arc):
            z.write(src, arc)
            top = "/".join(arc.split("/")[:2]) if arc.startswith(("Subset", "Result")) else arc.split("/")[0]
            counts[top] = counts.get(top, 0) + 1
        for f in sorted((data / "Original").glob("*.jpg")):
            add(f, f"Original/{f.name}")
        for f in sorted((data / "Mooney").glob("*.png")):
            add(f, f"Mooney/{f.name}")
        for k in sel:
            add(data / "Original" / f"{k}.jpg", f"Subset/Original/{k}.jpg")
            add(data / "Mooney" / f"{k}.png", f"Subset/Mooney/{k}.png")
        for n in range(1, 5):
            for s in ("Original", "Mooney"):
                for f in sorted((res / f"Result{n}" / s).glob("*")):
                    if f.is_file() and f.name not in EXCLUDE:
                        add(f, f"Result{n}/{s}/{f.name}")
        add(Path(a.pptx), f"{a.name}.pptx")
    md5 = hashlib.md5(zpath.read_bytes()).hexdigest()
    (out / f"{a.name}.zip.md5").write_text(f"{md5}  {zpath.name}\n")
    print(json.dumps(counts, indent=1))
    print(f"{zpath}  {zpath.stat().st_size / 1e6:.1f} MB  md5 {md5}")
    assert counts.get("Original") == 100 and counts.get("Mooney") == 100, "need exactly 100 originals and 100 Mooneys"
    assert counts.get("Subset/Original") == 10 and counts.get("Subset/Mooney") == 10


if __name__ == "__main__":
    main()
