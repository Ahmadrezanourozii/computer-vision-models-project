"""Collect numbers, texts and image paths for the deck -> work/deck_input.json.

    python slides/prepare_deck.py --data data --analysis work/analysis --shots work/shots \
        --selection work/selection.json -o work/deck_input.json
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image

LIGHTS = ["left", "right", "top", "bottom"]


def p(v, d=0):
    return "–" if v is None or v != v else f"{100 * v:.{d}f}%"


def f(v, d=2):
    return "–" if v is None or v != v else f"{v:.{d}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--analysis", default="work/analysis")
    ap.add_argument("--shots", default="work/shots")
    ap.add_argument("--selection", default="work/selection.json")
    ap.add_argument("--params", default="work/mooney_params.json")
    ap.add_argument("-o", "--out", default="work/deck_input.json")
    a = ap.parse_args()
    data, A, shots = Path(a.data), Path(a.analysis), Path(a.shots)
    S = json.load(open(A / "summary.json"))
    meta = pd.read_csv(data / "metadata.csv").set_index("id")
    sel = json.load(open(a.selection))["top10"]
    prm = json.load(open(a.params))
    blur, white = prm["blur"], prm["white_percent"]
    n = S["n"]
    D, V, F, M = S.get("depth", {}), S.get("vlm", {}), S.get("faces", {}), S.get("marigold", {})
    vt = V.get("table", {})
    vo, vm = vt.get("Original", {}), vt.get("Mooney", {})

    top10 = [dict(orig=str(data / "Original" / f"{k}.jpg"), mooney=str(data / "Mooney" / f"{k}.png"),
                  label=f"{k.replace('face_', '#')} · {meta.loc[k, 'light']}") for k in sel]
    mosaic = [str(data / "Mooney" / f"{k}.png") for k in sel[:9]]

    deck = dict(
        title="Do Vision Models See Mooney Faces?",
        subtitle=(f"Four state-of-the-art vision systems tested on {n} photorealistic portraits under extreme "
                  "single-source lighting and on their two-tone Mooney versions"),
        date="November 2026", title_tiles=mosaic,
        approach_title="Approach: extreme lighting, one Mooney tool, four vision systems",
        approach_cols=[
            dict(head="Stimuli", lines=[
                f"{n} photorealistic synthetic portraits (Z-Image-Turbo, 768 px): no real persons",
                "One hard light from the left, right, above or below (25 each), black background",
                "Balanced: 50 female / 50 male, 3 age groups, 6 ethnic backgrounds, frontal and 3/4 views",
                "Quality control: one clear face (SCRFD) and measured shading direction = intended light",
                "Ground truth from the design: light direction, gender, age (+ measured head yaw)"]),
            dict(head="Mooney tool", lines=[
                "Grayscale → Gaussian blur → threshold at the luminance percentile that leaves w % of the pixels white",
                "Closed form of the threshold search in Egger et al.; no per-image hand tuning",
                f"Same parameters for all images: σ = {blur:g} % of the image side, {white:g} % white (chosen with a sweep)",
                "GUI (live sliders, batch, top-10 picker) + CLI (batch 100+ images, CSV log); 3D-mesh input"]),
            dict(head="Evaluation", lines=[
                "All originals and Mooneys through 4 systems with an identical protocol",
                "Depth: agreement with the original's depth; 'bright = near' shortcut",
                "VLM: face, gender, age, head orientation and light direction vs. ground truth",
                "Face DCNN: detection and ArcFace identification (Paper 1 used VGG-Face)",
                "Inverse rendering: normals; are black regions explained as shadow or as albedo?"]),
        ],
        approach_question=("Question: humans see a 3D face in a good Mooney image; Paper 1 explains this with inverse "
                           "rendering (shape + extreme illumination). Do modern feed-forward vision models recover the face, "
                           "its 3D shape and the lighting from the same two-tone image?"),
        top10=top10,
        top10_criteria=[
            dict(head="Readable, not trivial", text="A face appears for a human observer, but no single blob is a face part on its own."),
            dict(head="Large coherent blobs", text="Shadows form a few big regions: no speckle from skin texture, hair strands or clothing."),
            dict(head="Lost contours", text="Part of the outline merges with the black background, so perceptual closure is needed (Mooney, 1957)."),
        ],
        systems=[
            dict(system="Monocular depth estimation", model="Depth Anything V2 Large (depth-anything/Depth-Anything-V2-Large-hf)",
                 output="Relative depth map; depth rendered as a lit 3D relief", source="Yang et al. 2024 · Hugging Face · CC-BY-NC-4.0"),
            dict(system="Vision-language model", model="Qwen3-VL-8B-Instruct (fp16, greedy decoding)",
                 output="Free description + structured answers: face, gender, age, head orientation, light direction, black regions",
                 source="Qwen Team 2025 · Hugging Face · Apache-2.0"),
            dict(system="Face analysis DCNN", model="InsightFace buffalo_l: SCRFD-10G, 106-pt landmarks, gender/age, ArcFace R50",
                 output="Face detection + score, landmarks, attributes, 512-d identity embedding",
                 source="Guo et al. 2022; Deng et al. 2019 · non-commercial research"),
            dict(system="Diffusion inverse rendering", model="Marigold v1-1: normals + IID-Lighting (albedo, diffuse shading, residual)",
                 output="Surface normals; albedo vs. shading decomposition", source="Ke et al. 2024/25 · Hugging Face · OpenRAIL++"),
        ],
        systems_note=("All systems are used off-the-shelf (pretrained, no fine-tuning) on Kaggle (2× NVIDIA T4, 16 GB). Originals "
                      "(768 px JPEG) and Mooneys (768 px binary PNG, fed as 3-channel RGB) get identical pre-processing. Per-image "
                      "outputs are in Result1–Result4/{Original, Mooney} of the zip."),
        results=[], sizes={},
    )

    # --- system 1 depth
    if D:
        deck["results"].append(dict(
            system="Depth Anything V2", title="Depth: the Mooney is read as an embossed pattern",
            panel=str(A / "depth_examples.png"),
            panel_caption="Original → predicted depth → depth rendered as 3D relief; right: the same for its Mooney image.",
            stats=[dict(big=f"r = {f(D['r_mean'])}", label="depth agreement Mooney vs. original (face region)"),
                   dict(big=f(D["partial_pattern"]), label="extra correlation with the white pattern (partial r)", color="moon")],
            chart=dict(title="|r| between predicted depth and the black/white pattern", labels=LIGHTS, fmt="0.00", max=1,
                       series=[dict(name="depth of original", values=[abs(D.get("by_light_pattern_orig", {}).get(l, 0)) for l in LIGHTS], color="orig"),
                               dict(name="depth of Mooney", values=[abs(D.get("by_light_pattern_mooney", {}).get(l, 0)) for l in LIGHTS], color="moon")]),
            takeaway=D.get("takeaway", "")))
    # --- system 2 VLM
    if V:
        keys = [("says_face", "Face present"), ("gender_acc", "Gender"), ("age_acc", "Age group"),
                ("light_acc", "Light direction"), ("orientation_acc", "Head orientation")]
        deck["results"].append(dict(
            system="Qwen3-VL-8B", title="VLM: the face disappears in the Mooney image",
            panel=str(A / "vlm_examples.png"), panel_caption="Unprimed descriptions (first sentence) of an original and its Mooney image.",
            stats=[dict(big=p(vo.get("says_face")), label="'contains a face' for originals"),
                   dict(big=p(vm.get("says_face")), label="'contains a face' for Mooneys", color="moon")],
            chart=dict(title="Answers correct (ground truth from the stimulus design)", labels=[k[1] for k in keys],
                       series=[dict(name="Original", values=[vo.get(k[0], 0) for k in keys], color="orig"),
                               dict(name="Mooney", values=[vm.get(k[0], 0) for k in keys], color="moon")]),
            takeaway=V.get("takeaway", "")))
    # --- system 3 faces
    if F:
        cmc = F.get("cmc_oracle", [])
        deck["results"].append(dict(
            system="InsightFace", title="Face DCNN: the detector misses Mooney faces",
            panel=str(A / "face_examples.png"), panel_caption="SCRFD detections (green: score ≥ 0.5) with landmarks, gender and age.",
            stats=[dict(big=p(F["det_rate_orig"]), label="faces detected in originals"),
                   dict(big=p(F["det_rate_mooney"]), label="faces detected in Mooneys", color="moon")],
            chart=dict(title="Mooney → original identification among 100 (ArcFace)", labels=["Rank-1", "Rank-5", "Rank-10"],
                       series=[dict(name="Mooney (aligned with original key points)", values=[cmc[0], cmc[4], cmc[9]] if cmc else [0, 0, 0], color="moon"),
                               dict(name="Chance", values=[0.01, 0.05, 0.10], color="grey")]),
            takeaway=F.get("takeaway", "")))
    # --- system 4 Marigold
    if M:
        deck["results"].append(dict(
            system="Marigold", title="Inverse rendering: black regions become paint, not shadow",
            panel=str(A / "marigold_examples.png"),
            panel_caption="Marigold normals, albedo and diffuse shading for an original (left) and its Mooney image (right).",
            stats=[dict(big=f(M["shadow_index_orig"]), label="shadow index, originals (1 = all contrast is shading)"),
                   dict(big=f(M["shadow_index_mooney"]), label="shadow index, Mooneys", color="moon")],
            chart=dict(title="Share of the black/white contrast explained as shading", labels=LIGHTS, fmt="0.00", max=1,
                       series=[dict(name="Original", values=[M["by_light_shadow_orig"].get(l, 0) for l in LIGHTS], color="orig"),
                               dict(name="Mooney", values=[M["by_light_shadow_mooney"].get(l, 0) for l in LIGHTS], color="moon")]),
            takeaway=M.get("takeaway", "")))

    # --- summary
    summ = S.get("summary_slide")
    if summ:
        deck["summary"] = summ

    deck.update(
        shot_gui=str(shots / "gui_image.png"), shot_cli=str(shots / "cli_batch.png"), shot_code=str(shots / "code.png"),
        tool_text=("GUI (python -m mooney.gui): live preview of grayscale, blur and Mooney, sliders for blur strength and % white, "
                   "histogram with the threshold; extra tabs for 3D meshes, batch processing and the top-10 selection.\n"
                   "CLI (python -m mooney convert): batch-processes folders of 100+ images in parallel and logs σ, threshold "
                   "and the achieved % white for every image."),
        code_title="The code that generates the Mooney images",
        code_text=("1) Luminance: ITU-R BT.601 grayscale.\n"
                   "2) Blur: Gaussian, σ given in % of the shorter image side, so the result does not depend on resolution.\n"
                   "3) Threshold: the (100 − w)-th percentile of the blurred image, so exactly w % of the pixels become white "
                   "(Egger et al. search the same threshold by bisection).\n"
                   "Options: despeckling, contrast-polarity inversion, % white inside an object mask (3D meshes)."),
        ai_cols=[
            dict(head="Where AI assistance was used", lines=[
                "Coding assistant (Claude, Anthropic): drafting and debugging the Python code of the Mooney tool (CLI + GUI), "
                "the GPU job scripts and the analysis scripts",
                "Drafting the wording and layout of these slides",
                "Discussing options: which open models fit a 16 GB T4 GPU, how to measure the results"]),
            dict(head="My own work and decisions", lines=[
                "Final choice of the idea, the stimulus design and the four vision systems",
                "Choice of the Mooney parameters and the manual selection of the top-10 images",
                "Running all experiments, checking the outputs, interpreting the results",
                "Reviewing the code; responsibility for everything in this submission"]),
        ],
        ai_note=("The 100 original images are AI-generated on purpose (Z-Image-Turbo, Apache-2.0) to control the lighting and to "
                 "avoid showing real persons. All vision systems are pretrained models used without fine-tuning."),
        references=[
            "[1] B. Egger, M. Siegel, R. Arora, A. A. Soltani, I. Yildirim, J. Tenenbaum. Inverse rendering best explains face "
            "perception under extreme illuminations. CogSci 2020.",
            "[2] C. M. Mooney. Age in the development of closure ability in children. Canadian Journal of Psychology, 1957.",
            "[3] J. K. Lappalainen et al. Connectome-constrained networks predict neural activity across the fly visual system. "
            "Nature 634, 2024.",
            "[4] L. Yang et al. Depth Anything V2. NeurIPS 2024.",
            "[5] Qwen Team. Qwen3-VL (Qwen3-VL-8B-Instruct), 2025.",
            "[6] J. Guo et al. Sample and computation redistribution for efficient face detection (SCRFD). ICLR 2022. "
            "J. Deng et al. ArcFace. CVPR 2019. InsightFace buffalo_l model pack.",
            "[7] B. Ke et al. Repurposing diffusion-based image generators for monocular depth estimation (Marigold). CVPR 2024; "
            "Marigold v1-1 normals and intrinsic image decomposition models, 2025.",
            "[8] Tongyi-MAI. Z-Image: an efficient image generation foundation model with single-stream diffusion transformer, 2025.",
            "[9] O. M. Parkhi, A. Vedaldi, A. Zisserman. Deep face recognition (VGG-Face). BMVC 2015.",
            "Code: github.com/Ahmadrezanourozii/computer-vision-models-project",
        ],
    )
    paths = [x for r in deck["results"] for x in [r["panel"]]] + [deck["shot_gui"], deck["shot_cli"], deck["shot_code"]]
    deck["sizes"] = {q: Image.open(q).size for q in paths if Path(q).exists()}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(deck, open(a.out, "w"), indent=1, ensure_ascii=False)
    print("->", a.out)


if __name__ == "__main__":
    main()
