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

    Hh0 = S.get("human", {})
    if D:
        D["takeaway"] = (f"Inside the face, depth predicted from the Mooney does not match the original's depth (mean r = {f(D['r_mean'])}, "
                         f"normals off by {D['normal_err']:.0f}°). It follows the paint instead: white = near (partial r = "
                         f"{f(D['partial_pattern'])} with the white pattern). The 3D relief is an embossed blob, not a face.")
    if V:
        V["takeaway"] = (f"Qwen3-VL calls {p(vm.get('says_face'))} of the Mooneys a face (originals {p(vo.get('says_face'))}); otherwise it "
                         f"sees ears, hands, ghosts or abstract shapes. Gender and age collapse ({p(vm.get('gender_acc'))}, "
                         f"{p(vm.get('age_acc'))}) and the black regions are never called shadows. Even for the originals the light "
                         f"direction is weak ({p(vo.get('light_acc'))}): it nearly always answers 'from the left' or 'from the front'.")
    if F:
        F["takeaway"] = (f"SCRFD finds {p(F['det_rate_mooney'])} of the Mooney faces (originals {p(F['det_rate_orig'])}). Even when the Mooney "
                         f"is aligned with the original's key points, ArcFace picks the right person out of 100 in only "
                         f"{p(F['rank1_oracle'])} of the cases (chance 1%; top-10: {p(F['cmc_oracle'][9])} vs. 10%).")
    if M:
        M["takeaway"] = (f"For originals {M['shadow_index_orig']:.2f} of the white/black contrast is explained as shading, for Mooneys only "
                         f"{M['shadow_index_mooney']:.2f}: the black regions are read as dark paint (albedo) and the normals are flat "
                         f"({M['normal_err']:.0f}° error). Humans read the same black regions as shadows.")
    if Hh0 and V and M:
        fs, vb = Hh0["face_share_by_light"], V.get("mooney_says_face_by_light", {})
        Hh0["points"] = [
            f"Human observer (the author, n = 1, each Mooney rated alone, without its original): a face (rating ≥ 4) in {p(fs['top'])} of the top-lit, "
            f"{p(fs['left'])}/{p(fs['right'])} of the left/right-lit and {p(fs['bottom'])} of the bottom-lit Mooneys.",
            f"Qwen3-VL shows the same order: 'contains a face' for {p(vb.get('top'))} of the top-lit vs. {p(vb.get('bottom'))} of "
            "the bottom-lit Mooneys.",
            f"Marigold, even on the ORIGINAL photos, explains bottom-lit contrast less as shading ({M['by_light_shadow_orig']['bottom']:.2f} "
            f"vs. {min(M['by_light_shadow_orig'][l] for l in ['left','right','top']):.2f}–{max(M['by_light_shadow_orig'][l] for l in ['left','right','top']):.2f}; "
            "also when the hairline is excluded): consistent with a light-from-above prior.",
            ("First Mooney version (no crop, 10% white): same ranking in the blind ratings (mean top "
             f"{Hh0['v1_mean_by_light']['top']:.1f} vs. bottom {Hh0['v1_mean_by_light']['bottom']:.1f}).") if "v1_mean_by_light" in Hh0 else "",
        ]
        Hh0["points"] = [x for x in Hh0["points"] if x]
        Hh0["takeaway"] = ("H2 supported: light from below removes the face for the human observer and for the VLM, and the "
                           "inverse-rendering model behaves as if light came from above.")
        corr = Hh0["corr_with_human"]
        cv = corr.get("VLM face confidence", {})
        ca = corr.get("ArcFace similarity to original", {})
        S["summary_title"] = "Summary: machines see paint, not faces"
        S["summary_takeaway"] = (f"Per image, only Qwen3-VL's face confidence follows the human ratings (ρ = {cv.get('rho', 0):.2f}, 95% CI "
                                 f"{cv.get('lo', 0):.2f}–{cv.get('hi', 0):.2f}), ArcFace weakly ({ca.get('rho', 0):.2f}). No system recovers "
                                 "3D shape or lighting from a Mooney face; consistent with Paper 1, feed-forward models lack the "
                                 "inverse-rendering explanation (shape + extreme light) that humans seem to use.")
        S["human"] = Hh0

    top10 = [dict(orig=str(data / "Original" / f"{k}.jpg"), mooney=str(data / "Mooney" / f"{k}.png"),
                  label=f"{k.replace('face_', '#')} · {meta.loc[k, 'light']}") for k in sel]
    order = ["face_031", "face_029", "face_010", "face_004", "face_027", "face_033", "face_048", "face_054", "face_058"]
    order = [k for k in order if k in sel] + [k for k in sel if k not in order]
    mosaic = [str(data / "Mooney" / f"{k}.png") for k in order[:9]]

    deck = dict(
        title="Do Modern Vision Systems Perceive Faces in Mooney Images?",
        subtitle=(f"Four state-of-the-art vision systems tested on {n} photorealistic portraits under extreme "
                  "single-source lighting and on their two-tone Mooney versions"),
        date="November 2026", title_tiles=mosaic,
        approach_title="Approach: extreme lighting, one Mooney tool, four vision systems",
        approach_cols=[
            dict(head="Stimuli", lines=[
                f"{n} photorealistic synthetic portraits (Z-Image-Turbo): no real persons",
                "One hard light from the left, right, above or below (25 each), black background",
                "Balanced: 50 female / 50 male, 3 age groups, 6 ethnic backgrounds, frontal and 3/4 views",
                "Quality control on 204 candidates: one clear face, light direction verified twice (shading gradient + "
                "facial key-point regions)",
                "Ground truth from the design: light direction, gender, age (+ measured head yaw)"]),
            dict(head="Mooney tool", lines=[
                "Grayscale → Gaussian blur → threshold at the luminance percentile that leaves w % of the pixels white "
                "(closed form of the threshold search in Egger et al.)",
                f"Same parameters for all images: σ = {blur:g} % of the image side, {white:g} % white (global sweep + visual check)",
                "Face-centred crops (1.7× face box, 512 px) normalise head size; v1 without crop was too hard for the "
                "blind human check (13% rated ≥ 3) and was revised",
                "GUI (live sliders, batch, top-10 picker) + CLI (batch 100+ images, CSV log); 3D-mesh input"]),
            dict(head="Evaluation", lines=[
                "All originals and Mooneys through 4 systems with an identical protocol",
                "Depth: agreement with the original's depth; 'bright = near' shortcut",
                "VLM: face, gender, age, head orientation and light direction vs. ground truth",
                "Face DCNN: detection and ArcFace identification (Paper 1 used VGG-Face)",
                "Inverse rendering: normals; are black regions explained as shadow or as albedo?"]),
        ],
        approach_question=("Hypotheses. H1: humans see a 3D face in a good Mooney image (Paper 1: inverse rendering of shape + "
                           "extreme light); feed-forward vision models do not. H2 (light-from-above prior; Ramachandran 1988, Johnston et "
                           "al. 1992): humans find faces lit from below harder; do vision models show the same bias for bottom-lit Mooneys?"),
        top10=top10,
        top10_note=("The 10 Mooneys most clearly seen as a face in the author's blind 1–5 rating (done before the originals "
                    "were shown): both images rated 5, then 8 of the 16 rated 4. All ten happen to be lit from "
                    "above, in line with the light-from-above prior."),
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
        systems_note=("All systems are used off-the-shelf (pretrained, no fine-tuning): R1, R2 and R4 on Kaggle (2× NVIDIA T4, 16 GB), "
                      "R3 on CPU (onnxruntime). Originals (512 px JPEG) and Mooneys (512 px binary PNG, fed as 3-channel RGB) get "
                      "identical pre-processing. Per-image outputs are in Result1–Result4/{Original, Mooney} of the zip."),
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
        keys = [("says_face", "Face"), ("gender_acc", "Gender"), ("age_acc", "Age"),
                ("light_acc", "Light dir."), ("orientation_acc", "Head pose")]
        deck["results"].append(dict(
            system="Qwen3-VL-8B", title="VLM: the face disappears in the Mooney image",
            panel=str(A / "vlm_examples.png"), panel_caption="Unprimed descriptions (first two sentences) of two originals and their Mooney images.",
            stats=[dict(big=p(vo.get("says_face")), label="'contains a face' for originals"),
                   dict(big=p(vm.get("says_face")), label="'contains a face' for Mooneys", color="moon")],
            chart=dict(title="Answers correct (ground truth from the stimulus design)", labels=[k[1] for k in keys], legend="b",
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
            panel_caption="Marigold normals, albedo and diffuse shading: originals (left) and their Mooney images (right); bottom row lit from below.",
            stats=[dict(big=f(M["shadow_index_orig"]), label="shadow index, originals (1 = all contrast is shading)"),
                   dict(big=f(M["shadow_index_mooney"]), label="shadow index, Mooneys", color="moon")],
            chart=dict(title="Share of the black/white contrast explained as shading", labels=LIGHTS, fmt="0.00", max=1,
                       series=[dict(name="Original", values=[M["by_light_shadow_orig"].get(l, 0) for l in LIGHTS], color="orig"),
                               dict(name="Mooney", values=[M["by_light_shadow_mooney"].get(l, 0) for l in LIGHTS], color="moon")]),
            takeaway=M.get("takeaway", "")))

    # --- light-from-above slide (H2) and summary
    Hh = S.get("human")
    if Hh and V and F and D and M:
        mo_face = V.get("mooney_says_face_by_light", {})
        mo_mention = V.get("mooney_mentions_face_by_light", {})
        deck["light_slide"] = dict(
            title="Light from below: no face for humans or machines",
            chart=dict(title="Mooney images: evidence for a face, by light direction", labels=["above", "left", "right", "below"],
                       fmt="0%", max=1, series=[
                           dict(name="Human: 'probably/clearly a face' (rating ≥ 4)",
                                values=[Hh["face_share_by_light"][l] for l in ["top", "left", "right", "bottom"]], color="dark"),
                           dict(name="Qwen3-VL: 'contains a face'", values=[mo_face.get(l, 0) for l in ["top", "left", "right", "bottom"]],
                                color="moon"),
                           dict(name="Qwen3-VL: face/head/person in free description",
                                values=[mo_mention.get(l, 0) for l in ["top", "left", "right", "bottom"]], color="teal"),
                       ]),
            points=Hh.get("points", []), takeaway=Hh.get("takeaway", ""))
        corr = Hh["corr_with_human"]
        names = [k for k in corr if corr[k]["rho"] is not None]
        deck["summary"] = dict(
            title=S.get("summary_title", "Summary: only the VLM partly follows human Mooney perception"),
            main=dict(title="Original vs. Mooney (all 100 images)", fmt="0%", max=1,
                      labels=["Face detected (SCRFD)", "VLM: 'a face'", "Black = shadow (VLM)", "Shadow index (Marigold)"],
                      series=[dict(name="Original", color="orig", values=[F["det_rate_orig"], vo.get("says_face", 0),
                                                                          vo.get("black_is_shadow", 0), M["shadow_index_orig"]]),
                              dict(name="Mooney", color="moon", values=[F["det_rate_mooney"], vm.get("says_face", 0),
                                                                        vm.get("black_is_shadow", 0), M["shadow_index_mooney"]])]),
            by_light=dict(title="Spearman ρ with the human rating (per Mooney image)", fmt="0.00", max=1, min=-0.4,
                          labels=[{"VLM face confidence": "VLM conf.", "VLM says 'face'": "VLM 'face'", "SCRFD detection score": "SCRFD",
                                   "ArcFace similarity to original": "ArcFace", "Depth agreement r": "Depth r",
                                   "Shadow index": "Shadow idx"}.get(n, n) for n in names], show_values=True,
                          series=[dict(name="ρ", color="dark", values=[round(corr[n]["rho"], 3) for n in names])]),
            takeaway=S.get("summary_takeaway", ""))

    deck.update(
        shot_gui=str(shots / "gui_image.png"), shot_cli=str(shots / "cli_batch.png"), shot_code=str(shots / "code.png"),
        shot_batch=str(shots / "gui_batch.png"),
        tool_text=("Left: GUI (python -m mooney.gui) with sliders for blur strength and % white, live preview and threshold "
                   "histogram. Right: the CLI batch run that produced the 100 Mooneys, and the GUI batch tab."),
        code_title="The code that generates the Mooney images",
        code_text=("1) Luminance: ITU-R BT.601 grayscale.\n"
                   "2) Blur: Gaussian, σ given in % of the shorter image side, so the result does not depend on resolution.\n"
                   "3) Threshold: the (100 − w)-th percentile of the blurred image, so exactly w % of the pixels become white "
                   "(Egger et al. search the same threshold by bisection).\n"
                   "Options: despeckling, contrast-polarity inversion, % white inside an object mask (3D meshes)."),
        ai_cols=[
            dict(head="Where AI assistance was used", lines=[
                "Claude (Anthropic) as coding and research assistant: wrote and debugged most of the Python code (Mooney "
                "tool CLI + GUI, image generation and vision-system jobs, analysis) and ran the GPU jobs on Kaggle",
                "Proposed options (stimulus design, models that fit a 16 GB T4 GPU, metrics) and drafted the wording and "
                "layout of these slides"]),
            dict(head="My own work and decisions", lines=[
                "Chose the idea and the four vision systems from the proposed options and revised the plan with my comments "
                "(e.g. focus on light from above, landmark-based light check, 1–5 ratings)",
                "Blind face ratings of all Mooney images (two rounds) and the manual top-10 selection",
                "Reviewed the results and the slides; I am responsible for this submission"]),
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
            "[10] V. S. Ramachandran. Perception of shape from shading. Nature 331, 163–166, 1988.",
            "[11] A. Johnston, H. Hill, N. Carman. Recognising faces: effects of lighting direction, inversion, and brightness "
            "reversal. Perception 21, 365–375, 1992.",
            "Code: github.com/Ahmadrezanourozii/computer-vision-models-project",
        ],
    )
    paths = [x for r in deck["results"] for x in [r["panel"]]] + [deck["shot_gui"], deck["shot_cli"], deck["shot_code"],
                                                                     deck["shot_batch"]]
    deck["sizes"] = {q: Image.open(q).size for q in paths if Path(q).exists()}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(deck, open(a.out, "w"), indent=1, ensure_ascii=False)
    print("->", a.out)


if __name__ == "__main__":
    main()
