"""One-minute findings video for Project Part 2.

    python part2_flyvis/make_video.py --results work/part2/Results --out work/part2/video [--voice <piper .onnx>]
                                      [--audio my_voice.wav]

Builds five 1920x1080 slides (findings only), the narration script, a PowerPoint file with the slides
and the script in the speaker notes (to record your own voice in PowerPoint: Record -> Export -> MP4),
and an MP4 (H.264 High / yuv420p + AAC 48 kHz, 'faststart', plays in VLC) of exactly --duration seconds.
Audio: --audio (your own recording, one file for the whole video) or --voice (local Piper text-to-speech
draft, run with the 'tts' virtual environment's python via --piper-python).
"""
import argparse
import json
import subprocess
import wave
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402
from PIL import Image  # noqa: E402

import common as C  # noqa: E402

BLUE, INK, GREY, MOON, PHOTO = "#04316A", "#1A1A1A", "#6B7680", "#04316A", "#8C9FB1"
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 20, "axes.edgecolor": "#444444"})
W, H = 19.2, 10.8           # inches at dpi 100 -> 1920 x 1080
FOOT = "Ahmadreza Nourozi  |  Computational Visual Perception, Project Part 2  |  FAU Erlangen-Nürnberg"

# narration: findings only (what we observed), ~140 words for one minute
SCRIPT = [
    ("How does a fly's visual system see a Mooney face? We showed our ten faces and their Mooney images to the "
     "connectome-constrained fly model, each as a static video."),
    ("First: to the fly eye, a Mooney is almost the photo. Its seven hundred and twenty-one ommatidia blur the "
     "details, so inputs and responses correlate at about zero point nine."),
    ("Second: the hard black and white of the Mooney drives the network harder, with about one point seven times "
     "stronger response patterns."),
    ("Third: after a switch, ON cells like Mi1 and Tm3 depolarize where the image got brighter, OFF cells like "
     "Mi9 and Tm9 where it got darker."),
    ("As the Mooney exaggerates the photo, the Mooney pops out after the photo, but the photo is masked after "
     "the Mooney."),
    ("Fourth: this fades within one second, then the activity equals the second image alone. Unlike humans, the "
     "fly model keeps no memory of the photo."),
]
# how the text-to-speech engine should pronounce cell-type names
TTS_FIX = {"Mi1": "M I one", "Tm3": "T M three", "Mi9": "M I nine", "Tm9": "T M nine", "ON cells": "On cells",
           "OFF cells": "Off cells", "Mooneys": "Mooney images", "connectome-constrained": "connectome constrained"}


def slide_base(title, number):
    fig = plt.figure(figsize=(W, H), dpi=100)
    fig.patch.set_facecolor("white")
    fig.text(0.04, 0.905, title, fontsize=38, fontweight="bold", color=BLUE, va="center")
    fig.text(0.04, 0.035, FOOT, fontsize=13, color=GREY, va="center")
    fig.text(0.96, 0.035, str(number), fontsize=13, color=GREY, va="center", ha="right")
    return fig


def add_img(fig, img, rect, title=None, title_size=17):
    ax = fig.add_axes(rect)
    ax.imshow(img, cmap="gray", vmin=0, vmax=1) if np.ndim(img) == 2 else ax.imshow(img)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=title_size, color=INK, pad=8)
    return ax


def add_hex(fig, values, rect, title=None, cmap="binary_r", lo=0.0, hi=1.0):
    from flyvis.analysis.visualization.plots import hex_scatter
    from flyvis.utils.hex_utils import get_hex_coords
    u, v = get_hex_coords(C.EXTENT)
    ax = fig.add_axes(rect)
    hex_scatter(u, v, values, fig=fig, ax=ax, cmap=plt.get_cmap(cmap), vmin=lo, vmax=hi, cbar=False, fill=True)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title, fontsize=17, color=INK, pad=8)
    return ax


def arrow(fig, x0, y0, x1, y1):
    fig.patches.append(FancyArrowPatch((x0, y0), (x1, y1), transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=30, color=GREY, lw=2.5))


def slides(res, out, example):
    S = dict(np.load(res.parent / "sim_000.npz"))
    meta = json.load(open(res.parent / "sim_000.json"))
    summ = json.load(open(res / "Comparisons" / "summary.json"))
    i = meta["ids"].index(example)
    _, gray, moon = C.load_pairs(ids=[example])
    paths = []

    # 1 ------------------------------------------------------------------ title + stimuli
    fig = slide_base("How does a fly's visual system see a Mooney face?", 1)
    fig.text(0.04, 0.83, "10 face pairs from Part 1, shown as static videos to the connectome-constrained model of the "
             "fly visual system (flyvis; Lappalainen et al., Nature 2024)", fontsize=19, color=INK, va="center")
    add_img(fig, gray[0], [0.05, 0.22, 0.2, 0.5], "photo (grey-scale)")
    add_img(fig, moon[0], [0.27, 0.22, 0.2, 0.5], "Mooney")
    arrow(fig, 0.49, 0.47, 0.54, 0.47)
    add_hex(fig, S["hex_orig"][i], [0.55, 0.22, 0.2, 0.5], "fly eye: photo")
    add_hex(fig, S["hex_moon"][i], [0.76, 0.22, 0.2, 0.5], "fly eye: Mooney")
    fig.text(0.26, 0.15, "512 x 512 pixels", fontsize=16, color=GREY, ha="center")
    fig.text(0.755, 0.15, "721 ommatidia (one grey value each)", fontsize=16, color=GREY, ha="center")
    fig.text(0.04, 0.085, "Ahmadreza Nourozi  |  M.Sc. Artificial Intelligence  |  student no. 23726011",
             fontsize=16, color=INK)
    paths.append(out / "slide1.png"); fig.savefig(paths[-1]); plt.close(fig)

    # 2 ------------------------------------------------------------------ similarity
    fig = slide_base("1   To the fly eye, a Mooney is almost the photo", 2)
    fig.text(0.05, 0.70, f"r = {summ['r_input_fly_eye']:.2f}", fontsize=72, fontweight="bold", color=BLUE)
    fig.text(0.05, 0.64, "photo vs. Mooney, input on the fly eye\n(face region, mean of 10 faces)", fontsize=19,
             color=INK, va="top")
    fig.text(0.05, 0.47, f"r = {summ['r_input_full_resolution']:.2f}", fontsize=44, fontweight="bold", color=GREY)
    fig.text(0.05, 0.42, "the same two images at full resolution", fontsize=19, color=INK, va="top")
    ax = fig.add_axes([0.45, 0.17, 0.5, 0.6])
    reg = [("photo-\nreceptors", "retina"), ("lamina,\nmedulla", "intermediate"), ("T-shaped,\ntransmedullary", "output")]
    rows = {t: v for t, v in zip(meta["cell_types"], meta["regions"])}
    from make_figures import analyse
    face = C.face_masks(meta["ids"])
    an, _ = analyse(S, face)
    for k, (lab, g) in enumerate(reg):
        vals = np.array([an[t]["r_face"] for t in an if rows[t] == g])
        ax.bar(k, vals.mean(), color=BLUE, width=0.55)
        ax.plot([k, k], [vals.min(), vals.max()], color=INK, lw=2)
        ax.text(k + 0.3, vals.mean(), f"{vals.mean():.2f}", ha="left", va="center", fontsize=20, fontweight="bold",
                color=BLUE)
    ax.axhline(summ["r_input_fly_eye"], color=BLUE, ls="--", lw=1.5)
    ax.axhline(summ["r_input_full_resolution"], color=GREY, ls=":", lw=2)
    ax.set_xticks(range(3)); ax.set_xticklabels([r[0] for r in reg], fontsize=17)
    ax.set_ylim(0.6, 1.0); ax.set_ylabel("photo vs. Mooney response similarity (r)", fontsize=17)
    ax.tick_params(axis="y", labelsize=15); ax.set_xlim(-0.5, 2.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("responses after 2 s (steady state); line = range over cell types", fontsize=16, color=GREY)
    paths.append(out / "slide2.png"); fig.savefig(paths[-1]); plt.close(fig)

    # 3 ------------------------------------------------------------------ stronger responses
    fig = slide_base("2   The Mooney drives the network harder", 3)
    a = Image.open(res / "SteadyState" / f"{example}_original.png")
    b = Image.open(res / "SteadyState" / f"{example}_mooney.png")
    crop = lambda im: im.crop((0, int(im.height * 0.13), im.width, im.height))  # drop the long figure title
    add_img(fig, np.asarray(crop(a)), [0.02, 0.47, 0.72, 0.36])
    add_img(fig, np.asarray(crop(b)), [0.02, 0.09, 0.72, 0.36])
    fig.text(0.03, 0.835, "photo, steady state", fontsize=18, fontweight="bold", color=INK)
    fig.text(0.03, 0.455, "Mooney, steady state", fontsize=18, fontweight="bold", color=INK)
    fig.text(0.77, 0.62, f"x {summ['contrast_ratio_median']:.1f}", fontsize=80, fontweight="bold", color=BLUE)
    fig.text(0.77, 0.55, "stronger response pattern\nfor the Mooney\n(median of 65 cell types,\n10 faces)",
             fontsize=19, color=INK, va="top")
    fig.text(0.77, 0.25, "each panel: one cell type\n(black = low, white = high\nvoltage, same scale in\nboth rows)",
             fontsize=15, color=GREY, va="top")
    paths.append(out / "slide3.png"); fig.savefig(paths[-1]); plt.close(fig)

    # 4 ------------------------------------------------------------------ switch transient
    fig = slide_base("3   After a switch, ON and OFF cells signal what changed", 4)
    im = Image.open(res / "Comparisons" / f"comparison_3_switch_transient_{example}.png")
    im = im.crop((0, int(im.height * 0.05), im.width, int(im.height * 0.95)))
    add_img(fig, np.asarray(im), [0.03, 0.1, 0.94, 0.72])
    on = summ["switch_on"]; off = summ["switch_off"]
    fig.text(0.5, 0.075, f"correlation with the brightness change (10 faces):  Mi1 {on['Mi1']:+.2f}   Tm3 {on['Tm3']:+.2f}"
             f"   |   Mi9 {off['Mi9']:+.2f}   Tm9 {off['Tm9']:+.2f}", fontsize=17, color=INK, ha="center")
    paths.append(out / "slide4.png"); fig.savefig(paths[-1]); plt.close(fig)

    # 5 ------------------------------------------------------------------ masking asymmetry
    fig = slide_base("3   ... so the Mooney pops out, and the photo is masked", 5)
    from flyvis.analysis.visualization.plots import hex_scatter
    cols = [("Mooney\nafter grey", "mf_p1_t120"), ("Mooney\nafter photo", "rf_p2_t120"),
            ("photo\nafter grey", "rf_p1_t120"), ("photo\nafter Mooney", "mf_p2_t120")]
    from make_figures import IDX, HEXAL, on_hexals
    for r, t in enumerate(["Mi1", "Tm3"]):
        vals = np.concatenate([S[k][i, IDX[t]] for _, k in cols])
        lo, hi = np.percentile(vals, [1, 99])
        for c, (lab, key) in enumerate(cols):
            x0 = 0.1 + c * 0.14 + (0.03 if c >= 2 else 0)
            ax = add_hex(fig, on_hexals(t, S[key][i, IDX[t]]), [x0, 0.47 - r * 0.33, 0.13, 0.3],
                         lab if r == 0 else None, lo=lo, hi=hi)
        fig.text(0.085, 0.62 - r * 0.33, t, fontsize=22, fontweight="bold", color=INK, ha="right", va="center")
    fig.text(0.38, 0.12, "120 ms after the image appeared (face_033), same grey scale per row", fontsize=15,
             color=GREY, ha="center")
    fig.text(0.72, 0.66, f"r = {summ['masking_mooney_after_photo_median']:.2f}", fontsize=50, fontweight="bold",
             color="#31a354")
    fig.text(0.72, 0.6, "Mooney after the photo\nvs. Mooney after grey", fontsize=18, color=INK, va="top")
    fig.text(0.72, 0.42, f"r = {summ['masking_photo_after_mooney_median']:.2f}", fontsize=50, fontweight="bold",
             color="#756bb1")
    fig.text(0.72, 0.36, "photo after the Mooney\nvs. photo after grey", fontsize=18, color=INK, va="top")
    fig.text(0.72, 0.2, f"median of 65 cell types, 10 faces\n(Mi1: {summ['masking_examples']['Mi1'][1]:.2f}, "
             f"Tm3: {summ['masking_examples']['Tm3'][1]:.2f})", fontsize=15, color=GREY, va="top")
    paths.append(out / "slide5.png"); fig.savefig(paths[-1]); plt.close(fig)

    # 6 ------------------------------------------------------------------ no memory
    fig = slide_base("4   The previous image is forgotten within one second", 6)
    t2 = np.arange(1, 2 * C.N_STIM + 1) * C.DT
    t1 = t2[:C.N_STIM]
    types = meta["cell_types"]
    for k, (t, rect) in enumerate([("Mi1", [0.06, 0.45, 0.42, 0.28]), ("Tm9", [0.06, 0.12, 0.42, 0.28])]):
        j = types.index(t)
        ax = fig.add_axes(rect)
        ax.plot(t2, S["rf_central"][j, i], color="#31a354", lw=3, label="photo -> Mooney")
        ax.plot(t2, S["mf_central"][j, i], color="#756bb1", lw=3, label="Mooney -> photo")
        ax.plot(t1 + C.T_STIM, S["ss_central"][j, 10 + i], color="#31a354", lw=1.6, ls="--", label="Mooney alone")
        ax.plot(t1 + C.T_STIM, S["ss_central"][j, i], color="#756bb1", lw=1.6, ls="--", label="photo alone")
        ax.axvline(C.T_STIM, color=GREY, lw=1, ls=":")
        ax.set_xlim(0, 4); ax.spines[["top", "right"]].set_visible(False)
        ax.text(0.02, 0.88, t, transform=ax.transAxes, fontsize=20, fontweight="bold", color=INK)
        ax.tick_params(labelsize=14)
        if k == 0:
            ax.legend(fontsize=14, frameon=False, ncol=4, loc="lower left", bbox_to_anchor=(0.0, 1.02),
                      handlelength=1.6, columnspacing=1.0)
        else:
            ax.set_xlabel("time (s); switch at 2 s", fontsize=16)
    ax = fig.add_axes([0.56, 0.3, 0.38, 0.47])
    tt = np.arange(1, C.N_STIM + 1) * C.DT
    for key, col, lab in [("rf_order", "#31a354", "Mooney after photo"), ("mf_order", "#756bb1", "photo after Mooney")]:
        y = S[key].reshape(-1, S[key].shape[-1])
        ax.plot(tt, np.median(y, 0) * 100, color=col, lw=3, label=lab)
    ax.axhline(5, color=GREY, ls="--", lw=1.2)
    ax.set_xlim(0, 2); ax.set_ylim(0, 130); ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("time after the switch (s)", fontsize=16); ax.set_ylabel("effect of the previous image (%)", fontsize=16)
    ax.tick_params(labelsize=14); ax.legend(fontsize=14, frameon=False)
    ax.set_title(f"median of 65 cell types x 10 faces; < 5 % for all after {summ['order_effect_gone_after_s']:.1f} s",
                 fontsize=14, color=GREY)
    fig.text(0.555, 0.17, "Unlike humans, who see the face in a Mooney once they\nhave seen the photo, the fly model "
             "keeps no trace of it.", fontsize=18, color=BLUE, fontweight="bold", va="top")
    paths.append(out / "slide6.png"); fig.savefig(paths[-1]); plt.close(fig)
    return paths


def wav_read(p):
    with wave.open(str(p)) as w:
        sr, n, ch = w.getframerate(), w.getnframes(), w.getnchannels()
        x = np.frombuffer(w.readframes(n), np.int16).astype(np.float32) / 32768
    return (x.reshape(-1, ch).mean(1) if ch > 1 else x), sr


def tts(texts, voice, piper_python, out, length_scale):
    clips = []
    for k, t in enumerate(texts, 1):
        for a, b in TTS_FIX.items():
            t = t.replace(a, b)
        p = out / f"voice{k}.wav"
        subprocess.run([piper_python, "-m", "piper", "-m", voice, "-f", str(p), "--length-scale", str(length_scale),
                        "--sentence-silence", "0.25"], input=t.encode(), check=True, capture_output=True)
        clips.append(wav_read(p))
    return clips


def build(paths, clips, out, duration, lead=0.35, gap=0.55):
    """Lay the narration clips one after the other, each slide as long as its clip; total = duration."""
    sr = 48000
    segs = []
    for x, s in clips:
        n = int(round(len(x) * sr / s))
        segs.append(np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32))
    lens = [len(s) / sr + gap for s in segs]
    lens[0] += lead
    spare = duration - sum(lens)
    if spare < 0:
        raise SystemExit(f"narration too long by {-spare:.2f} s: shorten the script or speed it up")
    lens[-1] += spare                                   # the last slide stays until the end
    audio = np.zeros(int(round(duration * sr)), np.float32)
    starts = np.concatenate([[0.0], np.cumsum(lens)[:-1]])
    for k, (s, t0) in enumerate(zip(segs, starts)):
        i0 = int(round((t0 + (lead if k == 0 else 0.15)) * sr))   # speech starts shortly after each slide
        audio[i0:i0 + len(s)] = s[: len(audio) - i0]
    peak = np.abs(audio).max() + 1e-9
    audio = np.clip(audio / peak * 0.89, -1, 1)
    wav = out / "narration.wav"
    with wave.open(str(wav), "w") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((audio * 32767).astype(np.int16).tobytes())
    return lens, wav


def encode(paths, lens, wav, mp4, duration):
    lst = mp4.parent / "slides.txt"
    with open(lst, "w") as f:
        for p, L in zip(paths, lens):
            f.write(f"file '{p.resolve()}'\nduration {L:.3f}\n")
        f.write(f"file '{paths[-1].resolve()}'\n")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-i", str(wav),
           "-vf", "fps=30,scale=1920:1080,format=yuv420p", "-c:v", "libx264", "-profile:v", "high", "-level", "4.0",
           "-preset", "slow", "-crf", "20", "-tune", "stillimage", "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
           "-ac", "2", "-t", f"{duration:.3f}", "-movflags", "+faststart", str(mp4)]
    subprocess.run(cmd, check=True)
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_name,width,height,"
                            "pix_fmt,sample_rate,channels", "-of", "json", str(mp4)], capture_output=True, text=True)
    return json.loads(probe.stdout)


def pptx(paths, texts, lens, out):
    from pptx import Presentation
    from pptx.util import Emu
    prs = Presentation()
    prs.slide_width, prs.slide_height = Emu(12192000), Emu(6858000)
    for p, t, L in zip(paths, texts, lens):
        s = prs.slides.add_slide(prs.slide_layouts[6])
        s.shapes.add_picture(str(p), 0, 0, prs.slide_width, prs.slide_height)
        s.notes_slide.notes_text_frame.text = f"[about {L:.0f} s]  {t}"
    prs.save(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(C.ROOT / "work/part2/Results"))
    ap.add_argument("--out", default=str(C.ROOT / "work/part2/video"))
    ap.add_argument("--example", default="face_033")
    ap.add_argument("--voice", default=None, help="Piper voice (.onnx) for a text-to-speech draft")
    ap.add_argument("--piper-python", default="/root/venvs/tts/bin/python")
    ap.add_argument("--length-scale", type=float, default=1.0)
    ap.add_argument("--audio", nargs="*", default=None, help="own recordings: one wav per slide")
    ap.add_argument("--duration", type=float, default=59.5)
    ap.add_argument("--name", default="Nourozi_23726011_CompVP_Part2_findings.mp4")
    a = ap.parse_args()
    res, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = slides(res, out, a.example)
    (out / "narration_script.txt").write_text(
        "\n\n".join(f"Slide {k}: {t}" for k, t in enumerate(SCRIPT, 1)) + "\n", encoding="utf-8")
    if a.audio:
        clips = [wav_read(p) for p in a.audio]
    elif a.voice:
        clips = tts(SCRIPT, a.voice, a.piper_python, out, a.length_scale)
    else:
        print("slides and script written; no audio given"); return
    lens, wav = build(paths, clips, out, a.duration)
    info = encode(paths, lens, wav, out / a.name, a.duration)
    pptx(paths, SCRIPT, lens, out / "video_slides_with_script.pptx")
    print("slide durations:", [round(x, 2) for x in lens])
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
