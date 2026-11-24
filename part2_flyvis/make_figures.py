"""Figures for Project Part 2 from the simulation results (simulate.py).

    python part2_flyvis/make_figures.py --sim work/part2/sim_000.npz --ensemble work/part2/sim_0*.npz \
        --out work/part2/Results

Writes
    SteadyState/   20 whole-network figures (steady state, 2 s after onset)
    RGBFirst/      20 whole-network figures (each image 120 ms after it appeared: original after grey,
    MooneyFirst/   20                         Mooney after original / Mooney after grey, original after Mooney)
    Comparisons/   timing, photo-vs-Mooney similarity, switch transients, per-pair panels, time courses,
                   robustness across the 10 best models, summary.json
The whole-network figures follow Fig. 1g of Lappalainen et al. (2024) / the flyvis Fig. 1 notebook:
one hexagonal map per cell type, black = low, white = high membrane voltage.
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from flyvis.analysis.visualization import plt_utils  # noqa: E402
from flyvis.analysis.visualization.network_fig import WholeNetworkFigure  # noqa: E402
from flyvis.analysis.visualization.plots import hex_scatter  # noqa: E402
from flyvis.connectome import ConnectomeFromAvgFilters  # noqa: E402
from flyvis.utils.color_utils import cell_type_colors  # noqa: E402
from flyvis.utils.hex_utils import get_hex_coords  # noqa: E402

import common as C  # noqa: E402

CTOME = ConnectomeFromAvgFilters()
NODES = CTOME.nodes.to_df()
TYPES = list(CTOME.unique_cell_types[:].astype(str))
LAYOUT = dict(CTOME.layout[:].astype(str))
IDX = {t: np.flatnonzero(NODES.type.values == t) for t in TYPES}
U, V = get_hex_coords(C.EXTENT)
_POS = {(int(u), int(v)): i for i, (u, v) in enumerate(zip(U, V))}
HEXAL = {t: np.array([_POS[(int(u), int(v))] for u, v in zip(NODES.u.values[IDX[t]], NODES.v.values[IDX[t]])])
         for t in TYPES}                                   # receptor (column) index of every cell of a type
REGION_NAME = {"retina": "retina", "intermediate": "lamina, medulla intrinsic, CT1",
               "output": "T-shaped, transmedullary"}
ON_TYPES = ["Mi1", "Tm3", "Mi4", "CT1(M10)"]              # known ON-contrast selective inputs to T4
OFF_TYPES = ["Mi9", "Tm1", "Tm2", "Tm9"]                  # known OFF-contrast selective inputs to T5 (Mi9: to T4)
EXAMPLE_TYPES = ["R1", "L1", "Mi1", "Tm3", "Mi9", "Tm9", "T4c", "T5c", "TmY18"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


def load(path):
    return dict(np.load(path)), json.load(open(Path(path).with_suffix(".json")))


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))


# ----------------------------------------------------------------------------------- whole network
def whole_network(job):
    """job: dict(cartesian, hexin, act (cells,), scale {type: (lo, hi)}, title, note, path)."""
    wnf = WholeNetworkFigure(CTOME, video=True, rendering=True)
    wnf.init_figure(
        figsize=(13, 3.25), decoder_box=False, cell_type_labels=False, fontsize=6,
        add_graph_kwargs={"constant_edge_width": 0.2, "edge_width": 0.15, "constant_edge_color": "k",
                          "edge_alpha": 0.1,
                          "nx_kwargs": {"min_source_margin": 0, "min_target_margin": 5, "selfloop_x0": 0,
                                        "selfloop_y0": 0, "selfloop_h_shift": 0.025, "selfloop_v_shift": 0.1275}},
        network_layout_axes_kwargs={"types_per_column": 4, "region_spacing": 1.25, "wspace": 0.0})
    for ax in wnf.axes:
        name = ax.get_label()
        if name == "video":
            ax.imshow(job["cartesian"], cmap="gray", vmin=0, vmax=1)
            ax.set_title("input image", fontsize=6, pad=2)
            plt_utils.rm_spines(ax)
            continue
        if name == "rendering":
            hex_scatter(U, V, job["hexin"], fig=wnf.fig, ax=ax, cmap=plt.cm.binary_r, vmin=0, vmax=1, cbar=False,
                        fill=True, label="fly eye", label_color="black", fontsize=6, labelxy="auto")
        else:
            lo, hi = job["scale"][name]
            u, v = NODES.u.values[IDX[name]], NODES.v.values[IDX[name]]
            hex_scatter(u, v, job["act"][IDX[name]], fig=wnf.fig, ax=ax, label=name,
                        label_color=cell_type_colors[name], cmap=plt.cm.binary_r, vmin=lo, vmax=hi, cbar=False,
                        fontsize=6, fill=True, labelxy="auto")
        (xmin, ymin, xmax, ymax) = ax.dataLim.extents
        ax.set_xlim(plt_utils.get_lims((xmin, xmax), 0.01))
        ax.set_ylim(plt_utils.get_lims((ymin, ymax), 0.02))
    wnf.fig.text(0.0, 1.10, job["title"], fontsize=11, fontweight="bold", ha="left", va="bottom")
    wnf.fig.text(0.0, 1.03, job["note"], fontsize=7.5, ha="left", va="bottom", color="0.3")
    wnf.fig.savefig(job["path"], dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(wnf.fig)
    return job["path"]


def scales(snapshots):
    """Per cell type grey-scale range shared by a set of snapshots (1st-99th percentile)."""
    out = {}
    for t in TYPES:
        vals = np.concatenate([s[:, IDX[t]].ravel() for s in snapshots])
        lo, hi = np.percentile(vals, [1, 99])
        if hi - lo < 1e-6:
            lo, hi = lo - 1e-3, hi + 1e-3
        out[t] = (float(lo), float(hi))
    return out


def deliverables(S, meta, out, gray, moon, workers):
    ids = meta["ids"]
    model = meta["model"]
    sc_ss = scales([S["ss_orig_steady"], S["ss_moon_steady"]])
    sc_tr = scales([S["rf_p1_t120"], S["rf_p2_t120"], S["mf_p1_t120"], S["mf_p2_t120"]])
    common_note = (f"flyvis model {model} (Lappalainen et al. 2024), dt = {meta['dt']*1000:.0f} ms, "
                   "pre-stimulus: steady state under uniform grey 0.5; each image is a static video. ")
    note_ss = common_note + "Grey scale per cell type shared by all 20 SteadyState figures (black = low, white = high voltage)."
    note_tr = common_note + "Grey scale per cell type shared by all 40 RGBFirst/MooneyFirst figures (black = low, white = high voltage)."
    jobs = []
    for i, k in enumerate(ids):
        for kind, cart, hexin, act in [("original", gray[i], S["hex_orig"][i], S["ss_orig_steady"][i]),
                                       ("mooney", moon[i], S["hex_moon"][i], S["ss_moon_steady"][i])]:
            jobs.append(dict(cartesian=cart, hexin=hexin, act=act, scale=sc_ss, note=note_ss,
                             title=f"{k} | {'Mooney' if kind == 'mooney' else 'original (grey-scale)'} | SteadyState: "
                                   f"activity {C.T_STIM:.1f} s after onset (steady state)",
                             path=out / "SteadyState" / f"{k}_{kind}.png"))
        jobs += [
            dict(cartesian=gray[i], hexin=S["hex_orig"][i], act=S["rf_p1_t120"][i], scale=sc_tr, note=note_tr,
                 title=f"{k} | RGBFirst, phase 1: original (grey-scale) | activity 120 ms after onset (after grey)",
                 path=out / "RGBFirst" / f"{k}_1_original_120ms.png"),
            dict(cartesian=moon[i], hexin=S["hex_moon"][i], act=S["rf_p2_t120"][i], scale=sc_tr, note=note_tr,
                 title=f"{k} | RGBFirst, phase 2: Mooney | activity 120 ms after the switch from the original (shown 2 s)",
                 path=out / "RGBFirst" / f"{k}_2_mooney_120ms_after_switch.png"),
            dict(cartesian=moon[i], hexin=S["hex_moon"][i], act=S["mf_p1_t120"][i], scale=sc_tr, note=note_tr,
                 title=f"{k} | MooneyFirst, phase 1: Mooney | activity 120 ms after onset (after grey)",
                 path=out / "MooneyFirst" / f"{k}_1_mooney_120ms.png"),
            dict(cartesian=gray[i], hexin=S["hex_orig"][i], act=S["mf_p2_t120"][i], scale=sc_tr, note=note_tr,
                 title=f"{k} | MooneyFirst, phase 2: original (grey-scale) | activity 120 ms after the switch from the Mooney (shown 2 s)",
                 path=out / "MooneyFirst" / f"{k}_2_original_120ms_after_switch.png"),
        ]
    for d in ["SteadyState", "RGBFirst", "MooneyFirst"]:
        (out / d).mkdir(parents=True, exist_ok=True)
    with Pool(workers) as p:
        for path in p.imap_unordered(whole_network, jobs):
            print("  ", path.relative_to(out), flush=True)


# ----------------------------------------------------------------------------------- analysis numbers
def type_face_masks(face):
    """face (pairs, 721) receptor mask -> {type: (pairs, n_cells)}."""
    return {t: face[:, HEXAL[t]] for t in TYPES}


def analyse(S, face):
    """Per cell type: photo-vs-Mooney similarity, response contrast ratio, switch-transient polarity."""
    fm = type_face_masks(face)
    n = face.shape[0]
    d_in = S["hex_moon"] - S["hex_orig"]                      # input change at the switch in RGBFirst
    rows = {}
    for t in TYPES:
        o, m = S["ss_orig_steady"][:, IDX[t]], S["ss_moon_steady"][:, IDX[t]]
        msk = fm[t]
        r = [corr(o[i][msk[i]], m[i][msk[i]]) if msk[i].sum() > 5 else np.nan for i in range(n)]
        so = np.array([o[i][msk[i]].std() for i in range(n)])
        sm = np.array([m[i][msk[i]].std() for i in range(n)])
        drf = S["rf_p2_t120"][:, IDX[t]] - S["rf_p1_end"][:, IDX[t]]
        dmf = S["mf_p2_t120"][:, IDX[t]] - S["mf_p1_end"][:, IDX[t]]
        din = d_in[:, HEXAL[t]]
        s_rf = [corr(drf[i][msk[i]], din[i][msk[i]]) for i in range(n)]
        s_mf = [corr(dmf[i][msk[i]], -din[i][msk[i]]) for i in range(n)]
        # same image 120 ms after it appeared: after its partner vs. after grey (1 = no masking by the partner)
        ix = IDX[t]
        m_mp = [corr(S["rf_p2_t120"][i, ix][msk[i]], S["mf_p1_t120"][i, ix][msk[i]]) for i in range(n)]
        m_pm = [corr(S["mf_p2_t120"][i, ix][msk[i]], S["rf_p1_t120"][i, ix][msk[i]]) for i in range(n)]
        rows[t] = dict(region=LAYOUT[t], r_face=np.nanmean(r), r_face_pairs=r,
                       contrast_ratio=float(sm.sum() / (so.sum() + 1e-12)),
                       r_switch_rgbfirst=float(np.nanmean(s_rf)), r_switch_mooneyfirst=float(np.nanmean(s_mf)),
                       r_mooney_after_photo=float(np.nanmean(m_mp)), r_photo_after_mooney=float(np.nanmean(m_pm)),
                       final_diff_rgbfirst=float(S["rf_final_diff"][TYPES.index(t)].max()),
                       final_diff_mooneyfirst=float(S["mf_final_diff"][TYPES.index(t)].max()))
    r_in_fly = [corr(S["hex_orig"][i][face[i]], S["hex_moon"][i][face[i]]) for i in range(n)]
    return rows, r_in_fly


def settle_time(S, thr=0.02):
    """Per type and stimulus: last time the distance to the final state exceeds thr (s)."""
    x = S["ss_settle"]                                          # (types, 20, T)
    bad = x > thr
    last = np.where(bad.any(-1), x.shape[-1] - np.argmax(bad[..., ::-1], axis=-1), 0)
    return last * C.DT


def fade_time(order, thr=0.05):
    """Time after the switch when the order effect stays below thr (per type and pair)."""
    bad = order > thr
    last = np.where(bad.any(-1), order.shape[-1] - np.argmax(bad[..., ::-1], axis=-1), 0)
    return last * C.DT


# ----------------------------------------------------------------------------------- comparison figures
def fig_timing(S, out):
    t = np.arange(1, C.N_STIM + 1) * C.DT
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    x = S["ss_settle"].reshape(len(TYPES) * 20, -1)
    ax = axs[0]
    ax.fill_between(t, np.percentile(x, 5, 0), np.percentile(x, 95, 0), color="#9ecae1", label="5-95 % of cell types x images")
    ax.plot(t, np.median(x, 0), color="#08519c", lw=2, label="median")
    ax.plot(t, x.max(0), color="#08519c", lw=1, ls=":", label="slowest")
    ax.axhline(0.02, color="0.4", lw=0.8, ls="--")
    ax.text(1.95, 0.024, "2 %", ha="right", fontsize=8, color="0.3")
    st = settle_time(S)
    ax.axvline(st.max(), color="#e6550d", lw=1)
    ax.text(st.max() + 0.03, 0.5, f"all 65 cell types\nsettled at {st.max():.2f} s", color="#e6550d", fontsize=8)
    ax.set_yscale("log"); ax.set_ylim(1e-4, 3); ax.set_xlim(0, 2)
    ax.set_xlabel("time after image onset (s)"); ax.set_ylabel("distance to final state\n(relative to grey -> final)")
    ax.set_title("a  SteadyState: when is the activity stable?", loc="left", fontweight="bold")
    ax.legend(fontsize=7, loc="lower left", frameon=False)

    ax = axs[1]
    tt = np.arange(1, C.N_STIM + 1) * C.DT * 1000
    for key, col, lab in [("rf_switch", "#31a354", "original -> Mooney (RGBFirst)"),
                          ("mf_switch", "#756bb1", "Mooney -> original (MooneyFirst)")]:
        y = S[key] / (S[key].max(-1, keepdims=True) + 1e-12)              # normalise per type and pair
        y = y.reshape(-1, y.shape[-1])
        ax.plot(tt, np.median(y, 0), color=col, lw=2, label=lab)
    y = S["ss_onset"] / (S["ss_onset"].max(-1, keepdims=True) + 1e-12)
    ax.plot(tt, np.median(y.reshape(-1, y.shape[-1]), 0), color="0.4", lw=1.5, ls="--", label="onset after grey")
    ax.axvline(120, color="#e6550d", lw=1); ax.text(130, 0.05, "read-out\n120 ms", color="#e6550d", fontsize=8)
    ax.set_xlim(0, 1000); ax.set_ylim(0, 1.05)
    ax.set_xlabel("time after onset / switch (ms)"); ax.set_ylabel("change of activity\n(normalised per cell type)")
    ax.set_title("b  How fast does the network react?", loc="left", fontweight="bold")
    ax.legend(fontsize=7, frameon=False, loc="lower right")

    ax = axs[2]
    for key, col, lab in [("rf_order", "#31a354", "Mooney after original vs. Mooney after grey"),
                          ("mf_order", "#756bb1", "original after Mooney vs. original after grey")]:
        y = S[key].reshape(-1, S[key].shape[-1])
        ax.fill_between(tt, np.percentile(y, 10, 0), np.percentile(y, 90, 0), color=col, alpha=0.15)
        ax.plot(tt, np.median(y, 0), color=col, lw=2, label=lab)
    ft = max(fade_time(S["rf_order"]).max(), fade_time(S["mf_order"]).max())
    ax.axvline(120, color="#e6550d", lw=1)
    ax.axhline(0.05, color="0.4", lw=0.8, ls="--"); ax.text(1950, 0.07, "5 %", ha="right", fontsize=8, color="0.3")
    ax.text(ft * 1000 + 20, 1.2, f"history gone for every\ncell type after {ft:.2f} s", fontsize=8)
    ax.set_xlim(0, 2000); ax.set_ylim(0, 2.2)
    ax.set_xlabel("time after the switch (ms)"); ax.set_ylabel("effect of the previous image\n(relative to the response range)")
    ax.set_title("c  How long does the previous image matter?", loc="left", fontweight="bold")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "comparison_1_timing.png", dpi=170, facecolor="white"); plt.close(fig)
    return float(st.max()), float(np.median(st)), float(ft)


def ordered_types():
    return [t for reg in ["retina", "intermediate", "output"] for t in TYPES if LAYOUT[t] == reg]


def fig_similarity(rows, r_in_fly, r_in_full, out):
    ts = ordered_types()
    x = np.arange(len(ts))
    cols = [cell_type_colors[t] for t in ts]
    fig, axs = plt.subplots(2, 1, figsize=(15, 6.8), sharex=True)
    ax = axs[0]
    for i, t in enumerate(ts):
        ax.scatter(np.full(10, i), rows[t]["r_face_pairs"], s=6, color="0.75", zorder=1)
    ax.scatter(x, [rows[t]["r_face"] for t in ts], s=28, c=cols, zorder=2, edgecolor="k", linewidth=0.3)
    ax.axhline(np.mean(r_in_fly), color="#08519c", lw=1.2, ls="--",
               label=f"input on the fly eye (721 ommatidia): r = {np.mean(r_in_fly):.2f}")
    ax.axhline(np.mean(r_in_full), color="#e6550d", lw=1.2, ls=":",
               label=f"the two images at full resolution (512 px): r = {np.mean(r_in_full):.2f}")
    ax.legend(loc="lower left", fontsize=8, frameon=True, framealpha=0.9)
    ax.set_ylim(0.6, 1.03); ax.set_ylabel("photo vs. Mooney\ncorrelation (face region)")
    ax.set_title("a  Steady state: how similar are the responses to the photo and to its Mooney? (grey dots: 10 pairs)",
                 loc="left", fontweight="bold")
    ax = axs[1]
    ax.bar(x, [rows[t]["contrast_ratio"] for t in ts], color=cols, edgecolor="k", linewidth=0.3)
    ax.axhline(1, color="k", lw=0.8)
    ax.set_ylabel("Mooney / photo\nspatial contrast of response")
    ax.set_title("b  Steady state: strength of the response pattern (std across the face region), Mooney relative to photo",
                 loc="left", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(ts, rotation=90, fontsize=7)
    for tl, t in zip(ax.get_xticklabels(), ts):
        tl.set_color(cell_type_colors[t])
    for reg in ["intermediate", "output"]:
        i0 = [i for i, t in enumerate(ts) if LAYOUT[t] == reg][0]
        for a in axs:
            a.axvline(i0 - 0.5, color="0.6", lw=0.8)
    for reg in ["retina", "intermediate", "output"]:
        ii = [i for i, t in enumerate(ts) if LAYOUT[t] == reg]
        axs[0].text(np.mean(ii), 1.003, REGION_NAME[reg], ha="center", va="bottom", fontsize=8, color="0.3")
    fig.tight_layout()
    fig.savefig(out / "comparison_2_steady_state_similarity.png", dpi=170, facecolor="white"); plt.close(fig)


def hexmap(ax, values, cmap, lo, hi, label=""):
    hex_scatter(U, V, values, fig=ax.figure, ax=ax, cmap=cmap, vmin=lo, vmax=hi, cbar=False, fill=True,
                label=label, label_color="black", fontsize=7, labelxy="auto")
    (xmin, ymin, xmax, ymax) = ax.dataLim.extents
    ax.set_xlim(plt_utils.get_lims((xmin, xmax), 0.01)); ax.set_ylim(plt_utils.get_lims((ymin, ymax), 0.02))


def on_hexals(t, values):
    """Cell-type values (n_cells,) -> receptor lattice (721,), NaN where the type has no cell."""
    full = np.full(721, np.nan, np.float32)
    full[HEXAL[t]] = values
    return full


def fig_switch(S, rows, meta, out, pair):
    i = meta["ids"].index(pair)
    d_in = S["hex_moon"][i] - S["hex_orig"][i]
    show = ["Mi1", "Tm3", "Mi9", "Tm9"]
    fig, axs = plt.subplots(2, 5, figsize=(14, 6))
    for r, (key, sign, lab) in enumerate([("rf", 1, "RGBFirst: original -> Mooney"), ("mf", -1, "MooneyFirst: Mooney -> original")]):
        lim = np.abs(d_in).max()
        hexmap(axs[r, 0], sign * d_in, plt.cm.RdBu_r, -lim, lim)
        axs[r, 0].set_title(f"{lab}\ninput change (new - old image)", fontsize=9)
        for j, t in enumerate(show):
            dv = S[f"{key}_p2_t120"][i, IDX[t]] - S[f"{key}_p1_end"][i, IDX[t]]
            lim2 = np.abs(np.concatenate([S["rf_p2_t120"][i, IDX[t]] - S["rf_p1_end"][i, IDX[t]]])).max() + 1e-9
            hexmap(axs[r, j + 1], on_hexals(t, dv), plt.cm.RdBu_r, -lim2, lim2)
            pol = "ON" if t in ON_TYPES else "OFF"
            axs[r, j + 1].set_title(f"{t} ({pol} pathway)\nchange 120 ms after switch", fontsize=9,
                                    color=cell_type_colors[t])
    fig.text(0.5, 0.005, "red = depolarised / brighter, blue = hyperpolarised / darker (relative to the last frame "
             "before the switch). ON cells copy the luminance change, OFF cells invert it; reversing the order flips the sign.",
             ha="center", fontsize=8.5, color="0.25")
    fig.suptitle(f"{pair}: right after the switch the fly network signals what changed between photo and Mooney",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(out / f"comparison_3_switch_transient_{pair}.png", dpi=170, facecolor="white"); plt.close(fig)

    ts = ordered_types()
    fig, ax = plt.subplots(figsize=(15, 3.4))
    x = np.arange(len(ts))
    ax.bar(x - 0.2, [rows[t]["r_switch_rgbfirst"] for t in ts], width=0.4, color="#31a354", label="original -> Mooney")
    ax.bar(x + 0.2, [rows[t]["r_switch_mooneyfirst"] for t in ts], width=0.4, color="#756bb1", label="Mooney -> original")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(ts, rotation=90, fontsize=7)
    for tl, t in zip(ax.get_xticklabels(), ts):
        tl.set_color(cell_type_colors[t]); tl.set_fontweight("bold" if t in ON_TYPES + OFF_TYPES else "normal")
    ax.set_ylabel("corr(activity change,\ninput change)")
    ax.set_title("Polarity of the switch transient per cell type (120 ms after the switch, face region, mean of 10 pairs): "
                 "+1 = follows the brightness change (ON), -1 = inverted (OFF)", loc="left", fontsize=9, fontweight="bold")
    ax.legend(fontsize=8, frameon=False, ncol=2, loc="lower left")
    ax.set_ylim(-1.05, 1.05)
    fig.tight_layout()
    fig.savefig(out / "comparison_3_switch_polarity_by_cell_type.png", dpi=170, facecolor="white"); plt.close(fig)


def fig_masking(rows, out):
    ts = ordered_types()
    x = np.arange(len(ts))
    fig, ax = plt.subplots(figsize=(15, 3.6))
    ax.bar(x - 0.2, [rows[t]["r_mooney_after_photo"] for t in ts], width=0.4, color="#31a354",
           label="Mooney 120 ms after the photo vs. Mooney 120 ms after grey")
    ax.bar(x + 0.2, [rows[t]["r_photo_after_mooney"] for t in ts], width=0.4, color="#756bb1",
           label="photo 120 ms after the Mooney vs. photo 120 ms after grey")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(ts, rotation=90, fontsize=7)
    for tl, t in zip(ax.get_xticklabels(), ts):
        tl.set_color(cell_type_colors[t])
    ax.set_ylim(-1.05, 1.1); ax.set_ylabel("pattern correlation\n(face region, 10 pairs)")
    ax.set_title("Masking right after the switch: 1 = the new image is represented as if it followed grey. "
                 "The Mooney pops out after the photo; the photo is masked after the Mooney", loc="left",
                 fontsize=9, fontweight="bold")
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "comparison_4_masking_by_cell_type.png", dpi=170, facecolor="white"); plt.close(fig)


def fig_pairs(S, meta, out, gray, moon):
    cols = [("photo\nsteady state", "ss_orig_steady"), ("Mooney\nsteady state", "ss_moon_steady"),
            ("Mooney 120 ms\nafter grey", "mf_p1_t120"), ("Mooney 120 ms\nafter photo", "rf_p2_t120"),
            ("photo 120 ms\nafter grey", "rf_p1_t120"), ("photo 120 ms\nafter Mooney", "mf_p2_t120")]
    pdir = out / "per_pair"
    pdir.mkdir(exist_ok=True)
    for i, k in enumerate(meta["ids"]):
        fig, axs = plt.subplots(len(EXAMPLE_TYPES) + 1, len(cols), figsize=(9.2, 14))
        for j, (lab, key) in enumerate(cols):
            src = S["hex_orig"][i] if "photo" in lab else S["hex_moon"][i]
            hexmap(axs[0, j], src, plt.cm.binary_r, 0, 1)
            axs[0, j].set_title(lab, fontsize=8.5)
        axs[0, 0].set_ylabel("input", fontsize=8)
        for r, t in enumerate(EXAMPLE_TYPES, start=1):
            vals = np.concatenate([S[key][i, IDX[t]] for _, key in cols])
            lo, hi = np.percentile(vals, [1, 99])
            for j, (lab, key) in enumerate(cols):
                hexmap(axs[r, j], on_hexals(t, S[key][i, IDX[t]]), plt.cm.binary_r, lo, hi)
            axs[r, 0].text(-0.15, 0.5, t, transform=axs[r, 0].transAxes, ha="right", va="center", fontsize=9,
                           color=cell_type_colors[t], fontweight="bold")
        fig.suptitle(f"{k}: selected cell types in all conditions (grey scale shared within each row)",
                     fontsize=10, fontweight="bold")
        fig.tight_layout(rect=(0.04, 0, 1, 0.98))
        fig.savefig(pdir / f"{k}.png", dpi=130, facecolor="white"); plt.close(fig)


def fig_timecourses(S, meta, out, pair):
    i = meta["ids"].index(pair)
    show = ["R1", "L1", "Mi1", "Tm3", "Mi9", "Tm9", "T4c", "T5c"]
    t2 = np.arange(1, 2 * C.N_STIM + 1) * C.DT
    t1 = t2[:C.N_STIM]
    fig, axs = plt.subplots(2, 4, figsize=(15, 6), sharex=True)
    for ax, t in zip(axs.ravel(), show):
        j = TYPES.index(t)
        ax.plot(t2, S["rf_central"][j, i], color="#31a354", lw=1.8, label="RGBFirst: photo -> Mooney")
        ax.plot(t2, S["mf_central"][j, i], color="#756bb1", lw=1.8, label="MooneyFirst: Mooney -> photo")
        ax.plot(t1 + C.T_STIM, S["ss_central"][j, 10 + i], color="#31a354", lw=1.1, ls="--",
                label="Mooney alone (after grey, SteadyState)")
        ax.plot(t1 + C.T_STIM, S["ss_central"][j, i], color="#756bb1", lw=1.1, ls="--",
                label="photo alone (after grey, SteadyState)")
        ax.axvline(2.0, color="0.6", lw=0.8, ls=":")
        ax.set_title(t, color=cell_type_colors[t], fontweight="bold")
        ax.set_xlim(0, 4)
    for ax in axs[1]:
        ax.set_xlabel("time (s); switch at 2 s")
    for ax in axs[:, 0]:
        ax.set_ylabel("central cell voltage (a.u.)")
    axs[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"{pair}: central-cell responses (as in flyvis tutorial 7). Dashed: the second image shown alone after grey, "
                 "aligned to the switch; the sequences join it within ~1 s", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / f"comparison_5_timecourses_{pair}.png", dpi=170, facecolor="white"); plt.close(fig)


def fig_ensemble(files, face, out):
    rows = []
    for f in files:
        S, meta = load(f)
        r, _ = analyse(S, face)
        st = settle_time(S)
        ft = max(fade_time(S["rf_order"]).max(), fade_time(S["mf_order"]).max())
        reg = {g: np.nanmean([v["r_face"] for v in r.values() if v["region"] == g]) for g in ["retina", "intermediate", "output"]}
        rows.append(dict(model=meta["model"].split("/")[-1], **{f"r_{g}": reg[g] for g in reg},
                         r_min=float(np.nanmin([v["r_face"] for v in r.values()])),
                         contrast_ratio=float(np.median([v["contrast_ratio"] for v in r.values()])),
                         on=float(np.mean([r[t]["r_switch_rgbfirst"] for t in ON_TYPES])),
                         off=float(np.mean([r[t]["r_switch_rgbfirst"] for t in OFF_TYPES])),
                         settle_max=float(st.max()), fade_max=float(ft),
                         mask_mp=float(np.nanmedian([v["r_mooney_after_photo"] for v in r.values()])),
                         mask_pm=float(np.nanmedian([v["r_photo_after_mooney"] for v in r.values()])),
                         final_diff=float(max(S["rf_final_diff"].max(), S["mf_final_diff"].max()))))
    m = [x["model"] for x in rows]
    x = np.arange(len(m))
    fig, axs = plt.subplots(1, 5, figsize=(20, 3.8))
    ax = axs[0]
    for g, c in [("retina", "#636363"), ("intermediate", "#e6550d"), ("output", "#3182bd")]:
        ax.plot(x, [r[f"r_{g}"] for r in rows], "o-", color=c, label=REGION_NAME[g])
    ax.plot(x, [r["r_min"] for r in rows], "v:", color="k", label="most different cell type")
    ax.set_ylim(0.5, 1); ax.set_title("photo vs. Mooney similarity", fontweight="bold", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax = axs[1]
    ax.bar(x, [r["contrast_ratio"] for r in rows], color="#756bb1"); ax.axhline(1, color="k", lw=0.8)
    ax.set_title("Mooney / photo response contrast\n(median over cell types)", fontweight="bold", fontsize=9)
    ax = axs[2]
    ax.bar(x - 0.2, [r["on"] for r in rows], 0.4, color="#d7301f", label="ON inputs (Mi1, Tm3, Mi4, CT1(M10))")
    ax.bar(x + 0.2, [r["off"] for r in rows], 0.4, color="#2b8cbe", label="OFF inputs (Mi9, Tm1, Tm2, Tm9)")
    ax.axhline(0, color="k", lw=0.8); ax.set_ylim(-1.05, 1.05)
    ax.set_title("switch transient vs. input change", fontweight="bold", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    ax = axs[3]
    ax.plot(x, [r["settle_max"] for r in rows], "o-", color="#08519c", label="settling after onset (2 %)")
    ax.plot(x, [r["fade_max"] for r in rows], "s-", color="#31a354", label="order effect gone (5 %)")
    ax.set_ylim(0, 2.0); ax.set_ylabel("s"); ax.set_title("slowest cell type", fontweight="bold", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax = axs[4]
    ax.plot(x, [r["mask_mp"] for r in rows], "o-", color="#31a354", label="Mooney after photo")
    ax.plot(x, [r["mask_pm"] for r in rows], "s-", color="#756bb1", label="photo after Mooney")
    ax.set_ylim(0, 1.05); ax.set_title("masking 120 ms after the switch\n(median over cell types)", fontweight="bold",
                                        fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    for a in axs:
        a.set_xticks(x); a.set_xticklabels(m, fontsize=8); a.set_xlabel("model (best task performance = 000)")
    fig.suptitle("Robustness: the same analysis in the 10 best models of the pretrained ensemble", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "comparison_6_ensemble_robustness.png", dpi=170, facecolor="white"); plt.close(fig)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default=str(C.ROOT / "work/part2/sim_000.npz"))
    ap.add_argument("--ensemble", nargs="*", default=[])
    ap.add_argument("--out", default=str(C.ROOT / "work/part2/Results"))
    ap.add_argument("--example", default="face_033")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--skip-network", action="store_true", help="skip the 60 whole-network figures")
    a = ap.parse_args()
    out = Path(a.out)
    S, meta = load(a.sim)
    ids = meta["ids"]
    _, gray, moon = C.load_pairs(ids=ids)
    face = C.face_masks(ids)
    comp = out / "Comparisons"
    comp.mkdir(parents=True, exist_ok=True)

    rows, r_in_fly = analyse(S, face)
    r_in_full = []
    for i, k in enumerate(ids):
        x0, y0, x1, y1 = [int(round(v)) for v in
                          json.load(open(C.ROOT / "results/Result3/Original" / f"{k}.json"))["detections"][0]["bbox"]]
        r_in_full.append(corr(gray[i][y0:y1, x0:x1].ravel(), moon[i][y0:y1, x0:x1].ravel()))
    st_max, st_med, fade = fig_timing(S, comp)
    fig_similarity(rows, r_in_fly, r_in_full, comp)
    fig_switch(S, rows, meta, comp, a.example)
    fig_timecourses(S, meta, comp, a.example)
    fig_masking(rows, comp)
    fig_pairs(S, meta, comp, gray, moon)
    ens = fig_ensemble(a.ensemble, face, comp) if a.ensemble else []
    reg = {g: float(np.nanmean([v["r_face"] for v in rows.values() if v["region"] == g])) for g in ["retina", "intermediate", "output"]}
    order = sorted(rows, key=lambda t: rows[t]["r_face"])
    summary = dict(
        model=meta["model"], pairs=ids, protocol=dict(dt=C.DT, t_stim=C.T_STIM, t_transient=C.T_TRANSIENT, pre="grey 0.5"),
        settle_time_max_s=st_max, settle_time_median_s=st_med, order_effect_gone_after_s=fade,
        verify_rel_dist_to_10s=meta.get("verify_rel_dist_to_10s"),
        r_input_fly_eye=float(np.mean(r_in_fly)), r_input_full_resolution=float(np.mean(r_in_full)),
        r_by_region=reg, most_different=[(t, round(rows[t]["r_face"], 3)) for t in order[:6]],
        most_similar=[(t, round(rows[t]["r_face"], 3)) for t in order[-4:]],
        contrast_ratio_median=float(np.median([v["contrast_ratio"] for v in rows.values()])),
        contrast_ratio_min=float(min(v["contrast_ratio"] for v in rows.values())),
        contrast_ratio_by_type={t: round(rows[t]["contrast_ratio"], 3) for t in TYPES},
        switch_on={t: round(rows[t]["r_switch_rgbfirst"], 3) for t in ON_TYPES},
        switch_off={t: round(rows[t]["r_switch_rgbfirst"], 3) for t in OFF_TYPES},
        masking_mooney_after_photo_median=float(np.nanmedian([v["r_mooney_after_photo"] for v in rows.values()])),
        masking_photo_after_mooney_median=float(np.nanmedian([v["r_photo_after_mooney"] for v in rows.values()])),
        masking_photo_after_mooney_below_05=int(sum(v["r_photo_after_mooney"] < 0.5 for v in rows.values())),
        masking_examples={t: (round(rows[t]["r_mooney_after_photo"], 2), round(rows[t]["r_photo_after_mooney"], 2))
                          for t in ["Mi1", "Tm3", "Tm2", "Mi9", "Tm9", "TmY18"]},
        final_state_diff_max=float(max(v["final_diff_rgbfirst"] for v in rows.values()) if rows else 0),
        final_state_diff_max_mooneyfirst=float(max(v["final_diff_mooneyfirst"] for v in rows.values())),
        ensemble=ens,
    )
    json.dump(summary, open(comp / "summary.json", "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("contrast_ratio_by_type", "ensemble")}, indent=1,
                     default=float))
    if not a.skip_network:
        deliverables(S, meta, out, gray, moon, a.workers)


if __name__ == "__main__":
    main()
