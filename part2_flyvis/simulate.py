"""Feed the 10 image pairs of Part 1 to a pretrained flyvis model in the three Part-2 conditions.

    python part2_flyvis/simulate.py --model flow/0000/000 --out work/part2/sim_000.npz --verify

Conditions (original = grey-scale version of the RGB photo, every image is a static video):
    SteadyState : grey -> image for 2 s                         (all 20 images)
    RGBFirst    : grey -> original for 2 s -> Mooney for 2 s     (10 pairs, same face)
    MooneyFirst : grey -> Mooney for 2 s -> original for 2 s     (10 pairs, same face)

Stored per model (npz): activity of all 45,669 cells at the read-out times and per-cell-type
time courses (distance to the final state, order effect after the switch, central-cell traces).
--verify additionally runs SteadyState for 10 s and checks that the state at 2 s is final.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import common as C


def simulate(net, movie, chunk=5):
    """movie (n, frames, 1, 721) -> activity (n, frames, cells); starts from the grey steady state.

    Runs a few stimuli at a time to keep the memory use low (flyvis keeps every frame of all cells).
    """
    out = None
    with torch.no_grad():
        for i in range(0, movie.shape[0], chunk):
            r = net.simulate(movie[i:i + chunk], C.DT, initial_state="auto").cpu().numpy()
            if out is None:
                out = np.empty((movie.shape[0], *r.shape[1:]), np.float32)
            out[i:i + chunk] = r
            del r
    return out


def log(*x):
    print(*x, flush=True)


def static(x, n_frames):
    return x.repeat(1, n_frames, 1, 1)


def per_type_norm(diff, groups):
    """diff (..., cells) -> (types, ...) L2 norm over the cells of each type."""
    return np.stack([np.linalg.norm(diff[..., ix], axis=-1) for ix in groups])


def central_index(nodes, types):
    out = []
    for t in types:
        n = nodes[nodes.type == t]
        r = np.hypot(n.u.values, n.v.values + n.u.values / 2)      # distance to the eye centre
        out.append(n.index.values[np.argmin(r)])
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=C.MODEL)
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify", action="store_true", help="10-s run to check that 2 s is steady")
    a = ap.parse_args()
    t0 = time.time()

    ids, gray, moon = C.load_pairs()
    x_orig, x_moon = C.render(gray), C.render(moon)                 # (10, 1, 1, 721) each
    net = C.load_network(a.model)
    types, layout, idx, nodes = C.cell_groups(net.connectome)
    groups = [idx[t] for t in types]
    cen = central_index(nodes, types)
    with torch.no_grad():
        grey = net.steady_state(1.0, C.DT, 1)
        base = net.simulate(torch.full((1, 2, 1, 721), 0.5), C.DT, initial_state=grey).cpu().numpy()[0, -1]
    n, T, iT, iS = len(ids), C.N_STIM, C.I_TRANSIENT, C.I_STEADY
    meta = dict(model=a.model, ids=ids, dt=C.DT, t_stim=C.T_STIM, t_transient=C.T_TRANSIENT,
                cell_types=types, regions=[layout[t] for t in types])

    # ---------------- SteadyState: grey -> image (2 s)
    log("SteadyState ...")
    ss = simulate(net, static(torch.cat([x_orig, x_moon]), T))       # (20, T, cells): 0-9 orig, 10-19 Mooney
    final = ss[:, iS]
    ref = per_type_norm(final - base, groups) + 1e-9                  # (types, 20) distance grey -> final
    settle = per_type_norm(ss - final[:, None], groups) / ref[..., None]          # (types, 20, T)
    onset = per_type_norm(ss - base, groups)                                       # distance from grey
    out = dict(
        base=base, hex_orig=x_orig[:, 0, 0].numpy(), hex_moon=x_moon[:, 0, 0].numpy(),
        ss_orig_steady=final[:n], ss_moon_steady=final[n:],
        ss_orig_t120=ss[:n, iT], ss_moon_t120=ss[n:, iT],
        ss_settle=settle.astype(np.float32), ss_onset=onset.astype(np.float32),
        ss_central=ss[:, :, cen].transpose(2, 0, 1),                              # (types, 20, T)
    )

    # ---------------- RGBFirst / MooneyFirst: grey -> A (2 s) -> B (2 s)
    for name, A, B, alone in [("rf", x_orig, x_moon, ss[n:]), ("mf", x_moon, x_orig, ss[:n])]:
        log(name, "...")
        seq = simulate(net, torch.cat([static(A, T), static(B, T)], dim=1))        # (10, 2T, cells)
        refB = per_type_norm(alone[:, iS] - base, groups) + 1e-9                    # (types, 10)
        out[f"{name}_order"] = (per_type_norm(seq[:, T:] - alone, groups) / refB[..., None]).astype(np.float32)
        out[f"{name}_final_diff"] = (per_type_norm(seq[:, -1] - alone[:, iS], groups) / refB).astype(np.float32)
        out[f"{name}_switch"] = per_type_norm(seq[:, T:] - seq[:, T - 1:T], groups).astype(np.float32)
        out[f"{name}_p1_t120"], out[f"{name}_p1_end"] = seq[:, iT], seq[:, T - 1]
        out[f"{name}_p2_t120"], out[f"{name}_p2_end"] = seq[:, T + iT], seq[:, -1]
        out[f"{name}_central"] = seq[:, :, cen].transpose(2, 0, 1)                 # (types, 10, 2T)
        del seq

    # ---------------- check: is 2 s already the steady state? (10 s run)
    if a.verify:
        del ss
        checks = [0.5, 1.0, 1.5, 2.0, 3.0]
        worst = {t: 0.0 for t in checks}
        x_all = torch.cat([x_orig, x_moon])
        for i in range(0, 2 * n, 2):                                   # two stimuli at a time
            long = simulate(net, static(x_all[i:i + 2], 1000), chunk=2)
            ref10 = per_type_norm(long[:, -1] - base, groups) + 1e-9
            for t in checks:
                d = per_type_norm(long[:, int(round(t / C.DT)) - 1] - long[:, -1], groups) / ref10
                worst[t] = max(worst[t], float(d.max()))
            del long
        meta["verify_rel_dist_to_10s"] = {f"{t:.1f}s": worst[t] for t in checks}
        log("verify", meta["verify_rel_dist_to_10s"])
    meta["runtime_s"] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **out)
    json.dump(meta, open(Path(a.out).with_suffix(".json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k not in ("cell_types", "regions")}, indent=1))


if __name__ == "__main__":
    main()
