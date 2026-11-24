"""Long presentations: does the order effect disappear (slow dynamics) or persist (memory)?

    python part2_flyvis/memory_check.py --models 000 001 ... --out work/part2/memory_check.npz

For every model and pair: B alone (grey -> B for T_LONG) and the sequence (grey -> A for 2 s -> B for T_LONG),
for both orders. Stores per cell type and pair the distance between "B after A" and "B alone" at
check-points after B appeared (relative to the distance grey -> final state of B alone), the settling of B alone,
and the photo-vs-Mooney similarity of the final states (face region).
"""
import argparse
import json

import numpy as np
import torch

import common as C
from simulate import per_type_norm, simulate, static

T_LONG = 10.0
CHECK = [0.12, 0.5, 1.0, 2.0, 5.0, 10.0]


def corr(a, b):
    a, b = a - a.mean(-1, keepdims=True), b - b.mean(-1, keepdims=True)
    return (a * b).sum(-1) / np.sqrt((a * a).sum(-1) * (b * b).sum(-1) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[f"{i:03d}" for i in range(10)])
    ap.add_argument("--out", default=str(C.ROOT / "work/part2/memory_check.npz"))
    ap.add_argument("--pairs-per-chunk", type=int, default=2)
    a = ap.parse_args()
    ids, gray, moon = C.load_pairs()
    x = {"orig": C.render(gray), "moon": C.render(moon)}
    face = C.face_masks(ids)
    nL, nA = int(round(T_LONG / C.DT)), C.N_STIM
    ck = [int(round(t / C.DT)) - 1 for t in CHECK]
    res = {}
    for m in a.models:
        net = C.load_network(f"flow/0000/{m}")
        types, layout, idx, nodes = C.cell_groups(net.connectome)
        groups = [idx[t] for t in types]
        from flyvis.utils.hex_utils import get_hex_coords
        U, V = get_hex_coords(C.EXTENT)
        pos = {(int(u), int(v)): i for i, (u, v) in enumerate(zip(U, V))}
        hexal = [np.array([pos[(int(u), int(v))] for u, v in zip(nodes.u.values[ix], nodes.v.values[ix])]) for ix in groups]
        with torch.no_grad():
            base = net.simulate(torch.full((1, 2, 1, 721), 0.5), C.DT, initial_state="auto").cpu().numpy()[0, -1]
        order = np.zeros((2, len(types), len(ids), len(CHECK)), np.float32)     # [RGBFirst, MooneyFirst]
        settle = np.zeros((2, len(types), len(ids), len(CHECK)), np.float32)   # B alone vs its 10-s state
        final = {"orig": np.zeros((len(ids), len(base)), np.float32), "moon": np.zeros((len(ids), len(base)), np.float32)}
        for d, (A, B) in enumerate([("orig", "moon"), ("moon", "orig")]):
            for i0 in range(0, len(ids), a.pairs_per_chunk):
                sl = slice(i0, i0 + a.pairs_per_chunk)
                alone = simulate(net, static(x[B][sl], nL), chunk=a.pairs_per_chunk)
                seq = simulate(net, torch.cat([static(x[A][sl], nA), static(x[B][sl], nL)], 1), chunk=a.pairs_per_chunk)
                fin = alone[:, -1]
                ref = per_type_norm(fin - base, groups) + 1e-9                        # (types, pairs)
                for j, c in enumerate(ck):
                    order[d, :, sl, j] = per_type_norm(seq[:, nA + c] - alone[:, c], groups) / ref
                    settle[d, :, sl, j] = per_type_norm(alone[:, c] - fin, groups) / ref
                final[B][sl] = fin
                del alone, seq
            print(m, "done", A, "->", B, flush=True)
        sim = np.full((len(types), len(ids)), np.nan, np.float32)
        for k, ix in enumerate(groups):
            msk = face[:, hexal[k]]
            for i in range(len(ids)):
                if msk[i].sum() > 5:
                    sim[k, i] = corr(final["orig"][i, ix][msk[i]], final["moon"][i, ix][msk[i]])
        res[m] = dict(order=order, settle=settle, sim10=sim)
    np.savez_compressed(a.out, models=np.array(a.models), check=np.array(CHECK), types=np.array(types),
                        regions=np.array([layout[t] for t in types]),
                        **{f"{k}_{m}": v for m, r in res.items() for k, v in r.items()})
    print(json.dumps({m: dict(order_median_at=dict(zip(map(str, CHECK), np.round(np.median(r["order"], (0, 1, 2)), 4).tolist())),
                              settle_median_at=dict(zip(map(str, CHECK), np.round(np.median(r["settle"], (0, 1, 2)), 4).tolist())),
                              sim10_median=float(np.nanmedian(r["sim10"])))
                      for m, r in res.items()}, indent=1))


if __name__ == "__main__":
    main()
