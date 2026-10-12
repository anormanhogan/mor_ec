"""Error of the compressed ROM against the compression tolerance, split as in Fig. 6.

    python tolerance_scan.py CAMPAIGN --configs CONFIG ... --tols 1e-3 1e-4 ... [--npts 100] [--out FILE]

At the final bath of every Fig. 6 point of each configuration's post sweep (fixed baths, no DMFT):
    EC channel         mean_n |G_ED - G_EC| / |G_ED|     EC ground state through the exact resolvent
    resolvent channel  mean_n |G_EC - G_ROM| / |G_EC|
    total              mean_n |G_ED - G_ROM| / |G_ED|
for the ROM as built and compressed at each tolerance.  Output: per-point values and sizes in
$MOREC_RUNS/CAMPAIGN/tolerance_scan_errors.json.
"""
import argparse
import json
import os

import numpy as np

from morec import AIM, DMFTConfig, EDSolver, MORSolver
from morec.ec import solve_reduced
from morec.ed import exact_gf
from run_campaign import CAMPAIGNS, _root, compressed_solver
from time_solvers import points


def relerr(a, b):
    """Mean over frequencies of |b - a| / |a| (norm over orbitals), as in paper/extract_data.py."""
    dn, en = np.linalg.norm(b - a, axis=0), np.linalg.norm(a, axis=0)
    return float(np.mean(dn[en > 1e-15] / en[en > 1e-15]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("campaign", choices=[c for c in CAMPAIGNS if "post_sweeps" in CAMPAIGNS[c]])
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--tols", nargs="+", type=float, required=True)
    ap.add_argument("--npts", type=int, default=100)
    ap.add_argument("--out")
    args = ap.parse_args()
    camp, root = CAMPAIGNS[args.campaign], _root(args.campaign)
    cfg = DMFTConfig(**camp["dmft"])
    aim = AIM(cfg.N, cfg.NI)
    ed = EDSolver(aim)
    iw = 1j * np.pi * (2 * np.arange(cfg.n_iw) + 1) / cfg.beta
    ed_dir = os.path.join(root, "ed")
    shared = [p for p in points(ed_dir) if all(os.path.isdir(os.path.join(root, c, "post_run", p)) for c in camp["configs"])]
    shared = shared[::max(1, len(shared) // args.npts)][:args.npts]  # the Fig. 6 points
    out = {"campaign": args.campaign, "points": shared, "tols": args.tols, "configs": {}}
    for config in args.configs:
        solver = MORSolver(os.path.join(root, config, "atlas"), interp="sigma", bw_ratios=camp["dmft"]["bw_ratios"])
        patch = solver.atlas.patches[0]
        patch.load()
        B = patch.B.toarray()
        ref = []
        for p in shared:
            d = os.path.join(root, config, "post_run", p)
            meta = json.load(open(os.path.join(d, "dmft_metadata.json")))
            h = np.load(os.path.join(d, "bath_history.npz"))
            bath = [(h["eb"][-1, o], h["vb"][-1, o]) for o in range(cfg.NI)] + [meta["dmu_abs"]]
            alpha, E = solve_reduced(patch.h_r, aim.theta(meta["U"], meta["J"], bath))
            _, H_p, H_h = aim.hamiltonians(meta["U"], meta["J"], bath)
            psi = B @ alpha
            psi /= np.linalg.norm(psi)
            G_ec = np.array([exact_gf(aim, float(E), psi, H_p, H_h, iw, o, o, gf_tol=1e-12) for o in range(cfg.NI)])
            ref.append((meta, bath, ed.solve(meta["U"], meta["J"], bath, iw)[0], G_ec))
        res = {"k": int(B.shape[1]), "ec": [relerr(G_ed, G_ec) for _, _, G_ed, G_ec in ref], "rows": []}
        for tol in [None] + list(args.tols):
            if tol is None:
                rom, m = solver, [int(x) for v in patch.meta["m"].values() for x in v]
            else:
                rom, record = compressed_solver(args.campaign, config, tol)
                m = [int(x) for v in record["m_compressed"].values() for x in v.values()]
            resolvent, total = [], []
            for meta, bath, G_ed, G_ec in ref:
                rom.set_bw_ratios(meta["bw_ratios"])
                G = rom.solve(meta["U"], meta["J"], bath, iw)[0]
                resolvent.append(relerr(G_ec, G))
                total.append(relerr(G_ed, G))
            res["rows"].append({"tol": tol, "m": m, "resolvent": resolvent, "total": total})
            print(f"{config:<12} eps_c = {'none' if tol is None else format(tol, 'g'):>6}: m = {min(m)}-{max(m)}, medians: "
                  f"EC {np.median(res['ec']):.1e}, resolvent {np.median(resolvent):.1e}, total {np.median(total):.1e}", flush=True)
        out["configs"][config] = res
    out_path = args.out or os.path.join(root, "tolerance_scan_errors.json")
    with open(out_path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
