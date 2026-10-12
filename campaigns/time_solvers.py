"""Per-call cost of ED and MOR+EC on identical problems, timed in turn in one process.

    python time_solvers.py CAMPAIGN [--npts 100] [--reps 3] [--configs CONFIG ...] [--compressed [TOL ...]] [--out FILE]

Replays a few iterations of about --npts points of the campaign's ED sweep (those every
configuration's post sweep also holds) with the recorded baths, giving each call to ED and then to
every ROM, so that all solvers see the same machine load.  Needs the ED sweep and the enriched
atlases.  --compressed also times the compressed ROMs (entries "<config>+c", or "<config>+c1e-06"
for a tolerance other than the campaign's).

Output: $MOREC_RUNS/CAMPAIGN/timing_back_to_back[_compressed].json in the layout of
paper/data/fig6_timing.json: per point [ground-state seconds, Green's-function seconds], one entry
per replayed iteration (median over --reps); downstream code drops the first.
"""
import argparse
import glob
import json
import os
import socket
from datetime import datetime

import numpy as np

from morec import MORSolver, make_solver
from run_campaign import CAMPAIGNS, COMPRESS_TOL, _root, compressed_solver, tag


def points(base):
    """Relative paths of the DMFT points under base (as paper/extract_data.points)."""
    return sorted(os.path.relpath(os.path.dirname(p), base)
                  for p in glob.glob(os.path.join(base, "**", "gf_sigma_current.npz"), recursive=True))


def replayed(n):
    """Iterations of an n-iteration point to replay: the first, then early, middle and last."""
    return [0] + sorted({1, n // 2, n - 1} - {0}) if n > 1 else []


def speedup(ed, rom, pts, col):
    """Median, 5th and 95th percentile over points of mean ED time / mean ROM time (first call dropped)."""
    t = lambda calls, p: np.sum([calls[p][i] for i in col], axis=0)[1:].mean()
    r = np.array([t(ed, p) / t(rom, p) for p in pts])
    return np.median(r), np.percentile(r, 5), np.percentile(r, 95)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("campaign", choices=[c for c in CAMPAIGNS if "ed_sweeps" in CAMPAIGNS[c]])
    ap.add_argument("--npts", type=int, default=100)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--configs", nargs="*", help="subset of the campaign's configurations (default: all)")
    ap.add_argument("--compressed", nargs="*", type=float, metavar="TOL",
                    help="also time the compressed ROMs (no value: the campaign's tolerance)")
    ap.add_argument("--out")
    args = ap.parse_args()
    camp, root = CAMPAIGNS[args.campaign], _root(args.campaign)
    N, NI = camp["dmft"]["N"], camp["dmft"]["NI"]
    ed_dir = os.path.join(root, "ed")
    configs = [c for c in camp["configs"] if args.configs is None or c in args.configs]
    tols = [COMPRESS_TOL] if args.compressed == [] else (args.compressed or [])
    shared = [p for p in points(ed_dir) if all(os.path.isdir(os.path.join(root, c, "post_run", p)) for c in configs)]
    shared = shared[::max(1, len(shared) // args.npts)][:args.npts]
    out_path = args.out or os.path.join(root, "timing_back_to_back_compressed.json" if tols else
                                        "timing_back_to_back.json")

    solvers = {"ED": make_solver("ED", N, NI)}
    for c in configs:
        solvers[c] = MORSolver(os.path.join(root, c, "atlas"), interp="sigma", bw_ratios=camp["dmft"]["bw_ratios"])
    compression = {}
    if tols:
        for c in configs:
            for tol in tols:
                solvers[f"{c}+c{tag(tol)}"], compression[f"{c}+c{tag(tol)}"] = compressed_solver(args.campaign, c, tol)
    roms = [name for name in solvers if name != "ED"]
    out = {"campaign": args.campaign, "host": socket.gethostname(), "started": datetime.now().isoformat(),
           "loadavg_start": os.getloadavg(), "reps_per_call": args.reps, "shared": [], "iterations": {},
           "ed": {}, "configs": {c: {"calls": {}} for c in roms}, "reps": {name: {} for name in solvers},
           "compression": compression}
    print(f"{args.campaign}: {len(shared)} points, {args.reps} repetitions per call, on {out['host']} "
          f"(load {out['loadavg_start'][0]:.1f}); solvers: {list(solvers)}", flush=True)

    def save():
        out.update(finished=datetime.now().isoformat(), loadavg_end=os.getloadavg())
        with open(out_path + ".tmp", "w") as f:
            json.dump(out, f, separators=(",", ":"))
        os.replace(out_path + ".tmp", out_path)

    warm = False
    for n_done, p in enumerate(shared, 1):
        d = os.path.join(ed_dir, p)
        meta = json.load(open(os.path.join(d, "dmft_metadata.json")))
        hist = np.load(os.path.join(d, "bath_history.npz"))
        its = replayed(hist["eb"].shape[0])
        if not its:
            continue
        iw = 1j * np.pi * (2 * np.arange(meta["n_iw"]) + 1) / meta["beta"]
        for s in solvers.values():
            if isinstance(s, MORSolver):
                s.set_bw_ratios(meta["bw_ratios"])
        times = {name: [] for name in solvers}  # [iteration][repetition] -> (t_gs, t_gf)
        for it in its:
            bath = [(hist["eb"][it, o], hist["vb"][it, o]) for o in range(NI)] + [meta.get("dmu_abs", 0.0)]
            if not warm:  # one untimed call each: the ROMs load their atlases on first use
                for s in solvers.values():
                    s.solve(meta["U"], meta["J"], bath, iw)
                warm = True
            rows = {name: [] for name in solvers}
            for _ in range(args.reps):
                for name, s in solvers.items():  # ED, then every ROM, within the same seconds
                    rows[name].append(s.solve(meta["U"], meta["J"], bath, iw)[3:5])
            for name in solvers:
                times[name].append(rows[name])
        out["shared"].append(p)
        out["iterations"][p] = its
        for name in solvers:
            med = np.median(np.array(times[name]), axis=1)  # (iterations, 2)
            calls = [med[:, 0].tolist(), med[:, 1].tolist()]
            if name == "ED":
                out["ed"][p] = calls
            else:
                out["configs"][name]["calls"][p] = calls
            out["reps"][name][p] = np.array(times[name]).tolist()
        if n_done % 10 == 0:
            save()
            tot = {c: speedup(out["ed"], out["configs"][c]["calls"], out["shared"], (0, 1))[0] for c in roms}
            print(f"  [{n_done}/{len(shared)}] total speedup so far: "
                  + "  ".join(f"{c} {v:.1f}x" for c, v in tot.items()), flush=True)

    save()  # also when the last points were skipped (single-iteration points have nothing to replay)
    pts = out["shared"]
    ms = lambda calls, i: 1e3 * np.mean([x for p in pts for x in calls[p][i][1:]])
    print(f"\nED per call: ground state {ms(out['ed'], 0):.1f} ms, Green's function {ms(out['ed'], 1):.1f} ms "
          f"({len(pts)} points; load at the end {out['loadavg_end'][0]:.1f})")
    print(f"{'config':<20}{'GS ms':>8}{'GF ms':>8} | speedup, median [5%-95%]: {'ground state':>24}{'Greens function':>22}{'total':>22}")
    for c in roms:
        calls = out["configs"][c]["calls"]
        f = lambda s: f"{s[0]:.1f}x [{s[1]:.1f}-{s[2]:.1f}]"
        print(f"{c:<20}{ms(calls, 0):>8.2f}{ms(calls, 1):>8.2f} | {'':26}{f(speedup(out['ed'], calls, pts, (0,))):>24}"
              f"{f(speedup(out['ed'], calls, pts, (1,))):>22}{f(speedup(out['ed'], calls, pts, (0, 1))):>22}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
