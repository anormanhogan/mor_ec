"""Reproduce the DMFT campaigns behind the paper figures.

Each campaign builds one offline atlas per (r, eps_off), copies it for every configuration,
enriches the copy during a coarse DMFT sweep and then runs the fine sweeps on the enriched
atlas, once with the ROM as built and once with the ROM compressed on the coarse sweep.  ED
reference sweeps run separately.

    python run_campaign.py n8ni2 build              # offline atlases
    python run_campaign.py n8ni2 enrich   [CONFIG]  # coarse sweep with online enrichment
    python run_campaign.py n8ni2 post     [CONFIG]  # fine sweeps on the enriched atlas
    python run_campaign.py n8ni2 compress [CONFIG]  # the same fine sweeps with the compressed ROM
    python run_campaign.py n8ni2 ed                 # ED reference sweeps
    python run_campaign.py n8ni2 list

Outputs go to ROOT/<campaign>/ (ROOT = $MOREC_RUNS or ./runs): <config>/atlas, enrich_run,
<sweep>_run, and for the compressed ROM <sweep>_c_run with the record compression.json.  Every
step resumes where it stopped, so a job killed by a wall-time limit can simply be resubmitted.
time_solvers.py and tolerance_scan.py (Table I, Fig. 6) work on these outputs.
"""
import json
import os
import shutil
import sys
import time

from morec import BuildConfig, DMFTConfig, EnrichConfig, MORSolver, build, make_solver, sweep, sweep_values

ENRICH = dict(tol=1e-2, variance_tol=1e-4, pool_size=16, scale=0.2, stagnation_iters=10)
N8 = dict(N=8, NI=2)
N8_BUILD = dict(**N8, U_lo=0.01, U_hi=6.0, bw_lo=[1.0, 0.01], bw_hi=[1.0, 0.75], n_test=5)
COMPRESS_TOL = 1e-5  # compression tolerance of the paper, both models (App. E)

CAMPAIGNS = {
    # Two-band OSMT at J = 0: Figs. 6, 9, 10, 11 and Table I.
    "n8ni2": {
        "build": N8_BUILD,
        "dmft": dict(**N8, bw_ratios=[1.0, 0.1]),
        "configs": {f"r{r}_thr1e-4": dict(r=r, ec_tol=1e-2, enrich=ENRICH) for r in range(1, 7)}
                   | {"r4_thr1e-8": dict(r=4, ec_tol=1e-8,
                                         enrich=ENRICH | dict(tol=1e-4, variance_tol=1e-8, pool_size=32))},
        "enrich_sweep": dict(U=(0, 10, 1), bwr=(0.1, 0.5, 0.1)),
        "post_sweeps": {"post": dict(U=(0, 10, 0.25), bwr=(0.1, 0.5, 0.05))},
        "extra_sweeps": {"r3_thr1e-4": {"fine": dict(U=(0, 8, 0.1), bwr=(0.1, 0.5, 0.01)),
                                        "fine_rev": dict(U=(0, 8, 0.1), bwr=(0.1, 0.5, 0.01), reverse=True)}},
        "ed_sweeps": {"ed": dict(U=(0, 10, 0.1), bwr=(0.1, 0.5, 0.05)),
                      "ed_coarse": dict(U=(0, 10, 1), bwr=(0.1, 0.5, 0.05))},
    },
    # Doped single-band Hubbard model: Figs. 7 and 8.
    "n10ni1": {
        "build": dict(N=10, NI=1, U_lo=0.01, U_hi=6.0, n_test=5, n_workers=10),
        "dmft": dict(N=10, NI=1, bw_ratios=[1.0]),
        "configs": {"r3_thr1e-1": dict(r=3, ec_tol=1e-2, enrich=ENRICH | dict(variance_tol=1e-1))},
        "enrich_sweep": dict(U=(0, 8, 1), mu=(-3, 4, 1)),
        "post_sweeps": {"post": dict(U=(0, 8, 0.25), mu=(-3, 4, 0.25))},
        "ed_sweeps": {"ed": dict(U=(0, 8, 0.25), mu=(-3, 4, 0.25))},
    },
    # SIAM with five bath sites, real-axis frequencies chosen by IRKA (Fig. 5): build and ED
    # comparison only.
    "siam_demo": {
        "build": dict(N=6, NI=1, r=6, U_lo=0.01, U_hi=8.0, bw_lo=[1.0, 0.01], bw_hi=[1.0, 0.6],
                      symmetric_baths=True, ec_tol=1e-6, k_max=100, frequencies="irka",
                      illustrate=True, n_test=200, n_test_w=500, test_eta=0.08),
        "configs": {"r6": dict(r=6, ec_tol=1e-6)},
    },
}


def _root(name):
    return os.path.join(os.environ.get("MOREC_RUNS", "runs"), name)


def _pristine(name, cfg):
    return os.path.join(_root(name), f"atlas_r{cfg['r']}_tol{cfg['ec_tol']:g}")


def _sweep(camp, solver, outdir, spec):
    cfg = DMFTConfig(**camp["dmft"])
    sweep(cfg, solver, outdir, sweep_values(*spec["U"], clamp_min=0.01), None,
          sweep_values(*spec["bwr"]) if "bwr" in spec else None,
          sweep_values(*spec["mu"]) if "mu" in spec else None, spec.get("reverse", False))


def tag(tol):
    """Suffix of the outputs of a compression tolerance: '' for COMPRESS_TOL, else the tolerance."""
    return "" if abs(tol / COMPRESS_TOL - 1.0) < 1e-9 else f"{tol:g}"


def compressed_solver(name, config, tol=COMPRESS_TOL):
    """(MORSolver on the enriched atlas of `config`, compressed in memory on the final baths of its
    coarse sweep; record of the compression)."""
    camp = CAMPAIGNS[name]
    solver = MORSolver(os.path.join(_root(name), config, "atlas"), interp="sigma", bw_ratios=camp["dmft"]["bw_ratios"])
    t0 = time.perf_counter()
    sizes = solver.compress(os.path.join(_root(name), config, "enrich_run"), tol=tol, final_only=True)
    patch = solver.atlas.patches[0]
    record = dict(training="final baths of the coarse sweep", n_solves=solver.compression["n_baths"], tol=tol,
                  n_freq=solver.compression["n_freq"], seconds=round(time.perf_counter() - t0, 2),
                  k=int(patch.B.shape[1]), m_full={ij: dict(zip("ph", m)) for ij, m in patch.meta["m"].items()},
                  m_compressed={f"{i},{j}": m for (i, j), m in sizes.items()})
    return solver, record


def run(name, step, only=None):
    camp = CAMPAIGNS[name]
    configs = {k: v for k, v in camp["configs"].items() if only in (None, k)}
    if step == "list":
        for k, v in configs.items():
            print(k, v)
        return
    if step == "build":
        for k, cfg in configs.items():
            pristine = _pristine(name, cfg)
            if not os.path.exists(os.path.join(pristine, "atlas.json")):
                build(BuildConfig(**(camp["build"] | dict(r=cfg["r"], ec_tol=cfg["ec_tol"]))), pristine)
            work = os.path.join(_root(name), k, "atlas")
            if "enrich" in cfg and not os.path.exists(work):
                shutil.copytree(pristine, work)
        return
    if step == "ed":
        solver = make_solver("ED", camp["dmft"]["N"], camp["dmft"]["NI"])
        for k, spec in camp["ed_sweeps"].items():
            _sweep(camp, solver, os.path.join(_root(name), k), spec)
        return
    for k, cfg in configs.items():
        atlas = os.path.join(_root(name), k, "atlas")
        if not os.path.exists(atlas):
            raise SystemExit(f"{atlas} missing: run the build step first")
        bw = camp["dmft"]["bw_ratios"]
        workers = camp["build"].get("n_workers", 4)
        if step == "enrich":
            solver = MORSolver(atlas, interp="sigma", bw_ratios=bw, n_workers=workers,
                               enrich=EnrichConfig(**cfg["enrich"]))
            _sweep(camp, solver, os.path.join(_root(name), k, "enrich_run"), camp["enrich_sweep"])
        elif step == "post":
            solver = MORSolver(atlas, interp="sigma", bw_ratios=bw)
            sweeps = camp["post_sweeps"] | camp.get("extra_sweeps", {}).get(k, {})
            for s, spec in sweeps.items():
                _sweep(camp, solver, os.path.join(_root(name), k, f"{s}_run"), spec)
        elif step == "compress":
            solver, record = compressed_solver(name, k)
            with open(os.path.join(_root(name), k, "compression.json"), "w") as f:
                json.dump(record, f, indent=1)
            print(f"=== {name} {k}: compressed in {record['seconds']} s on {record['n_solves']} coarse-sweep baths; "
                  f"m {record['m_full']} -> {record['m_compressed']} ===", flush=True)
            sweeps = camp["post_sweeps"] | camp.get("extra_sweeps", {}).get(k, {})
            for s, spec in sweeps.items():
                _sweep(camp, solver, os.path.join(_root(name), k, f"{s}_c_run"), spec)
        else:
            raise SystemExit(f"unknown step {step}")


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in CAMPAIGNS:
        raise SystemExit(__doc__ + "\ncampaigns: " + ", ".join(CAMPAIGNS))
    run(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
