"""Reproduce the DMFT campaigns behind the paper figures.

Each campaign builds one offline atlas per (r, eps_off), copies it for every configuration,
enriches the copy during a coarse DMFT sweep and then runs the fine sweeps on the enriched
atlas.  ED reference sweeps run separately.  Settings were recovered from the metadata of
the production runs.

    python run_campaign.py n8ni2 build            # offline atlases
    python run_campaign.py n8ni2 enrich [CONFIG]  # coarse sweep with online enrichment
    python run_campaign.py n8ni2 post   [CONFIG]  # fine sweeps on the enriched atlas
    python run_campaign.py n8ni2 ed               # ED reference sweeps
    python run_campaign.py n8ni2 list

Outputs go to ROOT/<campaign>/ (ROOT = $MOREC_RUNS or ./runs).  Every step resumes where it
stopped, so a job killed by a wall-time limit can simply be resubmitted.
"""
import os
import shutil
import sys

from morec import BuildConfig, DMFTConfig, EnrichConfig, MORSolver, build, make_solver, sweep, sweep_values

ENRICH = dict(tol=1e-2, variance_tol=1e-4, pool_size=16, scale=0.2, stagnation_iters=10)
N8 = dict(N=8, NI=2)
N8_BUILD = dict(**N8, U_lo=0.01, U_hi=6.0, bw_lo=[1.0, 0.01], bw_hi=[1.0, 0.75], n_test=5)
FIG5_FREQUENCIES = [[3.2888818034272242, 0.1], [-3.2888818034272242, 0.1], [0.2861615292297266, 0.1],
                    [-0.2861615292297266, 0.1], [0.6648120267507087, 0.1], [-0.6648120267507087, 0.1]]

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
    # Two-band model at finite Hund coupling J/U (not in the paper).
    "finite_j": {
        "build": dict(**N8, U_lo=0.01, U_hi=6.0, J_U_lo=0.0, J_U_hi=0.5, bw_lo=[1.0, 0.1], bw_hi=[1.0, 1.0],
                      n_test=5),
        "dmft": dict(**N8, bw_ratios=[1.0, 0.1]),
        "configs": {"r3_thr1e-3": dict(r=3, ec_tol=1e-2, enrich=ENRICH | dict(variance_tol=1e-3))},
        "enrich_sweep": dict(U=(0, 10, 1), JU=(0, 0.5, 0.1), bwr=(0.1, 1.0, 0.1)),
        "post_sweeps": {"post": dict(U=(0, 10, 0.25), JU=(0, 0.5, 0.05), bwr=(0.1, 1.0, 0.05))},
        "ed_sweeps": {"ed": dict(U=(0, 10, 1), JU=(0, 0.5, 0.05), bwr=(0.1, 1.0, 0.05))},
    },
    # SIAM with five bath sites, real-axis frequencies (Fig. 5): build and ED comparison only.  The
    # frequencies are the published ones, picked by the IRKA-style selection of the original code;
    # frequencies="irka" uses the current (textbook) IRKA instead and moves the Fig. 5 stars.
    "siam_demo": {
        "build": dict(N=6, NI=1, r=6, U_lo=0.01, U_hi=8.0, bw_lo=[1.0, 0.01], bw_hi=[1.0, 0.6],
                      symmetric_baths=True, ec_tol=1e-6, k_max=100, frequencies=FIG5_FREQUENCIES,
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
    if "JU" in spec:
        levels = [(f"JU{r:.4g}", lambda U, r=r: r * U) for r in sweep_values(*spec["JU"])]
    else:
        levels = [("J0", lambda U: 0.0)]
    sweep(cfg, solver, outdir, sweep_values(*spec["U"], clamp_min=0.01), levels,
          sweep_values(*spec["bwr"]) if "bwr" in spec else None,
          sweep_values(*spec["mu"]) if "mu" in spec else None, spec.get("reverse", False))


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
        else:
            raise SystemExit(f"unknown step {step}")


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in CAMPAIGNS:
        raise SystemExit(__doc__ + "\ncampaigns: " + ", ".join(CAMPAIGNS))
    run(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
