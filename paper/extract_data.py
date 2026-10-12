"""Reduce raw campaign outputs to the compact data bundle read by make_figures.py.

    python extract_data.py --raw /path/to/runs [--layout production|campaign] [fig5 fig6 ...]

--layout production reads the directory names of the original production archive;
--layout campaign reads the output of campaigns/run_campaign.py (ROOT = --raw).
--compressed TAG (campaign layout) takes the fine sweeps run with the compressed ROM (<sweep>_TAG_run of
`run_campaign.py CAMPAIGN compress`, TAG = c) and the timings measured with it (campaigns/time_solvers.py).
--logs DIR: captured output of the enrich steps, for atlases that did not record the wall time of their
enrichment events themselves (those enriched with morec 1.0).
Writes paper/data/*.json|npz (or $MOREC_PAPER_DATA).  Fig. 6 panels come from the error-channel CSVs when present in
the raw directory (computing them takes ~15 min, see fig6_compute).
"""
import argparse
import glob
import json
import os
import re
import shutil
import tempfile

import numpy as np

from morec import AIM, EDSolver, MORSolver
from morec.bethe import quasiparticle_weight
from morec.ed import exact_gf
from morec.atlas import Atlas
import legacy

DATA = os.environ.get("MOREC_PAPER_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
GS_GF = re.compile(r"GS:\s*([\d.]+)s\s*GF:\s*([\d.]+)s")
OCC = re.compile(r"Occupation:\s*n_0=[\d.eE+\-]+\s+sum=([\d.eE+\-]+)")
N8_CFGS = [f"r{r}_thr1e-4" for r in range(1, 7)] + ["r4_thr1e-8"]

LAYOUTS = {
    "production": {
        "demo": "_1B_5B_demo_1",
        "n8_ed": "final_production_data/N8_NI2_ED_DMFT",
        "n8_ed_coarse": "final_production_data/N8_NI2_ED_DMFT_coarse",
        "n8_atlas": lambda c: "final_production_data/N8NI2_atlas_{}_noirka_greedy_r4_try_again_thr{:g}".format(
            c.split("_")[0], float(c.split("thr")[1])),
        "n8_enrich": lambda c: f"final_production_data/N8NI2_MOREC_{c}_enrich_run_r4_try_again",
        "n8_post": lambda c: f"final_production_data/N8NI2_MOREC_{c}_post_enrich_run_r4_try_again",
        "n8_fine": "final_production_data/N8NI2_MOREC_r3_thr1e-4_post_enrich_run_r4_try_again_fine",
        "fig6_csv_r": "final_production_data/gf_error_channels_r4_try_again_thr0.0001.csv",
        "fig6_csv_k": "final_production_data/gf_error_channels_r4_try_again_r4_thr1e-08_k.csv",
        "n10_ed": "final_production_data/N10_NI1_ED_DMFT",
        "n10_post": "final_production_data/N10NI1_MOREC_r3_thr1e-1_post_enrich_run_r3_only",
        "n10_atlas": "final_production_data/N10NI1_atlas_r3_noirka_greedy_r3_only_thr0.1",
    },
    "campaign": {
        "demo": "siam_demo/atlas_r6_tol1e-06",
        "n8_ed": "n8ni2/ed",
        "n8_ed_coarse": "n8ni2/ed_coarse",
        "n8_atlas": lambda c: f"n8ni2/{c}/atlas",
        "n8_enrich": lambda c: f"n8ni2/{c}/enrich_run",
        "n8_post": lambda c: f"n8ni2/{c}/post_run",
        "n8_fine": "n8ni2/r3_thr1e-4/fine_run",
        "n8_timing": "n8ni2/timing_back_to_back.json",
        "fig6_csv_r": None,
        "fig6_csv_k": None,
        "n10_ed": "n10ni1/ed",
        "n10_post": "n10ni1/r3_thr1e-1/post_run",
        "n10_atlas": "n10ni1/r3_thr1e-1/atlas",
    },
}


# ----------------------------------------------------------------------------- run readers

def points(base):
    """Relative paths of every DMFT point (directories holding gf_sigma_current.npz)."""
    return sorted(os.path.relpath(os.path.dirname(p), base)
                  for p in glob.glob(os.path.join(base, "**", "gf_sigma_current.npz"), recursive=True))


def solver_calls(point):
    """Per-iteration (GS seconds, GF seconds) of one DMFT point."""
    try:
        text = open(os.path.join(point, "dmft_log.txt"), errors="replace").read()
    except OSError:
        return [], []
    calls = GS_GF.findall(text)
    return [float(a) for a, _ in calls], [float(b) for _, b in calls]


def last_occupation(point):
    found = OCC.findall(open(os.path.join(point, "dmft_log.txt"), errors="replace").read())
    return float(found[-1]) if found else float("nan")


def gf_data(point):
    try:
        d = np.load(os.path.join(point, "gf_sigma_current.npz"))
        return {k: d[k] for k in ("Sigma", "G_imp", "Delta", "iw_imag")}
    except Exception:
        return None


def z_sigma(d):
    iw = 1j * d["iw_imag"]
    return [quasiparticle_weight(d["Sigma"][o], iw) for o in range(len(d["Sigma"]))]


def z_dyson(d):
    iw = 1j * d["iw_imag"]
    return [quasiparticle_weight(d["Sigma"][o], iw, d["G_imp"][o], d["Delta"][o])
            for o in range(len(d["Sigma"]))]


def gf_relerr(a, b):
    """Mean over frequencies of |G_b - G_a| / |G_a| (norm over orbitals)."""
    dn = np.linalg.norm(b - a, axis=0)
    en = np.linalg.norm(a, axis=0)
    m = en > 1e-15
    return float(np.mean(dn[m] / en[m])) if m.any() else float("nan")


def _save(name, obj):
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, name)
    if name.endswith(".npz"):
        np.savez_compressed(path, **obj)
    else:
        with open(path, "w") as f:
            json.dump(obj, f, separators=(",", ":"))
    print(f"wrote {path} ({os.path.getsize(path) / 1e3:.0f} kB)")


def _new_layout(atlas_dir, tmp):
    """Atlas in the morec layout (original production atlases are converted on the fly)."""
    if not legacy.is_legacy(atlas_dir):
        return atlas_dir
    out = os.path.join(tmp, os.path.basename(atlas_dir))
    if not os.path.exists(out):
        legacy.convert(atlas_dir, out)
    return out


EVENT = re.compile(r"\[enrich\] .*k (\d+) -> (\d+) \(([\d.]+)s\)")


def build_cost(src, campaign, config):
    """Extra entries of the ROM summary: event_wall_s, the wall time of the enrichment events of `config`
    (candidate selection included) for an atlas that did not record it, read from the captured output
    of its enrich step under --logs, and compress_s / m_compressed from the compression record."""
    out = {}
    root = os.path.basename(os.path.normpath(src["raw"]))
    where = re.compile(r"outdir=\S*%s/%s/([^/\s]+)/(\w+_run)/" % (re.escape(root), campaign))
    for log in sorted(glob.glob(os.path.join(src["logs"], "*"))) if src.get("logs") else []:
        cur, walls = None, []
        for line in open(log, errors="replace"):
            m = where.search(line)
            if m:
                cur = m.group(1)
                continue
            m = EVENT.search(line)
            if m and cur == config:
                walls.append(float(m.group(3)))
        if walls:
            out["event_wall_s"] = float(sum(walls))
    rec = os.path.join(src["raw"], campaign, config, "compression.json")
    if src.get("compressed") and os.path.exists(rec):
        c = json.load(open(rec))
        out.update(compress_s=c["seconds"], compress_tol=c["tol"],
                   m_compressed=float(np.mean([x for v in c["m_compressed"].values() for x in v.values()])))
    return out


# ----------------------------------------------------------------------------- figures

def fig5(src, n_tests=15):
    """SIAM demo: ED and MOR+EC spectra at the first test points, plus the IRKA frequencies."""
    d = src["demo"]
    pm = json.load(open(glob.glob(os.path.join(d, "patch_*", "meta.json"))[0]))
    omega = pm["irka"]["omega"] if "irka" in pm else pm["omega"]
    tests = [np.load(os.path.join(d, f"test_{t}", "results.npz")) for t in range(n_tests)]
    _save("fig5_siam_demo.npz", dict(
        U=np.array([float(t["U_test"]) for t in tests]), w=tests[0]["freqs_real"].real,
        A_ed=np.array([-t["G_fom_real"][0, 0].imag / np.pi for t in tests]),
        A_rom=np.array([-t["G_rom1_real"][0, 0].imag / np.pi for t in tests]),
        omega=np.array([complex(*z) for z in omega])))


def fig6(src, npts=100):
    """Error channels vs r (panel a) and vs k (panel b), per-call solver timings (Table I)."""
    for key, name in (("fig6_csv_r", "fig6_channels_r.csv"), ("fig6_csv_k", "fig6_channels_k.csv")):
        if src.get(key) and os.path.exists(src[key]):
            shutil.copy(src[key], os.path.join(DATA, name))
            print(f"copied {src[key]}")
        else:
            print(f"{name}: no CSV in the raw data, run fig6_compute")
    ed = src["n8_ed"]
    runs = {c: src["n8_post"](c) for c in N8_CFGS}
    r_axis = [c for c in N8_CFGS if c.endswith("thr1e-4")]
    shared = [p for p in points(ed) if all(os.path.isdir(os.path.join(runs[c], p)) for c in r_axis)]
    shared = shared[::max(1, len(shared) // npts)][:npts]
    out = {"shared": shared, "ed": {p: solver_calls(os.path.join(ed, p)) for p in shared}, "configs": {}}
    for c in N8_CFGS:
        rels = [p for p in shared if os.path.isdir(os.path.join(runs[c], p))]
        s = legacy.rom_summary(src["n8_atlas"](c))
        out["configs"][c] = dict(r=int(c[1]), thr=float(c.split("thr")[1]), k=s["k"], k_off=s["k_off"],
                                 m=s["m"], ec_tol=s["ec_tol"],
                                 calls={p: solver_calls(os.path.join(runs[c], p)) for p in rels})
    b2b = src.get("n8_timing")
    if b2b and os.path.exists(b2b):
        # ED and the ROMs timed back to back on one node (campaigns/time_solvers.py) replace the
        # times logged by the separate sweeps, which do not share one machine load.
        t = json.load(open(b2b))
        out.update(shared=t["shared"], ed=t["ed"], timing=dict(source="back_to_back", host=t["host"],
                                                                 started=t["started"], reps=t["reps_per_call"]))
        for c in N8_CFGS:
            out["configs"][c]["calls"] = t["configs"][c]["calls"]
            cc = c + "+" + (src.get("compressed") or "")
            if cc in t["configs"]:  # the same ROM after compression, timed in the same job
                out["configs"][c]["compressed"] = dict(calls=t["configs"][cc]["calls"],
                                                       **build_cost(src, "n8ni2", c))
        print(f"timings: back to back on {t['host']} ({len(t['shared'])} points)")
    else:
        print("timings: from the sweep logs (no back-to-back file)")
    _save("fig6_timing.json", out)
    # Fig. 6(c): error channels and timings against the compression tolerance (campaigns/tolerance_scan.py and
    # time_solvers.py --compressed TOL ..., all tolerances timed in one job)
    scan = os.path.join(src["raw"], "n8ni2")
    if os.path.exists(os.path.join(scan, "tolerance_scan_errors.json")):
        shutil.copy(os.path.join(scan, "tolerance_scan_errors.json"), os.path.join(DATA, "fig6_tolerance.json"))
        print("copied tolerance_scan_errors.json")
    if os.path.exists(os.path.join(scan, "timing_tolerance_scan.json")):
        t = json.load(open(os.path.join(scan, "timing_tolerance_scan.json")))
        _save("fig6_tolerance_timing.json", dict(host=t["host"], shared=t["shared"], ed=t["ed"],
                                                 configs={c: v["calls"] for c, v in t["configs"].items()}))


def fig6_compute(src, npts=100, kstep=20):
    """Recompute both error-channel CSVs from the raw runs (EC channel = EC ground state driven
    through the exact resolvent, at each MOR run's final bath)."""
    import scipy.linalg as la
    aim = AIM(8, 2)
    ed = EDSolver(aim)
    tmp = tempfile.mkdtemp()

    def channels(atlas_dir, run, rels, ks=None):
        patch = Atlas(_new_layout(atlas_dir, tmp)).patches[0]
        patch.load()
        B = patch.B.toarray()
        res = []
        for rel in rels:
            u = os.path.join(run, rel)
            d = np.load(os.path.join(u, "gf_sigma_current.npz"))
            h = np.load(os.path.join(u, "bath_history.npz"))
            meta = json.load(open(os.path.join(u, "dmft_metadata.json")))
            bath = [(h["eb"][-1][o], h["vb"][-1][o]) for o in range(2)] + [meta["dmu_abs"]]
            iw = 1j * d["iw_imag"]
            _, H_p, H_h = aim.hamiltonians(meta["U"], meta["J"], bath)
            G_ed = ed.solve(meta["U"], meta["J"], bath, iw)[0]
            th = aim.theta(meta["U"], meta["J"], bath)
            h_full = sum(th.get(t, 0.0) * M for t, M in patch.h_r.items())
            row = []
            for k in (ks or [B.shape[1]]):
                w, v = la.eigh((h_full[:k, :k] + h_full[:k, :k].conj().T) / 2)
                psi = B[:, :k] @ v[:, 0]
                psi /= np.linalg.norm(psi)
                G_ec = np.array([exact_gf(aim, float(w[0]), psi, H_p, H_h, iw, o, o, gf_tol=1e-12)
                                 for o in range(2)])
                row.append((gf_relerr(G_ed, d["G_imp"]), gf_relerr(G_ed, G_ec), gf_relerr(G_ec, d["G_imp"]),
                            float(w[0])))
            res.append(row)
        return patch, np.array(res)

    ed_dir, r_axis = src["n8_ed"], [c for c in N8_CFGS if c.endswith("thr1e-4")]
    shared = [p for p in points(ed_dir)
              if all(os.path.isdir(os.path.join(src["n8_post"](c), p)) for c in r_axis)]
    shared = shared[::max(1, len(shared) // npts)][:npts]
    stats = lambda a: [np.median(a), np.percentile(a, 25), np.percentile(a, 75)]
    rows = []
    for c in r_axis:
        patch, res = channels(src["n8_atlas"](c), src["n8_post"](c), shared)
        m = np.mean([x for key, v in patch.meta["m"].items() if key[0] == key[-1] for x in v])
        rows.append([int(c[1]), patch.B.shape[1], m] + stats(res[:, 0, 0]) + stats(res[:, 0, 1]) + stats(res[:, 0, 2]))
    head = "r,k,m_avg,total_median,total_p25,total_p75,gs_median,gs_p25,gs_p75,rom_median,rom_p25,rom_p75"
    np.savetxt(os.path.join(DATA, "fig6_channels_r.csv"), rows, delimiter=",", header=head, comments="")

    run_b = src["n8_post"]("r4_thr1e-8")
    K = Atlas(_new_layout(src["n8_atlas"]("r4_thr1e-8"), tmp)).patches[0].meta["k"]
    ks = sorted({int(round(x)) for x in np.linspace(1, K, max(1, round((K - 1) / kstep)) + 1)})
    rels = [p for p in points(ed_dir) if os.path.isdir(os.path.join(run_b, p))]
    _, res = channels(src["n8_atlas"]("r4_thr1e-8"), run_b, rels[::max(1, len(rels) // npts)][:npts], ks)
    krows = [[k] + stats(res[:, i, 1]) + [np.mean(res[:, i, 3])] for i, k in enumerate(ks)]
    np.savetxt(os.path.join(DATA, "fig6_channels_k.csv"), krows, delimiter=",",
               header="k,gs_median,gs_p25,gs_p75,E0_mean", comments="")
    shutil.rmtree(tmp)


def fig7_8(src):
    """Doping scan: per-call timings, occupations and GF error on the shared (U, mu) grid."""
    ed, mo = src["n10_ed"], src["n10_post"]
    recs = []
    for p in points(ed):
        if not os.path.isdir(os.path.join(mo, p)):
            continue
        calls_ed, calls_mor = solver_calls(os.path.join(ed, p)), solver_calls(os.path.join(mo, p))
        if not (calls_ed[0] and calls_mor[0]):
            continue
        a, b = gf_data(os.path.join(ed, p)), gf_data(os.path.join(mo, p))
        u_dir, mu_dir = p.split(os.sep)[-2:]
        recs.append(dict(point=p, U=float(u_dir[1:]), mu=-float(mu_dir[2:]), ed=calls_ed, mor=calls_mor,
                         n_ed=last_occupation(os.path.join(ed, p)), n_mor=last_occupation(os.path.join(mo, p)),
                         err=gf_relerr(a["G_imp"], b["G_imp"]) if np.allclose(a["iw_imag"], b["iw_imag"])
                         else float("nan")))
    # The cumulative-cost curve runs through the points in (U, mu) order; mu is the physical
    # chemical potential, the negative of the value in the directory name.
    recs.sort(key=lambda r: (r["U"], r["mu"]))
    out = dict(points=recs, rom=dict(build_cost(src, "n10ni1", "r3_thr1e-1"), **legacy.rom_summary(src["n10_atlas"])))
    b2b = src.get("n10_timing")
    if b2b and os.path.exists(b2b):  # per-call speedups from ED and the ROM timed in turn on one node
        t = json.load(open(b2b))
        calls = t["configs"]["r3_thr1e-1" + ("+" + src["compressed"] if src.get("compressed") else "")]["calls"]
        out["timing"] = dict(host=t["host"], points=[[t["ed"][p], calls[p]] for p in t["shared"] if p in calls])
    _save("fig7_8_doping.json", out)


def fig9(src, bwrs=(0.1, 0.2, 0.5), cfg="r3_thr1e-4"):
    """Coarse scan with online enrichment: Z per band and solver time per (bwr, U)."""
    out = {"rom": dict(build_cost(src, "n8ni2", cfg), **legacy.rom_summary(src["n8_atlas"](cfg))), "bwr": {}}
    for bwr in bwrs:
        entry = {}
        for tag, base in (("ed", src["n8_ed_coarse"]), ("mor", src["n8_enrich"](cfg))):
            d = os.path.join(base, f"bwr{bwr:.4g}", "J0")
            rows = []
            for name in sorted(os.listdir(d)) if os.path.isdir(d) else []:
                g = gf_data(os.path.join(d, name))
                if g is not None:
                    gs, gf = solver_calls(os.path.join(d, name))
                    rows.append([float(name[1:])] + z_dyson(g) + [sum(gs) + sum(gf) if gs else None])
            entry[tag] = sorted(rows)
        out["bwr"][f"{bwr:g}"] = entry
    _save("fig9_coarse.json", out)


def fig10(src, cfg="r3_thr1e-4"):
    """Fine phase-diagram scan: Z grids of both solvers and matched per-call timings."""
    mor_base, ed_base = src["n8_fine"], src["n8_ed"]
    bwr_strs = sorted((d[3:] for d in os.listdir(mor_base) if re.match(r"^bwr[\d.]+$", d)), key=float)
    list_U = lambda d: {float(x[1:]) for x in os.listdir(d) if re.match(r"^U[\d.]+$", x)} if os.path.isdir(d) else set()
    bwr_strs = [b for b in bwr_strs if list_U(os.path.join(mor_base, f"bwr{b}", "J0"))]
    Us = sorted(set().union(*(list_U(os.path.join(base, f"bwr{b}", "J0"))
                              for b in bwr_strs for base in (mor_base, ed_base))))
    Z = {k: np.full((len(bwr_strs), len(Us)), np.nan) for k in ("Z0_mor", "Z1_mor", "Z0_ed", "Z1_ed")}
    calls = {}
    for ib, b in enumerate(bwr_strs):
        for iU, U in enumerate(Us):
            u = f"U{U:.4g}"
            pts = {tag: os.path.join(base, f"bwr{b}", "J0", u) for tag, base in (("mor", mor_base), ("ed", ed_base))}
            g = {tag: gf_data(p) if os.path.isdir(p) else None for tag, p in pts.items()}
            for tag in ("mor", "ed"):
                if g[tag] is not None:
                    Z[f"Z0_{tag}"][ib, iU], Z[f"Z1_{tag}"][ib, iU] = z_sigma(g[tag])
            if g["mor"] is not None and g["ed"] is not None:
                calls[f"{b}/{U:g}"] = [solver_calls(pts["ed"]), solver_calls(pts["mor"])]
    _save("fig10_phase.npz", dict(bwr=np.array([float(b) for b in bwr_strs]), U=np.array(Us), **Z))
    _save("fig10_timing.json", dict(calls=calls, rom=legacy.rom_summary(src["n8_atlas"](cfg))))


def fig11(src, cfg="r3_thr1e-4", bwr=0.15, Us=(2.0, 6.0), n_w=1000, eta=0.05, w_max=10.0):
    """Real-frequency narrow-band spectra of ED and MOR+EC, each at its own converged bath."""
    tmp = tempfile.mkdtemp()
    ed, mor = EDSolver(AIM(8, 2)), MORSolver(_new_layout(src["n8_atlas"](cfg), tmp))
    w = np.linspace(-w_max, w_max, n_w) + 1j * eta
    out = {"w": w.real, "U": np.array(Us)}
    for U in Us:
        for tag, base, solver in (("ed", src["n8_ed"], ed), ("mor", src["n8_post"](cfg), mor)):
            p = os.path.join(base, f"bwr{bwr:.4g}", "J0", f"U{U:.4g}")
            meta = json.load(open(os.path.join(p, "dmft_metadata.json")))
            h = np.load(os.path.join(p, "bath_history.npz"))
            bath = [(h["eb"][-1, o], h["vb"][-1, o]) for o in range(2)] + [meta["dmu_abs"]]
            if tag == "mor":
                solver.set_bw_ratios(meta["bw_ratios"])
            out[f"G_{tag}_U{U:g}"] = solver.solve(meta["U"], meta["J"], bath, w)[0][1]
    shutil.rmtree(tmp)
    _save("fig11_spectra.npz", out)


FIGS = {"fig5": fig5, "fig6": fig6, "fig7_8": fig7_8, "fig9": fig9, "fig10": fig10, "fig11": fig11}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="root of the raw campaign outputs")
    ap.add_argument("--layout", choices=list(LAYOUTS), default="production")
    ap.add_argument("--compressed", metavar="TAG", help="use the sweeps and timings of the compressed ROM")
    ap.add_argument("--logs", help="directory of the rebuild job logs")
    ap.add_argument("figs", nargs="*", help=f"subset of {list(FIGS) + ['fig6_compute']}")
    args = ap.parse_args()
    join = lambda v: (lambda c: os.path.join(args.raw, v(c))) if callable(v) else (
        os.path.join(args.raw, v) if v else None)
    src = {k: join(v) for k, v in LAYOUTS[args.layout].items()}
    src.update(raw=args.raw, logs=args.logs, compressed=args.compressed)
    if args.compressed:
        t = args.compressed
        src.update(n8_fine=join(f"n8ni2/r3_thr1e-4/fine_{t}_run"), n10_post=join(f"n10ni1/r3_thr1e-1/post_{t}_run"),
                   n8_timing=join("n8ni2/timing_back_to_back_compressed.json"),
                   n10_timing=join("n10ni1/timing_back_to_back_compressed.json"))
    for name in args.figs or list(FIGS):
        print(f"== {name}")
        (fig6_compute if name == "fig6_compute" else FIGS[name])(src)


if __name__ == "__main__":
    main()
