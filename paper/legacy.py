"""Readers for the original (pre-package) atlas layout used by the paper's production runs."""
import glob
import json
import os

import numpy as np
import scipy.sparse as sps

from morec.bath import base_bath
from morec.atlas import _save_json, save_patch


def is_legacy(atlas_dir):
    return os.path.exists(os.path.join(atlas_dir, "atlas_meta.json"))


def rom_summary(atlas_dir):
    """Build/enrichment cost and ROM size of an atlas in either layout.

    Returns dict(ec_s, mor_s, enrich_ec_s, enrich_mor_s, n_events, k_off, k, m, ec_tol), where
    ec_s/mor_s are the offline basis-selection/projection times and m is the mean m_p/m_h.
    """
    out = dict(ec_s=0.0, mor_s=0.0, enrich_ec_s=0.0, enrich_mor_s=0.0, n_events=0)
    legacy = is_legacy(atlas_dir)
    top = json.load(open(os.path.join(atlas_dir, "atlas_meta.json" if legacy else "atlas.json")))
    g = top["global_ec_basis"] if legacy else top["global_ec"]
    out["ec_s"] += g.get("time_s", 0.0)
    out["ec_tol"] = g.get("ec_variance_tol" if legacy else "tol")
    ms = []
    for pm_path in sorted(glob.glob(os.path.join(atlas_dir, "patch_*", "meta.json"))):
        pm = json.load(open(pm_path))
        if legacy:
            out["ec_s"] += pm["ec_basis"]["time_s"]
            out["mor_s"] += pm["mor_projections"]["total_time_s"]
            out["k"], out["k_off"] = pm["ec_basis"]["k_seed"], pm["ec_basis"].get("k_raw")
            ms += [v[s] for key, v in pm["mor_projections"]["pairs"].items()
                   for s in ("m_ij_p", "m_ij_h") if key[0] == key[-1] and not v.get("trivially_zero")]
        else:
            out["ec_s"] += pm["time"]["ec_s"]
            out["mor_s"] += pm["time"]["mor_s"]
            out["k"], out["k_off"] = pm["k"], pm["k_offline"]
            ms += [x for key, v in pm["m"].items() if key[0] == key[-1] for x in v]
        for e in pm.get("enrichment_history", []):
            out["enrich_ec_s"] += e.get("ec_time_s", 0.0)
            out["enrich_mor_s"] += e.get("mor_time_s", 0.0)
            out["n_events"] += 1
    out["m"] = float(np.mean(ms)) if ms else None
    return out


def _dict_dir(d, prefix):
    return {f[len(prefix):-4]: np.load(os.path.join(d, f)) for f in sorted(os.listdir(d)) if f.endswith(".npy")}


def convert(old, new, projectors=False):
    """Convert an original atlas directory to the morec layout (projectors only if requested)."""
    cfg = dict(np.load(os.path.join(old, "config.npz"), allow_pickle=True))
    am = json.load(open(os.path.join(old, "atlas_meta.json")))
    N, NI = int(cfg["N"]), int(cfg["NI"])
    bwr_orb = int(cfg["bwr_orb"]) if "bwr_orb" in cfg and int(cfg["bwr_orb"]) >= 0 else None
    base = base_bath(N - NI, NI)
    dirs = []
    for pdir in sorted(glob.glob(os.path.join(old, "patch_*"))):
        m = dict(np.load(os.path.join(pdir, "meta.npz"), allow_pickle=True))
        mj = json.load(open(os.path.join(pdir, "meta.json")))
        bwr_c = float(m["bwr_c"]) if "bwr_c" in m else None
        s = bwr_c if bwr_orb is not None and bwr_c is not None and not np.isclose(bwr_c, 1.0, rtol=1e-6) else 1.0
        bath_c = [(eb * s, vb * np.sqrt(s)) if o == bwr_orb else (eb, vb) for o, (eb, vb) in enumerate(base)]
        pairs, proj, mm = {}, {}, {}
        for i, j in [tuple(p) for p in m["pair_list"]]:
            if i > j:
                continue
            cv = np.load(os.path.join(pdir, f"coupling_{i}_{j}.npz"))
            pairs[(i, j)] = {"H_p": _dict_dir(os.path.join(pdir, f"H_hats_p_{i}_{j}"), "H_hat_"),
                             "H_h": _dict_dir(os.path.join(pdir, f"H_hats_h_{i}_{j}"), "H_hat_"),
                             "cv": {k: cv[k] for k in cv.files}}
            mm[f"{i},{j}"] = [int(cv["I_hat_p"].shape[0]), int(cv["I_hat_h"].shape[0])]
            if projectors:
                load = lambda n: sps.load_npz(os.path.join(pdir, f"{n}_{i}_{j}.npz"))
                Vp, Vh = load("Vp"), load("Vh")
                w = os.path.exists(os.path.join(pdir, f"Wp_{i}_{j}.npz"))
                proj[(i, j)] = (Vp, load("Wp") if w else Vp, Vh, load("Wh") if w else Vh)
        hist = [{"k_before": e.get("vector_idx"), "n_added": e.get("n_added", 1), "U": e["U"], "J": e["J"],
                 "bath": [[o["eb"], o["vb"]] for o in e["bath"]],
                 **{k: e.get(k) for k in ("ec_time_s", "mor_time_s", "h_hat_time_s", "total_time_s")}}
                for e in mj.get("enrichment_history", [])]
        meta = {"center": {"U": float(m["U_c"]), "J": float(m["J_c"]), "bwr": bwr_c, "E": float(m["E_c"]),
                           "bath": [[np.asarray(eb).tolist(), np.asarray(vb).tolist()] for eb, vb in bath_c]},
                "omega": [[float(z.real), float(z.imag)] for z in np.asarray(m["omega"])],
                "k_offline": int(mj["ec_basis"].get("k_raw", mj["ec_basis"]["k_seed"])), "m": mm,
                "iter_threshold": 50_000,
                "time": {"ec_s": mj["ec_basis"]["time_s"], "mor_s": mj["mor_projections"]["total_time_s"]},
                "enrichment_history": hist}
        name = os.path.basename(pdir)
        save_patch(os.path.join(new, name), meta, sps.load_npz(os.path.join(pdir, "B.npz")),
                   _dict_dir(os.path.join(pdir, "h_r"), "H_r_"), pairs, proj)
        dirs.append(name)
    g = am.get("global_ec_basis", {})
    _save_json(os.path.join(new, "atlas.json"), {
        "N": N, "NI": NI, "bwr_orb": bwr_orb, "patches": dirs, "config": {"legacy": am.get("atlas")},
        "global_ec": {"time_s": g.get("time_s", 0.0), "k": g.get("k_global_effective"),
                      "tol": g.get("ec_variance_tol"), "grid_size": g.get("ec_grid_size"),
                      "converged": g.get("converged")}})
