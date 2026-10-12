"""DMFT on the Bethe lattice with an ED or MOR+EC impurity solver, over parameter sweeps.

Each sweep point writes to <outdir>/[bwr{r}/]{J{J}|JU{J/U}}/[U{U}/][mu{mu}/]:
dmft_log.txt, dmft_metadata.json, gf_sigma_current.npz (G_imp, Sigma, G_latt, Delta, iw_imag)
and bath_history.npz (bath used at every iteration).
"""
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .bath import full_bath, reduce_full_to_free, sample_baths_local
from .bethe import bethe_gf, fit_bath_general, fit_bath_symmetric, hybridization, is_causal, quasiparticle_weight
from .ec import append_h2_terms, canonical_gs, greedy_grow, h2_terms, solve_reduced
from .ed import EDSolver
from .hamiltonian import AIM, affine_theta, mu_half_filling
from .atlas import Atlas

# Above this occupancy the bath fit uses only the warm start (extra starts only help n -> 0).
MULTISTART_N_MAX = 0.95
BATH_KEPT_TOL = 1e-6  # a fit that moves the bath by less than this has returned its input (to round-off)


@dataclass
class DMFTConfig:
    """Self-consistency settings; defaults follow the paper's campaigns."""
    N: int
    NI: int
    half_bw: float = 2.0
    bw_ratios: list = None          # per-orbital bandwidth ratios D_m / D
    beta: float = 232.0
    n_iw: int = 500
    n_iter_max: int = 150
    conv_tol: float = 1e-4          # rms |Sigma - Sigma_old|
    mix: float = 0.5
    bath_n_starts: int = 4
    warm_start: str = "any"         # 'any', 'converged' or 'none'
    resume: bool = True
    gf_tol: float = 1e-8


@dataclass
class EnrichConfig:
    """Online enrichment: trigger on the Ritz residual, then greedy local search."""
    tol: float = 1e-2               # Ritz residual that triggers an event (eps_trig)
    variance_tol: float = 1e-4      # online variance threshold eps_on
    pool_size: int = 16             # local Sobol candidates per event
    scale: float = 0.2              # local jitter of bath and U
    stagnation_iters: int = 10      # stop checking a point after this many fruitless events
    resolvent: str = "local"        # snapshots at the query point ('local') or at H0 ('reference')


class MORSolver:
    """MOR+EC impurity solver on an atlas, optionally enriching it online."""

    def __init__(self, atlas_dir, halo=1, interp="gf", diagonal_only=True, gf_tol=1e-8,
                 enrich=None, n_workers=1, bw_ratios=None, log=print):
        self.atlas = Atlas(atlas_dir)
        self.N, self.NI = self.atlas.N, self.atlas.NI
        self.halo, self.interp, self.gf_tol = halo, interp, gf_tol
        self.diagonal_only = diagonal_only or interp == "gf"
        self.enrich_cfg, self.n_workers, self.log = enrich, n_workers, log
        self.aim = AIM(self.N, self.NI) if enrich is not None else None
        self.set_bw_ratios(bw_ratios)
        self.nearest = None
        self.compression = None  # record of compress(), written to the metadata of every DMFT point
        self._h2 = {}  # patch path -> (k, h2 terms), updated incrementally across events

    def set_bw_ratios(self, bw_ratios):
        orb = self.atlas.bwr_orb
        self.bwr = float(bw_ratios[orb]) if orb is not None and bw_ratios is not None else None

    def solve(self, U, J, bath, iw):
        """(G_imp (NI, n_iw), E0, n_imp, t_gs, t_gf)."""
        t0 = time.perf_counter()
        G, E0, alpha, self.nearest, t_gs = self.atlas.gf(U, J, bath, iw, self.halo, self.interp,
                                                         self.diagonal_only, self.gf_tol, self.bwr)
        G_imp = np.array([G[m, m] for m in range(self.NI)]) #block diagonal GF
        n_imp = self.nearest.occupations(alpha)
        return G_imp, float(E0), n_imp, t_gs, time.perf_counter() - t0 - t_gs

    def compress(self, sweep_dir, tol=1e-5, n_freq=40, final_only=False):
        """Compress the ROM for Matsubara-axis solves by POD of its solutions at the baths a finished
        sweep visited (final_only: the last bath of each loop) and the n_freq lowest Matsubara
        frequencies, keeping relative singular values above `tol` (Patch.compress).  In memory;
        accurate only in the region the sweep covered.  Returns {(i, i): {'p': m, 'h': m}}."""
        if len(self.atlas.patches) != 1:
            raise ValueError("compress() supports single-patch atlases")
        thetas, beta = [], None
        for f in sorted(Path(sweep_dir).rglob("bath_history.npz")):
            meta, h = _load_json(f.parent / "dmft_metadata.json"), np.load(f)
            if meta is None or (meta["N"], meta["NI"]) != (self.N, self.NI):
                continue
            for i in range(len(h["eb"]))[-1 if final_only else 0:]:
                bath = [(h["eb"][i, o], h["vb"][i, o]) for o in range(self.NI)] + [meta["dmu_abs"]]
                thetas.append(affine_theta(meta["U"], meta["J"], bath, self.NI, self.atlas.NBpNI))
            beta = meta["beta"]
        if not thetas:
            raise ValueError(f"no DMFT points of this model under {sweep_dir}")
        patch = self.atlas.patches[0]
        patch.load()
        sizes = patch.compress(thetas, 1j * np.pi * (2 * np.arange(n_freq) + 1) / beta, tol, n_freq)
        self.compression = {"sweep_dir": str(sweep_dir), "n_baths": len(thetas), "final_only": final_only,
                            "tol": tol, "n_freq": n_freq}
        return sizes

    def try_enrich(self, U, J, bath):
        """None if the Ritz residual is below tol, else whether the nearest patch grew."""
        cfg, aim, patch = self.enrich_cfg, self.aim, self.nearest
        theta = aim.theta(U, J, bath)
        alpha, E = solve_reduced(patch.h_r, theta)
        psi = np.asarray(patch.B @ alpha).ravel()
        ritz = float(np.linalg.norm(aim.hamiltonian("gs", theta) @ psi - E * psi) / np.linalg.norm(psi))
        if ritz <= cfg.tol:
            return None
        t0, k0 = time.perf_counter(), patch.B.shape[1]
        self.log(f"  [enrich] Ritz residual {ritz:.2e} > tol={cfg.tol:.1e} at U={U:.3f}; k={k0}")
        ops = aim.ops["gs"]
        k_h2, h2_r = self._h2.get(patch.path, (None, None))
        if k_h2 != k0:
            _, h2_r = h2_terms(patch.B, ops, self.n_workers, also_h_r=False)
        grid = [(aim.theta(U_s, J, b), (U_s, b))
                for U_s, b in sample_baths_local(cfg.pool_size, bath, self.NI, cfg.scale, U0=U)]
        def solve(a):  # exact ground state, started from the ROM's approximation
            th = aim.theta(a[0], J, a[1])
            return canonical_gs(aim.hamiltonian("gs", th), guess=patch.B @ solve_reduced(patch.h_r, th)[0])[1]
        _, picks, _ = greedy_grow(patch.B, ops, grid, solve, cfg.variance_tol,
                                  h_r0=patch.h_r, h2_r0=h2_r, n_workers=self.n_workers)
        grown = patch.enrich(aim, U, J, bath, [p["psi"] for p in picks], ritz, self.n_workers,
                             resolvent=cfg.resolvent)
        if grown:
            h2_r = append_h2_terms(h2_r, ops, patch.B[:, :k0], patch.B[:, k0:])
            patch.record_event_time(time.perf_counter() - t0)
            if self.compression is not None:
                self.compression = None
                self.log("  [enrich] the ROM grew, which restores the full ROM: call compress() again")
        self._h2[patch.path] = (patch.B.shape[1], h2_r)
        self.log(f"  [enrich] {len(picks)} local candidate(s); k {k0} -> {patch.B.shape[1]} "
                 f"({time.perf_counter() - t0:.2f}s)")
        return grown


def make_solver(kind, N, NI, atlas_dir=None, **kw):
    if kind == "ED":
        return EDSolver(AIM(N, NI), gf_tol=kw.get("gf_tol", 1e-8))
    solver = MORSolver(atlas_dir, **kw)
    if (solver.N, solver.NI) != (N, NI):
        raise ValueError(f"atlas is for N={solver.N}, NI={solver.NI}")
    return solver


def initial_bath(NI, NB, D, iw):
    """Bath fitted to the non-interacting Bethe hybridization of each orbital."""
    m = (NB // NI - 1) // 2
    out = []
    for orb in range(NI):
        Dm = float(D[orb])
        t = Dm / 2.0
        eb = np.linspace(Dm / (m + 1), Dm * m / (m + 1), m) if m > 0 else np.array([])
        vb = np.full(m + 1, np.sqrt(Dm / (2.0 * (1 + m))))
        Delta0 = t**2 * bethe_gf(t, iw, np.zeros(len(iw), complex), np.linspace(-Dm, Dm, 1001), 0.0) #SE zero
        out.append(fit_bath_symmetric(Delta0, iw, m, eb, vb, t))
    return out


def _load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def is_complete(outdir):
    meta = _load_json(Path(outdir) / "dmft_metadata.json")
    return meta is not None and "end_time" in meta


def _partial_state(outdir, NI):
    try:
        hist = np.load(Path(outdir) / "bath_history.npz")
        eb, vb = hist["eb"], hist["vb"]
        sigma = np.load(Path(outdir) / "gf_sigma_current.npz")["Sigma"].copy()
    except Exception:
        return None
    if len(eb) == 0:
        return None
    return [(eb[-1, o], vb[-1, o]) for o in range(NI)], sigma, len(eb)


def _warm_start(candidates, N, NI, NBpNI, U, J, mu, require_converged):
    best = None
    for c in candidates:
        meta = _load_json(c / "dmft_metadata.json")
        if (meta is None or not (c / "bath_history.npz").exists()
                or (meta.get("N"), meta.get("NI"), meta.get("NBpNI")) != (N, NI, NBpNI)
                or (require_converged and not meta.get("converged", False))):
            continue
        dist = float(np.sqrt((meta["U"] - U) ** 2 + (meta.get("J", 0.0) - J) ** 2
                             + (meta.get("mu", 0.0) - mu) ** 2))
        if best is None or dist < best[0]:
            try:
                h = np.load(c / "bath_history.npz")
                best = (dist, c, [(h["eb"][-1, o], h["vb"][-1, o]) for o in range(NI)])
            except Exception:
                continue
    return best


def run_point(cfg, solver, U, J, bw_ratios, outdir, mu=None):
    """One DMFT self-consistency loop at (U, J, bw_ratios, mu); mu=None is half filling.

    Converged when rms |Sigma - Sigma_old| < cfg.conv_tol.  An iteration that only repeated the
    previous solve (the bath fit had returned its input, to within BATH_KEPT_TOL) counts only if
    its own fit keeps the bath as well.  Online enrichment runs when the solver is a MORSolver
    with an EnrichConfig.
    """
    enrich = getattr(solver, "enrich_cfg", None)
    N, NI = cfg.N, cfg.NI
    NB = N - NI
    NBpNI, m_free = NB // NI, (NB // NI - 1) // 2
    bw_ratios = np.asarray(bw_ratios, float)
    D = cfg.half_bw * bw_ratios
    mu_hf = mu_half_filling(U, J, NI)
    mu = mu_hf if mu is None else mu
    dmu = mu - mu_hf
    asym = np.full(NI, abs(dmu) > 1e-14)  # doping breaks each orbital's PH symmetry
    iw = 1j * np.pi * (2 * np.arange(cfg.n_iw) + 1) / cfg.beta

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "dmft_log.txt"
    open(log_path, "w").close()

    def log(msg=""):
        print(msg)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    meta = {"N": N, "NI": NI, "NB": NB, "NBpNI": NBpNI, "U": U, "J": J, "mu": float(mu),
            "dmu_abs": float(dmu), "half_bw": cfg.half_bw, "bw_ratios": bw_ratios.tolist(),
            **{k: v for k, v in asdict(cfg).items() if k not in ("N", "NI", "bw_ratios", "half_bw")},
            "solver": "MOR" if isinstance(solver, MORSolver) else "ED",
            "atlas_dir": solver.atlas.path if isinstance(solver, MORSolver) else None,
            "enrich": asdict(enrich) if enrich else None,
            "compression": getattr(solver, "compression", None),
            "start_time": datetime.now().isoformat(), "warm_start_source": None}

    def save_meta():
        with open(outdir / "dmft_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

    save_meta()
    log(f"=== DMFT run: {datetime.now().isoformat()} ===")
    log(f"N={N}  NI={NI}  NB={NB}  NBpNI={NBpNI}  m_free={m_free}")
    log(f"U={U}  J={J}  mu={mu:.4f} (shift from mu_hf={dmu:+.4f})  D={cfg.half_bw}  "
        f"bw_ratios={bw_ratios.tolist()}")
    log(f"beta={cfg.beta}  n_iw={cfg.n_iw}  mix={cfg.mix}  conv_tol={cfg.conv_tol}\n")

    # Resume this point, else warm start from the nearest finished sibling (in U), else cold.
    resume = _partial_state(outdir, NI) if cfg.resume else None
    sigma_old = None
    if resume is not None:
        bath, sigma_old, n_done = resume
        meta["resumed_from_iter"] = n_done
        save_meta()
        log(f"Resume: continuing from this run's own iteration {n_done}.")
        if sigma_old.shape != (NI, cfg.n_iw):
            sigma_old = None
    else:
        ws = None
        if cfg.warm_start != "none":
            if outdir.name.startswith("mu"):
                j_dir = outdir.parent.parent
                cands = sorted(j_dir.glob(f"U*/{outdir.name}")) if j_dir.is_dir() else []
            else:
                cands = sorted(outdir.parent.iterdir()) if outdir.parent.is_dir() else []
            ws = _warm_start(cands, N, NI, NBpNI, U, J, mu, cfg.warm_start == "converged")
        if ws is not None:
            dist, path, bath = ws
            meta["warm_start_source"] = str(path)
            save_meta()
            log(f"Warm start: '{path.name}' (|dU,dJ|={dist:.4f})")
            prev = path / "gf_sigma_current.npz"
            if prev.exists():
                s = np.load(prev)
                if "Sigma" in s and s["Sigma"].shape == (NI, cfg.n_iw):
                    sigma_old = s["Sigma"].copy()
        else:
            log("Warm start: none found; using cold start.")
            bath = initial_bath(NI, NB, D, iw)

    bath = list(bath)
    for o in range(NI):  # free form for symmetric orbitals, full form for asymmetric ones
        eb_o, vb_o = np.asarray(bath[o][0], float), bath[o][1]
        if asym[o] and len(eb_o) != NBpNI:
            e, v = full_bath([bath[o]], NBpNI)
            bath[o] = (e[0], v[0])
        elif not asym[o] and len(eb_o) != m_free:
            bath[o] = reduce_full_to_free(eb_o, vb_o, m_free)

    log("Initial bath parameters:")
    for o in range(NI):
        log(f"  orb {o}: eb=[{'  '.join(f'{e:.4f}' for e in bath[o][0]) or '(none)'}]  "
            f"vb=[{'  '.join(f'{v:.4f}' for v in bath[o][1])}]")
    log()

    eb_f, vb_f = full_bath(bath, NBpNI)
    Delta = np.array([hybridization(eb_f[o], vb_f[o], iw) for o in range(NI)])
    t_hop = D / 2.0
    x_grid = [np.linspace(-D[o], D[o], 1001) for o in range(NI)]
    hist_eb, hist_vb = [], []
    sigma_old = np.zeros((NI, cfg.n_iw), complex) if sigma_old is None else sigma_old
    mix, conv_prev, stagnant, gated = cfg.mix, np.inf, 0, False
    converged, bath_kept = False, False  # bath_kept: the last fit returned its input (to BATH_KEPT_TOL)
    t_start = time.perf_counter()

    for it in range(1, cfg.n_iter_max + 1):
        t_iter = time.perf_counter()
        G_imp, E0, n_imp, t_gs, t_gf = solver.solve(U, J, bath + [dmu], iw)
        G0_inv = iw[None, :] + mu - Delta
        Sigma = G0_inv - 1.0 / G_imp #valid for diagonal G_imp *only*

        if enrich is not None and stagnant < enrich.stagnation_iters:
            grown = solver.try_enrich(U, J, bath + [dmu])
            if grown:
                G_imp, E0, n_imp, t_gs2, t_gf2 = solver.solve(U, J, bath + [dmu], iw)
                Sigma = G0_inv - 1.0 / G_imp #valid for diagonal G_imp *only*
                t_gs, t_gf, stagnant = t_gs + t_gs2, t_gf + t_gf2, 0
            elif grown is False:
                stagnant += 1
                if stagnant >= enrich.stagnation_iters and not gated:
                    log(f"  [enrich] stagnation gate: {enrich.stagnation_iters} iterations with no "
                        f"basis growth -- no further enrichment at this point")
                    gated = True

        Sigma_latt = (1.0 - mix) * Sigma + mix * sigma_old
        G_latt = np.stack([bethe_gf(t_hop[o], iw, Sigma_latt[o], x_grid[o], mu) for o in range(NI)])
        Delta_new = np.stack([(D[o] / 2.0) ** 2 * G_latt[o] for o in range(NI)])
        sigma_conv = float(np.linalg.norm(Sigma - sigma_old) / np.sqrt(Sigma.size))
        delta_conv = float(np.linalg.norm(Delta_new - Delta) / np.sqrt(Delta_new.size))

        bath_new = []
        for o in range(NI):
            fit = fit_bath_general if asym[o] else fit_bath_symmetric
            bath_new.append(fit(Delta_new[o], iw, m_free, bath[o][0], bath[o][1], t_hop[o],
                                n_starts=cfg.bath_n_starts if n_imp[o] < MULTISTART_N_MAX else 1, U=U))

        Z = [quasiparticle_weight(Sigma[o], iw, G_imp[o], Delta[o]) for o in range(NI)]
        eb_old, vb_old = full_bath(bath, NBpNI)
        eb_new, vb_new = full_bath(bath_new, NBpNI)
        d_eb = float(np.max(np.abs(eb_new - eb_old))) if m_free > 0 else 0.0
        d_vb = float(np.max(np.abs(vb_new - vb_old)))

        np.savez_compressed(outdir / "gf_sigma_current.npz", G_imp=G_imp, Sigma=Sigma, G_latt=G_latt,
                            Delta=Delta, iw_imag=iw.imag, n_iter=it, E0=E0)
        if np.any(asym):
            hist_eb.append(eb_old), hist_vb.append(vb_old)
        else:
            hist_eb.append(np.array([bath[o][0] for o in range(NI)]))
            hist_vb.append(np.array([bath[o][1] for o in range(NI)]))
        np.savez(outdir / "bath_history.npz", eb=np.array(hist_eb), vb=np.array(hist_vb))

        log(f"\n--- Iteration {it:3d} | {datetime.now().strftime('%H:%M:%S')} ---")
        log(f"  E0={E0:.6f}   GS: {t_gs:.4f}s   GF: {t_gf:.4f}s   "
            f"total: {time.perf_counter() - t_iter:.4f}s")
        log(f"  Occupation: {'  '.join(f'n_{m}={n_imp[m]:.4f}' for m in range(NI))}   "
            f"sum={np.sum(n_imp):.4f}")
        for o in range(NI):
            okG, mG = is_causal(G_imp[o])
            okS, mS = is_causal(Sigma[o])
            log(f"  orb {o}: eb=[{'  '.join(f'{e:.4f}' for e in bath[o][0]) or '-'}]  "
                f"vb=[{'  '.join(f'{v:.4f}' for v in bath[o][1])}]")
            log(f"    Z={Z[o]:.4f}   Im[G(iw_0)]={G_imp[o, 0].imag:.5f} "
                f"({'causal' if okG else f'NOT CAUSAL (max={mG:+.3e})'})   "
                f"Im[Sigma(iw_0)]={Sigma[o, 0].imag:.5f} ({'causal' if okS else f'NOT CAUSAL (max={mS:+.3e})'})")
        log(f"  Bath change: max|deb|={d_eb:.4e}   max|dvb|={d_vb:.4e}")
        log(f"  Convergence: rms|Sigma - Sigma_old| = {sigma_conv:.4e}   "
            f"rms|Delta_new - Delta| = {delta_conv:.4e}  (tol = {cfg.conv_tol:.2e})")

        # Adaptive mixing: damp harder after a 3x jump in the residual, relax slowly otherwise.
        if it > 1:
            if sigma_conv > conv_prev * 3.0:
                mix = min(mix + 0.10, 0.95)
                log(f"  [adaptive mix] sigma_conv up ({conv_prev:.2e}->{sigma_conv:.2e}): mix={mix:.2f}")
            elif mix > cfg.mix:
                mix = max(mix - 0.02, cfg.mix)
        conv_prev = sigma_conv

        sigma_old, bath = Sigma, bath_new
        eb_f, vb_f = full_bath(bath, NBpNI)
        Delta = np.array([hybridization(eb_f[o], vb_f[o], iw) for o in range(NI)])
        # After a fit that kept the bath, this iteration repeated the last solve, so Sigma is trivially
        # unchanged.  That is convergence only if this fit keeps the bath too (a fixed point).
        moved = max(d_eb, d_vb) >= BATH_KEPT_TOL
        repeat_only = bath_kept and moved
        bath_kept = not moved
        if sigma_conv < cfg.conv_tol and not repeat_only:
            converged = True
            log(f"\n*** CONVERGED at iteration {it} (Sigma) ***")
            break
    else:
        log(f"\nDid not converge in {cfg.n_iter_max} iterations.")
    log(f"\n=== Done: {it} iterations in {time.perf_counter() - t_start:.1f}s ===")

    meta.update(n_iter_done=it, converged=converged,
                final_conv_metric=sigma_conv, end_time=datetime.now().isoformat())
    save_meta()


def sweep_values(lo, hi, step, clamp_min=None):
    """Inclusive grid lo, lo+step, ..., hi (values below clamp_min raised to it)."""
    n = int(round((hi - lo) / step + 1e-9)) + 1
    vals = [round(lo + i * step, 10) for i in range(n)]
    return [max(v, clamp_min) for v in vals] if clamp_min is not None else vals


def sweep(cfg, solver, outdir, U_vals, J_levels=None, bwr_vals=None, mu_vals=None, reverse=False):
    """Run every (bwr, mu, J level, U) point; bwr outermost, U innermost.

    J_levels: list of (name, J(U)) such as ('J0', lambda U: 0.0) or ('JU0.25', lambda U: 0.25*U).
    bwr_vals sweep the last entry of cfg.bw_ratios; mu_vals are absolute chemical potentials
    (None = half filling).  reverse walks every axis backwards (hysteresis scans).
    """
    J_levels = J_levels or [("J0", lambda U: 0.0)]
    bwr_list, mu_list = bwr_vals or [None], mu_vals or [None]
    U_list = list(U_vals)
    if reverse:
        U_list, J_levels, bwr_list, mu_list = U_list[::-1], J_levels[::-1], bwr_list[::-1], mu_list[::-1]
    base = np.ones(cfg.NI) if cfg.bw_ratios is None else np.asarray(cfg.bw_ratios, float)
    n_runs = len(bwr_list) * len(mu_list) * len(J_levels) * len(U_list)
    run, skipped, t0 = 0, 0, time.perf_counter()
    for bwr in bwr_list:
        bw = base.copy()
        if bwr is not None:
            bw[-1] = bwr
            if isinstance(solver, MORSolver):
                solver.set_bw_ratios(bw)
        d_bwr = Path(outdir) / f"bwr{bwr:.4g}" if bwr_vals else Path(outdir)
        for mu in mu_list:
            for j_name, J_of_U in J_levels:
                for U in U_list:
                    run += 1
                    d = d_bwr / j_name / f"U{U:.4g}"
                    d = d / f"mu{mu:.4g}" if mu_vals else d
                    if cfg.resume and is_complete(d):
                        skipped += 1
                        print(f"=== [{run}/{n_runs}] skip (done): {d} ===")
                        continue
                    print(f"\n=== [{run}/{n_runs}] U={U:g} J={J_of_U(U):g}"
                          + (f" bwr={bwr:g}" if bwr is not None else "")
                          + (f" mu={mu:g}" if mu is not None else "") + f"  outdir={d} ===")
                    run_point(cfg, solver, U, J_of_U(U), bw, d, mu=mu)
    print(f"\n=== Sweep done: {n_runs} runs in {time.perf_counter() - t0:.1f}s"
          + (f" ({skipped} already complete, skipped)" if skipped else "") + " ===")
