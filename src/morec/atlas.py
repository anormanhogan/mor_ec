"""Reduced-order model of the impurity Green's function: an atlas of patches.

A patch holds an EC basis B with its projected terms h_r, interpolation frequencies omega,
and a reference point (U_c, J_c, bath_c) with energy E_c whose resolvent (z + E_c - H0)^-1
builds the offline projectors.  For each orbital pair it stores projected H_hat terms and
coupling vectors; V/W projectors are kept on disk for enrichment only.

The paper uses one patch with diagonal (Galerkin) pairs.  Multi-patch atlases and
off-diagonal G_ij are retained but have not been validated against the paper results.
"""
import glob
import json
import os
import tempfile
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

import numpy as np
import scipy.linalg as la
import scipy.sparse as sps
from scipy.spatial import KDTree
from scipy.stats import qmc

from .bath import base_bath, sample_baths
from .ec import append_terms, canonical_gs, greedy_grow, mgs_residual, projected_terms, solve_reduced
from .ed import exact_gf
from .hamiltonian import AIM, affine_theta, mu_half_filling
from .irka import select_frequencies
from .lanczos import haydock_batch, shifted_solve
from .mor import (append_coupling_vecs_diag, append_orthonormal, biorthogonalize, build_pair,
                  compress_orth, coupling_vecs, gf_galerkin, gf_petrov, gram_regularize, orth_precomp,
                  solution_grams)
from .resolvent import ResolventSolver, ShiftedLanczos

UNTESTED = ("multi-patch atlases and off-diagonal G_ij are carried over from development "
            "and are not validated. Use at your own risk")


# ----------------------------------------------------------------------------- disk format

def _atomic(path, write):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", prefix=".tmp_", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as f:
            write(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _save_json(path, obj):
    _atomic(path, lambda f: f.write(json.dumps(obj, indent=2).encode()))


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _save_sparse(path, M):
    M = sps.csc_matrix(M)
    _atomic(path, lambda f: sps.save_npz(f, M, compressed=False))


def _bath_json(pairs):
    return [[np.asarray(eb, float).ravel().tolist(), np.asarray(vb, float).ravel().tolist()]
            for eb, vb in pairs]


def _pair_file(path, i, j):
    return os.path.join(path, f"pair_{i}_{j}.npz")


def _proj_file(path, name, i, j):
    return os.path.join(path, f"{name}_{i}_{j}.npz")


def _block_files(path, name, i, j):
    return sorted(glob.glob(os.path.join(path, f"{name}_{i}_{j}.add*.npz")))


def _load_projector(path, name, i, j, n_cols=None):
    """A projector with the column blocks appended by enrichment, cut to the n_cols recorded in meta."""
    parts = [sps.load_npz(_proj_file(path, name, i, j))] + [sps.load_npz(f) for f in _block_files(path, name, i, j)]
    M = sps.hstack(parts, format="csc") if len(parts) > 1 else parts[0].tocsc()
    return M if n_cols is None else M[:, :n_cols]


def _save_projector_block(path, name, i, j, start, M):
    """Columns start, start+1, ... of a projector, appended by one enrichment event."""
    _save_sparse(os.path.join(path, f"{name}_{i}_{j}.add{start:06d}.npz"), M)


def save_patch(path, meta, B, h_r, pairs, projectors=None):
    """Write a patch; projectors = {(i, j): (V_p, W_p, V_h, W_h)} (W is V for Galerkin pairs)."""
    os.makedirs(path, exist_ok=True)
    for (i, j), (V_p, W_p, V_h, W_h) in (projectors or {}).items():
        named = [("V_p", V_p), ("V_h", V_h)] + ([("W_p", W_p), ("W_h", W_h)] if W_p is not V_p else [])
        for name, M in named:
            _save_sparse(_proj_file(path, name, i, j), M)
            for f in _block_files(path, name, i, j):  # superseded by the full projector
                os.remove(f)
    _save_sparse(os.path.join(path, "B.npz"), B)
    _atomic(os.path.join(path, "h_r.npz"), lambda f: np.savez(f, **h_r))
    for (i, j), p in pairs.items():
        arrays = {f"Hp:{k}": v for k, v in p["H_p"].items()}
        arrays.update({f"Hh:{k}": v for k, v in p["H_h"].items()})
        arrays.update({f"cv:{k}": v for k, v in p["cv"].items()})
        _atomic(_pair_file(path, i, j), lambda f, a=arrays: np.savez(f, **a))
    meta["pairs"] = [f"{i},{j}" for i, j in sorted(pairs)]
    meta["k"] = int(B.shape[1])
    _save_json(os.path.join(path, "meta.json"), meta)


BATCH_MAX_M = 400  # largest compressed ROM run through haydock_batch (see Patch.compress)


class Patch:
    """One reduced-order model; heavy arrays are loaded on demand."""

    def __init__(self, path):
        self.path = path
        self.meta = _load_json(os.path.join(path, "meta.json"))
        c = self.meta["center"]
        self.U_c, self.J_c, self.bwr_c, self.E_c = c["U"], c["J"], c.get("bwr"), c["E"]
        self.J_U_c = self.J_c / self.U_c if self.U_c > 1e-14 else 0.0
        self.bath_c = [(np.array(eb), np.array(vb)) for eb, vb in c["bath"]]
        self.omega = np.array([complex(re, im) for re, im in self.meta["omega"]])
        self.loaded = False
        self._solvers = None
        self._proj = {}  # dense Galerkin projectors kept in memory between enrichment events

    def load(self):
        if self.loaded:
            return
        self.B = sps.load_npz(os.path.join(self.path, "B.npz")).tocsc()
        self.h_r = dict(np.load(os.path.join(self.path, "h_r.npz")))
        self.pairs = {}
        for key in self.meta["pairs"]:
            i, j = map(int, key.split(","))
            z = np.load(_pair_file(self.path, i, j))
            part = lambda tag: {n.split(":", 1)[1]: z[n] for n in z.files if n.startswith(tag + ":")}
            self.pairs[(i, j)] = {"H_p": part("Hp"), "H_h": part("Hh"), "cv": part("cv")}
        self._refresh()
        self.loaded = True

    def unload(self):
        for name in ("B", "h_r", "pairs", "orth", "_n_hat"):
            self.__dict__.pop(name, None)
        self._proj = {}
        self.loaded = False

    def _dense_projectors(self, i, j):
        if (i, j) not in self._proj:
            m = self.meta.get("m", {}).get(f"{i},{j}", [None, None])
            self._proj[(i, j)] = [_load_projector(self.path, "V_p", i, j, m[0]).toarray(),
                                  _load_projector(self.path, "V_h", i, j, m[1]).toarray()]
        return self._proj[(i, j)]

    def _refresh(self):
        self.orth = {ij: orth_precomp(p) for ij, p in self.pairs.items() if ij[0] == ij[1]}
        self._n_hat = None
        self._batch = None
        keys = sorted(self.h_r)  # stacked h_r for the online ground state
        stack = np.array([self.h_r[t] for t in keys])
        self._h_stack = (keys, stack.real if np.max(np.abs(stack.imag)) < 1e-12 * max(1.0, np.max(np.abs(stack.real))) else stack)

    def solution_grams(self, thetas, freqs, n_freq=40, normalize=None, freq_weight=0.0):
        """{(i, i): {sector: Gram}} of the uncompressed diagonal ROMs' solutions at `thetas` (mor.solution_grams)."""
        self._refresh()
        samples = [(th,) + self._solve_gs(th)[::-1] for th in thetas]
        return {ij: solution_grams(o, samples, freqs, n_freq, normalize, freq_weight) for ij, o in self.orth.items()}

    def compress(self, thetas=None, freqs=None, tol=1e-5, n_freq=40, normalize=None, freq_weight=0.0, grams=None):
        """Project the diagonal-pair ROMs onto the span their solutions use at `thetas` (or precomputed
        `grams` from solution_grams).  Undone by the next enrichment event (_refresh); returns m per sector."""
        grams = self.solution_grams(thetas, freqs, n_freq, normalize, freq_weight) if grams is None else grams
        self._refresh()
        self.orth = {ij: compress_orth(o, grams[ij], tol) for ij, o in self.orth.items()}
        secs = [(i, s, sd) for (i, _), o in sorted(self.orth.items()) for s, sd in sorted(o.items())]
        terms = secs[0][2]["terms"]
        M, k = max(sd["H_stack"].shape[1] for _, _, sd in secs), secs[0][2]["b_orth"].shape[1]
        if M > BATCH_MAX_M or any(sd["terms"] != terms for _, _, sd in secs):
            return {ij: {s: sd["H_stack"].shape[1] for s, sd in o.items()} for ij, o in self.orth.items()}
        H = np.zeros((len(terms), len(secs), M, M), dtype=complex)
        b = np.zeros((len(secs), M, k), dtype=complex)
        for n, (_, s, sd) in enumerate(secs):
            m = sd["H_stack"].shape[1]
            H[:, n, :m, :m] = sd["H_stack"] if s == "p" else -sd["H_stack"]
            b[n, :m] = sd["b_orth"]
        self._batch = {"secs": [(i, s) for i, s, _ in secs], "terms": terms, "H": H.reshape(len(terms), -1).T, "b": b, "M": M}
        return {ij: {s: sd["H_stack"].shape[1] for s, sd in o.items()} for ij, o in self.orth.items()}

    def _solve_gs(self, theta):
        """(alpha, E) as solve_reduced, from the stacked h_r."""
        keys, stack = self._h_stack
        c = np.array([theta.get(t, 0.0) for t in keys])
        k = stack.shape[1]
        h = (stack.reshape(len(keys), -1).T @ c).reshape(k, k)
        evals, evecs = la.eigh(h, subset_by_index=[0, 0])
        alpha = evecs[:, 0].real
        return alpha / np.linalg.norm(alpha), float(evals[0].real)

    @property
    def NI(self):
        return max(max(ij) for ij in self.pairs) + 1

    def gf(self, theta, freqs, diagonal_only=True, gf_tol=1e-8):
        """(G (NI, NI, n_freq), E, alpha, t_gs) at affine coefficients theta."""
        t0 = time.perf_counter()
        alpha, E = self._solve_gs(theta)
        t_gs = time.perf_counter() - t0
        G = np.zeros((self.NI, self.NI, len(freqs)), dtype=complex)
        # haydock_batch checks convergence only near the real axis: Matsubara-like frequencies only
        if (self._batch is not None and (diagonal_only or all(i == j for i, j in self.pairs))
                and np.ptp(np.real(freqs)) <= 1e-12 * max(1.0, float(np.max(np.abs(freqs))))):
            bt = self._batch
            c = np.array([theta.get(t, 0.0) for t in bt["terms"]])
            S = len(bt["secs"])
            Gs, _ = haydock_batch((bt["H"] @ c).reshape(S, bt["M"], bt["M"]), bt["b"] @ alpha,
                                  [freqs + E if s == "p" else freqs - E for _, s in bt["secs"]], tol=gf_tol)
            for (i, _), g in zip(bt["secs"], Gs):
                G[i, i] += g
            return G, E, alpha, t_gs
        for (i, j), pair in self.pairs.items():
            if i == j:
                G[i, i] = gf_galerkin(self.orth[(i, i)], theta, E, alpha, freqs, gf_tol)
            elif not diagonal_only:
                G[i, j] = G[j, i] = gf_petrov(pair, theta, E, alpha, freqs)
        return G, E, alpha, t_gs

    def occupations(self, alpha):
        """<n_i> (both spins) per orbital from the high-frequency tail of the hole block."""
        if self._n_hat is None:
            self._n_hat = [self.pairs[(m, m)]["cv"]["aL"].conj().T
                           @ np.linalg.solve(self.pairs[(m, m)]["cv"]["I_hat_h"],
                                             self.pairs[(m, m)]["cv"]["bL"])
                           for m in range(self.NI)]
        return np.array([2.0 * float(np.real(alpha @ n @ alpha)) for n in self._n_hat])

    def reference_solvers(self, aim, n_workers=1):
        """Resolvent solvers of the reference H0 (particle: H0_p - E_c, hole: E_c - H0_h).

        Created once per process; the 2r shifted factorizations are computed on first use and
        reused by every later enrichment event.
        """
        if self._solvers is None:
            th = aim.theta(self.U_c, self.J_c, self.bath_c)
            kw = dict(max_cached=len(self.omega), n_workers=n_workers,
                      iter_threshold=self.meta.get("iter_threshold", 50_000))
            A = {s: aim.hamiltonian(s, th) - self.E_c * sps.eye(aim.basis[s].size, format="csc")
                 for s in ("p", "h")}
            self._solvers = (ResolventSolver(A["p"], **kw), ResolventSolver(-A["h"], **kw))
        return self._solvers

    def enrich(self, aim, U, J, bath, extra_psi=(), ritz=None, n_workers=1,
               orth_tol=1e-8, ritz_frac=0.1, resolvent="local"):
        """Append the ground state at (U, J, bath) and the candidates in extra_psi to B, extend
        every projector with resolvent snapshots on the new columns, and save in place.
        """
        self.load()
        t0 = time.perf_counter()
        theta = aim.theta(U, J, bath)
        B = self.B
        E0, psi = canonical_gs(aim.hamiltonian("gs", theta), guess=B @ solve_reduced(self.h_r, theta)[0])
        new = []
        v = mgs_residual(B, np.asarray(psi, float).ravel())
        if np.linalg.norm(v) >= orth_tol:
            new.append(v / np.linalg.norm(v))
        if len(extra_psi):
            bar = max(orth_tol, ritz_frac * ritz) if ritz is not None else orth_tol
            cands = [np.asarray(p, float).ravel() for p in extra_psi]
            for v in sorted(cands, key=lambda v: -np.linalg.norm(mgs_residual(B, v, [], n_passes=1))):
                v = mgs_residual(B, v, new)
                if np.linalg.norm(v) >= bar:
                    new.append(v / np.linalg.norm(v))
        if not new:
            return False
        k_before, k_add = B.shape[1], len(new)
        B_add = sps.csc_matrix(np.column_stack(new))
        B_new = sps.hstack([B, B_add], format="csc")
        ops = aim.ops
        h_r = append_terms(self.h_r, ops["gs"], B, B_add)
        t_ec = time.perf_counter() - t0

        t1 = time.perf_counter()
        omega = np.unique(self.omega)
        if resolvent == "reference":
            solver_p, solver_h = self.reference_solvers(aim, n_workers)
        elif resolvent == "local":
            A = {s: (aim.hamiltonian(s, theta) - E0 * sps.eye(aim.basis[s].size, format="csc")).tocsc()
                 for s in ("p", "h")}
            solver_p = solver_h = None
        else:
            raise ValueError(f"unknown resolvent {resolvent!r}")
        gram_append = lambda G, V, a: G if a.shape[1] == 0 else (lambda C: np.vstack(
            [np.hstack([G, C]), np.hstack([C.conj().T, a.conj().T @ a])]))(V.conj().T @ a)
        orth_err = lambda G: float(np.max(np.abs(G - np.eye(G.shape[0]))))
        m_meta = self.meta.get("m", {})
        pairs, projectors, blocks, dense, t_hat = {}, {}, [], {}, 0.0
        for (i, j), old in self.pairs.items():
            gal = i == j
            if gal:
                Vp_old, Vh_old = self._dense_projectors(i, j)
                Wp_old, Wh_old = Vp_old, Vh_old
            else:
                m_p, m_h = m_meta.get(f"{i},{j}", [None, None])
                Vp_old, Vh_old = (_load_projector(self.path, n, i, j, c) for n, c in (("V_p", m_p), ("V_h", m_h)))
                Wp_old, Wh_old = (_load_projector(self.path, n, i, j, c) for n, c in (("W_p", m_p), ("W_h", m_h)))
            if resolvent == "local" and gal:  # one shifted Lanczos run per new column
                X_p = shifted_solve(A["p"], (aim.c_dag[j] @ B_add).toarray().real, omega, plus=False)
                X_h = shifted_solve(A["h"], (aim.c_ann[i] @ B_add).toarray().real, omega, plus=True)
                (Vp_add, mp), (Vh_add, mh) = gram_regularize(sps.csc_matrix(X_p)), gram_regularize(sps.csc_matrix(X_h))
                Wp_add, Wh_add = Vp_add, Vh_add
            else:
                if solver_p is None:
                    solver_p = ResolventSolver(A["p"], n_workers=n_workers)
                    solver_h = ResolventSolver(-A["h"], n_workers=n_workers)
                Vp_add, Wp_add, Vh_add, Wh_add, mp, mh = build_pair(
                    B_add, solver_p, solver_h, omega, aim.c_ann[i], aim.c_dag[j], aim.c_dag[i],
                    aim.c_ann[j], galerkin=gal, n_workers=n_workers, reuse=resolvent == "reference")

            Vp_app = Vh_app = G_p = G_h = None
            rep_p = rep_h = False
            Vp, Wp, Vh, Wh = Vp_old, Wp_old, Vh_old, Wh_old
            if mp > 0 and mh > 0 and gal:
                a_p = append_orthonormal(Vp_old, Vp_add).toarray()
                a_h = append_orthonormal(Vh_old, Vh_add).toarray()
                G_p = gram_append(old["cv"]["I_hat_p"], Vp_old, a_p)  # only the new blocks are computed
                G_h = gram_append(old["cv"]["I_hat_h"], Vh_old, a_h)
                Vp = np.hstack([Vp_old, a_p]) if a_p.shape[1] else Vp_old
                Vh = np.hstack([Vh_old, a_h]) if a_h.shape[1] else Vh_old
                Vp_app = a_p if a_p.shape[1] else None
                Vh_app = a_h if a_h.shape[1] else None
                if orth_err(G_p) > 1e-4:  # re-orthonormalise if the appends drifted
                    Vp, Vp_app, rep_p = gram_regularize(sps.csc_matrix(Vp))[0].toarray(), None, True
                    G_p = Vp.conj().T @ Vp
                if orth_err(G_h) > 1e-4:
                    Vh, Vh_app, rep_h = gram_regularize(sps.csc_matrix(Vh))[0].toarray(), None, True
                    G_h = Vh.conj().T @ Vh
                Wp, Wh = Vp, Vh
            elif mp > 0 and mh > 0:
                reg = [gram_regularize(sps.hstack([a, b], format="csc"))
                       for a, b in ((Vp_old, Vp_add), (Vh_old, Vh_add), (Wp_old, Wp_add), (Wh_old, Wh_add))]
                if min(n for _, n in reg) > 0:
                    Vp2, Wp2, np_ = biorthogonalize(reg[2][0], reg[0][0])
                    Vh2, Wh2, nh_ = biorthogonalize(reg[3][0], reg[1][0])
                    if np_ and nh_:
                        Vp, Wp, Vh, Wh = Vp2, Wp2, Vh2, Wh2
            if Vp.shape[1] == 0 or Vh.shape[1] == 0:
                continue

            t2 = time.perf_counter()
            if gal:
                H_p = (projected_terms(Vp, Vp, ops["p"], n_workers) if rep_p else
                       append_terms(old["H_p"], ops["p"], Vp_old, Vp_app) if Vp_app is not None else old["H_p"])
                H_h = (projected_terms(Vh, Vh, ops["h"], n_workers) if rep_h else
                       append_terms(old["H_h"], ops["h"], Vh_old, Vh_app) if Vh_app is not None else old["H_h"])
            else:
                H_p = projected_terms(Wp, Vp, ops["p"], n_workers)
                H_h = projected_terms(Wh, Vh, ops["h"], n_workers)
            t_hat += time.perf_counter() - t2
            if gal and not (rep_p or rep_h):
                cv = append_coupling_vecs_diag(old["cv"], Vp_old, Vp_app, Vh_old, Vh_app, B_new, k_add,
                                               aim.c_ann[i], aim.c_dag[j], G_p, G_h)
            else:
                cv = coupling_vecs(Vp, Wp, Vh, Wh, B_new, aim.c_ann[i], aim.c_dag[j],
                                   aim.c_dag[i], aim.c_ann[j])
            pairs[(i, j)] = {"H_p": H_p, "H_h": H_h, "cv": cv}
            if gal and not (rep_p or rep_h):  # only the appended columns go to disk
                blocks += [("V_p", i, j, Vp_old.shape[1], Vp_app), ("V_h", i, j, Vh_old.shape[1], Vh_app)]
                dense[(i, j)] = [Vp, Vh]
            elif gal:  # re-orthonormalised: rewrite the whole projectors
                Vp_s, Vh_s = sps.csc_matrix(Vp), sps.csc_matrix(Vh)
                projectors[(i, j)] = (Vp_s, Vp_s, Vh_s, Vh_s)
                dense[(i, j)] = [Vp, Vh]
            else:
                projectors[(i, j)] = (Vp, Wp, Vh, Wh)
        if not pairs:
            return False
        t_mor = time.perf_counter() - t1

        meta = self.meta
        meta["m"] = {**meta.get("m", {}),
                     **{f"{i},{j}": [int(P[0].shape[1]), int(P[2].shape[1])] for (i, j), P in projectors.items()},
                     **{f"{i},{j}": [int(P[0].shape[1]), int(P[1].shape[1])] for (i, j), P in dense.items()}}
        meta.setdefault("enrichment_history", []).append({
            "k_before": k_before, "n_added": k_add, "U": float(U), "J": float(J), "resolvent": resolvent,
            "bath": _bath_json(bath[:self.NI]),
            "ec_time_s": round(t_ec, 3), "mor_time_s": round(t_mor, 3),
            "h_hat_time_s": round(t_hat, 3), "total_time_s": round(t_ec + t_mor, 3)})
        for name, i, j, start, M in blocks:  # projector columns first: meta still describes the old patch
            if M is not None:
                _save_projector_block(self.path, name, i, j, start, sps.csc_matrix(M))
        save_patch(self.path, meta, B_new, h_r, pairs, projectors)
        self.B, self.h_r, self.pairs = B_new, h_r, pairs
        self._proj.update(dense)
        self._refresh()
        return True


# ----------------------------------------------------------------------------- atlas

def _patch_tree(patches):
    centers = np.array([[p.U_c, p.J_U_c] for p in patches])
    bwrs = [p.bwr_c for p in patches if p.bwr_c is not None]
    if not bwrs or max(bwrs) / max(min(bwrs), 1e-12) < 1.01:
        return KDTree(centers), lambda U, JU, bwr=None: [U, JU]
    lo, hi = centers.min(0), centers.max(0)
    lb, hb = np.log(min(bwrs)), np.log(max(bwrs))

    def norm(U, JU, bwr):
        return [(U - lo[0]) / max(hi[0] - lo[0], 1e-10),
                (JU - lo[1]) / max(hi[1] - lo[1], 1e-10) if hi[1] > lo[1] else 0.0,
                (np.log(max(float(bwr), 1e-8)) - lb) / max(hb - lb, 1e-10)]

    tree = KDTree(np.array([norm(p.U_c, p.J_U_c, p.bwr_c if p.bwr_c is not None else 1.0) for p in patches]))
    return tree, lambda U, JU, bwr=None: norm(U, JU, bwr if bwr is not None else max(bwrs))


class Atlas:
    """A directory of patches with nearest-patch lookup in (U, J/U[, bandwidth ratio])."""

    def __init__(self, path):
        self.path = path
        self.config = _load_json(os.path.join(path, "atlas.json"))
        self.N, self.NI = self.config["N"], self.config["NI"]
        self.NBpNI = (self.N - self.NI) // self.NI
        self.bwr_orb = self.config.get("bwr_orb")
        self.patches = [Patch(os.path.join(path, d)) for d in self.config["patches"]]
        if len(self.patches) > 1:
            warnings.warn(UNTESTED)
        self._tree, self._query = _patch_tree(self.patches)

    def nearest(self, U, J, bwr=None, n=1):
        J_U = J / U if U > 1e-14 else 0.0
        dists, idxs = self._tree.query(self._query(U, J_U, bwr), k=min(n, len(self.patches)))
        return np.atleast_1d(idxs), np.atleast_1d(dists)

    def gf(self, U, J, bath, freqs, halo=1, interp="gf", diagonal_only=True, gf_tol=1e-8, bwr=None):
        """(G (NI, NI, n_freq), E0, alpha, nearest patch, t_gs) averaged over the `halo` nearest
        patches with inverse-distance weights, directly ('gf') or through Sigma ('sigma')."""
        idxs, dists = self.nearest(U, J, bwr, halo)
        for i, p in enumerate(self.patches):
            if i not in idxs:
                p.unload()
        theta = affine_theta(U, J, bath, self.NI, self.NBpNI)
        results = []
        for i in idxs:
            self.patches[i].load()
            results.append(self.patches[i].gf(theta, freqs, diagonal_only, gf_tol))
        G, E0, alpha, _ = results[0]
        t_gs = sum(r[3] for r in results)
        if len(results) > 1:
            w = 1.0 / (dists**2 + 1e-14)
            w /= w.sum()
            if interp == "sigma":
                G0_inv = np.zeros_like(G)
                for m in range(self.NI):
                    G0_inv[m, m] = freqs + mu_half_filling(U, J, self.NI) + theta["dmu"]
                inv = lambda X: np.moveaxis(np.linalg.inv(np.moveaxis(X, -1, 0)), 0, -1)
                G = inv(G0_inv - sum(wi * (G0_inv - inv(r[0])) for wi, r in zip(w, results)))
            else:
                G = sum(wi * r[0] for wi, r in zip(w, results))
        return G, E0, alpha, self.patches[idxs[0]], t_gs


# ----------------------------------------------------------------------------- offline build

@dataclass
class BuildConfig:
    """Offline ROM construction; defaults follow the paper's N=8, NI=2 campaign."""
    N: int
    NI: int
    r: int = 3                            # interpolation frequencies
    beta: float = 232.0                   # Matsubara grid for the interpolation points
    U_lo: float = 0.01
    U_hi: float = 6.0
    J_U_lo: float = 0.0
    J_U_hi: float = 0.0
    bw_lo: list = None                    # per-orbital bandwidth-ratio sampling range
    bw_hi: list = None
    bath_scale: float = 0.4               # log-uniform bath jitter of the training grid
    symmetric_baths: bool = False         # share one bath jitter across orbitals
    ec_tol: float = 1e-2                  # offline variance threshold eps_off
    ec_grid_size: int = 1024              # Sobol candidates for the greedy search
    k_max: int = 500                      # cap on greedy additions
    frequencies: str = "matsubara"        # 'matsubara' (log-spaced iw_n), 'irka', or [[re, im], ...]
    n_matsubara_max: int = 40             # largest Matsubara index for 'matsubara'
    n_pin_iw: int = 0                     # also pin iw_0..iw_{n-1} when using 'irka'
    irka_eta: float = 0.08                # IRKA minimizes the error of G(w + i*irka_eta)
    diagonal_only: bool = True            # build only diagonal (Galerkin) pairs
    n_patches_U: int = 1                  # > 1 in any axis builds a multi-patch atlas (untested)
    n_patches_J: int = 1
    n_patches_bwr: int = 1
    U_grid_exp: float = 1.0
    k_local: int = 0                      # multi-patch: greedy local additions per patch
    k_anchor: int = 0                     # multi-patch: global vectors shared by all patches
    illustrate: bool = False              # narrow bath jitter (SIAM demo)
    iter_threshold: int = 50_000          # sector dimension above which solves are iterative
    projector: str = "lu"                 # offline resolvent solves: 'lu' (LU/ILU-GMRES per frequency)
                                          # or 'lanczos' (one shifted Lanczos run per column, all frequencies)
    n_workers: int = 4
    n_test: int = 0                       # post-build comparison against ED at random points
    n_test_w: int = 100
    test_eta: float = 0.01
    extra: dict = field(default_factory=dict)


def log_spaced_matsubara(r, n_max):
    """r Matsubara indices: 0, 1, then log-spaced up to n_max."""
    if r <= 1:
        return np.array([0])
    idx = {0, 1}
    if r > 2 and n_max > 2:
        idx |= set(np.round(np.geomspace(2, n_max, r - 2)).astype(int).tolist())
    idx = sorted(idx)
    cand = 2
    while len(idx) < min(r, n_max + 1) and cand <= n_max:
        if cand not in idx:
            idx = sorted(idx + [cand])
        cand += 1
    return np.array(idx[:r])


def orthonormalize(V, tol=1e-10):
    """Sparse orthonormal basis of span(V) by QR on the nonzero rows."""
    d, n = V.shape
    if n == 0:
        return sps.csc_matrix((d, 0), dtype=V.dtype)
    rows = np.unique(V.nonzero()[0])
    Q, R = np.linalg.qr(V.toarray() if rows.size == d else V[rows, :].toarray(), mode="reduced")
    r = np.abs(np.diag(R))
    Q = np.ascontiguousarray(Q[:, r > tol * r.max()] if r.max() > 0 else Q[:, :0])
    k = Q.shape[1]
    data = Q.ravel()
    mask = np.abs(data) > 1e-15
    return sps.csc_matrix((data[mask], (np.repeat(rows.astype(np.int32), k)[mask],
                                        np.tile(np.arange(k, dtype=np.int32), rows.size)[mask])),
                          shape=(d, k), dtype=V.dtype)


def select_anchors(B, k):
    """k columns of B chosen by pivoted QR of its Gram matrix, orthonormalised."""
    _, _, piv = la.qr((B.conj().T @ B).toarray(), pivoting=True)
    return orthonormalize(B[:, piv[:k]]), piv[:k]


def _log_uniform_U(raw, lo, hi):
    lo = max(lo, 1e-6)
    return np.exp(np.log(lo) + raw * np.log(hi / lo))


def _greedy(aim, B0, U_g, J_g, baths, tol, max_additions, n_workers, tag):
    grid = [(aim.theta(u, j, b), (u, j, b)) for u, j, b in zip(U_g, J_g, baths)]
    return greedy_grow(B0, aim.ops["gs"], grid,
                       lambda a: canonical_gs(aim.hamiltonian("gs", aim.theta(*a)))[1],
                       tol, max_additions, n_workers=n_workers, tag=tag)


def _sample_json(pick, NI):
    U, J, bath = pick["solve_args"]
    return {"U": U, "J": J, "bath": _bath_json(bath[:NI]), "variance": pick["variance"]}


def build(cfg, outdir, aim=None, verbose=True):
    """Build an atlas in outdir (offline stage). Returns the Atlas."""
    t_start = time.perf_counter()
    N, NI = cfg.N, cfg.NI
    aim = aim or AIM(N, NI)
    NB = N - NI
    base = base_bath(NB, NI)
    log = print if verbose else (lambda *a, **k: None)
    multi = cfg.n_patches_U * cfg.n_patches_J * cfg.n_patches_bwr > 1 or cfg.k_local > 0
    if multi or not cfg.diagonal_only:
        warnings.warn(UNTESTED)
    os.makedirs(outdir, exist_ok=True)
    log(f"MOR+EC build: N={N} NI={NI} d_gs={aim.basis['gs'].size} d_p={aim.basis['p'].size} -> {outdir}")

    # Stage 1a: global EC basis by greedy variance selection over a Sobol grid.
    t0 = time.perf_counter()
    G = cfg.ec_grid_size
    raw = qmc.Sobol(d=2, scramble=True, seed=0).random(n=1 << (G - 1).bit_length())[:G]
    U_g = _log_uniform_U(raw[:, 0], cfg.U_lo, cfg.U_hi)
    JU_g = (cfg.J_U_lo + raw[:, 1] * (cfg.J_U_hi - cfg.J_U_lo) if cfg.J_U_hi > cfg.J_U_lo
            else np.full(G, cfg.J_U_lo))
    baths = sample_baths(G, NB, NI, base, cfg.bath_scale, seed=1, symmetric=cfg.symmetric_baths,
                         bw_lo=cfg.bw_lo, bw_hi=cfg.bw_hi, illustrate=cfg.illustrate)
    B_glob, picks, converged = _greedy(aim, sps.csc_matrix((aim.basis["gs"].size, 0)), U_g, JU_g * U_g,
                                       baths, cfg.ec_tol, max(cfg.k_max, 1), cfg.n_workers, "EC" if verbose else "")
    t_ec = time.perf_counter() - t0
    log(f"  EC basis: k={B_glob.shape[1]} (converged={converged}, {t_ec:.1f}s)")
    B_anchor = B_glob if cfg.k_local == 0 else select_anchors(B_glob, cfg.k_anchor)[0]

    # Patch grid (a single patch at the domain centre unless requested otherwise).
    lo = np.ones(NI) if cfg.bw_lo is None else np.asarray(cfg.bw_lo, float)
    hi = np.ones(NI) if cfg.bw_hi is None else np.asarray(cfg.bw_hi, float)
    varying = [m for m in range(NI) if lo[m] < hi[m]]
    bwr_orb = varying[0] if varying else None
    n_bwr = cfg.n_patches_bwr if bwr_orb is not None else 1
    bwr_c = list(np.exp(np.linspace(np.log(lo[bwr_orb]), np.log(hi[bwr_orb]), n_bwr))) if n_bwr > 1 else [1.0]
    if cfg.n_patches_U == 1:
        U_c, U_hw = [0.5 * (cfg.U_lo + cfg.U_hi)], [0.5 * (cfg.U_hi - cfg.U_lo)]
    else:
        t = np.linspace(0.0, 1.0, cfg.n_patches_U)
        U_c = list(cfg.U_lo + (cfg.U_hi - cfg.U_lo) * (1.0 - (1.0 - t) ** cfg.U_grid_exp))
        U_hw = [(U_c[1] - U_c[0]) / 2] + [(U_c[i + 1] - U_c[i - 1]) / 4 for i in range(1, len(U_c) - 1)] \
            + [(U_c[-1] - U_c[-2]) / 2]
    if cfg.n_patches_J == 1:
        JU_c, JU_hw = [0.5 * (cfg.J_U_lo + cfg.J_U_hi)], 0.5 * (cfg.J_U_hi - cfg.J_U_lo)
    else:
        JU_c = list(np.linspace(cfg.J_U_lo, cfg.J_U_hi, cfg.n_patches_J))
        JU_hw = 0.5 * (cfg.J_U_hi - cfg.J_U_lo) / (cfg.n_patches_J - 1)

    dirs = []
    for pu, (Uc, Uhw) in enumerate(zip(U_c, U_hw)):
        for pj, JUc in enumerate(JU_c):
            for pb, bc in enumerate(bwr_c):
                name = f"patch_U{Uc:.4f}_JoU{JUc:.5f}" + (f"_bwr{bc:.4f}" if bwr_orb is not None else "")
                seed = pu * 97 + pj * 41 + pb * 13 + cfg.k_local
                _build_patch(cfg, aim, os.path.join(outdir, name), B_anchor, Uc, Uhw, JUc, JU_hw,
                             bc, bwr_orb, base, seed, log, verbose)
                dirs.append(name)

    _save_json(os.path.join(outdir, "atlas.json"), {
        "N": N, "NI": NI, "bwr_orb": bwr_orb, "patches": dirs, "config": asdict(cfg),
        "global_ec": {"time_s": round(t_ec, 2), "k": int(B_glob.shape[1]), "tol": cfg.ec_tol,
                      "grid_size": G, "converged": bool(converged),
                      "samples": [_sample_json(p, NI) for p in picks]},
        "build_time_s": round(time.perf_counter() - t_start, 2)})
    atlas = Atlas(outdir)
    if cfg.n_test > 0:
        validate(atlas, aim, cfg, verbose=verbose)
    return atlas


def _build_patch(cfg, aim, path, B_anchor, U_c, U_hw, JU_c, JU_hw, bwr_c, bwr_orb, base, seed, log, verbose=True):
    """Stages 1a (local EC growth, multi-patch only) and 1b (projections) for one patch."""
    NI, NB = cfg.NI, cfg.N - cfg.NI
    J_c = U_c * JU_c
    t0 = time.perf_counter()
    B_loc = sps.csc_matrix((aim.basis["gs"].size, 0))
    if cfg.k_local > 0:
        G = cfg.ec_grid_size
        raw = qmc.Sobol(d=2, scramble=True, seed=seed).random(n=1 << (G - 1).bit_length())[:G]
        U_g = _log_uniform_U(raw[:, 0], U_c - U_hw, U_c + U_hw)
        J_g = np.maximum(J_c + (raw[:, 1] - 0.5) * 2.0 * JU_hw, 1e-3)
        baths = sample_baths(G, NB, NI, base, cfg.bath_scale, seed=seed + 1, symmetric=cfg.symmetric_baths,
                             bw_lo=cfg.bw_lo, bw_hi=cfg.bw_hi, illustrate=cfg.illustrate)
        B_loc = _greedy(aim, B_loc, U_g, J_g, baths, cfg.ec_tol, cfg.k_local, cfg.n_workers,
                        os.path.basename(path) if verbose else "")[0]
    B = orthonormalize(sps.hstack([B_loc, B_anchor], format="csc"))
    t_ec = time.perf_counter() - t0

    t0 = time.perf_counter()
    bath_c = [(eb * bwr_c, vb * np.sqrt(bwr_c)) if m == bwr_orb else (eb, vb)
              for m, (eb, vb) in enumerate(base)]
    theta_c = aim.theta(U_c, J_c, bath_c)
    H_gs, H_p, H_h = (aim.hamiltonian(s, theta_c) for s in ("gs", "p", "h"))
    h_r = projected_terms(B, B, aim.ops["gs"], cfg.n_workers)
    E_c = solve_reduced(h_r, theta_c)[1]
    if isinstance(cfg.frequencies, list):
        omega = np.array([complex(re, im) for re, im in cfg.frequencies])
    elif cfg.frequencies == "irka":
        omega = select_frequencies(H_p, H_h, H_gs, E_c, B, cfg.r, aim.c_dag, aim.c_ann, eta=cfg.irka_eta,
                                   verbose=verbose)
        for n in range(cfg.n_pin_iw):
            iw = 1j * (2 * n + 1) * np.pi / cfg.beta
            if not np.any(np.abs(omega - iw) < 1e-10):
                omega = np.append(omega, iw)
    else:
        omega = 1j * (2 * log_spaced_matsubara(cfg.r, cfg.n_matsubara_max) + 1) * np.pi / cfg.beta

    # Stage 1b: projections from the reference resolvents (z + E_c - H0)^-1.
    A_p = H_p - E_c * sps.eye(H_p.shape[0], format="csc")
    A_h = -(H_h - E_c * sps.eye(H_h.shape[0], format="csc"))
    if cfg.projector == "lanczos":
        solver_p, solver_h = ShiftedLanczos(A_p), ShiftedLanczos(A_h)
    else:
        solver_p = ResolventSolver(A_p, max_cached=len(omega), iter_threshold=cfg.iter_threshold, n_workers=cfg.n_workers)
        solver_h = ResolventSolver(A_h, max_cached=len(omega), iter_threshold=cfg.iter_threshold, n_workers=cfg.n_workers)
    active = [(i, j) for i in range(NI) for j in range(i, NI) if i == j or not cfg.diagonal_only]
    n_conc = len(active) if max(1, cfg.n_workers // (2 * len(active))) >= 2 else 1
    per_pair = max(1, cfg.n_workers // n_conc)
    with ThreadPoolExecutor(max_workers=n_conc) as ex:
        futs = {ij: ex.submit(build_pair, B, solver_p, solver_h, omega, aim.c_ann[ij[0]], aim.c_dag[ij[1]],
                              aim.c_dag[ij[0]], aim.c_ann[ij[1]], ij[0] == ij[1], per_pair)
                for ij in active}
    pairs, projectors, m = {}, {}, {}
    for (i, j), fut in futs.items():
        V_p, W_p, V_h, W_h, mp, mh = fut.result()
        if mp == 0 or mh == 0:
            continue
        pairs[(i, j)] = {"H_p": projected_terms(W_p, V_p, aim.ops["p"], cfg.n_workers),
                         "H_h": projected_terms(W_h, V_h, aim.ops["h"], cfg.n_workers),
                         "cv": coupling_vecs(V_p, W_p, V_h, W_h, B, aim.c_ann[i], aim.c_dag[j],
                                             aim.c_dag[i], aim.c_ann[j])}
        projectors[(i, j)] = (V_p, W_p, V_h, W_h)
        m[f"{i},{j}"] = [int(mp), int(mh)]
    t_mor = time.perf_counter() - t0
    meta = {"center": {"U": float(U_c), "J": float(J_c), "bwr": float(bwr_c) if bwr_orb is not None else None,
                       "E": float(E_c), "bath": _bath_json(bath_c)},
            "omega": [[float(w.real), float(w.imag)] for w in omega],
            "k_offline": int(B.shape[1]), "m": m, "iter_threshold": cfg.iter_threshold,
            "time": {"ec_s": round(t_ec, 3), "mor_s": round(t_mor, 3)}}
    save_patch(path, meta, B, h_r, pairs, projectors)
    log(f"  {os.path.basename(path)}: k={B.shape[1]} r={len(omega)} "
        f"m_p/m_h={[tuple(v) for v in m.values()]} ({t_ec + t_mor:.1f}s)")


def validate(atlas, aim, cfg, seed=42, verbose=True):
    """Compare the ROM against ED at n_test random points; writes <atlas>/test_<i>/results.npz."""
    NI, NB = cfg.NI, cfg.N - cfg.NI
    raw = qmc.Sobol(d=2, scramble=True, seed=seed).random(n=1 << (cfg.n_test - 1).bit_length())[:cfg.n_test]
    U_t = cfg.U_lo + raw[:, 0] * (cfg.U_hi - cfg.U_lo)
    JU_t = cfg.J_U_lo + raw[:, 1] * (cfg.J_U_hi - cfg.J_U_lo)
    baths = sample_baths(cfg.n_test, NB, NI, base_bath(NB, NI), cfg.bath_scale, seed=seed,
                         symmetric=cfg.symmetric_baths, bw_lo=cfg.bw_lo, bw_hi=cfg.bw_hi)
    freqs = 1j * (2 * np.arange(cfg.n_test_w) + 1) * np.pi / cfg.beta
    freqs_real = np.linspace(-8.0, 8.0, cfg.n_test_w) + 1j * cfg.test_eta
    for t, (U, JU, bath) in enumerate(zip(U_t, JU_t, baths)):
        J = U * JU
        t0 = time.perf_counter()
        H_gs, H_p, H_h = aim.hamiltonians(U, J, bath)
        E0, psi = canonical_gs(H_gs)
        exact = lambda z: np.array([[exact_gf(aim, E0, psi, H_p, H_h, z, i, j) for j in range(NI)]
                                    for i in range(NI)])
        G_fom = exact(freqs)
        t_fom = time.perf_counter() - t0
        t0 = time.perf_counter()
        G_rom = atlas.gf(U, J, bath, freqs, diagonal_only=cfg.diagonal_only)[0]
        t_rom = time.perf_counter() - t0
        out = {"U_test": U, "J_test": J, "J_U_test": JU, "freqs": freqs, "freqs_real": freqs_real,
               "G_fom": G_fom, "G_rom1": G_rom, "G_fom_real": exact(freqs_real),
               "G_rom1_real": atlas.gf(U, J, bath, freqs_real, diagonal_only=cfg.diagonal_only)[0],
               "t_fom": t_fom, "t_rom1": t_rom}
        d = os.path.join(atlas.path, f"test_{t}")
        os.makedirs(d, exist_ok=True)
        np.savez_compressed(os.path.join(d, "results.npz"), **out)
        err = max(np.max(np.abs(G_rom[i, i] - G_fom[i, i])) for i in range(NI))
        if verbose:
            print(f"  test {t}: U={U:.3f} J/U={JU:.3f}  max|G_ROM - G_ED| (diag) = {err:.2e}  "
                  f"t_ED={t_fom:.3f}s t_ROM={t_rom:.4f}s")
