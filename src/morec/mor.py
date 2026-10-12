"""Model order reduction of the particle/hole resolvents.

For an orbital pair (i, j) and interpolation frequencies z, the particle projector V_p spans
(z + E_ref - H_p)^-1 c_j† B and the hole projector V_h spans (z - E_ref + H_h)^-1 c_i B.
Diagonal pairs are Galerkin (W = V); off-diagonal pairs use a biorthogonalised
Petrov-Galerkin pair (V, W).  V and W are stored as scipy.sparse csc matrices.
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.linalg as la
import scipy.sparse as sps

from .lanczos import haydock_gf


def _dense(x):
    return x.toarray() if sps.issparse(x) else np.asarray(x)


def _available_ram(default=8 * 1024**3):
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except ImportError:
        return default


def gram_regularize(V, tol=1e-6):
    """Orthonormal basis (sparse) of span(V), dropping singular values below tol * s_max."""
    d, n = V.shape
    if n == 0:
        return sps.csc_matrix((d, 0), dtype=V.dtype), 0
    dense = V.nnz / (d * n) > 0.3
    Vd = V.toarray() if dense else V
    G = Vd.conj().T @ Vd if dense else (V.conj().T @ V).toarray()
    evals, U = la.eigh(G)
    sig = np.sqrt(np.maximum(evals[::-1], 0))
    U = U[:, ::-1]
    keep = sig > tol * sig[0] if sig[0] > 0 else np.zeros(n, bool)
    if not keep.any():
        return sps.csc_matrix((d, 0), dtype=V.dtype), 0
    Vr = sps.csc_matrix(Vd @ (U[:, keep] / sig[keep]))
    Vr.data[np.abs(Vr.data) < 1e-14] = 0
    Vr.eliminate_zeros()
    return Vr, int(keep.sum())


def build_projector(solver, omega, op, B, tol=1e-6, n_workers=1, reuse=False):
    """Orthonormal basis of span{(z - H)^-1 op B : z in omega} via the Gram matrix of the blocks.

    reuse=True keeps every solve on the solver's cached factorizations (no worker processes).
    """
    r, k = len(omega), B.shape[1]
    rhs = _dense(op @ B)
    if hasattr(solver, "solve_all"):  # resolvent.ShiftedLanczos
        blocks = solver.solve_all(omega, rhs)
    elif n_workers > 1 and r > 1 and solver.iterative and not reuse:
        blocks = solver.solve_parallel(omega, rhs, tol=tol, n_workers=n_workers)
    else:
        blocks = [solver.solve(z, rhs, tol=tol) for z in omega]
    G = np.zeros((r * k, r * k), dtype=complex)
    for a in range(r):
        G[a * k:(a + 1) * k, a * k:(a + 1) * k] = blocks[a].conj().T @ blocks[a]
        for b in range(a):
            cross = blocks[a].conj().T @ blocks[b]
            G[a * k:(a + 1) * k, b * k:(b + 1) * k] = cross
            G[b * k:(b + 1) * k, a * k:(a + 1) * k] = cross.conj().T
    evals, U = la.eigh(G)
    sig = np.sqrt(np.maximum(evals[::-1], 0))
    U = U[:, ::-1]
    keep = sig > tol * sig[0] if sig[0] > 0 else np.zeros(r * k, bool)
    m = int(keep.sum())
    if m == 0:
        return sps.csc_matrix((op.shape[0], 0), dtype=complex), 0
    T = U[:, keep] / sig[keep]
    V = np.zeros((op.shape[0], m), dtype=complex)
    for s, blk in enumerate(blocks):
        V += blk @ T[s * k:(s + 1) * k]
    V = sps.csc_matrix(V)
    V.eliminate_zeros()
    return V, m


def biorthogonalize(W, V):
    """(V', W', m) spanning the same spaces with W'^H V' = I; m = 0 if the spaces are orthogonal."""
    Uc, s, Vt = np.linalg.svd(_dense(W.conj().T @ V), full_matrices=False)
    if s[0] < 1e-10:
        return None, None, 0
    keep = s > 1e-4 * s[0]
    root = np.sqrt(s[keep])
    return (sps.csc_matrix(V @ Vt[keep].conj().T / root),
            sps.csc_matrix(W @ Uc[:, keep] / root), int(keep.sum()))


def build_pair(B, solver_p, solver_h, omega, c_ann_i, c_dag_j, c_dag_i, c_ann_j,
               galerkin, n_workers=1, reuse=False):
    """(V_p, W_p, V_h, W_h, m_p, m_h) for orbital pair (i, j); solvers hold H_p - E and E - H_h."""
    d_p = c_dag_j.shape[0]
    concurrent = 2 if 2.5 * len(omega) * d_p * B.shape[1] * 16 < _available_ram() else 1
    per = max(1, n_workers // concurrent)

    def both(op_p, op_h):
        with ThreadPoolExecutor(max_workers=concurrent) as ex:
            fp = ex.submit(build_projector, solver_p, omega, op_p, B, 1e-6, per, reuse)
            fh = ex.submit(build_projector, solver_h, omega, op_h, B, 1e-6, per, reuse)
            return fp.result(), fh.result()

    (V_p, m_p), (V_h, m_h) = both(c_dag_j, c_ann_i)
    if m_p == 0 or m_h == 0:
        return None, None, None, None, 0, 0
    if galerkin:
        return V_p, V_p, V_h, V_h, m_p, m_h
    (W_p, n_p), (W_h, n_h) = both(c_dag_i, c_ann_j)
    if n_p == 0 or n_h == 0:
        return None, None, None, None, 0, 0
    V_p, W_p, m_p = biorthogonalize(W_p, V_p)
    V_h, W_h, m_h = biorthogonalize(W_h, V_h)
    return V_p, W_p, V_h, W_h, m_p, m_h


def append_orthonormal(V_old, V_new, tol=1e-6):
    """Orthonormal columns spanning V_new with span(V_old) projected out (Galerkin append)."""
    dim, n_new = V_new.shape
    if dim <= 10_000:
        Vo, Vn = _dense(V_old), _dense(V_new)
        Q, R = la.qr(Vn - Vo @ (Vo.conj().T @ Vn), mode="economic")
        r = np.abs(np.diag(R))
        keep = r > max((r.max() if r.size else 0.0) * tol, np.sqrt(dim) * np.finfo(Vn.dtype).eps * 100)
        return sps.csc_matrix(Q[:, keep]) if keep.any() else sps.csc_matrix((dim, 0), dtype=complex)
    if sps.issparse(V_old):
        Vo, Vn = V_old.tocsc(), V_new.tocsc()
    else:  # a dense V_old (kept in memory during enrichment): same algorithm with BLAS products
        Vo, Vn = V_old, _dense(V_new)
    C = _dense(Vo.conj().T @ Vn)
    evals, U = la.eigh(_dense(Vn.conj().T @ Vn) - C.conj().T @ C)
    idx = np.argsort(evals)[::-1]
    s = np.sqrt(np.maximum(evals[idx], 0.0))
    U = U[:, idx]
    keep = s > max((s[0] if s.size else 0.0) * tol, np.sqrt(dim) * np.finfo(complex).eps * 100)
    if not keep.any():
        return sps.csc_matrix((dim, 0), dtype=complex)
    T = U[:, keep] / s[keep]
    return sps.csc_matrix(Vn @ T - Vo @ (C @ T))


def coupling_vecs(V_p, W_p, V_h, W_h, B, c_ann_i, c_dag_j, c_dag_i, c_ann_j):
    """Reduced couplings of pair (i, j).

    G_p(z) = (aR alpha)^H [(z + E) I_p - H_p]^-1 (bR alpha),  G_h(z) likewise with aL, bL.
    """
    CBj_p = _dense(c_dag_j @ B)
    CBi_p = CBj_p if c_dag_i is c_dag_j else _dense(c_dag_i @ B)
    CBi_h = _dense(c_ann_i @ B)
    CBj_h = CBi_h if c_ann_j is c_ann_i else _dense(c_ann_j @ B)
    Wp = _dense(W_p)
    Vp = Wp if V_p is W_p else _dense(V_p)
    Wh = _dense(W_h)
    Vh = Wh if V_h is W_h else _dense(V_h)
    return {"aR": Vp.conj().T @ CBi_p, "bR": Wp.conj().T @ CBj_p,
            "aL": Vh.conj().T @ CBj_h, "bL": Wh.conj().T @ CBi_h,
            "I_hat_p": Wp.conj().T @ Vp, "I_hat_h": Wh.conj().T @ Vh}


def append_coupling_vecs_diag(cv, Vp_old, Vp_add, Vh_old, Vh_add, B_new, k_add, c_ann, c_dag,
                              I_hat_p, I_hat_h):
    """Block update of a diagonal pair's couplings after appending B and V columns."""
    k_old = B_new.shape[1] - k_add

    def sector(a, b, V_old, V_add, op):
        if V_add is None:
            strip = np.asarray(V_old.conj().T @ (op @ B_new[:, k_old:]).toarray())
            return np.hstack([a, strip]), np.hstack([b, strip])
        CB = (op @ B_new).toarray()
        strip = np.asarray(V_old.conj().T @ CB[:, k_old:])
        bot = _dense(V_add).conj().T @ CB
        return np.vstack([np.hstack([a, strip]), bot]), np.vstack([np.hstack([b, strip]), bot])

    aR, bR = sector(cv["aR"], cv["bR"], Vp_old, Vp_add, c_dag)
    aL, bL = sector(cv["aL"], cv["bL"], Vh_old, Vh_add, c_ann)
    return {"aR": aR, "bR": bR, "aL": aL, "bL": bL,
            "I_hat_p": cv["I_hat_p"] if I_hat_p is None else I_hat_p,
            "I_hat_h": cv["I_hat_h"] if I_hat_h is None else I_hat_h}


def orth_precomp(pair):
    """Per sector of a diagonal pair: H_stack = T^H H_hat T over sorted keys and b_orth = T^H b,
    with T = U diag(lam)^-1/2 whitening I_hat (dependent directions dropped)."""
    out = {}
    for s, (I_key, b_key) in (("p", ("I_hat_p", "bR")), ("h", ("I_hat_h", "bL"))):
        H_hat = pair["H_" + s]
        if not H_hat:
            continue
        lam, U = la.eigh(pair["cv"][I_key])
        lam, U = lam[::-1], U[:, ::-1]
        keep = lam > 1e-12 * max(float(lam[0]) if lam.size else 0.0, 1e-300)
        T = U[:, keep] / np.sqrt(lam[keep])[None, :]
        Th = T.conj().T
        terms = sorted(H_hat)
        out[s] = {"terms": terms, "b_orth": Th @ pair["cv"][b_key],
                  "H_stack": np.array([Th @ H_hat[t] @ T for t in terms])}
    return out


def solution_grams(orth, samples, freqs, n_freq=40, normalize=None, freq_weight=0.0):
    """{sector: X X^H} of the ROM solutions x = (z - H(theta))^-1 b(theta) at the n_freq lowest `freqs`,
    summed over `samples` [(theta, E, alpha)].  normalize ('column', 'call') and freq_weight rescale
    the snapshots; the defaults (raw norms, no weighting) worked best."""
    grams = {}
    for s, sd in orth.items():
        n_t, m, _ = sd["H_stack"].shape
        sign, gram = (1.0 if s == "p" else -1.0), np.zeros((m, m), dtype=complex)
        for theta, E, alpha in samples:
            c = np.array([theta.get(t, 0.0) for t in sd["terms"]])
            lam, U = la.eigh(sign * (sd["H_stack"].reshape(n_t, -1).T @ c).reshape(m, m))
            z = freqs[:n_freq] + (E if s == "p" else -E)
            X = U @ ((U.conj().T @ (sd["b_orth"] @ alpha))[:, None] / (z[None, :] - lam[:, None]))
            if freq_weight:
                X = X * (np.abs(freqs[0].imag) / np.abs(freqs[:n_freq].imag))[None, :]**freq_weight
            if normalize == "column":
                X = X / np.linalg.norm(X, axis=0)[None, :]
            elif normalize == "call":
                X = X / np.linalg.norm(X)
            gram += X @ X.conj().T
        grams[s] = gram
    return grams


def compress_orth(orth, grams, tol=1e-5):
    """POD of the ROM: project each sector onto the eigenvectors of its Gram (solution_grams) with
    relative singular values above tol.  Returns the layout of orth_precomp."""
    out = {}
    for s, sd in orth.items():
        ev, W = la.eigh(grams[s])
        P = W[:, np.sqrt(np.maximum(ev, 0.0) / ev[-1]) > tol]
        Ph = P.conj().T
        out[s] = {"terms": sd["terms"], "b_orth": Ph @ sd["b_orth"], "H_stack": Ph[None] @ sd["H_stack"] @ P[None]}
    return out


def gf_galerkin(orth, theta, E, alpha, freqs, gf_tol=1e-8):
    """Diagonal-pair ROM Green's function by adaptive Haydock recursion on the whitened H_hat."""
    G = 0.0
    for s, z, sign in (("p", freqs + E, 1.0), ("h", freqs - E, -1.0)):
        sd = orth[s]
        c = sign * np.array([theta.get(t, 0.0) for t in sd["terms"]])
        n_t, m, _ = sd["H_stack"].shape
        H = (sd["H_stack"].reshape(n_t, -1).T @ c).reshape(m, m)
        G = G + haydock_gf(H, sd["b_orth"] @ alpha, z, tol=gf_tol)[0]
    return G


def gf_petrov(pair, theta, E, alpha, freqs):
    """Pair ROM Green's function from the pole decomposition of (z I_hat -/+ H_hat)."""
    def assemble(H_hat):
        m = next(iter(H_hat.values())).shape[0]
        H = np.zeros((m, m), dtype=complex)
        for t, c in theta.items():
            if abs(c) >= 1e-14 and t in H_hat:
                H += c * H_hat[t]
        return H

    def residues(I_hat, H, a, b):
        if np.max(np.abs(I_hat - np.eye(I_hat.shape[0]))) < 1e-4:
            evals, X = la.eigh(H)
        else:
            try:
                evals, X = la.eigh(H, I_hat)
            except la.LinAlgError:
                lu = la.lu_factor(I_hat)
                evals, X = la.eig(la.lu_solve(lu, H))
                return evals, (a.conj() @ X) * (la.inv(X) @ la.lu_solve(lu, b))
        return evals, (a.conj() @ X) * (X.conj().T @ b)

    cv = pair["cv"]
    ep, rp = residues(cv["I_hat_p"], assemble(pair["H_p"]), cv["aR"] @ alpha, cv["bR"] @ alpha)
    eh, rh = residues(cv["I_hat_h"], -assemble(pair["H_h"]), cv["aL"] @ alpha, cv["bL"] @ alpha)
    return ((1.0 / ((freqs + E)[:, None] - ep[None, :])) @ rp
            + (1.0 / ((freqs - E)[:, None] - eh[None, :])) @ rh)
