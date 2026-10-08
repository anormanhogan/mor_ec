"""Eigenvector continuation: reduced ground states and variance-driven greedy basis growth.

The EC basis B (d_gs x k, orthonormal, scipy.sparse) spans exact ground states at selected
parameter points.  Projected terms h_r[key] = B^H H_key B give the reduced Hamiltonian
h(theta) = sum_key theta[key] h_r[key]; h2_r holds B^H H_l H_m B for the variance
Var(theta) = <H^2> - <H>^2 at the reduced ground state (= squared Ritz residual).
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.linalg as la
import scipy.sparse as sps
from scipy.sparse.linalg import eigsh


def canonical_gs(H, degen_tol=1e-8, guess=None):
    """Ground state of H; degeneracies are broken reproducibly by projecting a uniform vector.

    guess (e.g. the ROM's approximate ground state B alpha) starts ARPACK instead of the uniform
    vector; a 1% uniform admixture keeps every symmetry sector reachable, and a larger Krylov
    space (ncv = 40) helps with the clustered spectra near degeneracies.  The result is the same
    ground state to ARPACK's tolerance.
    """
    d = H.shape[0]
    u = np.ones(d) / np.sqrt(d)
    v0, ncv = u, None
    if guess is not None:
        g = np.asarray(guess, dtype=float).ravel()
        v0 = g / np.linalg.norm(g) + 1e-2 * u
        ncv = 40 if d > 40 else None
    vals, vecs = eigsh(H.astype(float), k=min(4, d - 1), which="SA", v0=v0, tol=1e-8, ncv=ncv)
    E0 = float(vals[0])
    degen = vecs[:, np.abs(vals - E0) < degen_tol]
    psi = degen @ (degen.T @ u)
    if np.linalg.norm(psi) < 1e-12:
        psi = degen[:, 0]
    return E0, (psi / np.linalg.norm(psi)).real


def solve_reduced(h_r, theta):
    """Lowest eigenpair (alpha, E) of h(theta) = sum theta[key] h_r[key]."""
    k = next(iter(h_r.values())).shape[0]
    h = np.zeros((k, k), dtype=complex)
    for key, c in theta.items():
        if abs(c) >= 1e-14 and key in h_r:
            h += c * h_r[key]
    if np.max(np.abs(h.imag)) < 1e-12 * max(1.0, np.max(np.abs(h.real))):
        evals, evecs = la.eigh(h.real, subset_by_index=[0, 0])
    else:
        evals, evecs = la.eigh(h)
    alpha = evecs[:, 0].real
    return alpha / np.linalg.norm(alpha), float(evals[0].real)


def _dense(x):
    return x.toarray() if sps.issparse(x) else np.asarray(x)


def _map(fn, items, n_workers):
    if n_workers > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            return list(pool.map(fn, items))
    return list(map(fn, items))


def _cap_workers(n_workers, d, m, itemsize):
    return max(1, min(int(n_workers), int(4 * 1024**3 / max(d * m * itemsize, 1))))


def projected_terms(W, V, ops, n_workers=1):
    """{key: W^H ops[key] V} (dense)."""
    Vd = _dense(V)
    WH = (Vd if W is V else _dense(W)).conj().T
    n_workers = _cap_workers(n_workers, *Vd.shape, Vd.itemsize)
    return dict(_map(lambda kv: (kv[0], WH @ (kv[1] @ Vd)), list(ops.items()), n_workers))


def h2_terms(B, ops, n_workers=1, also_h_r=True):
    """(h_r, h2_r) with h2_r[(l, m)] = Y_l^H Y_m + Y_m^H Y_l (l < m), Y_l^H Y_l (l = m).

    Pairs follow sorted(ops) so independently built dicts always agree on the pair order.
    """
    Bd = _dense(B)
    keys = sorted(ops)
    n_workers = _cap_workers(n_workers, *Bd.shape, Bd.itemsize)
    Y = dict(_map(lambda k: (k, ops[k] @ Bd), keys, n_workers))
    h_r = {k: Bd.conj().T @ Y[k] for k in keys} if also_h_r else None
    pairs = [(keys[i], keys[j]) for i in range(len(keys)) for j in range(i, len(keys))]

    def pair(p):
        l, m = p
        return p, (Y[l].conj().T @ Y[l] if l == m else Y[l].conj().T @ Y[m] + Y[m].conj().T @ Y[l])

    return h_r, dict(_map(pair, pairs, n_workers))


def append_terms(old, ops, L_old, R_add):
    """Block update of {key: L^H op L} when the columns R_add are appended to L_old.

    All cross blocks L_old^H (op R_add) come from one dense product.
    """
    R = _dense(R_add)
    keys, n = list(ops), R.shape[1]
    X = [ops[k] @ R for k in keys]
    cross = _dense(L_old).conj().T @ np.hstack(X)
    out = {}
    for i, key in enumerate(keys):
        c = cross[:, i * n:(i + 1) * n]
        out[key] = np.vstack([np.hstack([old[key], c]), np.hstack([c.conj().T, R.conj().T @ X[i]])])
    return out


def append_h2_terms(old, ops, B_old, B_add, chunk_bytes=2 * 1024**3):
    """Block update of h2_terms when the columns B_add are appended to B_old.

    The operators are Hermitian, so Y_l^H Y_m = B^H H_l H_m B and every cross block is
    B_old^H (H_l H_m B_add): dense products with B_old only, in chunks of about chunk_bytes.
    """
    Ro, Ra = _dense(B_old), _dense(B_add)
    keys = sorted(ops)
    Ya = {k: ops[k] @ Ra for k in keys}
    pairs = [(l, m) for i, l in enumerate(keys) for m in keys[i:]]
    n = Ra.shape[1]
    per = max(1, int(chunk_bytes // (Ra.shape[0] * n * Ra.itemsize)))
    RoH, out = Ro.conj().T, {}
    for c in range(0, len(pairs), per):
        chunk = pairs[c:c + per]
        Z = [ops[l] @ Ya[l] if l == m else ops[l] @ Ya[m] + ops[m] @ Ya[l] for l, m in chunk]
        cross = RoH @ np.hstack(Z)
        for i, (l, m) in enumerate(chunk):
            tr = cross[:, i * n:(i + 1) * n]
            br = Ya[l].conj().T @ Ya[l] if l == m else Ya[l].conj().T @ Ya[m] + Ya[m].conj().T @ Ya[l]
            out[(l, m)] = np.vstack([np.hstack([old[(l, m)], tr]), np.hstack([tr.conj().T, br])])
    return out


def variance(h_r, h2_r, thetas, n_workers=1):
    """Var(theta) = alpha^T h2(theta) alpha - E^2 at the reduced ground state, per theta."""
    keys = sorted(h_r)

    def one(theta):
        alpha, E = solve_reduced(h_r, theta)
        k = len(alpha)
        h2 = np.zeros((k, k))
        for i, l in enumerate(keys):
            cl = theta.get(l, 0.0)
            if abs(cl) < 1e-14:
                continue
            h2 += cl * cl * h2_r[(l, l)]
            for m in keys[i + 1:]:
                cm = theta.get(m, 0.0)
                if abs(cm) >= 1e-14:
                    h2 += cl * cm * h2_r[(l, m)]
        return max(float(alpha @ h2 @ alpha - E * E), 0.0)

    return np.array(_map(one, thetas, n_workers))


def mgs_residual(B, v, extra=(), n_passes=2):
    """v with span(B) and the vectors in `extra` projected out (repeated Gram-Schmidt)."""
    v = v.copy()
    for _ in range(n_passes):
        v -= np.asarray(B @ (B.T @ v)).ravel()
        for q in extra:
            v -= q * (q @ v)
    return v


def greedy_grow(B0, ops, grid, solve, tol, max_additions=None, h_r0=None, h2_r0=None,
                orth_tol=1e-8, n_workers=1, tag=""):
    """Weak-greedy growth of B over a fixed candidate grid.

    grid is a list of (theta, solve_args).  Each step ranks the remaining candidates by
    Var(theta) in the current basis, exact-solves only the argmax (solve(solve_args) -> psi)
    and appends its orthogonalised ground state.  Stops when max Var < tol, after
    max_additions vectors, or when the grid is exhausted.  An empty B0 bootstraps from the
    first candidate.  Returns (B, picks, converged); picks hold solve_args, variance and psi.
    """
    B = B0
    keys = sorted(ops)
    if B.shape[1] == 0:
        h_r = {k: np.zeros((0, 0)) for k in keys}
        h2_r = {(keys[i], keys[j]): np.zeros((0, 0)) for i in range(len(keys)) for j in range(i, len(keys))}
    elif h_r0 is not None and h2_r0 is not None:
        h_r, h2_r = h_r0, h2_r0
    else:
        h_r, h2_r = h2_terms(B, ops, n_workers)
    max_additions = len(grid) if max_additions is None else max_additions
    remaining = list(range(len(grid)))
    picks, converged = [], False

    while len(picks) < max_additions and remaining:
        if B.shape[1] == 0:
            i_loc, var = 0, None
        else:
            v = variance(h_r, h2_r, [grid[i][0] for i in remaining], n_workers)
            i_loc = int(np.argmax(v))
            var = float(v[i_loc])
            if tag:
                print(f"    [{tag}] greedy: max Var={var:.3e} (tol={tol:.3e}, k={B.shape[1]}, "
                      f"{len(remaining)} candidates)", flush=True)
            if var < tol:
                converged = True
                break
        args = grid[remaining.pop(i_loc)][1]
        psi = np.asarray(solve(args), dtype=float).ravel()
        r = mgs_residual(B, psi) if B.shape[1] > 0 else psi
        nr = float(np.linalg.norm(r))
        if nr < orth_tol:
            continue
        b_add = sps.csc_matrix((r / nr).reshape(-1, 1))
        B_old = B
        B = sps.hstack([B_old, b_add], format="csc") if B_old.shape[1] > 0 else b_add
        h_r = append_terms(h_r, ops, B_old, b_add)
        h2_r = append_h2_terms(h2_r, ops, B_old, b_add)
        picks.append({"solve_args": args, "variance": var, "psi": psi})
    return B, picks, converged
