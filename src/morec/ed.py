"""Exact diagonalization reference: ground state by Lanczos, Green's function by the same
storage-free Lanczos recursion as the ROM (lanczos.haydock_gf)."""
import time

import numpy as np
import scipy.sparse as sps

from .ec import canonical_gs
from .lanczos import haydock_gf


def _apply(op, v):
    r = op @ v
    return r.toarray().ravel() if sps.issparse(r) else np.asarray(r).ravel()


def exact_gf(aim, E0, psi, H_p, H_h, freqs, i, j, gf_tol=1e-8):
    """G_ij(z) = <c_i (z + E0 - H_p)^-1 c_j†> + <c_j† (z - E0 + H_h)^-1 c_i> in the sectors.

    G_ii comes from one haydock_gf run per sector; for i != j the polarization identity
    G_ij = (G[c_i + c_j] - G_ii - G_jj) / 2 applies because H and psi are real.
    """
    psi = np.asarray(psi).ravel()
    freqs = np.asarray(freqs)
    Hp, Hh = H_p.tocsr(), (-H_h).tocsr()
    gf = lambda H, v, z: haydock_gf(H, v, z, tol=gf_tol)[0] if np.linalg.norm(v) > 1e-15 else 0.0
    diag = lambda p, h: gf(Hp, p, freqs + E0) + gf(Hh, h, freqs - E0)
    p = [_apply(aim.c_dag[m], psi) for m in (i, j)]
    h = [_apply(aim.c_ann[m], psi) for m in (i, j)]
    if i == j:
        return diag(p[0], h[0])
    return 0.5 * (diag(p[0] + p[1], h[0] + h[1]) - diag(p[0], h[0]) - diag(p[1], h[1]))


class EDSolver:
    """Impurity solver returning the diagonal impurity GF from exact diagonalization."""

    def __init__(self, aim, gf_tol=1e-8):
        self.aim, self.gf_tol = aim, gf_tol

    def solve(self, U, J, bath, freqs):
        """(G_imp (NI, n_freq), E0, n_imp (NI,), t_gs, t_gf)."""
        t0 = time.perf_counter()
        H_gs, H_p, H_h = self.aim.hamiltonians(U, J, bath)
        t1 = time.perf_counter()
        E0, psi = canonical_gs(H_gs)
        t_gs = time.perf_counter() - t1
        NI = self.aim.NI
        n_imp = np.array([2.0 * float(np.linalg.norm(self.aim.c_ann[m] @ psi) ** 2) for m in range(NI)])
        G = np.array([exact_gf(self.aim, E0, psi, H_p, H_h, freqs, m, m, self.gf_tol) for m in range(NI)])
        return G, E0, n_imp, t_gs, time.perf_counter() - t0 - t_gs
