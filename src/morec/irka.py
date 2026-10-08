"""IRKA (Gugercin, Antoulas and Beattie 2008) for real-axis interpolation frequencies (SIAM demo).

G(z) = b^T (z - A)^-1 b is one SISO system with A = diag(H_p - E0, E0 - H_h) and b = [c†psi; c psi].
Its poles lie on the real axis, so the H2 norm is taken for the broadened G(w + i*eta).  The
reduced model of order r (bilinear projection V^T A V, which interpolates G and G' at every
point) is iterated until the points are the mirror images conj(lambda) + 2i*eta of its poles.
All shifted solves use one Lanczos run per sector.
"""
import numpy as np
import scipy.linalg as la
import scipy.sparse as sps
import scipy.sparse.linalg as spla

from .lanczos import lanczos_tridiag


def krylov_system(H_p, H_h, E0, psi, c_dag, c_ann, m=300):
    """(A, b) of G = G_p + G_h in the coordinates of one m-step Lanczos run per sector."""
    blocks, rhs = [], []
    for H, v, sign in ((H_p, c_dag @ psi, 1.0), (H_h, c_ann @ psi, -1.0)):
        a, bt, _, nb = lanczos_tridiag(H.tocsc(), np.asarray(v).ravel(), min(m, H.shape[0]))
        blocks.append(sign * (sps.diags([bt, a, bt], [-1, 0, 1]) - E0 * sps.eye(len(a))))
        rhs.append(np.r_[nb, np.zeros(len(a) - 1)])
    return sps.block_diag(blocks, format="csc"), np.concatenate(rhs)


def _sorted(z):
    return z[np.lexsort((z.imag, np.round(z.real, 8)))]


def select_frequencies(H_p, H_h, H_gs, E0, B, r, c_dag, c_ann, eta=0.08, max_iter=200, tol=1e-6,
                       lanczos_m=300, verbose=True):
    """r IRKA points for the reference model on basis B (one impurity orbital)."""
    if len(c_dag) != 1:
        raise NotImplementedError("IRKA is implemented for one impurity orbital (NI = 1)")
    log = print if verbose else (lambda *a, **k: None)
    _, vecs = la.eigh((B.conj().T @ (H_gs @ B)).toarray(), (B.conj().T @ B).toarray())
    psi = B @ vecs[:, 0]
    A, b = krylov_system(H_p, H_h, E0, psi, c_dag[0], c_ann[0], lanczos_m)
    I = sps.eye(A.shape[0], format="csc")

    a, bt, _, _ = lanczos_tridiag(A, b, r)  # seed: poles of the moment-matching model (z -> inf)
    z = _sorted(la.eigh_tridiagonal(a, bt, eigvals_only=True) + 2j * eta)
    for it in range(1, max_iter + 1):
        Q = la.qr(np.column_stack([spla.spsolve((zi * I - A).tocsc(), b.astype(complex)) for zi in z]),
                  mode="economic")[0]
        lam = la.eig(Q.T @ (A @ Q), Q.T @ Q, right=False)
        if not np.all(np.isfinite(lam)):
            log(f"[IRKA] singular reduced model at iteration {it}; keeping the previous points")
            return z
        z_new = _sorted(np.conj(lam) + 2j * eta)
        change = np.max(np.abs(z_new - z)) / max(1.0, np.max(np.abs(z_new)))
        z = z_new
        if change < tol:
            log(f"[IRKA] eta={eta}, r={len(z)}: converged after {it} iterations")
            return z
    log(f"[IRKA] eta={eta}, r={len(z)}: not converged after {max_iter} iterations "
        f"(last relative change {change:.1e}); using the last iterate")
    return z
