"""Lanczos tridiagonalisation and the Green's-function evaluations built on it."""
import numpy as np
import scipy.linalg as la


def _lanczos_steps(H, Q, alpha, beta, w, j, m, reorth_window):
    """Continue the recurrence of lanczos_tridiag from step j to m; returns (steps, w, breakdown)."""
    for j in range(j, m):
        beta[j - 1] = np.linalg.norm(w)
        if beta[j - 1] < 1e-14:
            return j, w, True
        Q[:, j] = w / beta[j - 1]
        for k in range(max(0, j - reorth_window), j):
            Q[:, j] -= np.vdot(Q[:, k], Q[:, j]) * Q[:, k]
        nrm = np.linalg.norm(Q[:, j])
        if nrm > 1e-14:
            Q[:, j] /= nrm
        w = H @ Q[:, j]
        alpha[j] = np.real(np.vdot(Q[:, j], w))
        w = w - alpha[j] * Q[:, j] - beta[j - 1] * Q[:, j - 1]
    return m, w, False


def _lanczos_start(H, b, m):
    b = np.asarray(b).ravel()
    b_norm = np.linalg.norm(b)
    if b_norm < 1e-15:
        raise ValueError("Lanczos starting vector has zero norm")
    # column-major: every Lanczos vector is contiguous (row-major strides cost 25x per step at N=12)
    Q = np.zeros((H.shape[0], m), dtype=np.result_type(H.dtype, b.dtype), order="F")
    alpha, beta = np.zeros(m), np.zeros(max(m - 1, 0))
    Q[:, 0] = b / b_norm
    w = H @ Q[:, 0]
    alpha[0] = np.real(np.vdot(Q[:, 0], w))
    return Q, alpha, beta, w - alpha[0] * Q[:, 0], b_norm


def lanczos_tridiag(H, b, m, reorth_window=10):
    """m-step Lanczos on H from b with windowed reorthogonalisation.

    Returns (alpha, beta, Q, |b|); stops early on breakdown.
    """
    Q, alpha, beta, w, b_norm = _lanczos_start(H, b, m)
    j, _, _ = _lanczos_steps(H, Q, alpha, beta, w, 1, m, reorth_window)
    return alpha[:j], beta[:j - 1], Q[:, :j], b_norm


def _haydock_sum(alpha, beta, b_norm, z):
    evals, evecs = la.eigh_tridiagonal(alpha, beta)
    return b_norm**2 * np.sum((evecs[0, :]**2)[None, :] / (z[:, None] - evals[None, :]), axis=1)


def _cf_at(alpha, beta, z_pts):
    """1/(z - a0 - b0^2/(z - a1 - ...)) at a few points by backward recursion (scalar, cheap)."""
    a, b2, out = alpha.tolist(), (beta**2).tolist(), []
    for z in z_pts:
        g = 1.0 / (z - a[-1])
        for j in range(len(a) - 2, -1, -1):
            g = 1.0 / (z - a[j] - b2[j] * g)
        out.append(g)
    return out


def haydock_gf(H, b, z, tol=1e-8, check_every=5, k_min=10, k_max=500, need_consec=2, n_check=4):
    """<b|(z - H)^-1|b> by storage-free Lanczos, stopped once G(z) is converged to tol.

    When z lies on a vertical line (Matsubara frequencies plus a real shift) the continued
    fraction converges slowest at the points nearest the real axis, so convergence is checked on
    the n_check nearest ones only and G is evaluated on all of z once at the end.  For any other
    z (e.g. real frequencies with a broadening) every point is checked.
    Returns (G, number of Lanczos steps).
    """
    m = len(b)
    k_max = min(k_max, m)
    alphas, betas = np.zeros(k_max), np.zeros(k_max)
    norm_b = float(np.linalg.norm(b))
    v, v_prev = b / norm_b, np.zeros(m, dtype=b.dtype)
    z = np.asarray(z)
    on_line = np.ptp(z.real) <= 1e-12 * max(1.0, float(np.max(np.abs(z))))
    z_chk = [complex(x) for x in z[np.argsort(np.abs(z.imag))[:n_check]]] if on_line else None
    g_prev, n_ok, j = None, 0, 0
    while j < k_max:
        w = H @ v
        if j > 0:
            w -= betas[j - 1] * v_prev
        alphas[j] = np.real(v.conj() @ w)
        w -= alphas[j] * v
        beta = float(np.linalg.norm(w))
        j += 1
        if beta < 1e-13:  # invariant subspace: the continued fraction is exact
            break
        betas[j - 1] = beta
        if j >= k_min and j % check_every == 0:
            g = np.array(_cf_at(alphas[:j], betas[:j - 1], z_chk)) if on_line else \
                _haydock_sum(alphas[:j], betas[:j - 1], 1.0, z)
            if g_prev is not None:
                d = np.max(np.abs(g - g_prev)) / max(np.max(np.abs(g)), 1e-300)
                n_ok = n_ok + 1 if d < tol else 0
                if n_ok >= need_consec:
                    break
            g_prev = g
        v_prev, v = v, w / beta
    return _haydock_sum(alphas[:j], betas[:j - 1], norm_b, z), j


def haydock_batch(H, b, z, tol=1e-8, check_every=5, k_min=10, need_consec=2, n_check=4):
    """haydock_gf for S independent small dense problems (H[s], b[s], z[s]) run in lockstep: one
    batched product per step instead of S, and one vectorised pole evaluation at the end.  Zero-padded
    rows/columns are harmless (b is zero there, so the Krylov space never reaches them).
    Each z[s] must lie on a vertical line (see haydock_gf): only its points nearest the real axis
    are checked for convergence.  Returns (G (S, n_z), steps (S,))."""
    S, m, _ = H.shape
    alphas, betas = np.zeros((S, m)), np.zeros((S, m))
    norm_b = np.linalg.norm(b, axis=1)
    v, v_prev = b / norm_b[:, None], np.zeros_like(b)
    z = np.asarray(z)
    z_chk = [[complex(x) for x in zs[np.argsort(np.abs(zs.imag))[:n_check]]] for zs in z]
    steps, live = np.full(S, m), np.ones(S, dtype=bool)
    g_prev, n_ok, j = [None] * S, [0] * S, 0
    while j < m:
        w = np.matmul(H, v[:, :, None])[:, :, 0]
        if j > 0:
            w -= betas[:, j - 1, None] * v_prev
        alphas[:, j] = (v.conj() * w).real.sum(axis=1)
        w -= alphas[:, j, None] * v
        beta = np.sqrt((w.real**2 + w.imag**2).sum(axis=1))
        betas[:, j] = beta
        j += 1
        stop = live & (beta < 1e-13)  # invariant subspace: exact
        if j >= k_min and j % check_every == 0:
            for s in np.flatnonzero(live & ~stop):
                g = _cf_at(alphas[s, :j], betas[s, :j - 1], z_chk[s])
                if g_prev[s] is not None:
                    d = max(abs(x - y) for x, y in zip(g, g_prev[s])) / max(max(abs(x) for x in g), 1e-300)
                    n_ok[s] = n_ok[s] + 1 if d < tol else 0
                    stop[s] = n_ok[s] >= need_consec
                g_prev[s] = g
        if stop.any():
            steps[stop], live[stop] = j, False
            if not live.any():
                break
        v_prev, v = v, w / np.where(beta > 1e-13, beta, 1.0)[:, None]  # finished sectors idle along
    # poles and weights per sector (zero-padded tridiagonals), then G in real arithmetic:
    # 1/(z - l) = (x - l - iy) / ((x - l)^2 + y^2)
    n = int(steps.max())
    ev, wt = np.zeros((S, n)), np.zeros((S, n))
    for s in range(S):
        k = steps[s]
        e, V = la.eigh_tridiagonal(alphas[s, :k], betas[s, :k - 1])
        ev[s, :k], wt[s, :k] = e, norm_b[s]**2 * V[0]**2
    dx = z.real[:, :, None] - ev[:, None, :]
    D = wt[:, None, :] / (dx * dx + (z.imag**2)[:, :, None])
    return (D * dx).sum(axis=2) - 1j * z.imag * D.sum(axis=2), steps


def shifted_solve(A, rhs, shifts, plus=False, tol=1e-6, stages=(400, 800, 1600), reorth_window=10):
    """X[:, (col, shift)] = (shift*I -/+ A)^-1 rhs[:, col] from one Lanczos run per column.

    The run is extended through the depths in `stages` until the shifted residual estimate is
    below tol.  Columns are grouped by right-hand side.
    """
    shifts = np.asarray(shifts, dtype=complex)
    sgn = 1.0 if plus else -1.0
    d, n_rhs = rhs.shape
    blocks = []
    for c in range(n_rhs):
        b = np.asarray(rhs[:, c], dtype=float)
        nb = np.linalg.norm(b)
        if nb < 1e-15:
            blocks.append(np.zeros((d, len(shifts)), dtype=complex))
            continue
        Q, alpha, beta, w, _ = _lanczos_start(A, b, min(stages[0], d))
        j = 1
        for m_try in stages:
            m = min(m_try, d)
            if m > Q.shape[1]:  # grow the storage (column-major) and continue the same run
                Q_new = np.zeros((d, m), dtype=Q.dtype, order="F")
                Q_new[:, :Q.shape[1]] = Q
                Q = Q_new
                alpha = np.concatenate([alpha, np.zeros(m - alpha.size)])
                beta = np.concatenate([beta, np.zeros(m - 1 - beta.size)])
            j, w, broke = _lanczos_steps(A, Q, alpha, beta, w, j, m, reorth_window)
            mm = j
            Y = np.empty((mm, len(shifts)), dtype=complex)
            ab = np.zeros((3, mm), dtype=complex)
            e1 = np.zeros(mm, dtype=complex)
            e1[0] = nb
            worst = 0.0
            for i, s in enumerate(shifts):
                ab[0, 1:], ab[1, :], ab[2, :-1] = sgn * beta[:mm - 1], s + sgn * alpha[:mm], sgn * beta[:mm - 1]
                Y[:, i] = la.solve_banded((1, 1), ab, e1)
                if mm > 1:
                    worst = max(worst, abs(beta[mm - 2] * Y[-1, i]) / nb)
            if worst < tol or mm >= d or broke:
                break
        Qm = Q[:, :mm]
        if np.iscomplexobj(Qm):
            blocks.append(Qm @ Y)
        else:
            blocks.append(Qm @ np.ascontiguousarray(Y.real) + 1j * (Qm @ np.ascontiguousarray(Y.imag)))
    return np.hstack(blocks)
