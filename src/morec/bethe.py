"""Bethe-lattice self-consistency ingredients: lattice GF, hybridization and bath fits."""
import numpy as np
from scipy.optimize import curve_fit

from .bath import reduce_full_to_free


def hybridization(eb_full, vb_full, iw):
    """Delta(iw) = sum_l V_l^2 / (iw - eps_l)."""
    iw = np.asarray(iw, dtype=complex)
    delta = np.zeros_like(iw)
    for eps, V in zip(eb_full, vb_full):
        delta += float(V) ** 2 / (iw - float(eps))
    return delta


def bethe_gf(t, freq, sigma, x, mu):
    """Local lattice GF for the semicircular DOS of half-bandwidth 2t, on the energy grid x."""
    t = float(t)
    x = np.asarray(x, dtype=float)
    rho = (1.0 / (2.0 * np.pi * t**2)) * np.sqrt(np.maximum(0.0, 4.0 * t**2 - x**2))
    z = np.asarray(freq, dtype=complex) + mu - np.asarray(sigma, dtype=complex)
    return np.sum((rho * (x[1] - x[0])) / (z[:, None] - x[None, :]), axis=1)


def quasiparticle_weight(Sigma, iw, G=None, Delta=None, n=0):
    """Z = 1 / (1 - Im Sigma(iw_n)/w_n) in [0, 1]; from Im[1/G + Delta] when G, Delta are given."""
    wn = iw[n].imag
    if G is not None and Delta is not None:
        y = (1.0 / G[n] + Delta[n]).imag
        slope = (wn * y) / (wn * wn)
        return 0.0 if slope <= 0.0 else float(np.clip(1.0 / slope, 0.0, 1.0))
    return float(np.clip(1.0 / (1.0 - Sigma[n].imag / wn), 0.0, 1.0))


def is_causal(G):
    """(Im G(iw_n) < 0 for all n, max Im G)."""
    m = float(np.max(np.asarray(G).imag))
    return m < 1e-10, m


def _pole_ceiling(w_n, t, U):
    # Hubbard-band poles sit near U/2 regardless of bandwidth, so the ceiling is max(5t, U).
    return min(5.0 * w_n[-1], max(5.0 * float(t), float(U)))


def _best_fit(model, w_n, y, starts, lower, upper, maxfev):
    best, best_r = None, np.inf
    for p0 in starts:
        try:
            p, _ = curve_fit(model, w_n, y, p0=p0, bounds=(lower, upper), maxfev=maxfev)
        except (RuntimeError, ValueError):
            continue
        r = float(np.mean((y - model(w_n, *p)) ** 2))
        if r < best_r:
            best, best_r = p, r
    return starts[0] if best is None else best


def fit_bath_symmetric(Delta, iw, m, eb0, vb0, t, n_starts=1, U=0.0, maxfev=10000):
    """Particle-hole symmetric bath (free form) fitted to Im Delta on the Matsubara axis.

    Poles at 0 and +/-eb_j; vb[0] is fixed by the sum rule vb0^2 + 2 sum vb_j^2 = t^2.
    Extra starts spread the poles geometrically over [0.1t, 3t].
    """
    w_n = np.asarray(iw).imag
    if m == 0:
        return np.array([]), np.array([float(t)])

    def model(w, *p):
        eb, vbf = np.array(p[:m]), np.array(p[m:])
        z = 1j * w
        d = np.sqrt(np.maximum(t**2 - 2.0 * np.sum(vbf**2), 0.0)) ** 2 / z
        for j in range(m):
            d += vbf[j]**2 / (z - eb[j]) + vbf[j]**2 / (z + eb[j])
        return d.imag

    lower = np.array([1e-4] * m + [0.0] * m)
    upper = np.array([_pole_ceiling(w_n, t, U)] * m + [float(t) / np.sqrt(2.0) - 1e-12] * m)
    eb0, vb0 = np.asarray(eb0, float), np.asarray(vb0, float)
    if len(eb0) != m:
        eb0, vb0 = reduce_full_to_free(eb0, vb0, m)
    starts = [np.clip(np.concatenate([np.abs(eb0), np.abs(vb0[1:])]), lower + 1e-12, upper - 1e-12)]
    lo_e, hi_e = max(lower[0] + 1e-4, 0.1 * t), min(upper[0] - 1e-4, 3.0 * t)
    spread = np.geomspace(0.4, 2.5, m) if m > 1 else np.array([1.0])
    if n_starts > 1 and lo_e < hi_e:
        for e in np.geomspace(lo_e, hi_e, n_starts - 1):
            p = np.concatenate([np.clip(e * spread, lo_e, hi_e), np.full(m, t / np.sqrt(2.0 * (m + 1)))])
            starts.append(np.clip(p, lower + 1e-12, upper - 1e-12))
    p = _best_fit(model, w_n, Delta.imag, starts, lower, upper, maxfev)
    order = np.argsort(p[:m])
    vbf = p[m:][order]
    return p[:m][order], np.concatenate([[np.sqrt(max(t**2 - 2.0 * np.sum(p[m:]**2), 0.0))], vbf])


def fit_bath_general(Delta, iw, m, eb0, vb0, t, n_starts=1, U=0.0, maxfev=10000):
    """Bath without particle-hole symmetry (full form) fitted to Re and Im Delta.

    Poles at 0, +ep_j and -em_j with independent couplings vp_j, vm_j; the central coupling
    is fixed by the sum rule.  Extra starts also spread the vp/vm ratio over [0.3, 3].  Each
    side is returned in ascending |energy| (the canonical order, see bath.canonical_bath).
    """
    w_n = np.asarray(iw).imag
    n = 2 * m + 1
    if m == 0:
        return np.array([0.0]), np.array([float(t)])

    def model(w, *p):
        ep, em, vp, vm = (np.array(p[a * m:(a + 1) * m]) for a in range(4))
        z = 1j * w
        d = np.sqrt(np.maximum(t**2 - np.sum(vp**2 + vm**2), 0.0)) ** 2 / z
        for j in range(m):
            d += vp[j]**2 / (z - ep[j]) + vm[j]**2 / (z + em[j])
        return np.concatenate([d.real, d.imag])

    e_max = _pole_ceiling(w_n, t, U)
    lower = np.concatenate([np.full(2 * m, 1e-4), np.zeros(2 * m)])
    upper = np.concatenate([np.full(2 * m, e_max), np.full(2 * m, float(t) - 1e-12)])
    eb0, vb0 = np.asarray(eb0, float), np.asarray(vb0, float)
    if len(eb0) == m:
        p0 = np.concatenate([np.abs(eb0), np.abs(eb0), np.abs(vb0[1:]), np.abs(vb0[1:])])
    else:
        p0 = np.concatenate([eb0[1:1 + m], -eb0[n - m:], vb0[1:1 + m], vb0[n - m:]])
    starts = [np.clip(p0, lower + 1e-12, upper - 1e-12)]
    v_mid = t / np.sqrt(2.0 * (m + 1))
    lo_e, hi_e = max(lower[0] + 1e-4, 0.1 * t), min(upper[0] - 1e-4, 3.0 * t)
    spread = np.geomspace(0.4, 2.5, m) if m > 1 else np.array([1.0])
    if n_starts > 1 and lo_e < hi_e:
        for e, r in zip(np.geomspace(lo_e, hi_e, n_starts - 1), np.geomspace(0.3, 3.0, n_starts - 1)):
            e_arr = np.clip(e * spread, lo_e, hi_e)
            vp = np.clip(v_mid * np.sqrt(r), lower[2 * m], upper[2 * m])
            vm = np.clip(v_mid / np.sqrt(r), lower[3 * m], upper[3 * m])
            p = np.concatenate([e_arr, e_arr, np.full(m, vp), np.full(m, vm)])
            starts.append(np.clip(p, lower + 1e-12, upper - 1e-12))
    y = np.concatenate([Delta.real, Delta.imag])
    p = _best_fit(model, w_n, y, starts, lower, upper, maxfev)
    ep, em, vp, vm = (p[a * m:(a + 1) * m] for a in range(4))
    op, om = np.argsort(ep), np.argsort(em)  # the two sides are independent: sort each
    eb, vb = np.zeros(n), np.zeros(n)
    vb[0] = np.sqrt(max(t**2 - np.sum(vp**2 + vm**2), 0.0))
    eb[1:1 + m], eb[n - m:] = ep[op], -em[om]
    vb[1:1 + m], vb[n - m:] = vp[op], vm[om]
    return eb, vb
