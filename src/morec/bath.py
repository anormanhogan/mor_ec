"""Bath parametrisations and Sobol sampling of bath configurations.

A bath is a list of NI (eb, vb) pairs, optionally followed by a float dmu (shift of the
chemical potential away from half filling).  Each pair is either free-parameter form
(m = (NBpNI-1)/2 positive energies, m+1 couplings, particle-hole symmetric) or full form
(NBpNI energies and couplings).  Full form orders sites as [0, +eb_1..+eb_m, -eb_1..-eb_m].
"""
import hashlib

import numpy as np
from scipy.stats import qmc


def split_bath(bath, NI):
    return list(bath[:NI]), (float(bath[NI]) if len(bath) > NI else 0.0)


def full_bath(pairs, NBpNI):
    """(eb, vb) arrays of shape (NI, NBpNI); free-parameter pairs are mirrored."""
    m = (NBpNI - 1) // 2
    NI = len(pairs)
    eb_f, vb_f = np.zeros((NI, NBpNI)), np.zeros((NI, NBpNI))
    for i, (eb, vb) in enumerate(pairs):
        eb, vb = np.asarray(eb, float), np.asarray(vb, float)
        if len(eb) == NBpNI:
            eb_f[i], vb_f[i] = eb, vb
            continue
        eb_f[i, 1:1 + m], eb_f[i, NBpNI - m:] = eb, -eb
        vb_f[i, 0] = vb[0]
        vb_f[i, 1:1 + m] = vb_f[i, NBpNI - m:] = vb[1:]
    return eb_f, vb_f


def canonical_bath(eb, vb):
    """Full-form (eb, vb) of shape (NI, NBpNI) with each side's poles in ascending |energy|.
    """
    eb, vb = np.array(eb, float), np.array(vb, float)
    n = eb.shape[1]
    m = (n - 1) // 2
    for i in range(eb.shape[0]):
        for side, sign in ((slice(1, 1 + m), 1.0), (slice(n - m, n), -1.0)):
            order = np.argsort(sign * eb[i, side], kind="stable")
            eb[i, side], vb[i, side] = eb[i, side][order], vb[i, side][order]
    return eb, vb


def reduce_full_to_free(eb_full, vb_full, m):
    """Free-parameter form of a full bath via the geometric mean of each mirrored pair."""
    eb_full, vb_full = np.asarray(eb_full, float), np.asarray(vb_full, float)
    n = 2 * m + 1
    eb_p, eb_m = eb_full[1:1 + m], -eb_full[n - m:]
    vb_p, vb_m = vb_full[1:1 + m], vb_full[n - m:]
    return (np.sqrt(np.abs(eb_p * eb_m)),
            np.concatenate([[vb_full[0]], np.sqrt(np.abs(vb_p * vb_m))]))


def base_bath(NB, NI):
    """Reference PH-symmetric bath: eb_j = j/m and equal spectral weight per free coupling."""
    m = (NB // NI - 1) // 2
    eb = np.array([(j + 1) / max(m, 1) for j in range(m)])
    vb = np.empty(1 + m)
    vb[0] = 1.0 / np.sqrt(1 + m)
    vb[1:] = 1.0 / np.sqrt(2 * (1 + m))
    return [(eb.copy(), vb.copy()) for _ in range(NI)]


def seed_from(*values, salt=0):
    """Deterministic Sobol seed derived from the given arrays/scalars."""
    h = hashlib.sha256(str(salt).encode())
    for v in values:
        h.update(np.ascontiguousarray(v, dtype=float).tobytes())
    return int.from_bytes(h.digest()[:4], "little")


def _sobol(n, d, seed):
    return qmc.Sobol(d=max(d, 1), scramble=True, seed=seed).random(
        n=1 << (n - 1).bit_length())[:n]


def sample_baths(n, NB, NI, base, scale=0.4, seed=None, symmetric=True,
                 bw_lo=None, bw_hi=None, illustrate=False):
    """n free-parameter baths around `base` with log-uniform jitter of width `scale`.

    symmetric=True shares one jitter across orbitals.  bw_lo/bw_hi (length NI) draw a
    per-orbital bandwidth ratio r ~ LogUniform(lo, hi) and rescale eb *= r, vb *= sqrt(r).
    illustrate=True shrinks the jitter tenfold and ignores bandwidth sampling.
    """
    if seed is None:
        seed = seed_from(*(v for pair in base for v in pair), salt=n)
    m = (NB // NI - 1) // 2
    n_sym = 2 * m + 1
    lo = np.ones(NI) if bw_lo is None else np.asarray(bw_lo, float)
    hi = np.ones(NI) if bw_hi is None else np.asarray(bw_hi, float)
    varying = lo != hi
    fixed = np.where(varying, np.nan, (lo + hi) / 2.0)
    n_bw = 0 if illustrate else int(varying.sum())
    symmetric = symmetric or illustrate
    n_bath_dims = n_sym if symmetric else NI * n_sym
    samples = _sobol(n, n_bath_dims + n_bw, seed)

    configs = []
    for s in samples:
        ratios = fixed.copy()
        if n_bw:
            ratios[varying] = np.exp(s[n_bath_dims:n_bath_dims + n_bw]
                                     * np.log(hi[varying] / lo[varying]) + np.log(lo[varying]))
        config = []
        for i in range(NI):
            off = 0 if symmetric else i * n_sym
            lnf = (s[off:off + n_sym] - 0.5) * 2.0 * scale
            if illustrate:
                lnf = lnf * 0.1
            eb0, vb0 = base[i][0] * ratios[i], base[i][1] * np.sqrt(ratios[i])
            config.append((eb0 * np.exp(lnf[:m]), vb0 * np.exp(lnf[m:])))
        configs.append(config)
    return configs


def sample_baths_local(n, bath, NI, scale=0.2, U0=None, seed=None):
    """n baths jittered log-uniformly around `bath`, keeping its form (free/full, dmu).

    dmu gets an additive jitter of the same width.  With U0 the interaction is jittered
    jointly and the result is a list of (U, bath) tuples.
    """
    has_dmu = len(bath) > NI
    dmu0 = float(bath[NI]) if has_dmu else 0.0
    eb0 = [np.asarray(bath[i][0], float) for i in range(NI)]
    vb0 = [np.asarray(bath[i][1], float) for i in range(NI)]
    n_bath_dims = sum(e.size + v.size for e, v in zip(eb0, vb0)) + int(has_dmu)
    if seed is None:
        seed = seed_from(*eb0, *vb0, dmu0, *([U0] if U0 is not None else []), salt=n)
    raw = _sobol(n, n_bath_dims + int(U0 is not None), seed)
    jitter = (raw[:, :n_bath_dims] - 0.5) * 2.0 * scale

    configs = []
    for row, s in enumerate(jitter):
        config, off = [], 0
        for e, v in zip(eb0, vb0):
            d_eb = s[off:off + e.size]; off += e.size
            d_vb = s[off:off + v.size]; off += v.size
            config.append((e * np.exp(d_eb), v * np.exp(d_vb)))
        if has_dmu:
            config.append(dmu0 + s[off])
        if U0 is not None:
            configs.append((U0 * np.exp((raw[row, n_bath_dims] - 0.5) * 2.0 * scale), config))
        else:
            configs.append(config)
    return configs
