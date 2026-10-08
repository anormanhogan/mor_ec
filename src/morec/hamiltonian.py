"""Anderson impurity model with Kanamori interaction, built directly in particle-number sectors.

Convention is 'udud' ordering: spatial site s owns modes 2s (up) and 2s+1 (down); impurities are
sites 0..NI-1 and the bath of impurity i occupies sites NI + i*NBpNI + j.  A basis state is an
integer whose bit (2N-1-k) is the occupation of mode k, and a ladder operator on mode k carries
the Jordan-Wigner sign (-1)^(number of occupied modes below k).
"""
import itertools
import warnings

import numpy as np
import scipy.sparse as sps

from .bath import canonical_bath, full_bath, split_bath

SECTORS = ("gs", "p", "h")


def mu_half_filling(U, J, NI):
    """Particle-hole symmetric chemical potential of the NI-orbital Kanamori impurity."""
    return U / 2.0 + (U - 2.5 * J) * (NI - 1)


def affine_theta(U, J, bath, NI, NBpNI):
    """Affine coefficients for (U, J, bath); bath = NI (eb, vb) pairs, optionally + dmu.

    The poles are put in canonical order first, so physically identical baths give identical
    coefficients for ED and for the reduced model.
    """
    pairs, dmu = split_bath(bath, NI)
    eb, vb = canonical_bath(*full_bath(pairs, NBpNI))
    th = {"U": float(U), "J": float(J), "dmu": float(dmu)}
    for i in range(NI):
        for j in range(NBpNI):
            th[f"eb{i}.{j}"] = float(eb[i, j])
            th[f"vb{i}.{j}"] = float(vb[i, j])
    return th


def sector_basis(N, n_up, n_dn):
    """Sorted basis integers with n_up spin-up and n_dn spin-down electrons on N sites."""
    n_q = 2 * N
    up = [sum(1 << (n_q - 1 - 2 * s) for s in c) for c in itertools.combinations(range(N), n_up)]
    dn = [sum(1 << (n_q - 2 - 2 * s) for s in c) for c in itertools.combinations(range(N), n_dn)]
    return np.sort((np.array(up, np.int64)[:, None] | np.array(dn, np.int64)[None, :]).ravel())


def _ladder(states, mode, create, n_q):
    bit = n_q - 1 - mode
    ok = ((states >> bit) & 1) == (0 if create else 1)
    sign = 1 - 2 * (np.bitwise_count(states >> (bit + 1)).astype(np.int64) & 1)
    return states ^ (1 << bit), ok, sign


def _operator(terms, src, dst, n_q):
    """Matrix of sum_t coeff_t * (product of ladder operators) from basis src to basis dst."""
    rows, cols, vals = [], [], []
    for coeff, ladders in terms:
        s, ok, sign = src, np.ones(src.size, bool), np.ones(src.size)
        for mode, create in reversed(ladders):
            s, ok_k, sg = _ladder(s, mode, create, n_q)
            ok &= ok_k
            sign = sign * sg
        j = np.nonzero(ok)[0]
        r = np.searchsorted(dst, s[j])
        if not np.array_equal(dst[np.minimum(r, dst.size - 1)], s[j]):
            raise ValueError("operator maps outside the target sector")
        rows.append(r), cols.append(j), vals.append(coeff * sign[j])
    M = sps.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(dst.size, src.size))
    M.eliminate_zeros()
    return M


def _one_body(pairs):
    """c†_p c_q terms for {(p, q): t}."""
    return [(t, [(p, True), (q, False)]) for (p, q), t in pairs.items() if abs(t) > 1e-12]


def _kanamori(NI, U, J):
    """Two-body integrals u[p,q,r,s] (term 0.5*u c†_p c†_q c_r c_s) of the Kanamori interaction.

    U n_up n_dn per orbital, U' = U-2J (opposite spin) and U'' = U-3J (parallel spin) between
    orbitals, spin flip -J c†_{i↑}c_{i↓}c†_{j↓}c_{j↑} and pair hopping J c†_{i↑}c†_{i↓}c_{j↓}c_{j↑}
    (i != j).
    """
    u = {}
    Up, Upp = U - 2.0 * J, U - 3.0 * J
    for i in range(NI):
        iu, id_ = 2 * i, 2 * i + 1
        u[iu, id_, id_, iu] = u[id_, iu, iu, id_] = U
    for i in range(NI):
        for j in range(i + 1, NI):
            iu, id_, ju, jd = 2 * i, 2 * i + 1, 2 * j, 2 * j + 1
            u[iu, ju, ju, iu] = u[ju, iu, iu, ju] = Upp
            u[iu, jd, jd, iu] = u[jd, iu, iu, jd] = Up
            u[id_, ju, ju, id_] = u[ju, id_, id_, ju] = Up
            u[id_, jd, jd, id_] = u[jd, id_, id_, jd] = Upp
            u[iu, jd, id_, ju] = u[jd, iu, ju, id_] = u[id_, ju, iu, jd] = u[ju, id_, jd, iu] = J
            u[iu, id_, jd, ju] = u[id_, iu, ju, jd] = u[ju, jd, id_, iu] = u[jd, ju, iu, id_] = J
    return [(0.5 * c, [(p, True), (q, True), (r, False), (s, False)])
            for (p, q, r, s), c in u.items() if abs(c) > 1e-12]


class AIM:
    """Affine operator family H(theta) = sum_key theta[key] * ops[sector][key] of the AIM.

    Keys are 'U', 'J' (NI > 1), 'dmu' (shift of mu away from half filling) and 'eb{i}.{j}',
    'vb{i}.{j}' for every bath site.  Sectors: 'gs' (half filling), 'p' (+1 spin-up electron)
    and 'h' (-1 spin-up electron).  c_ann[i] maps gs -> h and c_dag[i] maps gs -> p for the
    spin-up electron on impurity orbital i.
    """

    def __init__(self, N, NI):
        NB = N - NI
        if NB % NI or (NB // NI) % 2 == 0:
            raise ValueError("each impurity orbital needs an odd number of bath sites")
        self.N, self.NI, self.NB, self.NBpNI = N, NI, NB, NB // NI
        self.m_free = (self.NBpNI - 1) // 2
        n_q = 2 * N
        half = N // 2
        self.basis = {"gs": sector_basis(N, half, half),
                      "p": sector_basis(N, half + 1, half),
                      "h": sector_basis(N, half - 1, half)}
        terms = self._affine_terms()
        self.keys = list(terms)
        self.ops = {s: {k: _operator(t, b, b, n_q) for k, t in terms.items()}
                    for s, b in self.basis.items()}
        gs = self.basis["gs"]
        self.c_ann = [_operator([(1.0, [(2 * i, False)])], gs, self.basis["h"], n_q)
                      for i in range(NI)]
        self.c_dag = [_operator([(1.0, [(2 * i, True)])], gs, self.basis["p"], n_q)
                      for i in range(NI)]
        self._check_orbital_diagonal()

    def _bath_site(self, i, j):
        return self.NI + i * self.NBpNI + j

    def _check_orbital_diagonal(self):
        """Warn about terms that break the parity of a channel (orbital i and its bath).

        The solvers return only G_ii and DMFT inverts G elementwise.  Both are exact only if every
        term conserves the parity of each channel's electron count, which forces G_ij = 0 for i != j.
        """
        n_q, states = 2 * self.N, self.basis["gs"]
        label = np.zeros(states.size, np.int64)
        for i in range(self.NI):
            sites = [i] + [self._bath_site(i, j) for j in range(self.NBpNI)]
            mask = sum(3 << (n_q - 2 - 2 * s) for s in sites)
            label |= (np.bitwise_count(states & mask).astype(np.int64) & 1) << i
        bad = [k for k, M in self.ops["gs"].items()
               if np.any(label[M.indices] != label[np.repeat(np.arange(M.shape[1]), np.diff(M.indptr))])]
        if bad:
            warnings.warn(f"AIM terms {bad} couple different impurity orbitals, so G_ij can be nonzero "
                          f"for i != j; the solvers return only G_ii and the DMFT loop inverts G "
                          f"elementwise, which is then not exact")

    def _affine_terms(self):
        NI = self.NI
        imp_n = lambda c: {(p, p): c for i in range(NI) for p in (2 * i, 2 * i + 1)}
        terms = {"U": _kanamori(NI, 1.0, 0.0) + _one_body(imp_n(-mu_half_filling(1.0, 0.0, NI)))}
        if NI > 1:
            terms["J"] = _kanamori(NI, 0.0, 1.0) + _one_body(imp_n(-mu_half_filling(0.0, 1.0, NI)))
        terms["dmu"] = _one_body(imp_n(-1.0))
        for i in range(NI):
            for j in range(self.NBpNI):
                b = self._bath_site(i, j)
                terms[f"eb{i}.{j}"] = _one_body({(2 * b, 2 * b): 1.0, (2 * b + 1, 2 * b + 1): 1.0})
            for j in range(self.NBpNI):
                b = self._bath_site(i, j)
                terms[f"vb{i}.{j}"] = _one_body({(2 * i, 2 * b): 1.0, (2 * b, 2 * i): 1.0,
                                                 (2 * i + 1, 2 * b + 1): 1.0,
                                                 (2 * b + 1, 2 * i + 1): 1.0})
        return terms

    def theta(self, U, J, bath):
        return affine_theta(U, J, bath, self.NI, self.NBpNI)

    def hamiltonian(self, sector, theta):
        ops = self.ops[sector]
        order = [k for k in ("U", "J") if k in ops]
        order += [k for k in ops if k[:2] == "eb"] + [k for k in ops if k[:2] == "vb"] + ["dmu"]
        H = sps.csc_matrix(ops["U"].shape)
        for k in order:
            if theta.get(k, 0.0) != 0.0:
                H = H + theta[k] * ops[k]
        return H.tocsc()

    def hamiltonians(self, U, J, bath):
        """(H_gs, H_p, H_h) at (U, J, bath)."""
        th = self.theta(U, J, bath)
        return tuple(self.hamiltonian(s, th) for s in SECTORS)
