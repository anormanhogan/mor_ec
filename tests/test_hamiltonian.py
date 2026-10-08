import warnings

import numpy as np
import pytest
import scipy.linalg as la

from morec import AIM
from morec.hamiltonian import _kanamori, _one_body, _operator
from morec.bath import base_bath


def test_sector_dimensions():
    aim = AIM(6, 1)
    assert [aim.basis[s].size for s in ("gs", "p", "h")] == [400, 300, 300]
    assert aim.c_ann[0].shape == (300, 400) and aim.c_dag[0].shape == (300, 400)


def test_operators_hermitian_and_affine():
    aim = AIM(8, 2)
    for ops in aim.ops.values():
        for H in ops.values():
            assert abs(H - H.T).max() < 1e-14
    bath = [(np.array([0.7]), np.array([0.5, 0.4])), (np.array([0.2]), np.array([0.3, 0.1])), 0.3]
    th = aim.theta(3.0, 0.4, bath)
    H = aim.hamiltonian("gs", th)
    ref = sum(th[k] * op for k, op in aim.ops["gs"].items())
    assert abs(H - ref).max() < 1e-12


def test_particle_hole_symmetry_at_half_filling():
    aim = AIM(6, 1)
    _, H_p, H_h = aim.hamiltonians(2.5, 0.0, base_bath(5, 1))
    e_p, e_h = la.eigvalsh(H_p.toarray()), la.eigvalsh(H_h.toarray())
    assert np.allclose(e_p - e_p[0], e_h - e_h[0], atol=1e-10)


def test_orbital_coupling_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        AIM(8, 2)

    class Hopping(AIM):
        def _affine_terms(self):
            return {**super()._affine_terms(), "t01": _one_body({(0, 2): 1.0, (2, 0): 1.0})}

    with pytest.warns(UserWarning, match=r"\['t01'\] couple"):
        Hopping(8, 2)


def test_kanamori_spin_rotation_invariant():
    U, J = 4.0, 0.7
    for NI in (2, 3):
        n_q = 2 * NI
        b = np.arange(2 ** n_q, dtype=np.int64)
        H = _operator(_kanamori(NI, U, J), b, b, n_q).toarray()
        S_plus = _operator([(1.0, [(2 * m, True), (2 * m + 1, False)]) for m in range(NI)], b, b, n_q)
        assert abs(H @ S_plus - S_plus @ H).max() < 1e-12
    half = np.bitwise_count(b) == 3  # three electrons in three orbitals: S = 3/2 quartet at 3U - 9J
    e = np.linalg.eigvalsh(H[np.ix_(half, half)])
    assert np.isclose(e[0], 3 * U - 9 * J) and np.sum(np.isclose(e, e[0])) == 4
