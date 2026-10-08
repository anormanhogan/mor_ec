"""The ED Green's function (Lanczos recursion, shared with the ROM) against dense diagonalisation."""
import numpy as np
import pytest
import scipy.linalg as la

from morec import AIM
from morec.bath import base_bath
from morec.ec import canonical_gs
from morec.ed import exact_gf
from morec.hamiltonian import _one_body

MATSUBARA = 1j * np.pi / 232.0 * (2 * np.arange(100) + 1)
REAL_AXIS = np.linspace(-6.0, 6.0, 300) + 0.05j


def dense_gf(aim, E0, psi, H_p, H_h, z, i, j):
    """<c_i (z + E0 - H_p)^-1 c_j†> + <c_j† (z - E0 + H_h)^-1 c_i> from full eigendecompositions."""
    G = np.zeros(len(z), dtype=complex)
    for H, bra, ket, zz in ((H_p.toarray(), aim.c_dag[i] @ psi, aim.c_dag[j] @ psi, z + E0),
                            ((-H_h).toarray(), aim.c_ann[j] @ psi, aim.c_ann[i] @ psi, z - E0)):
        lam, Q = la.eigh(H)
        G += (1.0 / (zz[:, None] - lam[None, :])) @ ((Q.T @ bra) * (Q.T @ ket))
    return G


@pytest.mark.parametrize("z", [MATSUBARA, REAL_AXIS], ids=["matsubara", "real_axis"])
def test_diagonal_gf_matches_dense(z):
    aim = AIM(6, 1)
    for U in (0.5, 3.0, 7.0):
        H_gs, H_p, H_h = aim.hamiltonians(U, 0.0, base_bath(5, 1))
        E0, psi = canonical_gs(H_gs)
        ref = dense_gf(aim, E0, psi, H_p, H_h, z, 0, 0)
        G = exact_gf(aim, E0, psi, H_p, H_h, z, 0, 0)
        assert np.max(np.abs(G - ref)) / np.max(np.abs(ref)) < 1e-6


def test_offdiagonal_gf_matches_dense():
    class Hopping(AIM):  # inter-orbital hopping makes G_01 nonzero
        def _affine_terms(self):
            return {**super()._affine_terms(), "t01": _one_body({(0, 2): 1.0, (2, 0): 1.0})}

    with pytest.warns(UserWarning):
        aim = Hopping(4, 2)
    th = aim.theta(3.0, 0.5, base_bath(2, 2))  # AIM.hamiltonian sums the built-in terms only
    H_gs, H_p, H_h = (aim.hamiltonian(s, th) + 0.4 * aim.ops[s]["t01"] for s in ("gs", "p", "h"))
    E0, psi = canonical_gs(H_gs)
    for z in (MATSUBARA, REAL_AXIS):
        ref = dense_gf(aim, E0, psi, H_p, H_h, z, 0, 1)
        assert np.max(np.abs(ref)) > 1e-3
        assert np.max(np.abs(exact_gf(aim, E0, psi, H_p, H_h, z, 0, 1) - ref)) / np.max(np.abs(ref)) < 1e-6
