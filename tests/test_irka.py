import numpy as np
import pytest
import scipy.linalg as la
import scipy.sparse as sps
import scipy.sparse.linalg as spla

from morec import AIM
from morec.bath import base_bath
from morec.ec import canonical_gs
from morec.irka import krylov_system, select_frequencies


def test_irka_fixed_point():
    aim, eta = AIM(6, 1), 0.08
    H_gs, H_p, H_h = aim.hamiltonians(4.0, 0.0, base_bath(5, 1))
    E0, psi = canonical_gs(H_gs)
    z = select_frequencies(H_p, H_h, H_gs, E0, sps.csc_matrix(psi[:, None]), 4, aim.c_dag, aim.c_ann,
                           eta=eta, verbose=False)
    A, b = krylov_system(H_p, H_h, E0, psi, aim.c_dag[0], aim.c_ann[0])  # exact here: 300 steps >= d_p
    lu = [spla.splu((zi * sps.eye(A.shape[0], format="csc") - A).tocsc()) for zi in z]
    Q = la.qr(np.column_stack([f.solve(b.astype(complex)) for f in lu]), mode="economic")[0]
    Ar, Er, br = Q.T @ (A @ Q), Q.T @ Q, Q.T @ b
    mirror = np.conj(la.eigvals(Ar, Er)) + 2j * eta
    assert max(np.min(np.abs(mirror - zi)) for zi in z) < 1e-5  # points = mirrored reduced poles
    for zi, f in zip(z, lu):  # Hermite interpolation: G and G' match at every point
        x, xr = f.solve(b.astype(complex)), la.solve(zi * Er - Ar, br)
        assert np.isclose(b @ x, br @ xr, rtol=1e-8)
        assert np.isclose(x @ x, xr @ (Er @ xr), rtol=1e-6)


def test_irka_single_orbital_only():
    with pytest.raises(NotImplementedError):
        select_frequencies(None, None, None, 0.0, None, 3, [0, 1], [0, 1])
