"""Bath sites are interchangeable: the order of the poles must not matter to either solver."""
import numpy as np

from morec import AIM
from morec.bath import base_bath, canonical_bath, full_bath, sample_baths
from morec.bethe import fit_bath_general
from morec.dmft import make_solver
from morec.ec import canonical_gs

# one orbital, two poles on each side of zero: [central, +, +, -, -]
EB = np.array([0.0, 0.2, 0.9, -0.3, -1.4])
VB = np.array([0.30, 0.25, 0.40, 0.35, 0.45])
SWAP = [0, 2, 1, 4, 3]


def test_relabelling_bath_sites_is_a_symmetry():
    """Coefficients written out by hand in another site order: another matrix, the same physics."""
    aim = AIM(6, 1)
    th = aim.theta(2.0, 0.0, [(EB, VB), 0.4])
    th_swapped = dict(th)
    for j, k in enumerate(SWAP):
        th_swapped[f"eb0.{j}"], th_swapped[f"vb0.{j}"] = th[f"eb0.{k}"], th[f"vb0.{k}"]
    H, H_swapped = aim.hamiltonian("gs", th), aim.hamiltonian("gs", th_swapped)
    assert abs(H - H_swapped).max() > 0.1
    assert abs(canonical_gs(H)[0] - canonical_gs(H_swapped)[0]) < 1e-10


def test_solvers_see_one_parameter_point_per_bath():
    aim = AIM(6, 1)
    assert aim.theta(2.0, 0.0, [(EB, VB), 0.4]) == aim.theta(2.0, 0.0, [(EB[SWAP], VB[SWAP]), 0.4])
    iw = 1j * np.pi / 50.0 * (2 * np.arange(8) + 1)
    solver = make_solver("ED", 6, 1)
    G, E0, n = solver.solve(2.0, 0.0, [(EB, VB), 0.4], iw)[:3]
    G_s, E0_s, n_s = solver.solve(2.0, 0.0, [(EB[SWAP], VB[SWAP]), 0.4], iw)[:3]
    assert E0 == E0_s and np.array_equal(n, n_s) and np.array_equal(G, G_s)


def test_canonical_order_leaves_sorted_baths_alone():
    eb, vb = canonical_bath(EB[None], VB[None])
    assert np.array_equal(eb[0], EB) and np.array_equal(vb[0], VB)
    eb, vb = canonical_bath(EB[SWAP][None], VB[SWAP][None])
    assert np.array_equal(eb[0], EB) and np.array_equal(vb[0], VB)
    # a particle-hole symmetric (mirrored) bath stays mirrored, whatever order its poles came in
    for bath in sample_baths(64, 9, 1, base_bath(9, 1), 0.4, seed=1):
        eb, vb = canonical_bath(*full_bath(bath, 9))
        assert np.all(np.diff(eb[0, 1:5]) > 0) and np.array_equal(eb[0, 1:5], -eb[0, 5:]) and np.array_equal(vb[0, 1:5], vb[0, 5:])


def test_general_fit_returns_both_sides_sorted():
    iw = 1j * np.pi / 50.0 * (2 * np.arange(200) + 1)
    t = 1.0
    ep, em, vp, vm = np.array([0.15, 0.8]), np.array([0.25, 1.3]), np.array([0.35, 0.45]), np.array([0.30, 0.50])
    v0 = np.sqrt(t**2 - np.sum(vp**2 + vm**2))
    Delta = v0**2 / iw + sum(vp[j]**2 / (iw - ep[j]) + vm[j]**2 / (iw + em[j]) for j in range(2))
    # warm start with the negative poles paired the other way round
    eb0 = np.array([0.0, 0.2, 0.7, -1.2, -0.3])
    vb0 = np.array([0.1, 0.3, 0.4, 0.5, 0.3])
    eb, vb = fit_bath_general(Delta, iw, 2, eb0, vb0, t)
    assert np.all(np.diff(eb[1:3]) > 0) and np.all(np.diff(-eb[3:]) > 0)
    assert np.allclose(eb, [0.0, 0.15, 0.8, -0.25, -1.3], atol=1e-4) and np.allclose(vb, [v0, 0.35, 0.45, 0.30, 0.50], atol=1e-4)
