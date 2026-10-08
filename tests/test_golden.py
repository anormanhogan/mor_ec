"""Regression against values produced by the original implementation (same configuration).

The original did not order the bath poles, so its ROM is reproduced with the ordering switched
off (atlas_dir_unsorted); the exact reference Green's function must not depend on it.
"""
import json
import os

import numpy as np

from morec import Atlas

GOLDEN = json.load(open(os.path.join(os.path.dirname(__file__), "data", "golden_n6.json")))


def test_offline_build_matches_original(atlas_dir_unsorted):
    patch = Atlas(atlas_dir_unsorted).patches[0]
    patch.load()
    assert patch.B.shape[1] == GOLDEN["k"]
    assert abs(patch.E_c - GOLDEN["E_c"]) < 1e-8
    assert np.allclose(np.sort(patch.omega.imag), GOLDEN["omega_imag"], rtol=1e-12)
    assert patch.meta["m"]["0,0"] == GOLDEN["m"]


def test_validation_gfs_match_original(atlas_dir_unsorted):
    t = np.load(os.path.join(atlas_dir_unsorted, "test_0", "results.npz"))
    assert abs(float(t["U_test"]) - GOLDEN["U_test"]) < 1e-12
    for key, tol in (("G_ed", 1e-10), ("G_rom", 1e-6)):
        ref = np.array([complex(*z) for z in GOLDEN[key]])
        new = (t["G_fom"] if key == "G_ed" else t["G_rom1"])[0, 0][:len(ref)]
        assert np.max(np.abs(new - ref)) / np.max(np.abs(ref)) < tol


def test_exact_gf_does_not_depend_on_pole_order(atlas_dir):
    """Standard build (canonical order): the same physical test point, so the same exact G."""
    t = np.load(os.path.join(atlas_dir, "test_0", "results.npz"))
    ref = np.array([complex(*z) for z in GOLDEN["G_ed"]])
    assert np.max(np.abs(t["G_fom"][0, 0][:len(ref)] - ref)) / np.max(np.abs(ref)) < 1e-10
    assert np.max(np.abs(t["G_rom1"][0, 0][:len(ref)] - ref)) / np.max(np.abs(ref)) < 1e-3
