import json
import shutil

import numpy as np
import pytest

from morec import DMFTConfig, EnrichConfig, MORSolver, run_point, sweep

IW = 1j * np.pi * (2 * np.arange(100) + 1) / 232.0
CFG = DMFTConfig(N=6, NI=1, bw_ratios=[1.0], n_iw=100, n_iter_max=6, bath_n_starts=1)
U_COARSE = [1.0, 2.0, 3.0, 4.0]


def rel(a, b):
    return np.max(np.abs(a - b)) / np.max(np.abs(b))


def final_bath(sweep_dir, U):
    h = np.load(sweep_dir / "J0" / f"U{U:g}" / "bath_history.npz")
    return [(h["eb"][-1, 0], h["vb"][-1, 0])] + [0.0]


@pytest.fixture(scope="module")
def coarse(atlas_dir, tmp_path_factory):
    """A short DMFT sweep; the baths it visited train the compression."""
    out = tmp_path_factory.mktemp("coarse")
    sweep(CFG, MORSolver(atlas_dir), out, U_COARSE)
    return out


def test_compressed_rom_is_smaller_and_reproduces_the_sweep(atlas_dir, coarse):
    full, comp = MORSolver(atlas_dir), MORSolver(atlas_dir)
    sizes = comp.compress(coarse, tol=1e-6)
    full.solve(1.0, 0.0, final_bath(coarse, 1.0), IW)  # loads the patch
    m_full = {s: sd["H_stack"].shape[1] for s, sd in full.atlas.patches[0].orth[(0, 0)].items()}
    assert all(sizes[(0, 0)][s] < m_full[s] for s in "ph")
    for U in U_COARSE:
        a, b = full.solve(U, 0.0, final_bath(coarse, U), IW), comp.solve(U, 0.0, final_bath(coarse, U), IW)
        assert rel(b[0], a[0]) < 1e-5
        assert b[1] == a[1] and np.array_equal(b[2], a[2])  # ground state energy and occupation
    eb, vb = (np.mean([final_bath(coarse, U)[0][i] for U in (2.0, 3.0)], axis=0) for i in (0, 1))
    assert rel(comp.solve(2.5, 0.0, [(eb, vb), 0.0], IW)[0], full.solve(2.5, 0.0, [(eb, vb), 0.0], IW)[0]) < 1e-3


def test_compressed_solver_uses_the_full_rom_off_the_imaginary_axis(atlas_dir, coarse):
    full, comp = MORSolver(atlas_dir), MORSolver(atlas_dir)
    comp.compress(coarse, tol=1e-4)
    w = np.linspace(-6, 6, 200) + 0.05j
    bath = final_bath(coarse, 2.0)
    assert np.array_equal(comp.solve(2.0, 0.0, bath, w)[0], full.solve(2.0, 0.0, bath, w)[0])
    assert rel(comp.solve(2.0, 0.0, bath, IW)[0], full.solve(2.0, 0.0, bath, IW)[0]) > 1e-8  # compressed here


def test_dmft_point_records_the_compression(atlas_dir, coarse, tmp_path):
    solver = MORSolver(atlas_dir)
    solver.compress(coarse, tol=1e-6)
    run_point(CFG, solver, 2.5, 0.0, [1.0], tmp_path)
    assert json.load(open(tmp_path / "dmft_metadata.json"))["compression"]["tol"] == 1e-6


def test_enrichment_restores_the_full_rom(atlas_dir, coarse, tmp_path):
    work = str(tmp_path / "atlas")
    shutil.copytree(atlas_dir, work)
    solver = MORSolver(work, enrich=EnrichConfig(tol=1e-6), log=lambda *a: None)
    solver.compress(coarse, tol=1e-4)
    bath = [(np.array([0.3, 0.9]), np.array([0.45, 0.35, 0.3]))] + [0.0]
    solver.solve(7.5, 0.0, bath, IW)
    assert solver.try_enrich(7.5, 0.0, bath)
    patch = solver.atlas.patches[0]
    assert solver.compression is None and patch.orth is patch._orth_full


def test_training_set_is_every_visited_bath_unless_final_only(atlas_dir, coarse):
    n_iters = sum(len(np.load(f)["eb"]) for f in coarse.rglob("bath_history.npz"))
    solver = MORSolver(atlas_dir)
    solver.compress(coarse)
    assert solver.compression["n_baths"] == n_iters > len(U_COARSE)
    solver.compress(coarse, final_only=True)
    assert solver.compression["n_baths"] == len(U_COARSE)


def test_compress_needs_points_of_the_same_model(atlas_dir, tmp_path):
    with pytest.raises(ValueError):
        MORSolver(atlas_dir).compress(tmp_path)
