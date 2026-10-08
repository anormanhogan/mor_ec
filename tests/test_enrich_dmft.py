import json
import os
import shutil

import numpy as np
import pytest

from morec import AIM, DMFTConfig, EDSolver, EnrichConfig, MORSolver, run_point

IW = 1j * np.pi * (2 * np.arange(100) + 1) / 232.0
BATH = [(np.array([0.3, 0.9]), np.array([0.45, 0.35, 0.3]))] + [0.0]


def rel(a, b):
    return np.max(np.abs(a - b)) / np.max(np.abs(b))


@pytest.mark.parametrize("resolvent", ["local", "reference"])
def test_enrichment_grows_saves_and_reloads(atlas_dir, tmp_path, resolvent):
    work = str(tmp_path / "atlas")
    shutil.copytree(atlas_dir, work)
    solver = MORSolver(work, enrich=EnrichConfig(tol=1e-6, resolvent=resolvent), log=lambda *a: None)
    G_ed = EDSolver(AIM(6, 1)).solve(7.5, 0.0, BATH, IW)[0]
    err_before = rel(solver.solve(7.5, 0.0, BATH, IW)[0], G_ed)
    k0 = solver.nearest.B.shape[1]
    assert solver.try_enrich(7.5, 0.0, BATH)
    patch = solver.nearest
    B = patch.B.toarray()
    assert B.shape[1] > k0 and np.max(np.abs(B.T @ B - np.eye(B.shape[1]))) < 1e-10
    G_mem = solver.solve(7.5, 0.0, BATH, IW)[0]
    assert rel(G_mem, G_ed) < err_before
    G_disk = MORSolver(work).solve(7.5, 0.0, BATH, IW)[0]
    assert rel(G_disk, G_mem) < 1e-12
    assert len(json.load(open(os.path.join(patch.path, "meta.json")))["enrichment_history"]) == 1


def test_dmft_point_runs_with_both_solvers(atlas_dir, tmp_path):
    cfg = DMFTConfig(N=6, NI=1, bw_ratios=[1.0], n_iw=100, n_iter_max=4, bath_n_starts=1)
    for name, solver in (("ed", EDSolver(AIM(6, 1))), ("mor", MORSolver(atlas_dir))):
        out = tmp_path / name / "J0" / "U2"
        run_point(cfg, solver, 2.0, 0.0, [1.0], out)
        meta = json.load(open(out / "dmft_metadata.json"))
        assert meta["n_iter_done"] >= 1 and "end_time" in meta
        assert np.load(out / "gf_sigma_current.npz")["G_imp"].shape == (1, 100)
