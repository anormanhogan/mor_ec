import pytest

from morec import BuildConfig, build

GOLDEN_CFG = dict(N=6, NI=1, r=4, U_lo=0.01, U_hi=6.0, ec_tol=1e-4, ec_grid_size=256, k_max=100,
                  n_workers=1, n_test=2, n_test_w=20)


@pytest.fixture(scope="session")
def atlas_dir(tmp_path_factory):
    """Small N=6 single-band ROM, built once per test session."""
    path = tmp_path_factory.mktemp("atlas") / "n6"
    build(BuildConfig(**GOLDEN_CFG), str(path), verbose=False)
    return str(path)


@pytest.fixture(scope="session")
def atlas_dir_unsorted(tmp_path_factory):
    """The same ROM with the canonical pole order switched off, as the original implementation
    built it: some of its training baths have neighbouring poles the other way round."""
    path = tmp_path_factory.mktemp("atlas_unsorted") / "n6"
    patch = pytest.MonkeyPatch()
    patch.setattr("morec.hamiltonian.canonical_bath", lambda eb, vb: (eb, vb))
    try:
        build(BuildConfig(**GOLDEN_CFG), str(path), verbose=False)
    finally:
        patch.undo()
    return str(path)
