# morec: MOR+EC reduced-order impurity solver

Reduced-order models of single-particle Green's functions that combine eigenvector
continuation (EC) in parameter space with model order reduction (MOR) in frequency, applied
as an impurity solver for dynamical mean-field theory (DMFT) on the Bethe lattice.  This is
the code behind *Compact representation of strongly correlated Green's functions: the MOR+EC
way to explore phase space* (N. Hogan, A. F. Kemper, C. Mejuto-Zaera),
[arXiv:2609.28647](https://arxiv.org/abs/2609.28647).

## Install

```bash
pip install -e .              # numpy, scipy, psutil
pip install -e ".[paper]"     # + matplotlib, to regenerate the figures
pip install -e ".[test]"      # + pytest
```

## Quick start

```python
from morec import BuildConfig, DMFTConfig, EnrichConfig, MORSolver, build, sweep, sweep_values

# Offline stage: greedy EC basis over a Sobol grid, then MOR projections at r frequencies.
build(BuildConfig(N=8, NI=2, r=3, ec_tol=1e-2, bw_lo=[1.0, 0.01], bw_hi=[1.0, 0.75]), "atlas")

# Online stage (optional): a coarse DMFT sweep that enriches the ROM where it is inaccurate.
cfg = DMFTConfig(N=8, NI=2, bw_ratios=[1.0, 0.1])
solver = MORSolver("atlas", bw_ratios=cfg.bw_ratios, enrich=EnrichConfig(variance_tol=1e-4))
sweep(cfg, solver, "runs/coarse", sweep_values(0, 10, 1, clamp_min=0.01), bwr_vals=sweep_values(0.1, 0.5, 0.1))

# Fine scan entirely inside the ROM.
solver = MORSolver("atlas", bw_ratios=cfg.bw_ratios)
sweep(cfg, solver, "runs/fine", sweep_values(0, 8, 0.1, clamp_min=0.01), bwr_vals=sweep_values(0.1, 0.5, 0.01))
```

The same from the command line:

```bash
morec build atlas --N 8 --NI 2 --r 3 --ec_tol 1e-2 --bw_lo 1 0.01 --bw_hi 1 0.75
morec dmft runs/coarse --N 8 --NI 2 --bw_ratios 1 0.1 --atlas atlas --U 0 10 1 --bwr 0.1 0.5 0.1 --enrich
morec dmft runs/fine --N 8 --NI 2 --bw_ratios 1 0.1 --atlas atlas --U 0 8 0.1 --bwr 0.1 0.5 0.01
morec dmft runs/ed --N 8 --NI 2 --bw_ratios 1 0.1 --solver ED --U 0 10 1 --bwr 0.1 0.5 0.1
```

Axis options take one value or `lo hi step`; `--mu` sets an absolute chemical potential
(default half filling), `--JU` a Hund coupling proportional to U.  Every sweep point is written
to `<outdir>/[bwr*/]J*/U*/[mu*/]` (`dmft_log.txt`, `dmft_metadata.json`, `gf_sigma_current.npz`,
`bath_history.npz`); interrupted sweeps resume where they stopped.

## How the code maps onto the method

| Paper | Code |
|---|---|
| H(p) = sum_q p_q H_q in the ground-state, particle and hole sectors | `hamiltonian.AIM` (affine operators `U`, `J`, `dmu`, `eb{i}.{j}`, `vb{i}.{j}`) |
| Offline greedy EC basis, Var(p) criterion (App. A) | `ec.greedy_grow`, `ec.variance`, called by `atlas.build` |
| Reference resolvent R0(z) = (z + E_c - H0)^-1, projections V (Eq. 6, App. D) | `resolvent.ResolventSolver`, `mor.build_pair`; `BuildConfig(projector="lanczos")` uses `resolvent.ShiftedLanczos` instead (no factorization, for large sectors) |
| Projected H_hat, I_hat, a, b (Eq. 7) | `mor.coupling_vecs`, `ec.projected_terms` |
| ROM GF by Lanczos on the whitened H_hat | `mor.gf_galerkin`, `atlas.Patch.gf` |
| Online enrichment (Ritz trigger, local greedy, App. A) | `dmft.MORSolver.try_enrich`, `atlas.Patch.enrich` (`EnrichConfig`) |
| DMFT self-consistency (App. B) | `dmft.run_point`, `bethe` |
| ED reference solver | `ed.EDSolver` (Green's function by `lanczos.haydock_gf`, the recursion the ROM uses) |
| IRKA frequencies (SIAM demo, Fig. 5) | `irka.select_frequencies` |

An atlas directory holds `atlas.json` and one directory per patch (`meta.json`, the EC basis
`B.npz`, projected terms `h_r.npz`, per orbital pair `pair_i_j.npz`, and the projectors
`V_p_i_j.npz`/`V_h_i_j.npz`, which are only needed for enrichment).  B and V are stored as
scipy.sparse matrices.  Online enrichment rewrites the atlas in place, so work on a copy; it
appends the new projector columns as extra files (`V_p_i_j.addNNNNNN.npz`), which are read
together with the base file up to the column count recorded in `meta.json`.

## Reproducing the paper

```bash
cd paper
python make_figures.py          # Figs. 4-11 and Table I from paper/data (2 MB) -> paper/figures/
```

`paper/data` was reduced from the production runs by `extract_data.py --layout production`.  To
regenerate everything from scratch, run the campaigns and extract again:

```bash
export MOREC_RUNS=/scratch/runs
python campaigns/run_campaign.py n8ni2 build     # Figs. 6, 9, 10, 11, Table I
python campaigns/run_campaign.py n8ni2 enrich
python campaigns/run_campaign.py n8ni2 post
python campaigns/run_campaign.py n8ni2 ed
python campaigns/run_campaign.py n10ni1 ...      # Figs. 7, 8 (same steps)
python campaigns/run_campaign.py siam_demo build # Fig. 5
python paper/extract_data.py --raw $MOREC_RUNS --layout campaign fig5 fig6_compute fig6 fig7_8 fig9 fig10 fig11
```

Figs. 1-3 are drawings and are not generated.  `finite_j` (campaign and figures) covers the
finite-Hund's-coupling runs that are not part of the paper.  Exact figure reproduction needs
matplotlib 3.10 (pinned in the `paper` extra; later versions choose different log-axis ticks) and
`latex` on the PATH (otherwise the text falls back to matplotlib's fonts). 

## Notes and limitations

- Timings: ED and the ROM evaluate their Green's functions with the same Lanczos recursion
  (`lanczos.haydock_gf`), so per-call times compare like with like.  The ROM's gain is mostly in the
  ground state and grows with the size of the impurity model.
- Bath pole order: bath sites are interchangeable, but a reordered bath is a different parameter
  point for the reduced model.  Every bath is therefore brought to one order (each side of zero in
  ascending |energy|, `bath.canonical_bath`) before ED or the ROM uses it, offline and online, and
  the doped bath fit returns its poles in that order.  Results for models with one free pole per
  side (the paper's N=8, NI=2 runs) are unaffected.  Atlases built before this was added remain
  usable but were trained and enriched on unordered baths.
- `Patch.compress` (a second projection of the ROM onto the directions used at a given set of
  parameter points) is experimental and is not used in the paper.
- Multi-patch atlases (`n_patches_*`, `k_local`, `halo`) and off-diagonal G_ij are carried over
  from development and have not been validated; using them emits a warning.  For the models
  here (no inter-orbital hopping) G_ij vanishes for i != j, so off-diagonal pairs are dropped.
  The solvers and the DMFT loop assume an orbital-diagonal G; `AIM` warns if a term added to it
  couples the orbitals.

## License

MIT, see `LICENSE`.
