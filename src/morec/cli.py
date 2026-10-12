"""Command line interface.

  morec build ATLAS_DIR --N 8 --NI 2 [--r 3 --ec_tol 1e-2 --bw_lo 1 0.01 --bw_hi 1 0.75 ...]
  morec dmft OUTDIR --N 8 --NI 2 --solver MOR --atlas ATLAS_DIR --U 0 10 0.25 --bwr 0.1 0.5 0.05
             [--enrich --enrich_variance_tol 1e-4] [--JU 0 0.5 0.05 | --J 0.5] [--mu -3 4 0.25]

Axis options take one value or `lo hi step`.  Every BuildConfig / DMFTConfig / EnrichConfig
field is available as --<field> (enrichment fields as --enrich_<field>).
"""
import argparse
import dataclasses

from .dmft import DMFTConfig, EnrichConfig, make_solver, sweep, sweep_values
from .atlas import BuildConfig, build


def _add_fields(p, cls, prefix="", skip=()):
    for f in dataclasses.fields(cls):
        if f.name in skip or f.name == "extra":
            continue
        name = f"--{prefix}{f.name}"
        default = None if f.default is dataclasses.MISSING else f.default
        if isinstance(default, bool):
            p.add_argument(name, action=argparse.BooleanOptionalAction, default=default)
        elif f.type is list:
            p.add_argument(name, type=float, nargs="+", default=default)
        else:
            p.add_argument(name, type=f.type, default=default, required=default is None)


def _config(cls, args, prefix=""):
    return cls(**{f.name: getattr(args, prefix + f.name) for f in dataclasses.fields(cls)
                  if hasattr(args, prefix + f.name)})


def _axis(values, clamp_min=None):
    if values is None:
        return None
    if len(values) == 1:
        return [values[0]]
    if len(values) != 3:
        raise SystemExit("axis options take one value or `lo hi step`")
    return sweep_values(*values, clamp_min=clamp_min)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="morec", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="offline stage: EC basis and MOR projections")
    b.add_argument("atlas_dir")
    _add_fields(b, BuildConfig)

    d = sub.add_parser("dmft", help="DMFT sweep with an ED or MOR+EC impurity solver")
    d.add_argument("outdir")
    d.add_argument("--solver", choices=["ED", "MOR"], default="MOR")
    d.add_argument("--atlas", help="atlas directory (MOR solver)")
    d.add_argument("--U", type=float, nargs="+", required=True)
    d.add_argument("--U_min", type=float, default=0.01, help="U values below this are raised to it")
    d.add_argument("--J", type=float, nargs="+", help="absolute Hund coupling (default 0)")
    d.add_argument("--JU", type=float, nargs="+", help="Hund coupling as a ratio J/U")
    d.add_argument("--bwr", type=float, nargs="+", help="bandwidth ratio of the last orbital")
    d.add_argument("--mu", type=float, nargs="+", help="absolute chemical potential (default half filling)")
    d.add_argument("--reverse", action="store_true", help="walk every axis backwards")
    d.add_argument("--halo", type=int, default=1, help="patches averaged per solve (multi-patch only)")
    d.add_argument("--interp", choices=["gf", "sigma"], default="gf")
    d.add_argument("--off_diagonal", action="store_true", help="also evaluate G_ij, i != j (untested)")
    d.add_argument("--enrich", action="store_true", help="enrich the atlas online (modifies it in place)")
    d.add_argument("--compress", metavar="SWEEP_DIR",
                   help="compress the ROM on the baths of this finished sweep (MOR solver without --enrich)")
    d.add_argument("--compress_tol", type=float, default=1e-5)
    d.add_argument("--n_workers", type=int, default=1)
    _add_fields(d, DMFTConfig)
    _add_fields(d, EnrichConfig, prefix="enrich_")

    args = ap.parse_args(argv)
    if args.cmd == "build":
        build(_config(BuildConfig, args), args.atlas_dir)
        return

    cfg = _config(DMFTConfig, args)
    kw = {}
    if args.solver == "MOR":
        kw = dict(halo=args.halo, interp=args.interp, diagonal_only=not args.off_diagonal,
                  gf_tol=cfg.gf_tol, n_workers=args.n_workers, bw_ratios=cfg.bw_ratios,
                  enrich=_config(EnrichConfig, args, "enrich_") if args.enrich else None)
    solver = make_solver(args.solver, cfg.N, cfg.NI, atlas_dir=args.atlas, **kw)
    if args.compress:
        if args.solver != "MOR" or args.enrich:
            raise SystemExit("--compress needs the MOR solver without --enrich")
        solver.compress(args.compress, args.compress_tol)
    if args.JU is not None:
        levels = [(f"JU{r:.4g}", lambda U, r=r: r * U) for r in _axis(args.JU)]
    else:
        levels = [(f"J{J:.4g}", lambda U, J=J: J) for J in (_axis(args.J) or [0.0])]
    sweep(cfg, solver, args.outdir, _axis(args.U, args.U_min), levels,
          _axis(args.bwr), _axis(args.mu), args.reverse)


if __name__ == "__main__":
    main()
