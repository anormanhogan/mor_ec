"""Reproduce the paper figures (and Table I) from the data bundle in paper/data.

    python make_figures.py              # all figures -> paper/figures/
    python make_figures.py fig7 fig10   # a subset

Figs. 1-3 are hand-drawn and not generated here.  Speed-ups are the median over parameter
points of the per-call time ratio (first call of each run dropped), with 5th-95th percentile
whiskers; see `speedup`.
"""
import csv
import json
import os
import shutil
import sys
from collections import namedtuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.gridspec import GridSpec
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Rectangle
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, OUT = os.path.join(HERE, "data"), os.path.join(HERE, "figures")
TEX = dict(text_usetex=bool(shutil.which("latex")))
SERIF = {"text.usetex": TEX["text_usetex"], "font.family": "serif"}
W1 = 4.0  # full bandwidth of the wide band (half-bandwidth 2)


def _json(name):
    with open(os.path.join(DATA, name)) as f:
        return json.load(f)


def _save(fig, name, **kw):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"wrote figures/{name}")


def _label_panels(axes, fontsize=14, xloc=-0.15, yloc=1.04):
    for i, ax in enumerate(axes):
        ax.text(xloc, yloc, rf"$\mathrm{{{chr(ord('a') + i)})}}$", transform=ax.transAxes,
                fontsize=fontsize, fontweight="bold", va="bottom", ha="left")


# ----------------------------------------------------------------------------- speed-up statistics

SpeedUp = namedtuple("SpeedUp", "median lo hi aggregate n_points n_ed_calls n_mor_calls")


def speedup(points, drop_first=True):
    """Median/p5/p95 over points of mean(ED s/call) / mean(MOR+EC s/call), plus the pooled ratio.

    points: iterable of (ed_times, mor_times) per shared parameter point.  The first call of
    each run is dropped (MOR+EC loads its atlas lazily on the first call).
    """
    ratios, ed_all, mor_all = [], [], []
    for ed_t, mor_t in points:
        ed_t, mor_t = np.asarray(ed_t, float), np.asarray(mor_t, float)
        if drop_first:
            ed_t, mor_t = ed_t[1:], mor_t[1:]
        if ed_t.size == 0 or mor_t.size == 0 or ed_t.mean() <= 0 or mor_t.mean() <= 0:
            continue
        ratios.append(ed_t.mean() / mor_t.mean())
        ed_all.append(ed_t)
        mor_all.append(mor_t)
    if not ratios:
        return None
    ratios, ed_all, mor_all = np.array(ratios), np.concatenate(ed_all), np.concatenate(mor_all)
    return SpeedUp(float(np.median(ratios)), float(np.percentile(ratios, 5)), float(np.percentile(ratios, 95)),
                   float(ed_all.mean() / mor_all.mean()), ratios.size, ed_all.size, mor_all.size)


def fmt(s):
    f = lambda x: ("%.1f" if x < 10 else "%.0f") % x
    return "n/a" if s is None else f"{f(s.median)}x [{f(s.lo)}-{f(s.hi)}]"


def _sum_calls(a, b):
    return list(np.asarray(a[0]) + np.asarray(a[1])), list(np.asarray(b[0]) + np.asarray(b[1]))


# ----------------------------------------------------------------------------- Fig. 4

def fig4():
    """Schematic of the one- and two-band impurity models."""
    IMP_FC, IMP_EC, BATH_FC, BATH_EC = "#B98BD9", "#5B2C83", "#FFFFFF", "#3A5F8F"
    BOND, WIG, INK = "#2B2D42", "#C1420B", "#1A1A1A"
    R_IMP, R_BATH, CY = 3.4, 2.2, 25.0

    def wiggle(p0, p1, amp, ncyc):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        d = p1 - p0
        n = np.array([-d[1], d[0]]) / np.hypot(*d)
        t = np.linspace(0.0, 1.0, 400)
        return p0 + np.outer(t, d) + np.outer(amp * np.sin(ncyc * np.pi * t), n)

    def arc_wiggle(cx, cy, r, a0, a1, amp, ncyc):
        th, s = np.linspace(np.radians(a0), np.radians(a1), 400), np.linspace(0.0, 1.0, 400)
        rr = r + amp * np.sin(ncyc * np.pi * s)
        return np.column_stack([cx + rr * np.cos(th), cy + rr * np.sin(th)])

    def fan(cx, cy, rx, ry, angles):
        a = np.radians(np.asarray(angles, float))
        return np.column_stack([cx + rx * np.cos(a), cy + ry * np.sin(a)])

    with plt.rc_context({"text.usetex": TEX["text_usetex"], "font.family": "serif",
                         "mathtext.fontset": "dejavuserif", "svg.fonttype": "none"}):
        fig = plt.figure(figsize=(3.375, 3.375 * 52.0 / 100.0))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 100.0)
        ax.set_ylim(0, 52.0)
        ax.set_aspect("equal")
        ax.axis("off")
        draw = lambda pts, lw=1.0: ax.plot(pts[:, 0], pts[:, 1], color=WIG, lw=lw, solid_capstyle="round", zorder=6)
        imp = lambda x, y: ax.add_patch(Circle((x, y), R_IMP, facecolor=IMP_FC, edgecolor=IMP_EC, lw=1.3, zorder=5))
        bath = lambda x, y: ax.add_patch(Circle((x, y), R_BATH, facecolor=BATH_FC, edgecolor=BATH_EC, lw=1.1, zorder=5))
        bond = lambda p0, p1: ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=BOND, lw=0.9,
                                      solid_capstyle="round", zorder=4)

        def txt(x, y, s, fs, color=INK, bg=False):
            ax.text(x, y, s, ha="center", va="center", fontsize=fs, color=color, zorder=8,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.95) if bg else None)

        cx = 22.0
        sites = fan(cx, CY, 17.0, 17.0, np.linspace(-72, 72, 9))
        for s in sites:
            bond((cx, CY), s)
        for s in sites:
            bath(*s)
        imp(cx, CY)
        draw(arc_wiggle(cx, CY, R_IMP + 2.6, 108, 252, 0.7, 7))
        txt(cx - 10.4, CY, r"$U$", 8.0, color=WIG)
        txt(cx + 9.0, CY, r"$V_b$", 8.0, color=BOND, bg=True)
        txt(cx + 20.6, CY + 12.0, r"$\varepsilon_b$", 8.0, color=BATH_EC)

        lx, rx = 65.0, 84.0
        for c, sites in ((lx, fan(lx, CY, 11.5, 13.0, [140, 180, 220])), (rx, fan(rx, CY, 11.5, 13.0, [40, 0, -40]))):
            for s in sites:
                bond((c, CY), s)
            for s in sites:
                bath(*s)
        for c in (lx, rx):
            imp(c, CY)
            draw(arc_wiggle(c, CY, R_IMP + 2.2, 40, 140, 0.6, 5))
            txt(c, CY + 11.4, r"$U$", 8.0, color=WIG)
        draw(wiggle((lx + R_IMP, CY), (rx - R_IMP, CY), 1.0, 5), lw=1.1)
        txt((lx + rx) / 2, CY - 6.6, r"$U' = U - 2J,\ J$", 8.0, color=WIG, bg=True)
        txt(22.0, 48.0, r"$\mathrm{(a)}\ 1-\mathrm{Band}$", 9.0)
        txt(74.5, 48.0, r"$\mathrm{(b)}\ 2-\mathrm{Band}$", 9.0)
        os.makedirs(OUT, exist_ok=True)
        fig.savefig(os.path.join(OUT, "fig4_impurity_models.pdf"), dpi=400)
        plt.close(fig)
        print("wrote figures/fig4_impurity_models.pdf")


# ----------------------------------------------------------------------------- Fig. 5

def fig5():
    """SIAM demo: ED spectra (shaded), MOR+EC (dashed) and the IRKA snapshot frequencies."""
    d = np.load(os.path.join(DATA, "fig5_siam_demo.npz"))
    Us, w, n = d["U"], d["w"], len(d["U"])
    order = np.argsort(Us)
    A_ed, A_rom, Us_sorted = d["A_ed"][order], d["A_rom"][order], np.sort(Us)
    highlight = np.argsort(Us)[-3:]  # as in the original figure: indices of the unsorted list
    cmap, shift = plt.get_cmap("GnBu"), 0.3
    style = {"text.usetex": TEX["text_usetex"], "font.size": 15, "axes.labelsize": 17,
             "xtick.labelsize": 15, "ytick.labelsize": 15, "lines.linewidth": 0.8}
    with plt.rc_context(style):
        fig = plt.figure(figsize=(6, 5))
        for t in range(n - 1, -1, -1):
            base_z, base_y = (n - t) * 10, shift * t
            color = cmap(0.4 + 0.6 * t / (n - 1)) if t not in highlight else matplotlib.colors.to_rgba("orange")
            if t == 0:
                plt.text(w[-100], A_ed[t][-100] - 0.3, rf"$U={Us_sorted[t]:.2f}$", zorder=base_z + 6, fontsize=14,
                         color="black" if t not in highlight else "orange")
            elif t == n - 1:
                plt.text(w[-100], A_ed[t][-100] + shift * t + 0.3, rf"$U={Us_sorted[t]:.2f}$", color="black",
                         zorder=base_z + 6, fontsize=14)
            fade = 0.1 if t == 0 else 0.105
            spec_cmap = LinearSegmentedColormap.from_list("fade", [(0.0, "white"), (fade, color), (1.0, color)])
            plt.pcolormesh(np.tile(w, (2, 1)), np.vstack([np.full_like(w, base_y), A_ed[t] + base_y]),
                           np.vstack([np.zeros_like(w), np.ones_like(w)]), shading="gouraud", cmap=spec_cmap,
                           zorder=base_z + 2, antialiased=True)
            plt.plot(w, A_ed[t] + base_y, color=color, linewidth=0.5, alpha=0.7, zorder=base_z + 3)
            plt.plot(w, A_rom[t] + base_y, "--", color="purple", alpha=1, zorder=base_z + 4)
        for t in highlight:
            for w_i in d["omega"].real:
                plt.plot(w_i, np.interp(w_i, w, A_ed[t]) + shift * t, "*", color="magenta", markersize=8,
                         zorder=base_z + 5, markeredgecolor="darkmagenta", markeredgewidth=0.25)
        plt.xlabel(r"$\omega$")
        plt.ylabel(r"$-\frac{1}{\pi}\mathbf{G}^\mathrm{imp}(\omega)$")
        plt.text(w[-40], 1.1, r"$U\ \mathrm{increasing}\ \rightarrow$", fontsize=16, color="black", ha="right",
                 va="bottom", rotation=90, backgroundcolor="white", zorder=500)
        plt.yticks([], [])
        plt.ylim(-0.5, A_ed[-1].max() + (n + 1) * shift)
        plt.xlim(w[0] + 1, w[-1] - 1)

        class Gradient(HandlerBase):
            def create_artists(self, legend, handle, xd, yd, width, height, fontsize, trans):
                return [Rectangle((xd + i * width / 50, yd), width / 50, height,
                                  facecolor=cmap(0.4 + 0.6 * i / 49), edgecolor="none", transform=trans)
                        for i in range(50)]

        proxy = object()
        plt.legend(handles=[proxy, Line2D([0], [0], color="purple", linestyle="--"),
                            Patch(color="orange"),
                            Line2D([0], [0], color="magenta", marker="*", linestyle="", markeredgecolor="darkmagenta",
                                   markersize=8, markeredgewidth=0.25)],
                   labels=[r"$\mathrm{ED}$", r"$\mathrm{MOR+EC}$", r"$\mathrm{ED\ Basis\ State}$", r"$\mathrm{Snapshots}$"],
                   handler_map={proxy: Gradient()}, loc="upper left", fontsize=11, ncols=4, columnspacing=1.0,
                   handletextpad=0.5, borderpad=0.5)
        plt.tight_layout()
        _save(fig, "fig5_siam_demo.pdf", dpi=300)


# ----------------------------------------------------------------------------- Fig. 6 and Table I

def _fig6_data():
    rows = lambda f: sorted([[float(x) for x in r] for r in list(csv.reader(open(os.path.join(DATA, f))))[1:]])
    return rows("fig6_channels_r.csv"), rows("fig6_channels_k.csv"), _json("fig6_timing.json")


def _thr_tex(on, off):
    on_e = round(np.log10(on))
    if off is None:
        return r"$\varepsilon_\mathrm{on}=10^{%d}$" % on_e
    if np.isclose(off, on, rtol=1e-9):
        return r"$\varepsilon_\mathrm{off}=\varepsilon_\mathrm{on}=10^{%d}$" % on_e
    return r"$\varepsilon_\mathrm{off}=10^{%d},\ \varepsilon_\mathrm{on}=10^{%d}$" % (round(np.log10(off)), on_e)


def _config_speed(t, cfg):
    calls = t["configs"][cfg]["calls"]
    pts = [p for p in t["shared"] if p in calls]
    ed, mo = t["ed"], calls
    return (speedup([(ed[p][0], mo[p][0]) for p in pts]), speedup([(ed[p][1], mo[p][1]) for p in pts]),
            speedup([_sum_calls(ed[p], mo[p]) for p in pts]), pts)


def fig6():
    """GF error split into EC ground-state and ROM resolvent channels, vs r and vs k."""
    rows, krows, t = _fig6_data()
    STYLE = [(r"$\mathrm{Total}$", "steelblue", "o", "-"), (r"$\mathrm{EC\ ground\ state}$", "coral", "s", "--"),
             (r"$\mathrm{ROM\ resolvent}$", "forestgreen", "^", "-.")]
    rs = [z[0] for z in rows]
    a_cfgs = {int(z[0]): f"r{int(z[0])}_thr1e-4" for z in rows}
    with plt.rc_context({"text.usetex": TEX["text_usetex"]}):
        fig, (ax, axb) = plt.subplots(2, 1, figsize=(7, 8))
        for i, (lab, color, mk, ls) in enumerate(STYLE):
            med = [z[3 + 3 * i] for z in rows]
            ax.errorbar(rs, med, yerr=[[z[3 + 3 * i] - z[4 + 3 * i] for z in rows],
                                       [z[5 + 3 * i] - z[3 + 3 * i] for z in rows]],
                        fmt=mk, ls=ls, color=color, markersize=8, capsize=5, capthick=1.5, linewidth=2,
                        markeredgecolor="black", markeredgewidth=1.5, label=lab, zorder=3 - i)
        ax.set_yscale("log")
        ax.set_ylim(min(z[10] for z in rows) / 8, max(max(z[5], z[8], z[11]) for z in rows) * 800)
        ax.set_xticks(rs)
        ax.set_xlim(min(rs) - 0.3, max(rs) + 0.3)
        ax.set_xlabel(r"$\mathrm{Number\ of\ interpolation\ points}\ r$", fontsize=18)
        ylab = r"$\bar{\mathcal{E}}^{\mathrm{rel}}(\mathbf{G}^\mathrm{imp}(i\omega_n))$"
        ax.set_ylabel(ylab, fontsize=18)
        ax.tick_params(axis="both", which="major", labelsize=18)
        ax.grid(True, alpha=0.3)
        offs = {t["configs"][c]["ec_tol"] for c in a_cfgs.values()}
        ax.legend(fontsize=16, loc="lower left", framealpha=0.85, title_fontsize=16,
                  title=_thr_tex(1e-4, offs.pop() if len(offs) == 1 else None))
        for z in rows:
            s_tot = _config_speed(t, a_cfgs[int(z[0])])[2]
            ax.annotate(r"$%.0f\times$" % s_tot.median, (z[0], z[3]), textcoords="offset points", xytext=(0, 16),
                        ha="center", va="bottom", fontsize=14, color="steelblue",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.95, edgecolor="steelblue"))

        cb = t["configs"]["r4_thr1e-8"]
        kx = [z[0] for z in krows]
        axb.errorbar(kx, [z[1] for z in krows], yerr=[[z[1] - z[2] for z in krows], [z[3] - z[1] for z in krows]],
                     fmt="s", ls="--", color="coral", markersize=8, capsize=5, capthick=1.5, linewidth=2,
                     markeredgecolor="black", markeredgewidth=1.5)
        axb.text(0.97, 0.95, r"$r=%d,\ %s$" % (cb["r"], _thr_tex(cb["thr"], cb["ec_tol"]).strip("$")),
                 transform=axb.transAxes, ha="right", va="top", fontsize=16,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="0.7"))
        axb.axvline(cb["k_off"], color="0.45", ls=":", lw=1.5, zorder=1)
        axb.text(cb["k_off"] + 0.01 * (max(kx) - min(kx)), 0.06, r"$k_\mathrm{off}=%d$" % cb["k_off"],
                 transform=axb.get_xaxis_transform(), ha="left", va="bottom", fontsize=14, color="0.35")
        axb.set_yscale("log")
        axb.set_xlabel(r"$\mathrm{EC\ basis\ size}\ k$", fontsize=18)
        axb.set_ylabel(ylab, fontsize=18)
        axb.tick_params(axis="both", which="major", labelsize=18)
        axb.grid(True, alpha=0.3)
        _label_panels([ax, axb], fontsize=18, xloc=-0.15)
        fig.tight_layout()
        _save(fig, "fig6_gf_error_channels.pdf")


def table1():
    """Per-call solver cost of MOR+EC vs ED on the Fig. 6 points (LaTeX, figures/table1.tex)."""
    rows, _, t = _fig6_data()
    cfgs = [("(a)", f"r{int(z[0])}_thr1e-4") for z in rows] + [("(b)", "r4_thr1e-8")]
    ms = lambda calls, pts, i: 1e3 * np.mean([x for p in pts for x in calls[p][i][1:]])
    body, prev = [], None
    print(f"{'':4}{'r':>3}{'k_off':>6}{'k':>5}{'m':>6}{'n':>5} | {'GS ms':>7}{'GF ms':>8} | "
          f"{'GS speed-up':>16}{'GF speed-up':>16}{'total':>16}")
    for tag, c in cfgs:
        cfg = t["configs"][c]
        s_gs, s_gf, s_tot, pts = _config_speed(t, c)
        gs, gf = ms(cfg["calls"], pts, 0), ms(cfg["calls"], pts, 1)
        print(f"{tag:4}{cfg['r']:3d}{cfg['k_off']:6d}{cfg['k']:5d}{cfg['m']:6.0f}{len(pts):5d} | {gs:7.2f}{gf:8.2f} | "
              f"{fmt(s_gs):>16}{fmt(s_gf):>16}{fmt(s_tot):>16}")
        if prev is not None and prev != tag:
            body.append(r"    \midrule")
        prev = tag
        n = lambda x: ("%.1f" if x < 10 else "%.0f") % x
        body.append(rf"    {tag} & {cfg['r']} & {cfg['m']:.0f} & {gs:.2f} & {gf:.2f} & "
                    rf"{n(s_gs.median)}$\times$ & {n(s_gf.median)}$\times$ & {n(s_tot.median)}$\times$ \\")
    ed_ms = [ms(t["ed"], [p for p in t["shared"] if p in t["ed"]], i) for i in (0, 1)]
    tex = "\n".join([r"% requires \usepackage{booktabs}", r"\begin{tabular}{@{}llrrrrrr@{}}", r"    \toprule",
                     r"     & & & \multicolumn{2}{c}{MOR+EC (ms)} & \multicolumn{3}{c}{Speedup} \\",
                     r"    \cmidrule(lr){4-5}\cmidrule(lr){6-8}",
                     r"    Panel & $r$ & $m_{p/h}$ & GS & GF & GS & GF & total \\", r"    \midrule", *body,
                     r"    \bottomrule", r"\end{tabular}",
                     f"% ED reference: {ed_ms[0]:.0f} ms GS, {ed_ms[1]:.0f} ms GF per call", ""])
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "table1.tex"), "w").write(tex)
    print("wrote figures/table1.tex")


# ----------------------------------------------------------------------------- Figs. 7 and 8

def _doping():
    d = _json("fig7_8_doping.json")
    return d["points"], d["rom"]


def fig7():
    """Cumulative impurity-solver time vs solver calls: break-even of the ROM build."""
    pts, rom = _doping()
    ed_seq = np.concatenate([np.add(*p["ed"]) for p in pts])
    mo_seq = np.concatenate([np.add(*p["mor"]) for p in pts])
    rom_min = (rom["ec_s"] + rom["mor_s"] + rom["enrich_ec_s"] + rom["enrich_mor_s"]) / 60.0
    ed_cum, mo_cum = np.cumsum(ed_seq) / 60.0, rom_min + np.cumsum(mo_seq) / 60.0
    n_ed, n_mo = len(ed_seq), len(mo_seq)
    with plt.rc_context(SERIF):
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        x_max, y_top = int(max(n_ed, n_mo) * 1.04), ed_cum[-1] * 1.16
        ax.plot(np.arange(1, n_ed + 1), ed_cum, "-", color="k", linewidth=3.0, zorder=4, label=r"\textrm{ED}")
        ax.plot(np.arange(1, n_mo + 1), mo_cum, "-", color="steelblue", linewidth=3.5, zorder=4,
                label=r"\textrm{MOR+EC}")
        for n_, y, c in ((n_ed, ed_cum[-1], "k"), (n_mo, mo_cum[-1], "steelblue")):
            ax.plot([n_], [y], "o", color=c, markersize=12, markeredgecolor="w", markeredgewidth=1.5, zorder=6)
        ax.legend(loc="upper left", fontsize=16, framealpha=0.9)
        x_b = x_max * 0.34
        ax.annotate("", xy=(x_b, 0), xytext=(x_b, rom_min), arrowprops=dict(arrowstyle="<->", color="steelblue", lw=2))
        ax.text(x_b + x_max * 0.025, rom_min * 0.50, rf"\textrm{{ROM build}}: ${rom_min:.0f}$ \textrm{{min}}",
                fontsize=16, color="steelblue", ha="left", va="center")
        n_ov = min(n_ed, n_mo)
        below = np.nonzero(mo_cum[:n_ov] < ed_cum[:n_ov])[0]
        n_be = int(below[0] + 1) if below.size else None
        if n_be is not None:
            ax.plot([n_be, n_be], [0, ed_cum[n_be - 1]], "-.", color="firebrick", linewidth=2, zorder=3)
            ax.text(n_be + x_max * 0.0001 - 500, y_top * 0.32,
                    r"\textrm{break-even}" + "\n" + rf"$n_{{\mathrm{{solve}}}}={n_be}$",
                    fontsize=16, color="firebrick", ha="left", va="center")
        ins = ax.inset_axes([0.57, 0.23, 0.38, 0.27])
        solo = mo_cum - rom_min
        ins.plot(np.arange(1, n_mo + 1), solo, "-", color="steelblue", lw=2.5)
        ins.plot([n_mo], [solo[-1]], "o", color="steelblue", markersize=7, markeredgecolor="w",
                 markeredgewidth=1.0, zorder=6)
        ins.set_title(r"\textrm{MOR+EC solve only}", fontsize=16, color="steelblue", pad=4)
        ins.set_ylabel(r"\textrm{min}", fontsize=16)
        ins.set_xlim(0, x_max)
        ins.set_ylim(0, solo[-1] * 1.15)
        ins.set_xticks([0, 5000])
        ins.tick_params(axis="both", labelsize=16)
        ins.grid(True, alpha=0.3)
        ax.set_xlabel(r"\textrm{Impurity solver calls} $n_{\mathrm{solve}}$", fontsize=18)
        ax.set_ylabel(r"\textrm{Cumulative solver time (min)}", fontsize=18)
        ax.tick_params(axis="both", labelsize=16)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_top)
        plt.tight_layout()
        _save(fig, "fig7_break_even.pdf", dpi=300)
    s_tot = speedup([_sum_calls(p["ed"], p["mor"]) for p in pts])
    print(f"  {len(pts)} points; ED {n_ed} calls {ed_cum[-1]:.0f} min; MOR+EC {n_mo} calls "
          f"{np.sum(mo_seq) / 60:.1f} min + ROM {rom_min:.0f} min ({rom['n_events']} enrichment events); "
          f"break-even n_solve={n_be}; per-call speed-up {fmt(s_tot)}")


def fig8():
    """Doping scan: occupation vs mu/W with per-call speed-up inset (a), GF error map (b)."""
    pts, _ = _doping()
    W = 4.0
    Us, mus = np.array(sorted({p["U"] for p in pts})), np.array(sorted({p["mu"] for p in pts}))
    with plt.rc_context(SERIF | {"hatch.linewidth": 1.0}):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 8.6), sharex=True)
        cax1 = make_axes_locatable(ax1).append_axes("right", size="3.5%", pad=0.10)
        cax2 = make_axes_locatable(ax2).append_axes("right", size="3.5%", pad=0.10)
        unorm = plt.Normalize(vmin=Us.min() / W, vmax=Us.max() / W)
        by_U = {}
        for p in pts:
            by_U.setdefault(p["U"], []).append(p)
        for U in Us:
            ser = sorted(by_U[U], key=lambda p: p["mu"])
            ms = np.array([p["mu"] for p in ser])
            n_e, n_m = np.array([p["n_ed"] for p in ser]), np.array([p["n_mor"] for p in ser])
            ok = ~np.isnan(n_e) & ~np.isnan(n_m)
            col = cm.coolwarm(unorm(U / W))
            ax1.plot(ms[ok] / W, n_e[ok], "-", color=col, lw=1.5)
            ax1.plot(ms[~np.isnan(n_m)] / W, n_m[~np.isnan(n_m)], ".", color=col, alpha=0.8)
        ax1.legend(handles=[Line2D([0], [0], color="k", ls="-", lw=1.5, label=r"\textrm{ED}"),
                            Line2D([0], [0], color="k", ls="", marker=".", lw=1.5, label=r"\textrm{MOR+EC}")],
                   loc="lower left", fontsize=14)
        ax1.set_ylabel(r"$n_\mathrm{imp}$", fontsize=18)
        ax1.tick_params(axis="both", labelsize=16)
        sm = cm.ScalarMappable(cmap=cm.coolwarm, norm=unorm)
        sm.set_array([])
        cb1 = fig.colorbar(sm, cax=cax1)
        cb1.set_label(r"$U/W$", fontsize=18)
        cb1.ax.tick_params(labelsize=16)

        stats = [speedup([(p["ed"][i], p["mor"][i]) for p in pts]) for i in (0, 1)]
        ax_bar = ax1.inset_axes([0.70, 0.50, 0.28, 0.47])
        y_floor = min(s.lo for s in stats) * 0.55
        for pos, s, col in zip([0, 1], stats, ["#d62728", "#1f77b4"]):
            ax_bar.bar(pos, s.median - y_floor, bottom=y_floor, width=0.6,
                       yerr=[[s.median - s.lo], [s.hi - s.median]], capsize=3, ecolor="black", color=col,
                       alpha=0.85, zorder=3)
            ax_bar.annotate(rf"${s.median:.0f}\times$", xy=(pos, s.hi), xytext=(0, 4), textcoords="offset points",
                            ha="center", va="bottom", fontsize=12, fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.95, edgecolor=col))
        ax_bar.set_xticks([0, 1])
        ax_bar.set_xticklabels([r"\textrm{GS}", r"\textrm{GF}"], fontsize=12)
        ax_bar.set_ylabel(r"$\mathrm{Speedup}$" "\n" r"$\mathrm{(ED\ /\ MOR{+}EC)}$", fontsize=12)
        ax_bar.tick_params(axis="y", labelsize=12)
        ax_bar.set_yscale("log")
        ax_bar.set_ylim(y_floor, max(s.hi for s in stats) * 2.2)
        ax_bar.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=12))
        ax_bar.yaxis.set_major_formatter(ScalarFormatter())
        ax_bar.yaxis.set_minor_formatter(NullFormatter())
        ax_bar.set_xlim(-0.9, 1.9)

        iU, imu = {u: i for i, u in enumerate(Us)}, {m: i for i, m in enumerate(mus)}
        E, Nocc = np.full((len(mus), len(Us)), np.nan), np.full((len(mus), len(Us)), np.nan)
        for p in pts:
            E[imu[p["mu"]], iU[p["U"]]] = p["err"]
            Nocc[imu[p["mu"]], iU[p["U"]]] = p["n_ed"]
        pc = ax2.pcolormesh(mus / W, Us / W, E.T, cmap=plt.get_cmap("Blues"), shading="nearest",
                            norm=LogNorm(vmin=np.nanmin(E), vmax=np.nanmax(E)), rasterized=True)
        cb2 = fig.colorbar(pc, cax=cax2)
        cb2.set_label(r"$\bar{\mathcal{E}}^{\mathrm{rel}}(\mathbf{G}^\mathrm{imp}(i\omega_n))$", fontsize=16)
        cb2.ax.tick_params(labelsize=16)
        one = np.abs(Nocc - 1.0) < 1e-2
        wide = one.sum(axis=0, keepdims=True) >= 2
        band = ((np.abs(Nocc) < 1e-2) | (np.abs(Nocc - 2.0) < 1e-2) | (one & wide)).astype(float)
        cs = ax2.contourf(mus / W, Us / W, band.T, levels=[0.5, 1.5], colors="none", hatches=["//"], zorder=3)
        cs.set_edgecolor(matplotlib.colors.to_rgba("firebrick", 0.35))
        cs.set_linewidth(0)
        box = dict(boxstyle="round,pad=0.22", fc="white", ec="firebrick", lw=1.2, alpha=0.92)
        for xy, rot, lab in [((2.6 / W, 1.0), 90, r"$n_\mathrm{{imp}}=0$"), ((-2.8 / W, 0.52), 0, r"$n_\mathrm{{imp}}=2$")]:
            ax2.text(*xy, lab, color="firebrick", fontsize=16, ha="center", va="center", rotation=rot, zorder=5,
                     bbox=box)
        sel = Us < (Us[np.nonzero(wide[0])[0][0]] if wide.any() else np.inf)
        ax2.plot(-Us[sel] / (2.0 * W), Us[sel] / W, "--", color="firebrick", linewidth=2.5, zorder=4)
        ax2.text(-1.95 / W, 1.40, r"$n_\mathrm{{imp}}=1$", color="firebrick", fontsize=16, ha="center",
                 va="center", bbox=box, zorder=5)
        ax2.set_xlabel(r"$\mu/W$", fontsize=18)
        ax2.set_ylabel(r"$U/W$", fontsize=18)
        ax2.tick_params(axis="both", labelsize=16)
        for ax, lab in [(ax1, "a)"), (ax2, "b)")]:
            ax.text(-0.15, 1.04, rf"$\mathrm{{{lab}}}$", transform=ax.transAxes, fontsize=18, fontweight="bold",
                    va="bottom", ha="left")
        plt.tight_layout()
        _save(fig, "fig8_doping.pdf", dpi=300)
    print(f"  GS {fmt(stats[0])}  GF {fmt(stats[1])} per point; GF error median {np.nanmedian(E):.2e}, "
          f"max {np.nanmax(E):.2e}")


# ----------------------------------------------------------------------------- Fig. 9

def fig9(Z_thresh=1e-2):
    """Coarse scan: Z per band vs U/W1 at three bandwidth ratios (a) and total solver time (b)."""
    d = _json("fig9_coarse.json")
    rom = d["rom"]
    bwrs = sorted((float(b) for b in d["bwr"]), reverse=True)
    with plt.rc_context(SERIF):
        fig = plt.figure(figsize=(17.5, 5.1))
        gs = GridSpec(3, 2, figure=fig, width_ratios=[1, 1.1], hspace=0.0, wspace=0.22)
        ax_z, ax_t = [fig.add_subplot(gs[i, 0]) for i in range(3)], fig.add_subplot(gs[:, 1])
        colors = [(cm.Blues(0.85), cm.Blues(0.45)), (cm.Oranges(0.85), cm.Oranges(0.45)),
                  (cm.Greens(0.85), cm.Greens(0.45))]
        times = {}
        for idx, bwr in enumerate(bwrs):
            ax, (c_w, c_n) = ax_z[idx], colors[idx % 3]
            e = d["bwr"][f"{bwr:g}"]
            ed, mo = np.array(e["ed"], dtype=object), np.array(e["mor"], dtype=object)
            if len(ed):
                ax.plot(ed[:, 0].astype(float) / W1, ed[:, 1].astype(float), "-", color=c_w, linewidth=2.5, zorder=1)
                ax.plot(ed[:, 0].astype(float) / W1, ed[:, 2].astype(float), "--", color=c_n, linewidth=2.5, zorder=1)
            UW, Z0, Z1 = mo[:, 0].astype(float) / W1, mo[:, 1].astype(float), mo[:, 2].astype(float)
            ax.axhline(y=0, color="black", linestyle=":", linewidth=1)
            ax.plot(UW, Z0, "o", color=c_w, markersize=7, zorder=2)
            ax.plot(UW, Z1, "s", color=c_n, markersize=7, zorder=2)
            z_w = UW[np.where(Z0 < Z_thresh)[0][0]] if (Z0 < Z_thresh).any() else None
            z_n = UW[np.where(Z1 < Z_thresh)[0][0]] if (Z1 < Z_thresh).any() else None
            same = z_w is not None and z_n is not None and abs(z_w - z_n) < 0.01
            if z_w is not None:
                ax.axvline(x=z_w + (0.01 if same else 0.0), color=c_w, linestyle=":", linewidth=3)
            if z_n is not None:
                ax.axvline(x=z_n - (0.01 if same else 0.0), color=c_n, linestyle="--", linewidth=2)
            box = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
            if idx == 0:
                ax.text(0.04, 0.12, rf"$t_2/t_1 = {bwr:.2f}$", transform=ax.transAxes, ha="left", va="bottom",
                        fontsize=16, bbox=box)
            else:
                ax.text(0.96, 0.88, rf"$t_2/t_1 = {bwr:.2f}$", transform=ax.transAxes, ha="right", va="top",
                        fontsize=16, bbox=box, zorder=100)
            ax.set_ylabel(r"$Z_{i}$", fontsize=20)
            ax.tick_params(axis="both", labelsize=19)
            ax.set_ylim(-0.1, 1.1)
            ax.set_yticks([0.0, 0.4, 0.8, 1.0])
            ax.grid(True, alpha=0.3)
            if idx < 2:
                ax.tick_params(labelbottom=False)
            t_ed = {r[0]: r[3] for r in e["ed"] if r[3] is not None}
            t_mo = {r[0]: r[3] for r in e["mor"] if r[3] is not None}
            common = sorted(set(t_ed) & set(t_mo))
            times[bwr] = ([t_ed[U] for U in common], [t_mo[U] for U in common])
        gw, gn = cm.Greys(0.7), cm.Greys(0.4)
        hidden = Rectangle((0, 0), 0, 0, fill=False, edgecolor="none", visible=False)
        ax_z[0].legend([hidden, hidden, Line2D([0], [0], color=gw, linestyle="-", linewidth=2.5),
                        Line2D([0], [0], color=gw, marker="o", linestyle="None", markersize=7),
                        Line2D([0], [0], color=gn, linestyle="--", linewidth=2.5),
                        Line2D([0], [0], color=gn, marker="s", linestyle="None", markersize=7)],
                       [r"$\mathrm{ED}$", r"$\mathrm{MOR{+}EC}$", r"$Z_{1}$", r"$Z_{1}$", r"$Z_{2}$", r"$Z_{2}$"],
                       fontsize=14, loc="upper right", ncol=3, columnspacing=1.0, handletextpad=0.3)
        ax_z[-1].set_xlabel(r"$U/W_1$", fontsize=18)

        cmaps = [cm.Blues, cm.Oranges, cm.Greens]
        shades = lambda cmap, n: [cmap(0.3 + 0.6 * i / max(n - 1, 1)) for i in range(n)]

        def stack(ax, pos, which, width=0.6):
            bottom = 0.0
            for i_b, bwr in enumerate(sorted(times, reverse=True)):
                ts = times[bwr][which]
                for c, t_ in zip(shades(cmaps[i_b % 3], len(ts)), ts):
                    ax.bar(pos, t_ / 60.0, width, bottom=bottom, color=c, linewidth=0, edgecolor="none", rasterized=True)
                    bottom += t_ / 60.0
            return bottom

        def pair(ax, pos, ec, mor, width=0.6):
            ax.bar(pos, ec, width, bottom=0, color="mediumpurple", linewidth=0, edgecolor="none", rasterized=True)
            ax.bar(pos, mor, width, bottom=ec, color="gold", linewidth=0, edgecolor="none", rasterized=True)
            return ec + mor

        ed_total = stack(ax_t, 0, 0)
        off_total = pair(ax_t, 1, rom["ec_s"] / 60.0, rom["mor_s"] / 60.0)
        on_total = pair(ax_t, 2, rom["enrich_ec_s"] / 60.0, rom["enrich_mor_s"] / 60.0)
        mor_total = stack(ax_t, 3, 1)
        for pos, total in zip(range(4), [ed_total, off_total, on_total, mor_total]):
            ax_t.annotate(rf"$\mathbf{{{total:.2f}}}$", xy=(pos, total), xytext=(0, 5), textcoords="offset points",
                          ha="center", va="bottom", fontsize=20)
        ax_t.set_xticks(range(4))
        ax_t.set_xticklabels([r"\textrm{ED}", r"\textrm{Offline}", r"\textrm{Online}", r"\textrm{MOR+EC}"], fontsize=20)
        ax_t.set_ylabel(r"\textrm{Solver Time (min)}", fontsize=20, labelpad=4)
        ax_t.tick_params(axis="y", labelsize=19)
        ax_t.set_xlim(-0.5, 3.5)
        ax_t.set_ylim(0, ed_total * 1.85)
        ax_t.legend(handles=[Patch(facecolor=cmaps[i % 3](0.6), label=rf"$t_2/t_1={b:.2f}$")
                             for i, b in enumerate(sorted(times, reverse=True))]
                    + [Patch(facecolor="mediumpurple", label=r"\textrm{Basis Selection}"),
                       Patch(facecolor="gold", label=r"\textrm{Projection Build}")],
                    loc="upper left", fontsize=14, ncol=2)
        ins = ax_t.inset_axes([0.58, 0.32, 0.40, 0.4])
        pair(ins, 0, rom["ec_s"] / 60.0, rom["mor_s"] / 60.0)
        pair(ins, 1, rom["enrich_ec_s"] / 60.0, rom["enrich_mor_s"] / 60.0)
        stack(ins, 2, 1)
        for pos, total in zip(range(3), [off_total, on_total, mor_total]):
            ins.annotate(rf"$\mathbf{{{total:.2f}}}$", xy=(pos, total), xytext=(0, 3), textcoords="offset points",
                         ha="center", va="bottom", fontsize=16)
        ins.set_xticks(range(3))
        ins.set_xticklabels([r"\textrm{Offline}", r"\textrm{Online}", r"\textrm{MOR+EC}"], fontsize=16)
        ins.set_ylabel(r"\textrm{Solver Time (min)}", fontsize=16)
        ins.tick_params(axis="both", labelsize=14)
        ins.set_xlim(-0.5, 2.5)
        ins.set_ylim(0, max(off_total, on_total, mor_total, 1e-9) * 1.2)
        ax_z[0].text(-0.15, 1.02, r"$\mathrm{a)}$", transform=ax_z[0].transAxes, fontsize=20, fontweight="bold",
                     va="bottom", ha="left")
        ax_t.text(-0.1, 1.02, r"$\mathrm{b)}$", transform=ax_t.transAxes, fontsize=20, fontweight="bold",
                  va="bottom", ha="left")
        _save(fig, "fig9_Z_and_timing.pdf", dpi=300)
    print(f"  ED {ed_total:.2f} min, offline {off_total:.2f}, online {on_total:.2f} "
          f"({rom['n_events']} events), MOR+EC {mor_total:.2f} min")


# ----------------------------------------------------------------------------- Fig. 10

def fig10(Z_thr=1e-2, Z_floor=1e-2, exclude_ed_bwr=(0.25, 0.35, 0.45)):
    """Fine phase diagram: Z of both bands over (t2/t1, U/W1) with Mott lines (a), speed-up (b)."""
    d = np.load(os.path.join(DATA, "fig10_phase.npz"))
    calls = _json("fig10_timing.json")["calls"]
    bwrs, U = d["bwr"], d["U"]
    Z0_ed, Z1_ed = d["Z0_ed"].copy(), d["Z1_ed"].copy()
    excl = np.isin(np.round(bwrs, 6), np.round(exclude_ed_bwr, 6))
    Z0_ed[excl], Z1_ed[excl] = np.nan, np.nan
    UW = U / W1

    def onset(field):  # first crossing of Z below Z_thr along U, per bwr column
        xs, ys = [], []
        for ib, b in enumerate(bwrs):
            row = field[ib]
            valid = np.where(~np.isnan(row))[0]
            for i0, i1 in zip(valid[:-1], valid[1:]):
                if row[i0] >= Z_thr > row[i1]:
                    xs.append(b)
                    ys.append(UW[i0] + (Z_thr - row[i0]) / (row[i1] - row[i0]) * (UW[i1] - UW[i0]))
                    break
        return np.array(xs), np.array(ys)

    def edges(a):
        if len(a) == 1:
            w = abs(a[0]) * 0.1 or 0.5
            return np.array([a[0] - w, a[0] + w])
        e = np.empty(len(a) + 1)
        e[1:-1] = (a[:-1] + a[1:]) / 2
        e[0], e[-1] = a[0] - (a[1] - a[0]) / 2, a[-1] + (a[-1] - a[-2]) / 2
        return e

    ZZ0, ZZ1 = d["Z0_mor"].T, d["Z1_mor"].T
    vis = [i for i in range(len(U)) if not (np.all(np.isnan(ZZ0[i])) and np.all(np.isnan(ZZ1[i])))] or list(range(len(U)))
    y_edges = edges(np.array([UW[i] for i in vis]))
    y_min, y_max = y_edges[0], y_edges[-1]
    b_edges = edges(bwrs)
    yticks = [round(u / W1, 3) for u in np.arange(0.0, U.max() + 1.0, 1.0) if u / W1 <= y_max][::2]
    cmap = LinearSegmentedColormap.from_list("inferno_log", [plt.cm.inferno(x) for x in np.linspace(0, 1, 170)], N=512)
    norm = LogNorm(vmin=Z_floor, vmax=1.0)
    style = {"text.usetex": TEX["text_usetex"], "font.size": 14, "axes.labelsize": 16, "axes.titlesize": 16,
             "xtick.labelsize": 14, "ytick.labelsize": 14}
    with plt.rc_context(style):
        fig = plt.figure(figsize=(11.5, 4.0))
        gs = GridSpec(1, 5, figure=fig, width_ratios=[0.90, 0.90, 0.045, 0.13, 0.78], left=0.07, right=0.97,
                      top=0.95, bottom=0.14, wspace=0.02)
        ax0, ax1, ax_cb, ax_r = (fig.add_subplot(gs[0, i]) for i in (0, 1, 2, 4))
        p0, p1 = ax0.get_position(), ax1.get_position()
        ax1.set_position([p0.x1, p1.y0, p1.width, p1.height])
        mesh = lambda ax, ZZ: ax.pcolormesh(*np.meshgrid(b_edges, y_edges), ZZ[np.ix_(vis, range(len(bwrs)))],
                                            cmap=cmap, norm=norm, rasterized=True)
        pcm = mesh(ax0, ZZ0)
        mesh(ax1, ZZ1)
        for ax in (ax0, ax1):
            ax.set_yticks(yticks)
            ax.set_xlim(b_edges[0], b_edges[-1])
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel(r"$t_2/t_1$")
            ax.tick_params(axis="both", which="major", labelsize=14)
        ax0.set_ylabel(r"$U/W_1$")
        ax1.tick_params(axis="y", which="both", left=False, labelleft=False)
        for ax, lab in ((ax0, r"$\mathrm{Band~1~(Wide)}$"), (ax1, r"$\mathrm{Band~2~(Narrow)}$")):
            ax.text(0.02, 0.95, lab, transform=ax.transAxes, va="top", ha="left", fontsize=14, color="white",
                    fontweight="bold")
        ax1.legend(handles=[Line2D([0], [0], color="lime", lw=1.5, ls="solid", label=r"$Z_1\to 0$"),
                            Line2D([0], [0], color="lime", lw=1.5, ls="dashed", label=r"$Z_2\to 0$")],
                   fontsize=12, loc="lower right", framealpha=0.7)
        for ax, fld, ls in ((ax0, d["Z0_mor"], "solid"), (ax1, d["Z0_mor"], "solid"), (ax1, d["Z1_mor"], "dashed")):
            bx, uy = onset(fld)
            if len(bx):
                ax.plot(bx, uy, color="lime", lw=2.0, ls=ls, alpha=1.0, solid_capstyle="round", zorder=4)
        xs, ys = (np.concatenate(v) for v in zip(onset(Z0_ed), onset(Z1_ed)))
        inr = (ys >= y_min) & (ys <= y_max)
        if inr.any():
            ax1.scatter(xs[inr], ys[inr], marker="*", color="red", s=50, zorder=5, clip_on=False)
        lbl = dict(ha="center", va="center", fontsize=10, color="white", fontweight="bold", zorder=6)
        mb, mu = onset(d["Z0_mor"])
        for ax in (ax0, ax1):
            if len(mb):
                i, j = 2 * len(mb) // 3, 3 * len(mb) // 4
                ax.text(mb[i], 0.5 * (mu[i] + y_max), r"$\mathrm{INSULATING}$", **lbl)
                ax.text(mb[j], 0.55 * mu[j], r"$\mathrm{METALLIC}$", **lbl)
        nb, nu = onset(d["Z1_mor"])
        wmap = dict(zip(mb, mu))
        osmp = [(b, lo, wmap[b]) for b, lo in zip(nb, nu) if b in wmap and wmap[b] > lo]
        if osmp:
            ox, olo, ohi = (np.array(v) for v in zip(*osmp))
            sel = (ox >= 0.10) & (ox <= 0.15)
            sel = sel if sel.sum() >= 2 else np.ones_like(ox, bool)
            wx, wm = ox[sel], 0.5 * (olo[sel] + ohi[sel])
            cx = 0.5 * (wx[0] + wx[-1])
            (x0, y0), (x1, y1) = ax1.transData.transform(np.column_stack([wx[[0, -1]], wm[[0, -1]]]))
            ax1.text(cx, float(np.interp(cx, wx, wm)) + 0.1, r"$\mathrm{OSMP}$",
                     rotation=float(np.degrees(np.arctan2(y1 - y0, x1 - x0))), rotation_mode="anchor", **lbl)
        step = max(1, (len(bwrs) - 1) // 5)
        ax0.set_xticks(bwrs[:-1][::step])
        ax1.set_xticks(bwrs[1:][::step])
        cp = ax_cb.get_position()
        ax_cb.set_visible(False)
        cb = fig.colorbar(pcm, cax=fig.add_axes([cp.x0, cp.y0 + 0.2 * cp.height, cp.width, 0.6 * cp.height]))
        cb.ax.set_title(r"$Z_i$", fontsize=18, pad=10)
        cb.ax.tick_params(labelsize=14)

        per_bwr = {}
        for key, (ed_c, mo_c) in calls.items():
            b = float(key.split("/")[0])
            if round(b, 6) in np.round(exclude_ed_bwr, 6):
                continue
            per_bwr.setdefault(b, []).append((ed_c, mo_c))
        bars = {i: [] for i in (0, 1)}
        for b in bwrs:
            for i in (0, 1):
                s = speedup([(e[i], m[i]) for e, m in per_bwr.get(b, [])])
                if s is not None:
                    bars[i].append((b, s))
        for i, lab in ((0, "GS"), (1, "GF")):
            s = speedup([(e[i], m[i]) for v in per_bwr.values() for e, m in v])
            print(f"  {lab} speed-up (all bwr): {fmt(s)} per point over {s.n_points} points; {s.aggregate:.0f}x aggregate")
        slots = {b: k for k, b in enumerate(sorted({b for i in (0, 1) for b, _ in bars[i]}))}
        floor = min(s.lo for i in (0, 1) for _, s in bars[i]) * 0.55
        top = max(s.hi for i in (0, 1) for _, s in bars[i]) * 2.0
        for i, off, col, lab in ((0, -0.19, "#d62728", r"$\mathrm{GS}$"), (1, 0.19, "#1f77b4", r"$\mathrm{GF}$")):
            x = np.array([slots[b] for b, _ in bars[i]], float) + off
            med = np.array([s.median for _, s in bars[i]])
            err = np.array([[s.median - s.lo for _, s in bars[i]], [s.hi - s.median for _, s in bars[i]]])
            ax_r.bar(x, med - floor, bottom=floor, width=0.34, yerr=err, capsize=3, ecolor="black", color=col,
                     alpha=0.85, zorder=3, label=lab)
            for xi, m, h in zip(x, med, err[1]):
                ax_r.annotate(rf"${m:.0f}\times$", xy=(xi, m + h), xytext=(0, 3), textcoords="offset points",
                              ha="center", va="bottom", fontsize=11, zorder=6,
                              bbox=dict(boxstyle="round,pad=0.18", facecolor="white", alpha=0.95, edgecolor=col,
                                        linewidth=0.9))
        ax_r.legend(fontsize=11, framealpha=0.7, loc="upper right", ncol=2)
        ax_r.set_xlabel(r"$t_2/t_1$", fontsize=14)
        ax_r.set_ylabel(r"$\mathrm{Speedup\ (ED\ /\ MOR{+}EC)}$", fontsize=14)
        ax_r.set_yscale("log")
        ax_r.set_ylim(floor, top)
        ax_r.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=12))
        ax_r.yaxis.set_major_formatter(ScalarFormatter())
        ax_r.yaxis.set_minor_formatter(NullFormatter())
        ax_r.yaxis.set_label_position("right")
        ax_r.yaxis.tick_right()
        ax_r.tick_params(axis="both", which="major", labelsize=14)
        ax_r.grid(True, linestyle="--", alpha=0.5, axis="y", which="major")
        ax_r.grid(True, linestyle=":", alpha=0.25, axis="y", which="minor")
        ax_r.set_xlim(-0.65, len(slots) - 0.35)
        ax_r.set_xticks(np.arange(len(slots)))
        ax_r.set_xticklabels([rf"${b:g}$" for b in slots])
        _label_panels([ax0, ax_r], fontsize=18, xloc=-0.15)
        _save(fig, "fig10_phase_diagram.pdf")


# ----------------------------------------------------------------------------- Fig. 11

def fig11():
    """Real-frequency narrow-band spectra of ED and MOR+EC at weak and strong coupling."""
    d = np.load(os.path.join(DATA, "fig11_spectra.npz"))
    w = d["w"]
    with plt.rc_context(SERIF):
        fig, axes = plt.subplots(2, 1, figsize=(7.5, 5.0), sharex="col")
        for ax, U in zip(axes, d["U"]):
            G_ed, G_mo = d[f"G_ed_U{U:g}"], d[f"G_mor_U{U:g}"]
            ax.plot(w, -G_ed.imag / np.pi, "k-", linewidth=1.8)
            ax.plot(w, -G_mo.imag / np.pi, "--", color="blue", linewidth=1.1)
            ax.tick_params(axis="both", labelsize=18)
            ax.grid(True, alpha=0.3)
            ax.set_title(f"$U/W_1={U / W1:.2f}$", fontsize=20)
            ax.set_ylabel(r"$-\frac{1}{\pi}\mathrm{Im}[\mathbf{G}^\mathrm{imp}(\omega)]_{22}$", fontsize=18)
            ax.set_xlim(w[0], w[-1])
            ins = ax.inset_axes([0.80, 0.55, 0.18, 0.38])
            ins.semilogy(w, np.abs(G_mo - G_ed) / np.max(np.abs(G_ed)), "r-", linewidth=1.0)
            ins.set_ylim(1e-7, 1e-1)
            ins.set_yticks([1e-6, 1e-4, 1e-2])
            ins.set_xlabel(r"$\omega$", fontsize=16)
            ins.tick_params(axis="both", labelsize=16)
            ins.grid(True, alpha=0.3)
        axes[0].legend(handles=[Line2D([0], [0], color="black", linestyle="-", label=r"\textrm{ED}"),
                                Line2D([0], [0], color="blue", linestyle="--", label=r"\textrm{MOR+EC}")],
                       fontsize=16, loc="upper left")
        axes[1].set_xlabel(r"$\omega$", fontsize=18)
        _label_panels(list(axes), fontsize=18, xloc=-0.1)
        plt.subplots_adjust(left=0.08, right=0.99, bottom=0.10, top=0.94, wspace=0.22, hspace=0.25)
        _save(fig, "fig11_real_frequency.pdf", dpi=300)


# ----------------------------------------------------------------------------- finite J (not in the paper)

def finite_j(ju_max=0.6, err_floor=1e-4, err_ceil=1.0):
    """ED vs MOR+EC at finite J/U: Z maps, Z in the (J/U, U/W1) plane, relative Z error, speed-up."""
    recs = _json("finite_j.json")["points"]
    for r in recs:
        r["ju"] = float(r["level"][2:])
    jus, bwrs, Us = (sorted({r[k] for r in recs}) for k in ("ju", "bwr", "U"))
    y = np.array(Us) / W1
    cmap = LinearSegmentedColormap.from_list("inferno_log", [plt.cm.inferno(x) for x in np.linspace(0, 1, 170)], N=512)
    znorm = LogNorm(vmin=1e-2, vmax=1.0)
    style = {"text.usetex": TEX["text_usetex"], "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13,
             "xtick.labelsize": 11, "ytick.labelsize": 11}

    def grid(key, o):
        g = np.full((len(jus), len(bwrs), len(Us)), np.nan)
        for r in recs:
            g[jus.index(r["ju"]), bwrs.index(r["bwr"]), Us.index(r["U"])] = r[key][o]
        return g

    def edges(a):
        a = np.asarray(a, float)
        if len(a) == 1:
            return np.array([a[0] - (abs(a[0]) * 0.1 or 0.5), a[0] + (abs(a[0]) * 0.1 or 0.5)])
        e = np.empty(len(a) + 1)
        e[1:-1] = (a[:-1] + a[1:]) / 2
        e[0], e[-1] = a[0] - (a[1] - a[0]) / 2, a[-1] + (a[-1] - a[-2]) / 2
        return e

    def mesh(ax, xs, ys, ZZ, cm_, norm):
        Xe, Ye = np.meshgrid(edges(xs), edges(ys))
        p = ax.pcolormesh(Xe, Ye, ZZ, cmap=cm_, norm=norm, rasterized=True)
        ax.set_xlim(Xe[0, 0], Xe[0, -1])
        ax.set_ylim(Ye[0, 0], Ye[-1, 0])
        return p

    G = {(k, o): grid(k, o) for k in ("Z_ed", "Z_mor") for o in (0, 1)}
    fields = [(("Z_ed", 0), r"$\mathrm{ED}$", r"$Z_1$"), (("Z_mor", 0), r"$\mathrm{MOR{+}EC}$", r"$Z_1$"),
              (("Z_ed", 1), r"$\mathrm{ED}$", r"$Z_2$"), (("Z_mor", 1), r"$\mathrm{MOR{+}EC}$", r"$Z_2$")]
    tag = r"$r=3,\ \mathrm{thr}=\mathrm{1e{-}3}$"
    with plt.rc_context(style):
        fig, axs = plt.subplots(len(jus), 4, figsize=(9.0, 1.55 * len(jus) + 1.0), squeeze=False, sharex=True,
                                sharey=True)
        for ij, ju in enumerate(jus):
            for ic, (key, who, zl) in enumerate(fields):
                ax = axs[ij][ic]
                pcm = mesh(ax, bwrs, y, G[key][ij].T, cmap, znorm)
                if ij == 0:
                    ax.set_title(zl + r"$,\ $" + who, fontsize=12, pad=6)
                if ic == 0:
                    ax.set_ylabel(r"$U/W_1$")
                    ax.text(-0.55, 0.5, rf"$J/U={ju:g}$", transform=ax.transAxes, rotation=90, va="center",
                            ha="center", fontsize=12)
                if ij == len(jus) - 1:
                    ax.set_xlabel(r"$t_2/t_1$")
                ax.set_xticks(bwrs)
                ax.tick_params(labelsize=10)
        fig.suptitle(tag, fontsize=13)
        fig.tight_layout(rect=[0.03, 0.0, 0.90, 0.97])
        cb = fig.colorbar(pcm, cax=fig.add_axes([0.92, 0.18, 0.018, 0.64]))
        cb.ax.set_title(r"$Z_i$", fontsize=14, pad=8)
        _save(fig, "finite_j_Z_maps.pdf")

        cols = [i for i, ju in enumerate(jus) if ju <= ju_max]
        fig, axs = plt.subplots(4, len(bwrs), figsize=(2.4 * len(bwrs) + 1.2, 8.4), squeeze=False, sharex=True,
                                sharey=True)
        for ir, (key, who, zl) in enumerate(fields):
            for ib, b in enumerate(bwrs):
                ax = axs[ir][ib]
                m = mesh(ax, [jus[i] for i in cols], y, G[key][cols][:, ib, :].T, cmap, znorm)
                ax.set_xticks([jus[i] for i in cols][::-(-len(cols) // 6)])
                if ir == 0:
                    ax.set_title(rf"$t_2/t_1={b:g}$", fontsize=12)
                if ir == 3:
                    ax.set_xlabel(r"$J/U$")
                if ib == 0:
                    ax.set_ylabel(zl + r"$,\ $" + who + "\n" + r"$U/W_1$", fontsize=11)
                ax.tick_params(labelsize=10)
        fig.suptitle(tag, fontsize=13)
        fig.tight_layout(rect=[0.0, 0.0, 0.90, 0.96])
        cb = fig.colorbar(m, cax=fig.add_axes([0.92, 0.18, 0.018, 0.64]))
        cb.ax.set_title(r"$Z_i$", fontsize=14, pad=8)
        _save(fig, "finite_j_Z_JU.pdf")

        with np.errstate(divide="ignore", invalid="ignore"):
            E = [np.abs(G[("Z_mor", o)] - G[("Z_ed", o)]) / np.where(G[("Z_ed", o)] > 0, G[("Z_ed", o)], np.nan)
                 for o in (0, 1)]
        fig, axs = plt.subplots(2, len(jus), figsize=(1.35 * len(jus) + 2.0, 5.2), squeeze=False, sharex=True,
                                sharey=True)
        for row in range(2):
            for ij, ju in enumerate(jus):
                ax = axs[row][ij]
                pcm = mesh(ax, bwrs, y, np.clip(E[row][ij].T, err_floor, err_ceil), plt.cm.viridis,
                           LogNorm(vmin=err_floor, vmax=err_ceil))
                if row == 0:
                    ax.set_title(rf"$J/U={ju:g}$", fontsize=12)
                else:
                    ax.set_xlabel(r"$t_2/t_1$")
                if ij == 0:
                    ax.set_ylabel(rf"$Z_{row + 1}$" + "\n" + r"$U/W_1$")
                ax.set_xticks(bwrs)
                ax.tick_params(labelsize=9)
        fig.suptitle(tag + r"$\ \ |\Delta Z|/Z_{\mathrm{ED}}$", fontsize=13)
        fig.tight_layout(rect=[0.0, 0.0, 0.88, 0.95])
        cb = fig.colorbar(pcm, cax=fig.add_axes([0.90, 0.15, 0.02, 0.7]), extend="max")
        cb.ax.set_title(r"$|\Delta Z|/Z$", fontsize=12, pad=8)
        _save(fig, "finite_j_Z_error.pdf")

        fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.8))
        for ax, field, xlabel, ticks in ((axs[0], "ju", r"$J/U$", jus), (axs[1], "bwr", r"$t_2/t_1$", bwrs)):
            groups = {v: [(speedup([(r["ed"][i], r["mor"][i]) for r in recs if r[field] == v])) for i in (0, 1)]
                      for v in ticks}
            vals = [v for v in ticks if groups[v][0] or groups[v][1]]
            lo_all, hi_all = [], []
            for off, i, col, lab in ((-0.19, 0, "#d62728", r"$\mathrm{GS}$"), (0.19, 1, "#1f77b4", r"$\mathrm{GF}$")):
                st = [(k, groups[v][i]) for k, v in enumerate(vals) if groups[v][i]]
                med = np.array([s.median for _, s in st])
                err = np.array([[s.median - s.lo for _, s in st], [s.hi - s.median for _, s in st]])
                lo_all += [s.lo for _, s in st]
                hi_all += [s.hi for _, s in st]
                xs = np.array([k for k, _ in st]) + off
                ax.bar(xs, med, width=0.34, yerr=err, capsize=3, ecolor="black", color=col, alpha=0.85, zorder=3,
                       label=lab)
                for xi, m, h in zip(xs, med, err[1]):
                    ax.annotate((r"$%.1f\times$" if m < 10 else r"$%.0f\times$") % m, xy=(xi, m + h), xytext=(0, 3),
                                textcoords="offset points", ha="center", va="bottom", fontsize=9, zorder=6)
            ax.set_xticks(np.arange(len(vals)))
            ax.set_xticklabels([rf"${v:g}$" for v in vals])
            ax.set_xlabel(xlabel)
            ax.set_yscale("log")
            ax.set_ylim(min(lo_all) * 0.6, max(hi_all) * 2.5)
            ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=12))
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.grid(True, linestyle="--", alpha=0.5, axis="y", which="major")
            ax.grid(True, linestyle=":", alpha=0.25, axis="y", which="minor")
        axs[0].set_ylabel(r"$\mathrm{Speedup\ (ED\,/\,MOR{+}EC)}$")
        axs[0].legend(fontsize=10, framealpha=0.7, loc="upper right", ncol=2)
        fig.suptitle(tag, fontsize=13)
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
        _save(fig, "finite_j_timing.pdf")
    e0 = np.array([abs(r["Z_mor"][0] - r["Z_ed"][0]) / r["Z_ed"][0] for r in recs if r["Z_ed"][0] > 0])
    e1 = np.array([abs(r["Z_mor"][1] - r["Z_ed"][1]) / r["Z_ed"][1] for r in recs if r["Z_ed"][1] > 0])
    print(f"  {len(recs)} matched points; |dZ|/Z median {np.median(e0):.2e} (Z_1), {np.median(e1):.2e} (Z_2)")


FIGS = {"fig4": fig4, "fig5": fig5, "fig6": fig6, "table1": table1, "fig7": fig7, "fig8": fig8, "fig9": fig9,
        "fig10": fig10, "fig11": fig11, "finite_j": finite_j}

if __name__ == "__main__":
    for name in sys.argv[1:] or list(FIGS):
        print(f"== {name}")
        FIGS[name]()
