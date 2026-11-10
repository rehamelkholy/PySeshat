"""
Appendix figures for the real-data Capella (Torres+2009) and alpha Cen AB
(Akeson+2021) benchmarks.  Combined visual+RV fit with dataset-level
weighting (the PySeshat default).  Layout fixes the overprint seen in the
previous synthetic-data Fig. A.6a: small semi-transparent markers, model
curve drawn last (on top), RV phase-folded, residuals in mas.

Run:  MPLBACKEND=Agg python scripts/generate_real_data_figures.py
Writes validation/results/plots/<System>_real/{visual_orbit,residuals,rv_orbit_phased}.pdf
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import capella_real_data as cap
import alpha_cen_real_data as acen
from pyseshat.inference.optimizer import fit_visual_orbit
from pyseshat.orbit.orbit_model import compute_visual_orbit
from pyseshat.orbit.rv_model import compute_rv_orbit
from pyseshat.orbit.phase import orbital_phase

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation/results/plots"
C_OBS, C_MOD = "#1f4e79", "#c0392b"
plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8})


def scale_of(f, key_part):
    for k, v in f.metadata["error_scales"].items():
        if key_part in str(k).lower():
            return float(v)
    return 1.0


def make(name, el, t, X, Y, sx, sy, rv, f, tag_vis, tag_rv):
    d = OUT / f"{name}_real"
    d.mkdir(parents=True, exist_ok=True)
    P = el.P
    # --- visual orbit
    tm = el.T0 + np.linspace(0, P, 800)
    orb = compute_visual_orbit(tm, el)
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.errorbar(Y * 1e3, X * 1e3, xerr=sy * 1e3, yerr=sx * 1e3, fmt="none",
                ecolor=C_OBS, alpha=0.25, lw=0.6, zorder=1)
    ax.scatter(Y * 1e3, X * 1e3, s=9, color=C_OBS, alpha=0.55, lw=0, zorder=2,
               label="Observations")
    ax.plot(orb["Y"] * 1e3, orb["X"] * 1e3, color=C_MOD, lw=1.3, zorder=3,
            label="Fitted orbit")
    ax.plot(0, 0, marker="*", color="k", ms=9, ls="none", zorder=4, label="Primary")
    ax.set_xlabel("East (mas)"); ax.set_ylabel("North (mas)")
    ax.invert_xaxis(); ax.set_aspect("equal"); ax.legend(frameon=False, loc="best")
    fig.savefig(d / "visual_orbit.pdf", bbox_inches="tight"); plt.close(fig)
    # --- residuals
    mod = compute_visual_orbit(t, el)
    s_v = scale_of(f, tag_vis)
    rx, ry = (X - mod["X"]) * 1e3, (Y - mod["Y"]) * 1e3
    fig, axs = plt.subplots(2, 1, figsize=(6.0, 3.6), sharex=True)
    for a, r, s, lab in ((axs[0], rx, sx, r"$\Delta X$ (North)"),
                         (axs[1], ry, sy, r"$\Delta Y$ (East)")):
        a.axhline(0, color="k", lw=0.6)
        a.errorbar(t, r, yerr=s * s_v * 1e3, fmt="o", ms=2.5, color=C_OBS,
                   ecolor=C_OBS, elinewidth=0.5, alpha=0.6, lw=0)
        a.set_ylabel(lab + " O$-$C (mas)")
    axs[1].set_xlabel("Epoch (yr)")
    fig.tight_layout(); fig.savefig(d / "residuals.pdf"); plt.close(fig)
    # --- RV (phase-folded, or vs epoch when the RV span covers < 0.5 P)
    s_r = scale_of(f, tag_rv)
    span = rv[:, 0].max() - rv[:, 0].min()
    fold = span > 0.5 * P
    x_obs = orbital_phase(rv[:, 0], P, el.T0) if fold else rv[:, 0]
    if fold:
        x_mod = np.linspace(0, 1, 600); t_mod = el.T0 + x_mod * P
        xl = "Orbital phase"
    else:
        t_mod = np.linspace(rv[:, 0].min(), rv[:, 0].max(), 800); x_mod = t_mod
        xl = "Epoch (yr)"
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    for c, col, lab in ((0, C_OBS, "Primary"), (1, "#2e8b57", "Secondary")):
        m = rv[:, 3] == c
        ax.errorbar(x_obs[m], rv[m, 1], yerr=rv[m, 2] * s_r, fmt="o", ms=2.5,
                    color=col, ecolor=col, elinewidth=0.4, alpha=0.45, lw=0,
                    zorder=1, label=lab)
        ax.plot(x_mod, compute_rv_orbit(t_mod, el, component=c), color=C_MOD,
                lw=1.2, zorder=3)
    ax.set_xlabel(xl); ax.set_ylabel(r"Radial velocity (km s$^{-1}$)")
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(); fig.savefig(d / "rv_orbit_phased.pdf"); plt.close(fig)
    print(name, "scales", f.metadata["error_scales"], "->", d)


def capella():
    rv = cap.rv_cfa()
    t, X, Y, sx, sy = cap.astrometry(**cap.SUBSETS["B"])
    f = fit_visual_orbit(cap.start(), t, X, Y, sx, sy, rv_data=rv, weighting="dataset")
    make("Capella", f.elements, t, X, Y, sx, sy, rv, f, "vis", "rv")


def alpha():
    ep, X, Y, sx, sy, rv = acen.load()
    f = fit_visual_orbit(acen.start_elements(), ep, X, Y, sx, sy, rv_data=rv,
                         weighting="dataset")
    make("AlphaCen", f.elements, ep, X, Y, sx, sy, rv, f, "vis", "rv")


if __name__ == "__main__":
    capella()
    alpha()
