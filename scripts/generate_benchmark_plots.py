"""
Generate the full available plot set for the benchmark systems
that have a matching visualization function, and save the PNGs
under validation/results/plots/<system>/ for a one-time visual
pass, per docs/development_plan_phase2.md Phase 4 item 3
("once Phase 0 unblocks all 8 systems, generate the full plot
set for each and do one visual pass before considering the
validation phase closed").

Scope, and why it is 2 of 8 systems rather than 8 of 8
--------------------------------------------------------
`src/pyseshat/visualization/` has exactly five plotting
functions: ``plot_visual_orbit``, ``plot_residuals``,
``plot_rv_orbit`` (all operating on a flat, two-body
``OrbitalElements`` model), ``plot_separation_vs_time`` (same),
and ``plot_binary_hrdiagram`` (operating on evolutionary-track
data, ``HRDiagramData``, entirely unrelated to an orbital fit).

Of the 8 registered benchmark systems:

- FIN379, GL765 are plain visual+RV binaries -- a flat
  ``OrbitalElements`` model is exactly correct for them, so all
  four orbit/dynamics plotting functions apply directly. This
  script fits both (mirroring
  ``validation/benchmark_runner.py``'s ``run_benchmark()``) and
  generates the full set.
- ZetaAqr, HIP111805 are hierarchical triples fit through the
  20-parameter ORBIT3 formalism
  (``triple.orbit3_fit.fit_orbit3``). ``TripleOrbitalElements``
  does carry a real ``outer: OrbitalElements`` sub-object, so
  ``plot_visual_orbit``/``plot_separation_vs_time`` COULD be
  pointed at it -- but the outer-orbit astrometric observations
  these systems are fit against include the unresolved inner
  subsystem's astrometric wobble (the ``f`` parameter;
  see docs/project_decisions.md, "`f` unbounded" and the ZetaAqr
  residual-decomposition entry). A flat two-body curve plotted
  directly against those observations would show a real,
  wobble-driven residual pattern that a reader could easily
  mistake for a fit-quality problem rather than an intentional,
  already-modeled physical effect. No triple-aware orbit plot
  (one that overlays the wobble-corrected curve, or explicitly
  separates the two effects) exists in this codebase yet -- see
  the "Future work" entry in docs/paper_reference_notes.md.
  Rather than fabricate a plot that risks being scientifically
  misleading under this task's time pressure, this is
  intentionally left undone and flagged in project_decisions.md/
  paper_reference_notes.md rather than silently worked around.
- HD214222, HD191854, HD130669, HD184467 are Al-Wardat
  unresolved-binary SED fits -- there is no orbital solution
  being fit at all (see examples/04_alwardat_sed_fit.py's
  docstring: "Unlike Examples 1-3, there is no orbital fit here
  at all"), so none of the orbit-plotting functions apply.
  ``plot_binary_hrdiagram`` could in principle be used, but
  requires a full evolutionary-track cross-check
  (``evolution/binary.py``) to be run per system first -- a
  separate, heavier pipeline this pass did not have time to wire
  up and verify for all four systems. Flagged as future work.

Run

    MPLBACKEND=Agg python validation/../scripts/generate_benchmark_plots.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np

from pyseshat.io.orbit_parser import parse_orbit_file
from pyseshat.inference.optimizer import fit_visual_orbit
from pyseshat.visualization.orbit_plot import (
    plot_visual_orbit,
    plot_residuals,
    plot_rv_orbit,
)
from pyseshat.visualization.dynamics_plot import plot_separation_vs_time
from pyseshat.analysis.dynamics import separation_vs_time
from pyseshat.inference.likelihood import visual_xy_residuals


REPO_ROOT = Path(__file__).resolve().parent.parent

SYSTEMS_DIR = REPO_ROOT / "validation" / "benchmark_systems"

PLOTS_DIR = REPO_ROOT / "validation" / "results" / "plots"

BINARY_SYSTEMS = {
    "FIN379": SYSTEMS_DIR / "FIN379" / "FIN379.inp",
    "GL765": SYSTEMS_DIR / "GL765" / "GL765.inp",
}


def _fit(data_file):

    parsed = parse_orbit_file(str(data_file))

    elements = parsed["elements"]

    visual = parsed["visual_data"]

    rv = parsed["rv_data"]

    t = visual[:, 0]
    theta = np.deg2rad(visual[:, 1])
    rho = visual[:, 2]
    sigma = visual[:, 3]

    X = rho * np.cos(theta)
    Y = rho * np.sin(theta)

    fit_result = fit_visual_orbit(
        elements, t, X, Y, sigma, sigma, rv_data=rv,
    )

    return fit_result.elements, t, X, Y, sigma, rv


def generate_for_system(name, data_file):

    out_dir = PLOTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    fitted, t, X_obs, Y_obs, sigma, rv = _fit(data_file)

    # Dense model curve over the observed time span for the
    # visual-orbit overlay.
    from pyseshat.orbit.orbit_model import compute_visual_orbit

    t_model = np.linspace(t.min(), t.max(), 500)
    orbit = compute_visual_orbit(t_model, fitted)

    fig, ax = plot_visual_orbit(X_obs, Y_obs, orbit, show=False)
    fig.savefig(out_dir / "visual_orbit.png", dpi=150)
    matplotlib.pyplot.close(fig)

    res = visual_xy_residuals(
        fitted.fitted_parameters(), fitted, t, X_obs, Y_obs, sigma, sigma,
    )
    fig, axes = plot_residuals(t, res, show=False)
    fig.savefig(out_dir / "residuals.png", dpi=150)
    matplotlib.pyplot.close(fig)

    if len(rv) > 0:
        fig, ax = plot_rv_orbit(rv, fitted, phase=True, show=False)
        fig.savefig(out_dir / "rv_orbit_phased.png", dpi=150)
        matplotlib.pyplot.close(fig)

    epochs = np.linspace(t.min(), t.max(), 500)
    dyn = separation_vs_time(fitted, epochs)
    fig, ax = plot_separation_vs_time(
        dyn["epoch"], dyn["rho"], show=False,
    )
    fig.savefig(out_dir / "separation_vs_time.png", dpi=150)
    matplotlib.pyplot.close(fig)

    print(f"{name}: wrote 4 plots to {out_dir}")


def main():

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    for name, data_file in BINARY_SYSTEMS.items():
        generate_for_system(name, data_file)

    print(
        "\nZetaAqr, HIP111805 (hierarchical triples) and the 4 "
        "Al-Wardat SED systems are intentionally NOT plotted here "
        "-- see this script's module docstring for why, and "
        "docs/paper_reference_notes.md / docs/project_decisions.md "
        "for the corresponding future-work entries."
    )


if __name__ == "__main__":
    main()
