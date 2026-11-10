"""
Generate the Capella MCMC posterior corner plot for the paper's
Section 2.4/2.7 representative-MCMC-posterior figure (the
near-circular-orbit eccentricity-argument-of-periastron
degeneracy).

Why this script exists
------------------------------------------------------------
``MCMCResult.corner_plot()`` (src/pyseshat/inference/mcmc_result.py)
is unit-tested (tests/test_corner_plot.py) and is demonstrated in
examples/02_mcmc_posterior_exploration.py -- but neither of those
ever writes a corner-plot figure to disk (the test calls
``plt.close(fig)`` immediately; the example script only calls
``plt.show()``), and no benchmark-generation script
(``scripts/generate_benchmark_plots.py``,
``scripts/generate_triple_orbit_plots.py``) has ever invoked it
either -- both of those only call
``pyseshat.inference.optimizer.fit_visual_orbit`` (the point-estimate
optimizer), never ``fit_visual_orbit_mcmc``.

So: corner_plot() is tested and demonstrated in one example, but no
benchmark-generation script runs it, and no MCMC chain/samples for
Capella (or any other system) exist on disk apart from those this
script produces.

Why Capella
------------------------------------------------------------
Capella's own point-estimate fit (validation/results/Capella_orbit.json)
already shows the signature of this degeneracy directly:
eccentricity e=0.00089 (expected/literature) vs e=0.003265 (fitted)
-- a near-circular orbit -- while omega (argument of periastron) is
fit to 285.94 deg against an expected 342.6 deg, a huge (31%
relative) discrepancy. This is exactly what the e-omega degeneracy
predicts: when e -> 0, omega becomes only weakly constrained by the
data (periastron itself becomes an ill-defined reference point), so
the point estimate can land far from the literature value even
though the orbit shape (and the RV/astrometric curves themselves)
are barely affected. An MCMC posterior is the correct way to show
this -- a corner plot reveals the e/omega panel's covariance
structure (or, for CI-only marginals, that e is pinned near 0 while
omega's posterior is broad/multi-modal) directly, which a single
point estimate cannot.

What this script does
------------------------------------------------------------
1. Parses Capella_orbit's data file exactly as
   scripts/generate_benchmark_plots.py's own ``_fit()`` helper does.
2. Runs ``fit_visual_orbit_mcmc`` (least-squares starting point,
   then emcee posterior sampling) -- this is the MCMC run itself.
3. Saves the resulting chain/samples as a compressed .npz archive
   under validation/results/ (so the MCMC run itself is a durable,
   inspectable artifact, not just a transient in-memory object) --
   this makes future_run=False in generate_for_system() below
   possible: the same saved samples can be re-loaded for a corner
   plot regeneration without re-running the (~1-2 minute) MCMC.
4. Generates and saves the corner plot as PDF, with the paper
   pipeline's grid-off convention (show_grid=False, which is also
   corner_plot()'s own default -- see mcmc_result.py's docstring).

Run
    MPLBACKEND=Agg PYTHONPATH=src python scripts/generate_capella_corner_plot.py
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from pyseshat.io.orbit_parser import parse_orbit_file
from pyseshat.inference.optimizer import fit_visual_orbit_mcmc


REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    REPO_ROOT
    / "validation"
    / "benchmark_systems"
    / "Capella_orbit"
    / "Capella_orbit.inp"
)

PLOTS_DIR = REPO_ROOT / "validation" / "results" / "plots"

SAMPLES_PATH = (
    REPO_ROOT / "validation" / "results" / "Capella_orbit_mcmc_samples.npz"
)

# Same walker/step/burn-in/thin/seed settings as
# tests/test_corner_plot.py's FIN379 fit, for consistency across
# this codebase's MCMC runs; Capella has both visual and RV data
# (10 free parameters, same as FIN379), so the same walker count is
# appropriate. n_steps is higher than the test's 500 (which only
# needs to *run*, not converge tightly) since this chain is a paper
# figure.
N_WALKERS = 32
N_STEPS = 4000
BURN_IN = 1000
THIN = 4
RANDOM_SEED = 42


def run_capella_mcmc():

    parsed = parse_orbit_file(str(DATA_FILE))

    elements = parsed["elements"]
    visual = parsed["visual_data"]
    rv_data = parsed["rv_data"]

    t = visual[:, 0]
    theta = np.deg2rad(visual[:, 1])
    rho = visual[:, 2]
    sigma = visual[:, 3]

    X_obs = rho * np.cos(theta)
    Y_obs = rho * np.sin(theta)

    posterior = fit_visual_orbit_mcmc(
        elements,
        t,
        X_obs,
        Y_obs,
        sigma,
        sigma,
        rv_data=rv_data,
        n_walkers=N_WALKERS,
        n_steps=N_STEPS,
        burn_in=BURN_IN,
        thin=THIN,
        rng=np.random.default_rng(RANDOM_SEED),
    )

    return posterior


def save_samples(posterior):

    SAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        SAMPLES_PATH,
        samples=posterior.samples,
        parameter_names=np.array(posterior.parameter_names),
    )

    print(
        f"Capella_orbit: saved MCMC chain "
        f"({posterior.samples.shape[0]} samples x "
        f"{posterior.samples.shape[1]} parameters) to {SAMPLES_PATH}"
    )


def _recenter_omega_for_plotting(posterior):
    """
    Return a PLOTTING-ONLY copy of ``posterior`` with the ``omega``
    column remapped from this codebase's canonical [0, 360) range
    (see ``optimizer.py``'s ``_wrap_periodic_mcmc_samples`` and its
    docstring) into (-180, 180].

    Why this exists
    --------------------------------------------------------------
    Wrapping ``omega`` into [0, 360) fixed the real bug (an
    unbounded, unwrapped raw MCMC chain blowing the axis out to
    +/-50000) but left a cosmetic side effect for THIS system
    specifically: Capella's omega posterior (median ~282 deg, 68% CI
    roughly [248, 307] deg, with a longer tail toward higher values)
    sits close enough to the 360/0 wrap point that a real, modest
    fraction of samples (~1.6%) land pressed against both the 0-deg
    and 360-deg panel edges -- physically one continuous mode, but
    visually split across the two ends of the marginal histogram and
    every omega-involving 2D panel.

    Recentering the SAME samples into (-180, 180] moves this
    system's mode away from either edge (282 deg wraps to 282-360 =
    -78 deg, comfortably inside (-180, 180] with its full +/-~35 deg
    spread nowhere near either boundary), eliminating the visual
    split without changing the underlying posterior at all -- exactly
    as legitimate as the [0, 360) wrap itself, since both are just
    different representative choices for the same periodic
    coordinate.

    This is PLOTTING-ONLY: the saved ``.npz`` chain
    (``save_samples()``) keeps the codebase's canonical [0, 360)
    convention (matching ``OrbitalElements.normalize_angles()``
    elsewhere), so downstream numeric consumers (credible intervals,
    future re-plots, other scripts) are unaffected -- only this
    function's returned copy, used for one figure, is remapped.

    Returns
    -------
    MCMCResult
        A new object (``dataclasses.replace``) sharing every field
        with ``posterior`` except ``samples``, whose ``omega`` column
        has been remapped via ``((omega + 180) % 360) - 180``.
    """

    names = list(posterior.parameter_names)

    remapped = np.array(posterior.samples, dtype=float, copy=True)

    if "omega" in names:

        idx = names.index("omega")

        remapped[:, idx] = (
            (remapped[:, idx] + 180.0) % 360.0
        ) - 180.0

    return dataclasses.replace(
        posterior,
        samples=remapped,
    )


def generate_corner_plot(posterior, output_format="pdf", show_grid=False):

    out_dir = PLOTS_DIR / "Capella_orbit"
    out_dir.mkdir(parents=True, exist_ok=True)

    save_path = out_dir / f"corner.{output_format}"

    plot_posterior = _recenter_omega_for_plotting(posterior)

    names = list(plot_posterior.parameter_names)

    omega_idx = names.index("omega")

    # corner.corner()'s ``range`` kwarg requires one entry per
    # displayed dimension (no per-column "auto" opt-out once any
    # entry is given) -- so every OTHER column gets its own natural
    # [min, max] (matching corner.corner()'s own default behavior),
    # and only ``omega`` gets the fixed (-180, 180) window that
    # actually does the recentering.
    ranges = [
        (
            (-180.0, 180.0)
            if i == omega_idx
            else (
                float(np.min(plot_posterior.samples[:, i])),
                float(np.max(plot_posterior.samples[:, i])),
            )
        )
        for i in range(len(names))
    ]

    fig = plot_posterior.corner_plot(
        show_grid=show_grid,
        output_format=output_format,
        save_path=save_path,
        range=ranges,
    )

    plt.close(fig)

    print(f"Capella_orbit: wrote corner plot to {save_path}")

    return save_path


def load_saved_posterior():
    """
    Reconstruct a plotting-capable ``MCMCResult`` from the saved
    ``.npz`` chain, without re-running the ~1-2 minute MCMC.

    Only ``samples``/``parameter_names``/``log_probability`` are
    populated with real data (everything ``corner_plot()`` and
    ``_recenter_omega_for_plotting()`` actually use); the remaining
    dataclass fields (``elements``, ``optimizer_result``, etc.) are
    not part of the saved archive and are left at benign defaults --
    fine for regenerating a plot, not a substitute for a real
    ``fit_visual_orbit_mcmc()`` result if those fields are needed.
    """

    from pyseshat.inference.mcmc_result import MCMCResult

    data = np.load(SAMPLES_PATH, allow_pickle=True)

    samples = data["samples"]

    parameter_names = list(data["parameter_names"])

    return MCMCResult(
        elements=None,
        optimizer_result=None,
        residuals=None,
        covariance=None,
        uncertainties=None,
        samples=samples,
        log_probability=np.zeros(samples.shape[0]),
        parameter_names=parameter_names,
    )


def main():

    # Reuse the saved chain when one already exists, rather than
    # re-running the MCMC: this script's RNG seed does not give a
    # bit-reproducible chain across separate process runs
    # (non-angular columns differ at the float level
    # between two same-seed runs, most likely due to BLAS/scipy
    # optimizer floating-point-order nondeterminism upstream of the
    # sampler, not this script's own logic), so re-running here would
    # silently produce a slightly different "final" posterior than
    # the one already checked and reported. A plot-only change (like
    # recentering omega's display range) should not perturb those
    # already-reported numbers.

    if SAMPLES_PATH.exists():

        print(f"Capella_orbit: reusing saved MCMC chain at {SAMPLES_PATH}")

        posterior = load_saved_posterior()

    else:

        posterior = run_capella_mcmc()

        save_samples(posterior)

    generate_corner_plot(posterior)


if __name__ == "__main__":
    main()
