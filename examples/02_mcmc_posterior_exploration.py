"""
Example 2: MCMC posterior exploration.

Takes the same FIN379 fit from Example 1, but samples the full
Bayesian posterior with emcee instead of stopping at the
least-squares point estimate: produces a corner plot, reports the
standard MCMC convergence diagnostics (acceptance fraction,
per-parameter autocorrelation time), and compares the posterior
median/68% credible interval against the least-squares
uncertainties.

Run

    python examples/02_mcmc_posterior_exploration.py

This uses a short chain (n_steps=2000) so the example finishes in
under a minute -- for a publication-quality posterior, increase
n_steps and check that n_steps/burn_in comfortably exceeds ~50x
the reported autocorrelation times (see emcee's own convergence
guidance; PySeshat's MCMCResult.autocorrelation_time exposes the
per-parameter estimate needed to check this).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pyseshat.io.orbit_parser import (
    parse_orbit_file,
)

from pyseshat.inference.optimizer import (
    fit_visual_orbit_mcmc,
)


REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    REPO_ROOT
    / "validation"
    / "benchmark_systems"
    / "FIN379"
    / "FIN379.inp"
)

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    parsed = parse_orbit_file(
        str(DATA_FILE)
    )

    elements = parsed["elements"]

    visual = parsed["visual_data"]

    rv = parsed["rv_data"]

    t = visual[:, 0]

    theta = np.deg2rad(
        visual[:, 1]
    )

    rho = visual[:, 2]

    sigma = visual[:, 3]

    X_obs = rho * np.cos(theta)

    Y_obs = rho * np.sin(theta)

    # --------------------------------------------------
    # fit_visual_orbit_mcmc() first runs the same
    # least-squares fit as Example 1 to find the maximum-
    # likelihood point and its covariance, then seeds an
    # emcee ensemble sampler around it (random_scale
    # controls the initial walker spread as a fraction of
    # each parameter's least-squares uncertainty).
    # --------------------------------------------------

    mcmc_result = fit_visual_orbit_mcmc(
        elements,
        t,
        X_obs,
        Y_obs,
        sigma,
        sigma,
        rv_data=rv,
        n_walkers=32,
        n_steps=2000,
        burn_in=400,
        thin=5,
        rng=np.random.default_rng(42),
    )

    print(
        f"Drew {mcmc_result.n_samples} posterior samples "
        f"over {mcmc_result.n_parameters} parameters "
        f"({mcmc_result.n_walkers} walkers x "
        f"{mcmc_result.n_steps} steps, "
        f"burn_in={mcmc_result.burn_in}, "
        f"thin={mcmc_result.thin})."
    )

    # --------------------------------------------------
    # Convergence diagnostics. A low acceptance fraction
    # (<~0.1 or >~0.7) usually means the proposal spread
    # needs adjusting; an autocorrelation time close to or
    # exceeding n_steps/thin means the chain hasn't
    # decorrelated enough to trust the posterior yet.
    # --------------------------------------------------

    print()

    print(
        "Mean acceptance fraction: "
        f"{np.mean(mcmc_result.acceptance_fraction):.3f}"
    )

    print()

    if mcmc_result.autocorrelation_time is None:

        # emcee's autocorrelation-time estimator refuses to
        # return a value when the chain is too short relative
        # to its own estimate (fewer than ~50x tau steps) --
        # exactly the tradeoff made here for a fast-running
        # example. Increase n_steps for a chain long enough to
        # get a real estimate.

        print(
            "Autocorrelation time: not available -- the chain "
            "is too short relative to its own decorrelation "
            "time for emcee's estimator to trust (increase "
            "n_steps for a real estimate; this is expected for "
            "this example's deliberately short chain)."
        )

    else:

        print("Per-parameter autocorrelation time:")

        for name, tau in zip(
            mcmc_result.parameter_names,
            mcmc_result.autocorrelation_time,
        ):

            print(
                f"  {name:>6}: {tau:.1f} steps"
            )

    # --------------------------------------------------
    # Posterior median +/- 68% credible interval,
    # compared against the least-squares point estimate
    # carried in mcmc_result.elements (the ML solution
    # used to seed the walkers).
    # --------------------------------------------------

    print()

    print(
        f"{'Parameter':>10}"
        f"{'Least sq.':>14}"
        f"{'Posterior med.':>16}"
        f"{'-1 sigma':>12}"
        f"{'+1 sigma':>12}"
    )

    lower, median, upper = np.percentile(
        mcmc_result.samples,
        [16, 50, 84],
        axis=0,
    )

    for i, name in enumerate(
        mcmc_result.parameter_names
    ):

        least_squares_value = getattr(
            mcmc_result.elements,
            name,
        )

        print(
            f"{name:>10}"
            f"{least_squares_value:>14.4f}"
            f"{median[i]:>16.4f}"
            f"{median[i] - lower[i]:>12.4f}"
            f"{upper[i] - median[i]:>12.4f}"
        )

    # --------------------------------------------------
    # Corner plot of the full posterior.
    # --------------------------------------------------

    fig = mcmc_result.corner_plot()

    fig.savefig(
        OUTPUT_DIR / "fin379_corner.png",
        dpi=150,
    )

    print()

    print(
        "Saved corner plot to "
        f"{OUTPUT_DIR / 'fin379_corner.png'}"
    )


if __name__ == "__main__":

    main()
