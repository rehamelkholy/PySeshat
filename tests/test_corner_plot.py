"""
Regression test for MCMCResult.corner_plot(), rewritten
2026-08-27 from a print-only demo script (no ``def test_...``
function, no assertions -- pytest collected 0 items from it
regardless of correctness, the same defect fixed for
test_rv_model.py/test_statistics.py/test_reduced_chi_square_rv.py
in docs/project_decisions.md's "Whole-codebase audit" section)
into a real assertion-bearing test, per
docs/development_plan_phase2.md Phase 4's charge to strengthen
plot-correctness coverage.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from pyseshat.io.orbit_parser import (
    parse_orbit_file
)

from pyseshat.inference.optimizer import (
    fit_visual_orbit_mcmc
)


def test_corner_plot_has_one_panel_row_column_per_parameter():

    parsed = parse_orbit_file(
        "validation/benchmark_systems/FIN379/FIN379.inp"
    )

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
        n_walkers=20,
        n_steps=500,
        burn_in=100,
        thin=2,
        rng=np.random.default_rng(42),
    )

    fig = posterior.corner_plot()

    n_parameters = posterior.n_parameters

    # corner.corner() lays out an n x n grid of axes (upper
    # triangle blanked, not removed) for n parameters.

    assert len(fig.axes) == n_parameters ** 2

    plt.close(fig)


def test_corner_plot_respects_a_parameters_subset():

    parsed = parse_orbit_file(
        "validation/benchmark_systems/FIN379/FIN379.inp"
    )

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
        n_walkers=20,
        n_steps=500,
        burn_in=100,
        thin=2,
        rng=np.random.default_rng(42),
    )

    subset = ["P", "T0", "e"]

    fig = posterior.corner_plot(parameters=subset)

    assert len(fig.axes) == len(subset) ** 2

    plt.close(fig)
