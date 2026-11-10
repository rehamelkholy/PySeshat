"""
Regression test for MCMCResult.corner_plot(), with assertions on
the plot's content rather than just checking that it runs.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pyseshat.io.orbit_parser import (
    parse_orbit_file
)

from pyseshat.inference.optimizer import (
    fit_visual_orbit_mcmc
)


def _fit_fin379_posterior():

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

    return fit_visual_orbit_mcmc(
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


@pytest.fixture(scope="module")
def fin379_posterior():

    return _fit_fin379_posterior()


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


# ==========================================================
# show_grid / output_format / save_path
#
# Tests for the show_grid/output_format/save_path parameters.
# Unlike every other plotting
# function in this codebase, corner_plot()'s show_grid defaults
# to False (not True) -- corner.corner() draws no grid lines by
# default and this function never added any, so False is what
# actually preserves this function's *original* behavior. These
# tests both confirm that default and confirm show_grid=True is
# still available as an opt-in. Uses the module-scoped
# fin379_posterior fixture so the (~5s) MCMC fit only runs once
# for this whole file, not once per test.
# ==========================================================

def _lower_triangle_axes(fig, ndim):

    axes = np.asarray(fig.axes, dtype=object).reshape(ndim, ndim)

    return [
        axes[row, column]
        for row in range(ndim)
        for column in range(row + 1)
    ]


def test_corner_plot_show_grid_defaults_to_off(fin379_posterior):

    fig = fin379_posterior.corner_plot()

    ndim = fin379_posterior.n_parameters

    for ax in _lower_triangle_axes(fig, ndim):

        assert not ax.xaxis.get_gridlines()[0].get_visible()

    plt.close(fig)


def test_corner_plot_show_grid_true_is_opt_in(fin379_posterior):

    fig = fin379_posterior.corner_plot(show_grid=True)

    ndim = fin379_posterior.n_parameters

    for ax in _lower_triangle_axes(fig, ndim):

        assert ax.xaxis.get_gridlines()[0].get_visible()

    plt.close(fig)


def test_corner_plot_save_path_writes_requested_format(
    fin379_posterior, tmp_path,
):

    out = tmp_path / "corner.pdf"

    fig = fin379_posterior.corner_plot(
        output_format="pdf",
        save_path=out,
    )

    assert out.exists()
    assert out.stat().st_size > 0
    assert out.read_bytes()[:4] == b"%PDF"

    plt.close(fig)
