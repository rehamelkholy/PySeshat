import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pyseshat.orbit.orbital_elements import OrbitalElements
from pyseshat.orbit.orbit_model import compute_visual_orbit
from pyseshat.orbit.rv_model import compute_rv_model
from pyseshat.visualization.orbit_plot import (
    plot_visual_orbit,
    plot_residuals,
    plot_rv_orbit,
)


# ==========================================================
# plot_visual_orbit
# ==========================================================
#
# Correctness, not just "runs without raising": the actual
# scatter/line data drawn on the axes must reproduce the
# X_obs/Y_obs and orbit-model arrays passed in, per
# docs/development_plan_phase2.md Phase 4 item 1
# ("Strengthening these to assert on actual plotted data...
# is real, missing test coverage").

def test_plot_visual_orbit():

    x = np.linspace(-1.0, 1.0, 20)
    y = np.sin(x)

    orbit = {
        "X": x,
        "Y": y,
    }

    fig, ax = plot_visual_orbit(
        x,
        y,
        orbit,
        show=False,
    )

    assert fig is not None
    assert ax is not None
    assert ax.get_xlabel() == "East (arcsec)"
    assert ax.get_ylabel() == "North (arcsec)"

    plt.close(fig)


def test_plot_visual_orbit_draws_the_actual_observations():

    # Distinct, easy-to-mis-transpose-or-swap X/Y values so a
    # swapped-axis or transposed-data regression would be
    # caught, not just "some points near the right place."
    #
    # X_obs/Y_obs are this codebase's internal North/East
    # convention (X=North, Y=East -- settled 2026-08-27, see
    # docs/project_decisions.md and plot_visual_orbit's own
    # docstring). The standard North-up/East-left display swaps
    # them onto the screen axes: North (X) plotted vertically,
    # East (Y) plotted horizontally (with the axis inverted, but
    # the *plotted data itself* is un-negated -- see below).

    X_obs = np.array([0.10, 0.25, -0.30, -0.15])
    Y_obs = np.array([0.40, -0.05, 0.20, -0.35])

    orbit = {
        "X": np.linspace(-0.5, 0.5, 50),
        "Y": np.linspace(-0.5, 0.5, 50) ** 2,
    }

    fig, ax = plot_visual_orbit(
        X_obs,
        Y_obs,
        orbit,
        show=False,
    )

    # The observation scatter is the first PathCollection
    # added to the axes (see plot_visual_orbit's
    # "Observations" block, added before "Primary star").

    scatter = ax.collections[0]

    plotted_offsets = np.asarray(
        scatter.get_offsets()
    )

    # Horizontal (matplotlib x) data is East (Y_obs); vertical
    # (matplotlib y) data is North (X_obs). The axis itself is
    # inverted (checked separately below) rather than the data
    # negated, matching standard practice for a plotted East/RA
    # axis.

    np.testing.assert_allclose(
        plotted_offsets[:, 0],
        Y_obs,
    )

    np.testing.assert_allclose(
        plotted_offsets[:, 1],
        X_obs,
    )

    # The fitted-orbit curve is the first Line2D.

    line = ax.get_lines()[0]

    np.testing.assert_allclose(
        line.get_xdata(),
        orbit["Y"],
    )

    np.testing.assert_allclose(
        line.get_ydata(),
        orbit["X"],
    )

    plt.close(fig)


def test_plot_visual_orbit_is_north_up_east_left():

    # Literature-convention regression test (settled 2026-08-27,
    # see docs/project_decisions.md): the plot must show North up
    # (unmodified on the vertical axis) and East left (horizontal
    # axis inverted). A point at PA=90 deg (due East: X=North=0,
    # Y=East=rho) must render to the LEFT of the origin -- matching
    # the IAU's own definition that PA increases counterclockwise
    # when North is up.

    X_obs = np.array([0.0])   # North component
    Y_obs = np.array([1.0])   # East component (PA = 90 deg)

    orbit = {"X": np.array([0.0]), "Y": np.array([0.0])}

    fig, ax = plot_visual_orbit(X_obs, Y_obs, orbit, show=False)

    assert ax.xaxis_inverted()

    # Data coordinates are un-negated (East's true value, 1.0,
    # still sits on the x-data axis) -- inversion is what actually
    # moves it to the left on screen, not a sign flip of the data.

    scatter = ax.collections[0]
    plotted_offsets = np.asarray(scatter.get_offsets())

    assert plotted_offsets[0, 0] == pytest.approx(1.0)
    assert plotted_offsets[0, 1] == pytest.approx(0.0)

    plt.close(fig)


def test_plot_visual_orbit_axes_are_not_swapped_for_a_real_orbit():

    # Regression guard against a transposed-axis bug: build a
    # real orbit model from OrbitalElements and confirm
    # compute_visual_orbit's own X (North)/Y (East) -- not some
    # other permutation of them -- is exactly what lands on the
    # plot's vertical/horizontal axes respectively.

    elements = OrbitalElements(
        P=10.0, T0=0.0, e=0.3, a=1.0,
        Omega=45.0, omega=90.0, i=60.0,
    )

    t_model = np.linspace(0.0, 10.0, 30)

    orbit = compute_visual_orbit(t_model, elements)

    fig, ax = plot_visual_orbit(
        orbit["X"],
        orbit["Y"],
        orbit,
        show=False,
    )

    line = ax.get_lines()[0]

    np.testing.assert_allclose(line.get_xdata(), orbit["Y"])
    np.testing.assert_allclose(line.get_ydata(), orbit["X"])

    # Sanity: X and Y are not simply equal to each other (which
    # would pass a swapped-axis bug silently for a degenerate
    # test orbit) -- this orbit's e/i/Omega/omega give it real
    # asymmetry.

    assert not np.allclose(orbit["X"], orbit["Y"])

    plt.close(fig)


# ==========================================================
# plot_residuals
# ==========================================================

def test_plot_residuals():

    t = np.arange(10.0)

    residuals = np.zeros(20)

    fig, axes = plot_residuals(
        t,
        residuals,
        show=False,
    )

    assert fig is not None
    assert len(axes) == 2

    plt.close(fig)


def test_plot_residuals_draws_the_actual_resX_resY_split():

    t = np.arange(5.0)

    resX = np.array([0.1, -0.2, 0.3, -0.4, 0.5])
    resY = np.array([-0.5, 0.4, -0.3, 0.2, -0.1])

    residuals = np.concatenate((resX, resY))

    fig, axes = plot_residuals(
        t,
        residuals,
        show=False,
    )

    x_scatter = axes[0].collections[0]
    y_scatter = axes[1].collections[0]

    np.testing.assert_allclose(
        np.asarray(x_scatter.get_offsets())[:, 1],
        resX,
    )

    np.testing.assert_allclose(
        np.asarray(y_scatter.get_offsets())[:, 1],
        resY,
    )

    plt.close(fig)


# ==========================================================
# plot_rv_orbit
# ==========================================================

def test_plot_rv_orbit():

    elements = OrbitalElements(
        P=10.0,
        T0=0.0,
        e=0.2,
        a=1.0,
        Omega=30.0,
        omega=60.0,
        i=70.0,
        K1=20.0,
        K2=30.0,
        V0=0.0,
    )

    rv = np.array([
        [0.0, 20.0, 1.0, 0],
        [2.5, -30.0, 1.0, 1],
        [5.0, -20.0, 1.0, 0],
        [7.5, 30.0, 1.0, 1],
    ])

    fig, ax = plot_rv_orbit(
        rv,
        elements,
        show=False,
    )

    assert fig is not None
    assert ax is not None

    plt.close(fig)


def test_plot_rv_orbit_draws_the_actual_rv_observations():

    elements = OrbitalElements(
        P=10.0, T0=0.0, e=0.2, a=1.0,
        Omega=30.0, omega=60.0, i=70.0,
        K1=20.0, K2=30.0, V0=5.0,
    )

    rv = np.array([
        [0.0, 20.0, 1.0, 0],
        [2.5, -30.0, 1.0, 1],
        [5.0, -20.0, 1.0, 0],
        [7.5, 30.0, 1.0, 1],
    ])

    fig, ax = plot_rv_orbit(
        rv,
        elements,
        show=False,
    )

    # Both primary (component 0) and secondary (component 1)
    # scatter series must be present, each drawing exactly the
    # RV values of its own rows -- not the other component's,
    # and not some component-blind average.

    primary_mask = rv[:, 3].astype(int) == 0
    secondary_mask = rv[:, 3].astype(int) == 1

    all_plotted_y = [
        np.asarray(collection.get_offsets())[:, 1]
        for collection in ax.collections
    ]

    plotted_primary = next(
        values for values in all_plotted_y
        if np.allclose(np.sort(values), np.sort(rv[primary_mask, 1]))
    )

    plotted_secondary = next(
        values for values in all_plotted_y
        if np.allclose(np.sort(values), np.sort(rv[secondary_mask, 1]))
    )

    np.testing.assert_allclose(
        np.sort(plotted_primary),
        np.sort(rv[primary_mask, 1]),
    )

    np.testing.assert_allclose(
        np.sort(plotted_secondary),
        np.sort(rv[secondary_mask, 1]),
    )

    plt.close(fig)


def test_plot_rv_orbit_phase_folded_model_is_centered_at_zero_phase():

    # Literature-convention check: phase-folded RV curves are
    # conventionally zeroed at periastron passage (T0), per
    # docs/development_plan_phase2.md Phase 4 item 2. Confirm
    # plot_rv_orbit's own phase=True model curve construction
    # does exactly this: at model-phase 0.0, its time argument
    # is elements.T0 (matching compute_rv_model's own T0-relative
    # phase convention used throughout inference/).

    elements = OrbitalElements(
        P=10.0, T0=3.0, e=0.2, a=1.0,
        Omega=30.0, omega=60.0, i=70.0,
        K1=20.0, K2=30.0, V0=0.0,
    )

    rv = np.array([
        [3.0, 20.0, 1.0, 0],
        [8.0, -20.0, 1.0, 0],
    ])

    fig, ax = plot_rv_orbit(
        rv,
        elements,
        phase=True,
        show=False,
    )

    model_line = ax.get_lines()[0]

    model_phase = model_line.get_xdata()

    model_rv = model_line.get_ydata()

    # RV at phase 0 (periastron) from the smooth model curve
    # must match compute_rv_model evaluated directly at T0.

    index_zero_phase = np.argmin(np.abs(model_phase))

    direct_rv_at_T0 = compute_rv_model(
        np.array([[elements.T0, 0.0, 1.0, 0]]),
        elements,
    )[0]

    assert model_phase[index_zero_phase] == pytest.approx(0.0, abs=1e-3)

    assert model_rv[index_zero_phase] == pytest.approx(
        direct_rv_at_T0,
        abs=0.5,
    )

    plt.close(fig)


def test_plot_rv_orbit_empty():

    elements = OrbitalElements(
        P=10.0,
        T0=0.0,
        e=0.2,
        a=1.0,
        Omega=30.0,
        omega=60.0,
        i=70.0,
        K1=20.0,
        K2=30.0,
        V0=0.0,
    )

    with pytest.raises(ValueError):

        plot_rv_orbit(
            np.empty((0, 4)),
            elements,
            show=False,
        )
