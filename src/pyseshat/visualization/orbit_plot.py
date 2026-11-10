import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from pyseshat.orbit.rv_model import (
        compute_rv_orbit
    )

from pyseshat.orbit.phase import (
    orbital_phase
)

# ==========================================================
# Visual orbit plot
# ==========================================================

def plot_visual_orbit(
    X_obs,
    Y_obs,
    orbit,
    title="Visual Binary Orbit",
    show=True,
    show_grid=True,
    output_format="png",
    save_path=None
):
    """
    Plot observed and modeled visual binary orbit, in the
    standard astronomical North-up/East-left orientation.

    Parameters
    ----------
    X_obs, Y_obs : ndarray
        Observed Cartesian coordinates, in this codebase's
        internal convention ``X = rho*cos(theta)`` (North/Dec
        offset), ``Y = rho*sin(theta)`` (East/RA*cos(Dec) offset)
        -- see `Orientation convention` below.

    orbit : dict
        Output from compute_visual_orbit()

    title : str
        Plot title

    show_grid : bool, default=True
        Whether to draw axes grid lines. Default preserves this
        function's original (always-on) behavior, so existing
        callers are unaffected.

    output_format : str, default="png"
        File format passed to ``Figure.savefig()`` (any format
        Matplotlib's ``savefig`` natively supports, e.g. "png",
        "pdf", "svg"). Only takes effect when ``save_path`` is
        given.

    save_path : str or Path, optional
        If given, save the figure to this path (in ``output_format``)
        before returning. If None (default), the figure is not
        saved internally -- matching this function's original
        behavior, where the caller retrieves ``(fig, ax)`` and
        saves it externally.

    Orientation convention
    --------------------------------------------------------
    This codebase's internal `X = rho*cos(theta)`,
    `Y = rho*sin(theta)` convention (see
    `orbit.orbit_model.compute_visual_orbit`'s docstring) uses
    `theta` as the position angle read directly from `.inp`
    files' PA column, which follows the standard astronomical/WDS
    definition: measured from North (0 deg), turning positive
    toward increasing right ascension (East) -- confirmed via the
    IAU's own definition ("the angle measured relative to the
    north celestial pole, turning positive into the direction of
    the right ascension") and the Washington Double Star
    Catalog's own usage (PA=180 deg = "due south"). Since `theta`
    is measured FROM North, `cos(theta)` is by construction the
    North-component coefficient and `sin(theta)` the
    East-component coefficient: this codebase's `X` is the North
    (Dec) offset, `Y` is the East (RA*cos(Dec)) offset.

    The standard published convention for a visual-binary-orbit
    figure is North up, East left (used uniformly across the
    field -- e.g. the `orbitize` package's own documented
    convention, and every WDS/speckle-astrometry orbit figure).
    With `X` = North and `Y` = East as derived above, that means:
    North (`X`) goes on the vertical axis, unmodified; East (`Y`)
    goes on the horizontal axis with the axis *inverted* (not
    negated -- axis labels still show the true East-offset value,
    matching standard practice, e.g. how a plotted RA axis is
    conventionally inverted) so that increasing East appears to
    the left. This is exactly the transformation implemented
    below. `tests/test_orbit_plot.py` regression-tests both the
    axis-inversion call and the counterclockwise PA sense this
    produces (a point at PA=90 deg, due East, plots to the left of
    the origin, matching the IAU's own statement that PA increases
    counterclockwise when North is up).
    """

    X_model = orbit["X"]
    Y_model = orbit["Y"]

    fig, ax = plt.subplots(figsize=(7, 7))

    # ------------------------------------------------------
    # Observations
    #
    # North (X) is plotted on the vertical axis, East (Y) on
    # the horizontal axis (with the axis inverted below) -- see
    # the "Orientation convention" note above.
    # ------------------------------------------------------

    ax.scatter(
        Y_obs,
        X_obs,
        s=40,
        label="Observations"
    )

    # ------------------------------------------------------
    # Orbit model
    # ------------------------------------------------------

    ax.plot(
        Y_model,
        X_model,
        linewidth=2,
        label="Fitted Orbit"
    )

    # ------------------------------------------------------
    # Primary star
    # ------------------------------------------------------

    ax.scatter(
        0,
        0,
        marker="*",
        s=200,
        label="Primary"
    )

    # ------------------------------------------------------
    # Formatting
    # ------------------------------------------------------

    ax.set_xlabel("East (arcsec)")
    ax.set_ylabel("North (arcsec)")

    # East increases to the left, per the standard North-up/
    # East-left convention -- see the docstring above.
    ax.invert_xaxis()

    ax.set_title(title)

    ax.legend()

    ax.set_aspect("equal")

    if show_grid:
        ax.grid(True)

    if save_path is not None:
        fig.savefig(
            save_path,
            format=output_format,
            dpi=150,
            bbox_inches="tight"
        )

    if show:
        plt.show()

    return fig, ax


# ==========================================================
# Residual plots
# ==========================================================

def plot_residuals(
    t,
    residuals,
    title="Residual Diagnostics",
    show=True,
    show_grid=True,
    output_format="png",
    save_path=None
):
    """
    Plot X (North) and Y (East) residuals versus time.

    Parameters
    ----------
    t : ndarray
        Observation epochs

    residuals : ndarray
        Concatenated residual vector:
            [resX, resY]
        In this codebase's internal convention (see
        `plot_visual_orbit`'s docstring), `resX` is the North
        (declination) residual and `resY` the East
        (right-ascension*cos(declination)) residual -- the same
        `(observed - model) / sigma` values
        `inference.terms.visual.visual_xy_residuals` computes.
        Unlike `plot_visual_orbit`, this is two independent
        time-series subplots rather than a 2D spatial plot, so no
        North-up/East-left axis transform applies here -- both
        panels simply plot residual value vs. epoch.

    show_grid : bool, default=True
        Whether to draw axes grid lines on both panels. Default
        preserves this function's original (always-on) behavior.

    output_format : str, default="png"
        File format passed to ``Figure.savefig()`` (e.g. "png",
        "pdf", "svg"). Only takes effect when ``save_path`` is
        given.

    save_path : str or Path, optional
        If given, save the figure to this path (in ``output_format``)
        before returning. If None (default), matches this
        function's original behavior of not saving internally.
    """

    n = len(t)

    resX = residuals[:n]
    resY = residuals[n:]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 6),
        sharex=True
    )

    # ------------------------------------------------------
    # X residuals
    # ------------------------------------------------------

    axes[0].scatter(
        t,
        resX
    )

    axes[0].axhline(
        0,
        linestyle="--"
    )

    axes[0].set_ylabel("North Residuals")

    if show_grid:
        axes[0].grid(True)

    # ------------------------------------------------------
    # Y residuals
    # ------------------------------------------------------

    axes[1].scatter(
        t,
        resY
    )

    axes[1].axhline(
        0,
        linestyle="--"
    )

    axes[1].set_ylabel("East Residuals")

    axes[1].set_xlabel("Epoch")

    if show_grid:
        axes[1].grid(True)

    fig.suptitle(title)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(
            save_path,
            format=output_format,
            dpi=150,
            bbox_inches="tight"
        )

    if show:
        plt.show()

    return fig, axes

# ==========================================================
# RV orbit plot
# ==========================================================

def plot_rv_orbit(
    rv_data,
    elements,
    title="Radial Velocity Orbit",
    phase=False,
    show=True,
    show_grid=True,
    output_format="png",
    save_path=None
):
    """
    Plot observed and modeled radial velocity curves for both
    components of a spectroscopic binary.

    Parameters
    ----------
    rv_data : ndarray, shape (N, 4)
        Rows of ``(t, rv, sigma_rv, component)``, where
        ``component`` is 0 for the primary and 1 for the
        secondary (the same layout ``io.orbit_parser`` produces
        and ``inference.terms.radial_velocity`` consumes).
    elements : OrbitalElements
        Orbital elements (including ``K1``/``K2``/``V0``) used to
        draw the smooth model curve for each component.
    title : str
        Plot title.
    phase : bool
        If True, fold both the observations and the model curve
        on orbital phase (0-1), with phase 0 at ``elements.T0``
        (periastron) -- the standard literature convention; see
        ``tests/test_orbit_plot.py::test_plot_rv_orbit_phase_folded_model_is_centered_at_zero_phase``.
        If False, plot against raw time/epoch instead.
    show : bool
        If True, call ``plt.show()`` before returning.

    show_grid : bool, default=True
        Whether to draw axes grid lines. Default preserves this
        function's original (always-on) behavior.

    output_format : str, default="png"
        File format passed to ``Figure.savefig()`` (e.g. "png",
        "pdf", "svg"). Only takes effect when ``save_path`` is
        given.

    save_path : str or Path, optional
        If given, save the figure to this path (in ``output_format``)
        before returning. If None (default), matches this
        function's original behavior of not saving internally.

    Returns
    -------
    (Figure, Axes)

    Raises
    ------
    ValueError
        If ``rv_data`` is empty -- there is nothing to plot.
    """

    if len(rv_data) == 0:

        raise ValueError(
            "No RV observations available."
        )

    t = rv_data[:,0]

    if phase:

        x = orbital_phase(
            t,
            elements.P,
            elements.T0
        )

    else:

        x = t

    rv = rv_data[:,1]

    component = (
        rv_data[:,3]
        .astype(int)
    )

    fig, ax = plt.subplots(
        figsize=(8,5)
    )

    # ------------------------------------------------------
    # Smooth model curve
    # ------------------------------------------------------

    if phase:

        phase_model = np.linspace(
            0.0,
            1.0,
            1000
        )

        t_model = (
            elements.T0
            +
            phase_model
            * elements.P
        )

        x_model = phase_model

    else:

        t_model = np.linspace(
            np.min(t),
            np.max(t),
            1000
        )

        x_model = t_model

    rv1_model = compute_rv_orbit(
        t_model,
        elements,
        component=0
    )

    rv2_model = compute_rv_orbit(
        t_model,
        elements,
        component=1
    )

    ax.plot(
        x_model,
        rv1_model,
        linewidth=2,
        label="Primary Model"
    )

    ax.plot(
        x_model,
        rv2_model,
        linewidth=2,
        label="Secondary Model"
    )

    # ------------------------------------------------------
    # Observations
    # ------------------------------------------------------

    mask1 = (
        component == 0
    )

    mask2 = (
        component == 1
    )

    ax.scatter(
        x[mask1],
        rv[mask1],
        label="Primary RV"
    )

    ax.scatter(
        x[mask2],
        rv[mask2],
        label="Secondary RV"
    )

    if phase:

        ax.set_xlabel(
            "Orbital Phase"
        )

    else:

        ax.set_xlabel(
            "Epoch"
        )

    ax.set_ylabel(
        "RV (km/s)"
    )

    ax.set_title(
        title
    )

    if show_grid:
        ax.grid(True)

    ax.legend()

    if save_path is not None:
        fig.savefig(
            save_path,
            format=output_format,
            dpi=150,
            bbox_inches="tight"
        )

    if show:
        plt.show()

    return fig, ax