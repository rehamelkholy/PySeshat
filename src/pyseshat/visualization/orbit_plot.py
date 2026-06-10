import numpy as np
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
    show=True
):
    """
    Plot observed and modeled visual binary orbit.

    Parameters
    ----------
    X_obs, Y_obs : ndarray
        Observed Cartesian coordinates

    orbit : dict
        Output from compute_visual_orbit()

    title : str
        Plot title
    """

    X_model = orbit["X"]
    Y_model = orbit["Y"]

    fig, ax = plt.subplots(figsize=(7, 7))

    # ------------------------------------------------------
    # Observations
    # ------------------------------------------------------

    ax.scatter(
        X_obs,
        Y_obs,
        s=40,
        label="Observations"
    )

    # ------------------------------------------------------
    # Orbit model
    # ------------------------------------------------------

    ax.plot(
        X_model,
        Y_model,
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

    ax.set_xlabel("X (arcsec)")
    ax.set_ylabel("Y (arcsec)")

    ax.set_title(title)

    ax.legend()

    ax.set_aspect("equal")

    ax.grid(True)

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
    show=True
):
    """
    Plot X and Y residuals versus time.

    Parameters
    ----------
    t : ndarray
        Observation epochs

    residuals : ndarray
        Concatenated residual vector:
            [resX, resY]
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

    axes[0].set_ylabel("X Residuals")

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

    axes[1].set_ylabel("Y Residuals")

    axes[1].set_xlabel("Epoch")

    axes[1].grid(True)

    fig.suptitle(title)

    plt.tight_layout()

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
    show=True
):
    """
    Plot observed and modeled radial
    velocity curves.
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

    ax.grid(True)

    ax.legend()

    if show:
        plt.show()

    return fig, ax