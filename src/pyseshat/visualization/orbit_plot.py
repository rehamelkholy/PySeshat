import numpy as np
import matplotlib.pyplot as plt


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