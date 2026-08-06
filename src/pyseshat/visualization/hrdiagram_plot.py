from __future__ import annotations

import matplotlib.pyplot as plt

from pyseshat.evolution.hrdiagram import (
    HRDiagramData,
)


def plot_binary_hrdiagram(
    data: HRDiagramData,
    *,
    ax=None,
    save_path=None,
):
    """
    Plot a publication-quality binary Hertzsprung-Russell
    diagram.

    Parameters
    ----------
    data
        Scientific data required for the H-R diagram.

    ax
        Existing matplotlib Axes.

    save_path
        Optional output filename.

    Returns
    -------
    matplotlib.axes.Axes
    """

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(6.5, 5.0),
        )
    else:
        fig = ax.figure

    #
    # Evolutionary tracks.
    #
    ax.plot(
        data.primary_track.log_teff,
        data.primary_track.log_luminosity,
        linewidth=2.0,
        linestyle="-",
        label="Primary Track",
    )

    ax.plot(
        data.secondary_track.log_teff,
        data.secondary_track.log_luminosity,
        linewidth=2.0,
        linestyle="--",
        label="Secondary Track",
    )

    #
    # Observed primary.
    #
    ax.errorbar(
        data.primary_observables.log_teff,
        data.primary_observables.log_luminosity,
        xerr=data.primary_observables.sigma_log_teff,
        yerr=data.primary_observables.sigma_log_luminosity,
        fmt="o",
        markersize=7,
        capsize=3,
        linewidth=1.2,
        zorder=10,
        label="Primary Observation",
    )

    #
    # Observed secondary.
    #
    ax.errorbar(
        data.secondary_observables.log_teff,
        data.secondary_observables.log_luminosity,
        xerr=data.secondary_observables.sigma_log_teff,
        yerr=data.secondary_observables.sigma_log_luminosity,
        fmt="D",
        markersize=7,
        capsize=3,
        linewidth=1.2,
        zorder=10,
        label="Secondary Observation",
    )

    #
    # Best-fitting models.
    #
    ax.scatter(
        data.solution.primary.log_teff,
        data.solution.primary.log_luminosity,
        s=90,
        marker="o",
        edgecolors="black",
        linewidths=0.8,
        zorder=20,
        label="Primary Model",
    )

    ax.scatter(
        data.solution.secondary.log_teff,
        data.solution.secondary.log_luminosity,
        s=90,
        marker="D",
        edgecolors="black",
        linewidths=0.8,
        zorder=20,
        label="Secondary Model",
    )

    #
    # Astronomy convention.
    #
    ax.invert_xaxis()

    ax.set_xlabel(
        r"$\log T_{\rm eff}\;(\mathrm{K})$"
    )

    ax.set_ylabel(
        r"$\log\left(L/L_\odot\right)$"
    )

    ax.grid(
        alpha=0.3,
    )

    ax.legend(
        frameon=True,
    )

    ax.margins(
        x=0.05,
        y=0.05,
    )

    if save_path is not None:
        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    return ax