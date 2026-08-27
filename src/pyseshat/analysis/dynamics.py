import numpy as np
import pandas as pd

from pyseshat.orbit.orbit_model import (
    compute_visual_orbit
)

from pyseshat.orbit.orbital_elements import (
    OrbitalElements
)


# ==========================================================
# Separation versus time
# ==========================================================

def separation_vs_time(
    elements: OrbitalElements,
    epochs
):
    """
    Compute projected separation
    as a function of epoch.

    Parameters
    ----------
    elements : OrbitalElements

    epochs : array_like

    Returns
    -------
    dict
    """

    orbit = compute_visual_orbit(
        epochs,
        elements
    )

    return {

        "epoch": np.asarray(
            epochs
        ),

        "rho": orbit["rho"]
    }

def separation_table(
    elements,
    epochs
):
    """
    Tabulate projected separation vs. epoch as a pandas
    DataFrame, for CSV export or notebook display.

    Thin wrapper around ``separation_vs_time()`` -- same values,
    reshaped into a two-column ``DataFrame`` with columns
    ``"Epoch"`` and ``"Separation"`` (note the capitalization:
    these differ from ``separation_vs_time()``'s own dict keys
    ``"epoch"``/``"rho"``).

    Parameters
    ----------
    elements : OrbitalElements
    epochs : array_like

    Returns
    -------
    pandas.DataFrame
        Columns ``"Epoch"`` and ``"Separation"`` (arcsec).
    """

    data = separation_vs_time(
        elements,
        epochs
    )

    return pd.DataFrame({

        "Epoch":
            data["epoch"],

        "Separation":
            data["rho"]
    })