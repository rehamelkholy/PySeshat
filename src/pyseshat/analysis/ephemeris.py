import numpy as np
import pandas as pd

from pyseshat.orbit.orbit_model import (
    compute_visual_orbit
)

from pyseshat.orbit.rv_model import (
    compute_rv_orbit
)


def generate_visual_ephemeris(
    elements,
    epochs
):
    """
    Generate visual ephemeris.

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

        "theta": orbit["theta"],

        "rho": orbit["rho"],

        "X": orbit["X"],

        "Y": orbit["Y"]
    }

def visual_ephemeris_table(
    elements,
    epochs
):
    """
    Return visual ephemeris as a DataFrame.
    """

    eph = generate_visual_ephemeris(
        elements,
        epochs
    )

    return pd.DataFrame({

        "Epoch":
            eph["epoch"],

        "Theta":
            eph["theta"],

        "Rho":
            eph["rho"],

        "X":
            eph["X"],

        "Y":
            eph["Y"]
    })

def generate_rv_ephemeris(
    elements,
    epochs
):
    """
    Generate RV ephemeris.

    Parameters
    ----------
    elements : OrbitalElements

    epochs : array_like

    Returns
    -------
    dict
    """

    epochs = np.asarray(
        epochs
    )

    rv1 = compute_rv_orbit(
        epochs,
        elements,
        component=0
    )

    rv2 = compute_rv_orbit(
        epochs,
        elements,
        component=1
    )

    return {

        "epoch": epochs,

        "rv1": rv1,

        "rv2": rv2
    }

def rv_ephemeris_table(
    elements,
    epochs
):
    """
    Return RV ephemeris as DataFrame.
    """

    eph = generate_rv_ephemeris(
        elements,
        epochs
    )

    return pd.DataFrame({

        "Epoch":
            eph["epoch"],

        "RV1":
            eph["rv1"],

        "RV2":
            eph["rv2"]
    })

def generate_combined_ephemeris(
    elements,
    epochs
):
    """
    Generate combined visual + RV ephemeris.
    """

    visual = generate_visual_ephemeris(
        elements,
        epochs
    )

    rv = generate_rv_ephemeris(
        elements,
        epochs
    )

    return {

        "epoch": visual["epoch"],

        "theta": visual["theta"],

        "rho": visual["rho"],

        "X": visual["X"],

        "Y": visual["Y"],

        "rv1": rv["rv1"],

        "rv2": rv["rv2"]
    }

def combined_ephemeris_table(
    elements,
    epochs
):
    """
    Return combined ephemeris as DataFrame.
    """

    eph = generate_combined_ephemeris(
        elements,
        epochs
    )

    return pd.DataFrame({

        "Epoch":
            eph["epoch"],

        "Theta":
            eph["theta"],

        "Rho":
            eph["rho"],

        "X":
            eph["X"],

        "Y":
            eph["Y"],

        "RV1":
            eph["rv1"],

        "RV2":
            eph["rv2"]
    })