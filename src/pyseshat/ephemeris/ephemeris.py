import numpy as np
import pandas as pd

from pyseshat.orbit.orbit_model import (
    compute_visual_orbit
)

from pyseshat.orbit.rv_model import (
    compute_rv_orbit
)

from pyseshat.ephemeris.ephemeris_result import (
    EphemerisResult
)

def generate_visual_ephemeris(
    elements,
    epochs
):
    """
    Generate visual ephemeris.
    """

    orbit = compute_visual_orbit(
        epochs,
        elements
    )

    return EphemerisResult(

        epoch=np.asarray(
            epochs
        ),

        theta=orbit["theta"],

        rho=orbit["rho"],

        X=orbit["X"],

        Y=orbit["Y"]
    )

def visual_ephemeris_table(
    elements,
    epochs
):

    return (
        generate_visual_ephemeris(
            elements,
            epochs
        )
        .to_dataframe()
    )

def generate_rv_ephemeris(
    elements,
    epochs
):
    """
    Generate RV ephemeris.
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

    return EphemerisResult(

        epoch=epochs,

        rv1=rv1,

        rv2=rv2
    )

def rv_ephemeris_table(
    elements,
    epochs
):

    return (
        generate_rv_ephemeris(
            elements,
            epochs
        )
        .to_dataframe()
    )

def generate_combined_ephemeris(
    elements,
    epochs
):
    """
    Generate combined ephemeris.
    """

    visual = generate_visual_ephemeris(
        elements,
        epochs
    )

    rv = generate_rv_ephemeris(
        elements,
        epochs
    )

    return EphemerisResult(

        epoch=visual.epoch,

        theta=visual.theta,

        rho=visual.rho,

        X=visual.X,

        Y=visual.Y,

        rv1=rv.rv1,

        rv2=rv.rv2
    )

def combined_ephemeris_table(
    elements,
    epochs
):

    return (
        generate_combined_ephemeris(
            elements,
            epochs
        )
        .to_dataframe()
    )