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
    Predict visual-orbit position (position angle, separation,
    and Cartesian X/Y) at a set of epochs from a fitted or
    literature ``OrbitalElements`` solution -- e.g. for planning
    future speckle-interferometry/adaptive-optics observations,
    or for reproducing a paper's ephemeris table.

    Parameters
    ----------
    elements : OrbitalElements
    epochs : array_like
        Epochs (decimal years) to evaluate the orbit at.

    Returns
    -------
    EphemerisResult
        Carries ``epoch``, ``theta`` (position angle, radians),
        ``rho`` (separation, arcsec), ``X``, ``Y``.
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
    """
    ``generate_visual_ephemeris()``, returned as a pandas
    DataFrame (``EphemerisResult.to_dataframe()``) for CSV
    export or notebook display.

    Parameters
    ----------
    elements : OrbitalElements
    epochs : array_like

    Returns
    -------
    pandas.DataFrame
    """

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
    Predict both components' radial velocities at a set of
    epochs from a fitted or literature ``OrbitalElements``
    solution (requires ``K1``/``K2``/``V0`` to be set) -- e.g.
    for planning future spectroscopic observations.

    Parameters
    ----------
    elements : OrbitalElements
    epochs : array_like
        Epochs (decimal years) to evaluate the RV curves at.

    Returns
    -------
    EphemerisResult
        Carries ``epoch``, ``rv1`` (primary, km/s), ``rv2``
        (secondary, km/s).
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
    """
    ``generate_rv_ephemeris()``, returned as a pandas DataFrame
    (``EphemerisResult.to_dataframe()``) for CSV export or
    notebook display.

    Parameters
    ----------
    elements : OrbitalElements
    epochs : array_like

    Returns
    -------
    pandas.DataFrame
    """

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
    Predict both visual-orbit position and both components' RVs
    at the same set of epochs in one call -- the union of
    ``generate_visual_ephemeris()`` and
    ``generate_rv_ephemeris()``, sharing one ``epoch`` array.

    Parameters
    ----------
    elements : OrbitalElements
    epochs : array_like

    Returns
    -------
    EphemerisResult
        Carries ``epoch``, ``theta``, ``rho``, ``X``, ``Y``,
        ``rv1``, ``rv2``.
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
    """
    ``generate_combined_ephemeris()``, returned as a pandas
    DataFrame (``EphemerisResult.to_dataframe()``) for CSV
    export or notebook display.

    Parameters
    ----------
    elements : OrbitalElements
    epochs : array_like

    Returns
    -------
    pandas.DataFrame
    """

    return (
        generate_combined_ephemeris(
            elements,
            epochs
        )
        .to_dataframe()
    )
