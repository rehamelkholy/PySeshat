import numpy as np

from pyseshat.photometry.parameters import (
    PhotometricParameters
)


# ==========================================================
# Constants
# ==========================================================

SOLAR_EFFECTIVE_TEMPERATURE = 5772.0  # K


# ==========================================================
# Stellar luminosity
# ==========================================================

def compute_luminosity(
    radius_solar,
    temperature_kelvin
):
    """
    Compute stellar luminosity in solar units.

    Parameters
    ----------
    radius_solar : float
        Stellar radius in solar radii.

    temperature_kelvin : float
        Effective temperature.

    Returns
    -------
    float
        Luminosity in solar luminosities.
    """

    return (

        radius_solar**2

        *

        (

            temperature_kelvin
            /
            SOLAR_EFFECTIVE_TEMPERATURE

        )**4

    )


def compute_log_luminosity(
    radius_solar,
    temperature_kelvin
):
    """
    Compute log10(L/Lsun) from radius and effective temperature
    via the Stefan-Boltzmann law, normalized to solar units.

    Thin log10-wrapping convenience around
    ``compute_luminosity()`` -- used wherever a log-luminosity is
    needed directly (e.g. building ``HRDiagramData``/
    ``StellarObservables`` for an evolutionary-track comparison),
    to avoid every caller repeating ``np.log10(compute_luminosity(...))``.

    Parameters
    ----------
    radius_solar : float
        Stellar radius, solar radii.
    temperature_kelvin : float
        Effective temperature, Kelvin.

    Returns
    -------
    float
        log10(L / L_sun).
    """

    luminosity = compute_luminosity(

        radius_solar=radius_solar,

        temperature_kelvin=temperature_kelvin

    )

    return np.log10(

        luminosity

    )


# ==========================================================
# Binary luminosities
# ==========================================================

def component_luminosities(
    parameters: PhotometricParameters
):
    """
    Compute the luminosities of both
    stellar components.

    Parameters
    ----------
    parameters : PhotometricParameters

    Returns
    -------
    dict
    """

    primary = compute_luminosity(

        radius_solar=parameters.primary_radius,

        temperature_kelvin=parameters.primary_temperature

    )

    secondary = compute_luminosity(

        radius_solar=parameters.secondary_radius,

        temperature_kelvin=parameters.secondary_temperature

    )

    return {

        "L1": primary,

        "L2": secondary

    }


def combined_luminosity(
    parameters: PhotometricParameters
):
    """
    Compute the total luminosity
    of the binary system.

    Returns
    -------
    float
        Total luminosity in solar units.
    """

    luminosities = component_luminosities(

        parameters

    )

    return (

        luminosities["L1"]

        +

        luminosities["L2"]

    )