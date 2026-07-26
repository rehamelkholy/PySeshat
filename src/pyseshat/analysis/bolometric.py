import numpy as np


# ==========================================================
# Constants
# ==========================================================

SOLAR_BOLOMETRIC_MAGNITUDE = 4.74


# ==========================================================
# Luminosity <-> Bolometric Magnitude
# ==========================================================

def luminosity_to_mbol(
    luminosity_solar
):
    """
    Convert luminosity to absolute bolometric
    magnitude.

    Parameters
    ----------
    luminosity_solar : float

    Returns
    -------
    float
        Absolute bolometric magnitude.
    """

    return (

        SOLAR_BOLOMETRIC_MAGNITUDE

        -

        2.5 * np.log10(
            luminosity_solar
        )

    )


def mbol_to_luminosity(
    mbol
):
    """
    Convert absolute bolometric magnitude
    to luminosity in solar units.
    """

    return 10.0 ** (

        (
            SOLAR_BOLOMETRIC_MAGNITUDE
            -
            mbol
        )

        / 2.5

    )


# ==========================================================
# Distance Modulus
# ==========================================================

def distance_modulus(
    distance_pc
):
    """
    Compute the distance modulus.

    Parameters
    ----------
    distance_pc : float

    Returns
    -------
    float
    """

    return (

        5.0

        * np.log10(
            distance_pc
        )

        - 5.0

    )


# ==========================================================
# Magnitude Conversions
# ==========================================================

def absolute_to_apparent(
    absolute_magnitude,
    distance_pc,
    extinction=0.0
):
    """
    Convert absolute magnitude to apparent
    magnitude.

    Parameters
    ----------
    absolute_magnitude : float

    distance_pc : float

    extinction : float, optional

    Returns
    -------
    float
    """

    return (

        absolute_magnitude

        +

        distance_modulus(
            distance_pc
        )

        +

        extinction

    )


def apparent_to_absolute(
    apparent_magnitude,
    distance_pc,
    extinction=0.0
):
    """
    Convert apparent magnitude to absolute
    magnitude.

    Parameters
    ----------
    apparent_magnitude : float

    distance_pc : float

    extinction : float, optional

    Returns
    -------
    float
    """

    return (

        apparent_magnitude

        -

        distance_modulus(
            distance_pc
        )

        -

        extinction

    )