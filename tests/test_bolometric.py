import numpy as np

from pyseshat.analysis.bolometric import (
    SOLAR_BOLOMETRIC_MAGNITUDE,
    luminosity_to_mbol,
    mbol_to_luminosity,
    distance_modulus,
    absolute_to_apparent,
    apparent_to_absolute,
)


# ==========================================================
# Luminosity <-> Bolometric Magnitude
# ==========================================================

def test_solar_bolometric_magnitude():

    mbol = luminosity_to_mbol(1.0)

    assert np.isclose(
        mbol,
        SOLAR_BOLOMETRIC_MAGNITUDE
    )


def test_solar_bolometric_luminosity():

    luminosity = mbol_to_luminosity(
        SOLAR_BOLOMETRIC_MAGNITUDE
    )

    assert np.isclose(
        luminosity,
        1.0
    )


def test_luminosity_round_trip():

    luminosity = 12.345

    recovered = mbol_to_luminosity(

        luminosity_to_mbol(
            luminosity
        )

    )

    assert np.isclose(
        recovered,
        luminosity
    )


# ==========================================================
# Distance Modulus
# ==========================================================

def test_distance_modulus_10pc():

    dm = distance_modulus(10.0)

    assert np.isclose(
        dm,
        0.0
    )


def test_distance_modulus_100pc():

    dm = distance_modulus(100.0)

    assert np.isclose(
        dm,
        5.0
    )


def test_distance_modulus_1000pc():

    dm = distance_modulus(1000.0)

    assert np.isclose(
        dm,
        10.0
    )


# ==========================================================
# Apparent / Absolute Magnitude
# ==========================================================

def test_absolute_to_apparent_no_extinction():

    apparent = absolute_to_apparent(

        absolute_magnitude=5.0,

        distance_pc=100.0

    )

    assert np.isclose(
        apparent,
        10.0
    )


def test_absolute_to_apparent_with_extinction():

    apparent = absolute_to_apparent(

        absolute_magnitude=5.0,

        distance_pc=100.0,

        extinction=0.35

    )

    assert np.isclose(
        apparent,
        10.35
    )


def test_apparent_to_absolute_no_extinction():

    absolute = apparent_to_absolute(

        apparent_magnitude=10.0,

        distance_pc=100.0

    )

    assert np.isclose(
        absolute,
        5.0
    )


def test_round_trip_with_extinction():

    absolute = -1.25

    apparent = absolute_to_apparent(

        absolute_magnitude=absolute,

        distance_pc=347.2,

        extinction=0.61

    )

    recovered = apparent_to_absolute(

        apparent_magnitude=apparent,

        distance_pc=347.2,

        extinction=0.61

    )

    assert np.isclose(
        recovered,
        absolute
    )