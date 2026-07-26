import numpy as np

from pyseshat.analysis.luminosity import (
    SOLAR_EFFECTIVE_TEMPERATURE,
    compute_luminosity,
    compute_log_luminosity,
    component_luminosities,
    combined_luminosity,
)

from pyseshat.photometry.parameters import (
    PhotometricParameters,
)


# ==========================================================
# Single-star luminosity
# ==========================================================

def test_compute_luminosity_solar():

    luminosity = compute_luminosity(

        radius_solar=1.0,

        temperature_kelvin=SOLAR_EFFECTIVE_TEMPERATURE

    )

    assert luminosity == 1.0


def test_compute_luminosity_scaling():

    luminosity = compute_luminosity(

        radius_solar=2.0,

        temperature_kelvin=SOLAR_EFFECTIVE_TEMPERATURE

    )

    assert luminosity == 4.0


def test_compute_log_luminosity_solar():

    log_l = compute_log_luminosity(

        radius_solar=1.0,

        temperature_kelvin=SOLAR_EFFECTIVE_TEMPERATURE

    )

    assert log_l == 0.0


# ==========================================================
# Binary luminosities
# ==========================================================

def test_component_luminosities():

    parameters = PhotometricParameters(

        primary_temperature=SOLAR_EFFECTIVE_TEMPERATURE,

        secondary_temperature=SOLAR_EFFECTIVE_TEMPERATURE,

        primary_radius=1.0,

        secondary_radius=2.0,

        ebv=0.0

    )

    luminosities = component_luminosities(

        parameters

    )

    assert luminosities["L1"] == 1.0

    assert luminosities["L2"] == 4.0


def test_combined_luminosity():

    parameters = PhotometricParameters(

        primary_temperature=SOLAR_EFFECTIVE_TEMPERATURE,

        secondary_temperature=SOLAR_EFFECTIVE_TEMPERATURE,

        primary_radius=1.0,

        secondary_radius=2.0,

        ebv=0.0

    )

    total = combined_luminosity(

        parameters

    )

    assert total == 5.0


# ==========================================================
# General numerical consistency
# ==========================================================

def test_log_matches_linear_luminosity():

    luminosity = compute_luminosity(

        radius_solar=1.7,

        temperature_kelvin=6400.0

    )

    log_l = compute_log_luminosity(

        radius_solar=1.7,

        temperature_kelvin=6400.0

    )

    assert log_l == np.log10(luminosity)