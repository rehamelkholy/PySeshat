import numpy as np

import pytest

from pyseshat.photometry.parameters import (
    PhotometricParameters,
    update_parameters
)


# ==========================================================
# Construction
# ==========================================================

def test_construction():

    parameters = PhotometricParameters(

        primary_temperature=6000.0,

        secondary_temperature=5000.0,

        primary_radius=1.20,

        secondary_radius=0.90,

        ebv=0.15

    )

    assert parameters.primary_temperature == 6000.0
    assert parameters.secondary_temperature == 5000.0
    assert parameters.primary_radius == 1.20
    assert parameters.secondary_radius == 0.90
    assert parameters.ebv == 0.15


# ==========================================================
# Parameter vector
# ==========================================================

def test_fitted_parameters():

    parameters = PhotometricParameters(

        primary_temperature=6100.0,

        secondary_temperature=4800.0,

        primary_radius=1.35,

        secondary_radius=0.82,

        ebv=0.08

    )

    expected = np.array(

        [

            6100.0,

            4800.0,

            1.35,

            0.82,

            0.08

        ],

        dtype=float

    )

    np.testing.assert_allclose(

        parameters.fitted_parameters(),

        expected

    )


# ==========================================================
# Parameter names
# ==========================================================

def test_fitted_parameter_names():

    assert (

        PhotometricParameters.fitted_parameter_names()

        ==

        [

            "primary_temperature",

            "secondary_temperature",

            "primary_radius",

            "secondary_radius",

            "ebv"

        ]

    )


# ==========================================================
# Update
# ==========================================================

def test_update_parameters():

    original = PhotometricParameters(

        primary_temperature=6000.0,

        secondary_temperature=5000.0,

        primary_radius=1.0,

        secondary_radius=0.8,

        ebv=0.10

    )

    updated = update_parameters(

        original,

        np.array(

            [

                6500.0,

                5200.0,

                1.45,

                0.95,

                0.20

            ]

        )

    )

    assert isinstance(

        updated,

        PhotometricParameters

    )

    assert updated.primary_temperature == 6500.0
    assert updated.secondary_temperature == 5200.0
    assert updated.primary_radius == 1.45
    assert updated.secondary_radius == 0.95
    assert updated.ebv == 0.20


# ==========================================================
# Invalid vector length
# ==========================================================

def test_update_parameters_invalid_length():

    parameters = PhotometricParameters(

        primary_temperature=6000.0,

        secondary_temperature=5000.0,

        primary_radius=1.0,

        secondary_radius=0.8,

        ebv=0.10

    )

    with pytest.raises(

        ValueError,

        match="exactly five"

    ):

        update_parameters(

            parameters,

            np.ones(

                4

            )

        )


# ==========================================================
# Integer vector is accepted
# ==========================================================

def test_update_parameters_integer_vector():

    parameters = PhotometricParameters(

        primary_temperature=6000.0,

        secondary_temperature=5000.0,

        primary_radius=1.0,

        secondary_radius=1.0,

        ebv=0.0

    )

    updated = update_parameters(

        parameters,

        np.array(

            [

                7000,

                5600,

                2,

                1,

                0

            ]

        )

    )

    assert updated.primary_temperature == 7000.0
    assert updated.secondary_temperature == 5600.0
    assert updated.primary_radius == 2.0
    assert updated.secondary_radius == 1.0
    assert updated.ebv == 0.0