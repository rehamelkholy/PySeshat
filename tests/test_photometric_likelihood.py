import numpy as np

import pytest

from pyseshat.photometry.observations import (
    PhotometricMeasurement
)

from pyseshat.inference.photometric_likelihood import (
    photometric_residuals,
    chi_square,
    reduced_chi_square,
    log_likelihood
)


# ==========================================================
# Residual vector
# ==========================================================

def test_photometric_residuals(

    monkeypatch

):

    def fake_model(

        **kwargs

    ):

        return {

            "B": 10.20,

            "V": 9.80

        }

    monkeypatch.setattr(

        "pyseshat.inference.photometric_likelihood."
        "synthetic_binary_magnitudes",

        fake_model

    )

    measurements = [

        PhotometricMeasurement(

            passband="B",

            magnitude=10.00,

            uncertainty=0.10

        ),

        PhotometricMeasurement(

            passband="V",

            magnitude=10.00,

            uncertainty=0.20

        )

    ]

    residuals = photometric_residuals(

        parameters=np.array(

            [

                6000,

                5000,

                1.2,

                0.9,

                0.1

            ]

        ),

        atmosphere_grid=None,

        passbands=None,

        zero_points=None,

        measurements=measurements,

        primary_logg=4.5,

        secondary_logg=4.5,

        metallicity=0.0,

        distance=10.0

    )

    np.testing.assert_allclose(

        residuals,

        np.array(

            [

                2.0,

                -1.0

            ]

        )

    )


# ==========================================================
# Chi-square
# ==========================================================

def test_chi_square():

    residuals = np.array(

        [

            2.0,

            -1.0,

            3.0

        ]

    )

    assert chi_square(

        residuals

    ) == pytest.approx(

        14.0

    )


# ==========================================================
# Reduced chi-square
# ==========================================================

def test_reduced_chi_square():

    residuals = np.array(

        [

            2.0,

            -1.0,

            3.0,

            0.0,

            1.0

        ]

    )

    expected = (

        15.0

        /

        2.0

    )

    assert reduced_chi_square(

        residuals,

        n_parameters=3

    ) == pytest.approx(

        expected

    )


# ==========================================================
# Invalid degrees of freedom
# ==========================================================

def test_reduced_chi_square_invalid_dof():

    residuals = np.ones(

        5

    )

    with pytest.raises(

        ValueError,

        match="degrees of freedom"

    ):

        reduced_chi_square(

            residuals,

            n_parameters=5

        )


# ==========================================================
# Log likelihood
# ==========================================================

def test_log_likelihood():

    residuals = np.array(

        [

            2.0,

            -1.0,

            3.0

        ]

    )

    assert log_likelihood(

        residuals

    ) == pytest.approx(

        -7.0

    )


# ==========================================================
# Zero residuals
# ==========================================================

def test_zero_residuals():

    residuals = np.zeros(

        4

    )

    assert chi_square(

        residuals

    ) == 0.0

    assert log_likelihood(

        residuals

    ) == 0.0