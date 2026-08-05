import numpy as np

import pytest

from pyseshat.analysis.error_propagation import (
    sigma_log10,
    sigma_log_temperature,
    sigma_log_radius,
    sigma_log_luminosity,
)


def test_sigma_log10():

    value = 6000.0

    sigma = 100.0

    expected = sigma / (

        value * np.log(10.0)

    )

    assert np.isclose(

        sigma_log10(

            value,

            sigma,

        ),

        expected,

    )


def test_sigma_log_temperature():

    temperature = 5772.0

    sigma = 50.0

    expected = sigma / (

        temperature * np.log(10.0)

    )

    assert np.isclose(

        sigma_log_temperature(

            temperature,

            sigma,

        ),

        expected,

    )


def test_sigma_log_radius():

    radius = 1.5

    sigma = 0.05

    expected = sigma / (

        radius * np.log(10.0)

    )

    assert np.isclose(

        sigma_log_radius(

            radius,

            sigma,

        ),

        expected,

    )


def test_sigma_log_luminosity():

    sigma_log_t = (

        sigma_log_temperature(

            6000.0,

            100.0,

        )

    )

    sigma_log_r = (

        sigma_log_radius(

            1.2,

            0.05,

        )

    )

    expected = np.sqrt(

        4.0 * sigma_log_r**2

        +

        16.0 * sigma_log_t**2

    )

    assert np.isclose(

        sigma_log_luminosity(

            temperature=6000.0,

            sigma_temperature=100.0,

            radius=1.2,

            sigma_radius=0.05,

        ),

        expected,

    )


@pytest.mark.parametrize(

    "value,sigma",

    [

        (0.0, 1.0),

        (-1.0, 1.0),

    ],

)

def test_sigma_log10_requires_positive_value(

    value,

    sigma,

):

    with pytest.raises(

        ValueError

    ):

        sigma_log10(

            value,

            sigma,

        )


def test_sigma_log10_requires_nonnegative_sigma():

    with pytest.raises(

        ValueError

    ):

        sigma_log10(

            10.0,

            -1.0,

        )