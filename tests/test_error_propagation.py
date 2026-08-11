import numpy as np

import pytest

from pyseshat.analysis.error_propagation import (
    sigma_log10,
    sigma_log_temperature,
    sigma_log_radius,
    sigma_log_luminosity,
    sigma_total_mass,
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


def test_sigma_total_mass():

    semi_major_axis = 0.5

    sigma_semi_major_axis = 0.01

    period = 10.0

    sigma_period = 0.1

    parallax = 0.05

    sigma_parallax = 0.002

    mass = (
        semi_major_axis
        / parallax
    ) ** 3 / period**2

    expected = mass * np.sqrt(

        9.0
        * (
            sigma_semi_major_axis
            / semi_major_axis
        ) ** 2

        +

        4.0
        * (
            sigma_period
            / period
        ) ** 2

        +

        9.0
        * (
            sigma_parallax
            / parallax
        ) ** 2

    )

    assert np.isclose(

        sigma_total_mass(

            semi_major_axis=(
                semi_major_axis
            ),

            sigma_semi_major_axis=(
                sigma_semi_major_axis
            ),

            period=period,

            sigma_period=sigma_period,

            parallax=parallax,

            sigma_parallax=sigma_parallax,

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


@pytest.mark.parametrize(

    "kwargs",

    [

        {
            "semi_major_axis": 0.0,
            "sigma_semi_major_axis": 0.01,
            "period": 10.0,
            "sigma_period": 0.1,
            "parallax": 0.05,
            "sigma_parallax": 0.002,
        },

        {
            "semi_major_axis": 0.5,
            "sigma_semi_major_axis": 0.01,
            "period": 0.0,
            "sigma_period": 0.1,
            "parallax": 0.05,
            "sigma_parallax": 0.002,
        },

        {
            "semi_major_axis": 0.5,
            "sigma_semi_major_axis": 0.01,
            "period": 10.0,
            "sigma_period": 0.1,
            "parallax": 0.0,
            "sigma_parallax": 0.002,
        },

    ],

)


def test_sigma_total_mass_requires_positive_values(
    kwargs,
):

    with pytest.raises(
        ValueError
    ):

        sigma_total_mass(
            **kwargs
        )


def test_sigma_total_mass_requires_nonnegative_uncertainties():

    with pytest.raises(
        ValueError
    ):

        sigma_total_mass(

            semi_major_axis=0.5,

            sigma_semi_major_axis=-0.01,

            period=10.0,

            sigma_period=0.1,

            parallax=0.05,

            sigma_parallax=0.002,

        )


def test_sigma_total_mass_zero_uncertainty():

    assert sigma_total_mass(

        semi_major_axis=0.5,

        sigma_semi_major_axis=0.0,

        period=10.0,

        sigma_period=0.0,

        parallax=0.05,

        sigma_parallax=0.0,

    ) == pytest.approx(
        0.0
    )