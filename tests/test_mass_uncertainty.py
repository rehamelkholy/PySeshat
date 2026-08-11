import numpy as np

import pytest

from pyseshat.analysis.masses import (
    total_mass_uncertainty_from_elements,
)

from pyseshat.orbit.orbital_elements import (
    OrbitalElements,
)


def make_elements():

    return OrbitalElements(

        P=10.0,

        T0=2000.0,

        e=0.0,

        a=0.5,

        Omega=0.0,

        omega=0.0,

        i=90.0,
    )


def test_total_mass_uncertainty_from_elements():

    elements = make_elements()

    uncertainties = {

        "P": 0.1,

        "a": 0.01,

    }

    parallax_mas = 50.0

    sigma_parallax_mas = 2.0

    mass = (
        elements.a
        /
        (parallax_mas / 1000.0)
    ) ** 3 / elements.P**2

    expected = mass * np.sqrt(

        9.0
        *
        (
            uncertainties["a"]
            /
            elements.a
        ) ** 2

        +

        4.0
        *
        (
            uncertainties["P"]
            /
            elements.P
        ) ** 2

        +

        9.0
        *
        (
            sigma_parallax_mas
            /
            parallax_mas
        ) ** 2
    )

    result = (
        total_mass_uncertainty_from_elements(

            elements=elements,

            uncertainties=uncertainties,

            parallax_mas=parallax_mas,

            sigma_parallax_mas=(
                sigma_parallax_mas
            ),
        )
    )

    assert result == pytest.approx(
        expected
    )


def test_total_mass_uncertainty_zero_errors():

    result = (
        total_mass_uncertainty_from_elements(

            elements=make_elements(),

            uncertainties={

                "P": 0.0,

                "a": 0.0,

            },

            parallax_mas=50.0,

            sigma_parallax_mas=0.0,

        )
    )

    assert result == pytest.approx(
        0.0
    )


def test_total_mass_uncertainty_requires_period_uncertainty():

    with pytest.raises(
        KeyError
    ):

        total_mass_uncertainty_from_elements(

            elements=make_elements(),

            uncertainties={
                "a": 0.01,
            },

            parallax_mas=50.0,

            sigma_parallax_mas=2.0,

        )


def test_total_mass_uncertainty_requires_semimajor_axis_uncertainty():

    with pytest.raises(
        KeyError
    ):

        total_mass_uncertainty_from_elements(

            elements=make_elements(),

            uncertainties={
                "P": 0.1,
            },

            parallax_mas=50.0,

            sigma_parallax_mas=2.0,

        )


def test_total_mass_uncertainty_requires_dict():

    with pytest.raises(
        TypeError
    ):

        total_mass_uncertainty_from_elements(

            elements=make_elements(),

            uncertainties=None,

            parallax_mas=50.0,

            sigma_parallax_mas=2.0,

        )


def test_total_mass_uncertainty_rejects_negative_uncertainties():

    with pytest.raises(
        ValueError
    ):

        total_mass_uncertainty_from_elements(

            elements=make_elements(),

            uncertainties={

                "P": -0.1,

                "a": 0.01,

            },

            parallax_mas=50.0,

            sigma_parallax_mas=2.0,

        )


def test_total_mass_uncertainty_rejects_negative_parallax_uncertainty():

    with pytest.raises(
        ValueError
    ):

        total_mass_uncertainty_from_elements(

            elements=make_elements(),

            uncertainties={

                "P": 0.1,

                "a": 0.01,

            },

            parallax_mas=50.0,

            sigma_parallax_mas=-2.0,

        )