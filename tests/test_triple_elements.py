import numpy as np

import pytest

from pyseshat.orbit.orbital_elements import (
    OrbitalElements
)

from pyseshat.triple.triple_elements import (
    TripleOrbitalElements
)


def make_outer():

    return OrbitalElements(

        P=30.0,

        T0=2010.0,

        e=0.30,

        a=0.50,

        Omega=120.0,

        omega=80.0,

        i=70.0
    )


def make_inner():

    return OrbitalElements(

        P=3.0,

        T0=2012.0,

        e=0.10,

        a=0.05,

        Omega=250.0,

        omega=140.0,

        i=85.0
    )


def make_triple():

    return TripleOrbitalElements(

        outer=make_outer(),

        inner=make_inner(),

        K1=13.1,

        K2=19.2,

        f=-0.33,

        K3=6.06,

        K4=8.60,

        V0=-22.575,

        B=1.0
    )


def test_orbit3_parameter_names():

    names = (
        TripleOrbitalElements.parameter_names()
    )

    assert names == [

        "P",
        "T0",
        "e",
        "a",
        "Omega",
        "omega",
        "i",

        "K1",
        "K2",
        "f",

        "P2",
        "T02",
        "e2",
        "a2",
        "Omega2",
        "omega2",
        "i2",

        "K3",
        "K4",
        "V0"
    ]


def test_orbit3_parameter_count():

    names = (
        TripleOrbitalElements.parameter_names()
    )

    assert len(names) == 20


def test_orbit3_values_are_stored():

    triple = make_triple()

    assert triple.K1 == pytest.approx(
        13.1
    )

    assert triple.K2 == pytest.approx(
        19.2
    )

    assert triple.f == pytest.approx(
        -0.33
    )

    assert triple.K3 == pytest.approx(
        6.06
    )

    assert triple.K4 == pytest.approx(
        8.60
    )

    assert triple.V0 == pytest.approx(
        -22.575
    )


def test_orbit3_fitted_mask_has_twenty_entries():

    triple = make_triple()

    mask = triple.fitted_mask()

    assert isinstance(
        mask,
        np.ndarray
    )

    assert mask.dtype == bool

    assert mask.shape == (
        20,
    )


def test_orbit3_fitted_parameter_names_follow_order():

    triple = make_triple()

    triple.fit_flags.update({

        "P": True,
        "e": True,
        "K1": True,
        "f": True,
        "P2": True,
        "K4": True,
        "V0": True
    })

    assert (
        triple.fitted_parameter_names()
        ==
        [
            "P",
            "e",
            "K1",
            "f",
            "P2",
            "K4",
            "V0"
        ]
    )


def test_orbit3_wobble_factor_can_be_negative():

    triple = make_triple()

    assert triple.f == pytest.approx(
        -0.33
    )


def test_legacy_B_is_retained():

    triple = make_triple()

    assert triple.B == pytest.approx(
        1.0
    )


def test_legacy_scale_inner_uses_B():

    triple = make_triple()

    assert triple.scale_inner == pytest.approx(
        triple.B
    )


def test_default_modern_parameters():

    triple = TripleOrbitalElements(

        outer=make_outer(),

        inner=make_inner()
    )

    assert triple.K1 == pytest.approx(
        0.0
    )

    assert triple.K2 == pytest.approx(
        0.0
    )

    assert triple.f == pytest.approx(
        0.0
    )

    assert triple.K3 == pytest.approx(
        0.0
    )

    assert triple.K4 == pytest.approx(
        0.0
    )

    assert triple.V0 == pytest.approx(
        0.0
    )

    assert triple.B == pytest.approx(
        1.0
    )