import numpy as np

import pytest

from pyseshat.orbit.orbital_elements import (
    OrbitalElements
)

from pyseshat.triple.triple_elements import (
    TripleOrbitalElements
)

from pyseshat.triple.triple_parameters import (
    LEGACY_TRIPLE_PARAMETER_ORDER,
    ORBIT3_PARAMETER_ORDER,
    array_to_triple,
    legacy_triple_parameter_names,
    orbit3_array_to_triple,
    orbit3_fitted_parameters,
    orbit3_parameter_names,
    triple_to_array,
    triple_to_orbit3_array,
    update_orbit3_fitted_parameters,
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

        B=0.75
    )


# ==========================================================
# Legacy representation
# ==========================================================

def test_legacy_parameter_order():

    assert (
        legacy_triple_parameter_names()
        ==
        LEGACY_TRIPLE_PARAMETER_ORDER
    )

    assert len(
        LEGACY_TRIPLE_PARAMETER_ORDER
    ) == 15


def test_legacy_triple_to_array():

    triple = make_triple()

    values = triple_to_array(
        triple
    )

    assert values.shape == (
        15,
    )

    assert np.allclose(

        values,

        [

            # Outer orbit

            30.0,
            2010.0,
            0.30,
            0.50,
            120.0,
            80.0,
            70.0,

            # Inner orbit

            3.0,
            2012.0,
            0.10,
            0.05,
            250.0,
            140.0,
            85.0,

            # B

            0.75
        ]
    )


def test_legacy_array_to_triple_round_trip():

    triple = make_triple()

    values = triple_to_array(
        triple
    )

    restored = array_to_triple(

        values,

        triple
    )

    assert np.allclose(

        triple_to_array(
            restored
        ),

        values
    )


# ==========================================================
# ORBIT3 representation
# ==========================================================

def test_orbit3_parameter_order():

    assert (
        orbit3_parameter_names()
        ==
        ORBIT3_PARAMETER_ORDER
    )

    assert len(
        ORBIT3_PARAMETER_ORDER
    ) == 20


def test_orbit3_parameter_vector():

    triple = make_triple()

    values = triple_to_orbit3_array(
        triple
    )

    assert values.shape == (
        20,
    )

    assert np.allclose(

        values,

        [

            # Inner orbit

            3.0,
            2012.0,
            0.10,
            0.05,
            250.0,
            140.0,
            85.0,

            # Inner RV / wobble

            13.1,
            19.2,
            -0.33,

            # Outer orbit

            30.0,
            2010.0,
            0.30,
            0.50,
            120.0,
            80.0,
            70.0,

            # Outer RV

            6.06,
            8.60,
            -22.575
        ]
    )


def test_orbit3_array_to_triple_round_trip():

    triple = make_triple()

    values = triple_to_orbit3_array(
        triple
    )

    restored = orbit3_array_to_triple(

        values,

        triple
    )

    restored_values = (
        triple_to_orbit3_array(
            restored
        )
    )

    assert np.allclose(

        restored_values,

        values
    )


def test_orbit3_does_not_modify_legacy_B():

    triple = make_triple()

    values = triple_to_orbit3_array(
        triple
    )

    restored = orbit3_array_to_triple(

        values,

        triple
    )

    assert restored.B == pytest.approx(
        triple.B
    )


def test_orbit3_accepts_negative_wobble():

    triple = make_triple()

    values = triple_to_orbit3_array(
        triple
    )

    assert values[9] == pytest.approx(
        -0.33
    )


# ==========================================================
# ORBIT3 fitted parameters
# ==========================================================

def test_orbit3_fitted_parameters():

    triple = make_triple()

    triple.fit_flags.update({

        "P": True,

        "K1": True,

        "f": True,

        "P2": True,

        "K4": True,

        "V0": True
    })

    values = orbit3_fitted_parameters(
        triple
    )

    assert np.allclose(

        values,

        [

            3.0,
            13.1,
            -0.33,
            30.0,
            8.60,
            -22.575
        ]
    )


def test_orbit3_fitted_parameter_update():

    triple = make_triple()

    triple.fit_flags.update({

        "P": True,

        "K1": True,

        "f": True,

        "P2": True,

        "K4": True,

        "V0": True
    })

    updated = (
        update_orbit3_fitted_parameters(

            triple,

            [

                3.5,

                14.0,

                -0.40,

                32.0,

                9.20,

                -21.0
            ]
        )
    )

    assert updated.inner.P == pytest.approx(
        3.5
    )

    assert updated.K1 == pytest.approx(
        14.0
    )

    assert updated.f == pytest.approx(
        -0.40
    )

    assert updated.outer.P == pytest.approx(
        32.0
    )

    assert updated.K4 == pytest.approx(
        9.20
    )

    assert updated.V0 == pytest.approx(
        -21.0
    )


def test_orbit3_fitted_parameter_count_mismatch():

    triple = make_triple()

    triple.fit_flags.update({

        "P": True,

        "K1": True,

        "f": True
    })

    with pytest.raises(ValueError):

        update_orbit3_fitted_parameters(

            triple,

            [

                3.5,

                14.0
            ]
        )


# ==========================================================
# Shape validation
# ==========================================================

def test_legacy_array_requires_fifteen_values():

    triple = make_triple()

    with pytest.raises(ValueError):

        array_to_triple(

            np.zeros(14),

            triple
        )


def test_orbit3_array_requires_twenty_values():

    triple = make_triple()

    with pytest.raises(ValueError):

        orbit3_array_to_triple(

            np.zeros(19),

            triple
        )