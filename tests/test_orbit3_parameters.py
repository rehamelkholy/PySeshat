import numpy as np
import pytest


from pyseshat.orbit.orbital_elements import (
    OrbitalElements
)


from pyseshat.triple.triple_elements import (
    TripleOrbitalElements
)


from pyseshat.triple.triple_fit import (
    ORBIT3_PARAMETER_ORDER,
    array_to_orbit3,
    orbit3_fitted_parameter_names,
    orbit3_fitted_parameters,
    orbit3_to_array,
    update_orbit3_fitted_parameters,
)


def make_elements():

    inner = OrbitalElements(

        P=10.0,

        T0=2000.0,

        e=0.2,

        a=0.5,

        Omega=30.0,

        omega=50.0,

        i=70.0,

    )


    outer = OrbitalElements(

        P=50.0,

        T0=2001.0,

        e=0.4,

        a=2.0,

        Omega=120.0,

        omega=140.0,

        i=110.0,

    )


    return TripleOrbitalElements(

        outer=outer,

        inner=inner,

        K1=20.0,

        K2=30.0,

        f=0.25,

        K3=10.0,

        K4=15.0,

        V0=-20.0,

    )


# ==========================================================
# Parameter order
# ==========================================================

def test_orbit3_parameter_order():

    assert ORBIT3_PARAMETER_ORDER == [

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
        "V0",

    ]


# ==========================================================
# Full conversion
# ==========================================================

def test_orbit3_to_array():

    elements = make_elements()


    array = orbit3_to_array(
        elements
    )


    expected = np.array([

        10.0,
        2000.0,
        0.2,
        0.5,
        30.0,
        50.0,
        70.0,

        20.0,
        30.0,
        0.25,

        50.0,
        2001.0,
        0.4,
        2.0,
        120.0,
        140.0,
        110.0,

        10.0,
        15.0,
        -20.0,

    ])


    np.testing.assert_allclose(

        array,

        expected,

    )


# ==========================================================
# Round trip
# ==========================================================

def test_orbit3_array_round_trip():

    elements = make_elements()


    array = orbit3_to_array(
        elements
    )


    reconstructed = array_to_orbit3(

        array,

        elements

    )


    np.testing.assert_allclose(

        orbit3_to_array(
            reconstructed
        ),

        array,

    )


# ==========================================================
# Fitted parameter extraction
# ==========================================================

def test_orbit3_fitted_parameters():

    elements = make_elements()


    elements.fit_flags.update({

        "P": True,
        "T0": True,
        "e": True,
        "a": False,
        "Omega": True,
        "omega": False,
        "i": True,

        "K1": True,
        "K2": False,
        "f": True,

        "P2": True,
        "T02": False,
        "e2": True,
        "a2": False,
        "Omega2": True,
        "omega2": False,
        "i2": True,

        "K3": True,
        "K4": False,
        "V0": True,

        "B": False,

    })


    expected_names = [

        "P",
        "T0",
        "e",
        "Omega",
        "i",

        "K1",
        "f",

        "P2",
        "e2",
        "Omega2",
        "i2",

        "K3",
        "V0",

    ]


    assert (

        orbit3_fitted_parameter_names(
            elements
        )

        ==

        expected_names

    )


    expected_values = np.array([

        10.0,
        2000.0,
        0.2,
        30.0,
        70.0,

        20.0,
        0.25,

        50.0,
        0.4,
        120.0,
        110.0,

        10.0,
        -20.0,

    ])


    np.testing.assert_allclose(

        orbit3_fitted_parameters(
            elements
        ),

        expected_values,

    )


# ==========================================================
# Updating fitted parameters
# ==========================================================

def test_update_orbit3_fitted_parameters():

    elements = make_elements()


    elements.fit_flags.update({

        "P": True,

        "K1": True,

        "f": True,

        "P2": True,

        "K3": True,

        "V0": True,

    })


    original = orbit3_to_array(
        elements
    )


    fitted_values = np.array([

        12.0,
        25.0,
        0.30,
        55.0,
        12.5,
        -15.0,

    ])


    updated = update_orbit3_fitted_parameters(

        elements,

        fitted_values

    )


    result = orbit3_to_array(
        updated
    )


    expected = original.copy()


    expected[0] = 12.0

    expected[7] = 25.0

    expected[9] = 0.30

    expected[10] = 55.0

    expected[17] = 12.5

    expected[19] = -15.0


    np.testing.assert_allclose(

        result,

        expected,

    )


# ==========================================================
# Fixed parameters remain fixed
# ==========================================================

def test_update_orbit3_preserves_fixed_parameters():

    elements = make_elements()


    elements.fit_flags.update({

        "P": True,

    })


    original = orbit3_to_array(
        elements
    )


    updated = update_orbit3_fitted_parameters(

        elements,

        np.array([
            17.5
        ])

    )


    result = orbit3_to_array(
        updated
    )


    assert result[0] == pytest.approx(
        17.5
    )


    np.testing.assert_allclose(

        result[1:],

        original[1:],

    )


# ==========================================================
# Incorrect vector length
# ==========================================================

def test_orbit3_array_requires_20_parameters():

    elements = make_elements()


    with pytest.raises(
        ValueError
    ):

        array_to_orbit3(

            np.zeros(19),

            elements

        )


# ==========================================================
# Incorrect fitted-vector length
# ==========================================================

def test_orbit3_update_requires_correct_fitted_length():

    elements = make_elements()


    elements.fit_flags["P"] = True

    elements.fit_flags["K1"] = True


    with pytest.raises(
        ValueError
    ):

        update_orbit3_fitted_parameters(

            elements,

            np.array([
                10.0
            ])

        )


# ==========================================================
# B is deliberately excluded
# ==========================================================

def test_orbit3_vector_does_not_include_legacy_B():

    elements = make_elements()


    array = orbit3_to_array(
        elements
    )


    assert len(
        array
    ) == 20


    assert (

        "B"

        not in

        ORBIT3_PARAMETER_ORDER

    )


    #
    # The legacy value remains available on the object,
    # but it is not part of the modern ORBIT3 optimizer
    # vector.
    #

    assert elements.B == pytest.approx(
        1.0
    )