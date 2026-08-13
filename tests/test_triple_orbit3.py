import numpy as np
import pytest


from pyseshat.orbit.orbit_model import (
    compute_visual_orbit
)


from pyseshat.orbit.orbital_elements import (
    OrbitalElements
)


from pyseshat.orbit.rv_model import (
    compute_rv_orbit
)


from pyseshat.triple.triple_elements import (
    TripleOrbitalElements
)


from pyseshat.triple.triple_orbit import (
    compute_orbit3_model,
    compute_orbit3_radial_velocities,
    compute_orbit3_visual,
)


# ==========================================================
# Test elements
# ==========================================================

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
# Visual model
# ==========================================================

def test_orbit3_visual_pos1_is_inner_orbit():

    elements = make_elements()


    epochs = np.array([

        2000.0,

        2002.5,

        2007.5,

    ])


    result = compute_orbit3_visual(

        epochs,

        elements,

    )


    expected = compute_visual_orbit(

        epochs,

        elements.inner,

    )


    np.testing.assert_allclose(

        result["POS1"]["X"],

        expected["X"],

    )


    np.testing.assert_allclose(

        result["POS1"]["Y"],

        expected["Y"],

    )


# ==========================================================
# ORBIT3 wobble
# ==========================================================

def test_orbit3_visual_pos2_contains_inner_wobble():

    elements = make_elements()


    epochs = np.array([

        2000.0,

        2002.5,

        2007.5,

    ])


    result = compute_orbit3_visual(

        epochs,

        elements,

    )


    inner = compute_visual_orbit(

        epochs,

        elements.inner,

    )


    outer = compute_visual_orbit(

        epochs,

        elements.outer,

    )


    expected_X = (

        outer["X"]

        +

        elements.f

        * inner["X"]

    )


    expected_Y = (

        outer["Y"]

        +

        elements.f

        * inner["Y"]

    )


    np.testing.assert_allclose(

        result["POS2"]["X"],

        expected_X,

    )


    np.testing.assert_allclose(

        result["POS2"]["Y"],

        expected_Y,

    )


# ==========================================================
# Wobble-factor sensitivity
# ==========================================================

def test_orbit3_visual_wobble_factor():

    elements = make_elements()


    epochs = np.array([

        2000.0,

        2003.0,

        2008.0,

    ])


    zero_wobble = compute_orbit3_visual(

        epochs,

        elements,

    )


    elements.f = 0.0


    no_wobble = compute_orbit3_visual(

        epochs,

        elements,

    )


    outer = compute_visual_orbit(

        epochs,

        elements.outer,

    )


    np.testing.assert_allclose(

        no_wobble["POS2"]["X"],

        outer["X"],

    )


    np.testing.assert_allclose(

        no_wobble["POS2"]["Y"],

        outer["Y"],

    )


    #
    # Restore the non-zero value and verify that the
    # original model was actually different.
    #

    assert not np.allclose(

        zero_wobble["POS2"]["X"],

        no_wobble["POS2"]["X"],

    )


# ==========================================================
# Radial-velocity model
# ==========================================================

def test_orbit3_radial_velocity_components():

    elements = make_elements()


    epochs = np.array([

        2000.0,

        2002.5,

        2007.5,

    ])


    result = compute_orbit3_radial_velocities(

        epochs,

        elements,

    )


    inner_elements = OrbitalElements(

        P=elements.inner.P,

        T0=elements.inner.T0,

        e=elements.inner.e,

        a=elements.inner.a,

        Omega=elements.inner.Omega,

        omega=elements.inner.omega,

        i=elements.inner.i,

        K1=elements.K1,

        K2=elements.K2,

        V0=0.0,

    )


    outer_elements = OrbitalElements(

        P=elements.outer.P,

        T0=elements.outer.T0,

        e=elements.outer.e,

        a=elements.outer.a,

        Omega=elements.outer.Omega,

        omega=elements.outer.omega,

        i=elements.outer.i,

        K1=elements.K3,

        K2=elements.K4,

        V0=elements.V0,

    )


    inner_Aa = compute_rv_orbit(

        epochs,

        inner_elements,

        component=0,

    )


    inner_Ab = compute_rv_orbit(

        epochs,

        inner_elements,

        component=1,

    )


    outer_A = compute_rv_orbit(

        epochs,

        outer_elements,

        component=0,

    )


    outer_B = compute_rv_orbit(

        epochs,

        outer_elements,

        component=1,

    )


    np.testing.assert_allclose(

        result["Aa"],

        inner_Aa + outer_A,

    )


    np.testing.assert_allclose(

        result["Ab"],

        inner_Ab + outer_A,

    )


    np.testing.assert_allclose(

        result["B"],

        outer_B,

    )


# ==========================================================
# Systemic velocity
# ==========================================================

def test_orbit3_systemic_velocity_affects_all_outer_components():

    elements = make_elements()


    epochs = np.array([

        2000.0,

        2005.0,

        2010.0,

    ])


    baseline = compute_orbit3_radial_velocities(

        epochs,

        elements,

    )


    elements.V0 += 7.5


    shifted = compute_orbit3_radial_velocities(

        epochs,

        elements,

    )


    np.testing.assert_allclose(

        shifted["Aa"]
        -
        baseline["Aa"],

        7.5,

    )


    np.testing.assert_allclose(

        shifted["Ab"]
        -
        baseline["Ab"],

        7.5,

    )


    np.testing.assert_allclose(

        shifted["B"]
        -
        baseline["B"],

        7.5,

    )


# ==========================================================
# Complete forward model
# ==========================================================

def test_orbit3_complete_model():

    elements = make_elements()


    epochs = np.array([

        2000.0,

        2003.0,

        2008.0,

    ])


    result = compute_orbit3_model(

        epochs,

        elements,

    )


    assert set(
        result
    ) == {

        "visual",

        "radial_velocity",

    }


    assert set(
        result["visual"]
    ) == {

        "inner",

        "outer",

        "POS1",

        "POS2",

    }


    assert set(
        result["radial_velocity"]
    ) == {

        "inner_primary",

        "inner_secondary",

        "outer_primary",

        "outer_secondary",

        "Aa",

        "Ab",

        "B",

    }


    for key in (
        "X",
        "Y",
        "rho",
        "theta",
    ):

        assert len(
            result["visual"]["POS1"][key]
        ) == len(
            epochs
        )


        assert len(
            result["visual"]["POS2"][key]
        ) == len(
            epochs
        )


    for key in (
        "Aa",
        "Ab",
        "B",
    ):

        assert len(
            result["radial_velocity"][key]
        ) == len(
            epochs
        )