import numpy as np
import pytest


from pyseshat.orbit.orbit_model import (
    compute_visual_orbit
)


from pyseshat.orbit.orbital_elements import (
    OrbitalElements
)


from pyseshat.triple.triple_elements import (
    TripleOrbitalElements
)


from pyseshat.triple.triple_observation import (
    TripleObservation
)


from pyseshat.triple.triple_orbit import (
    compute_orbit3_visual,
    compute_orbit3_radial_velocities,
)


from pyseshat.triple.triple_residuals import (
    orbit3_chi_square,
    orbit3_radial_velocity_residuals,
    orbit3_residuals,
    orbit3_visual_residuals,
)


# ==========================================================
# Test system
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
# Helper: visual observation from a model point
# ==========================================================

def make_visual_observation(
    epoch,
    X,
    Y,
    sigma,
    orbit_id
):

    rho = np.sqrt(

        X**2
        +
        Y**2

    )


    theta = (

        np.degrees(

            np.arctan2(
                Y,
                X
            )

        )

        %

        360.0

    )


    return TripleObservation(

        epoch=epoch,

        theta=theta,

        rho=rho,

        sigma=sigma,

        orbit_id=orbit_id,

    )


# ==========================================================
# Helper: RV observation from model
# ==========================================================

def make_rv_observation(
    epoch,
    velocity,
    sigma,
    component
):

    return TripleObservation(

        epoch=epoch,

        theta=None,

        rho=None,

        sigma=sigma,

        orbit_id=component,

        velocity=velocity,

    )


# ==========================================================
# I1 residuals
# ==========================================================

def test_orbit3_i1_residuals_are_zero_for_model_data():

    elements = make_elements()


    epoch = 2004.0


    model = compute_orbit3_visual(

        np.array([
            epoch
        ]),

        elements

    )


    observation = make_visual_observation(

        epoch,

        model["POS1"]["X"][0],

        model["POS1"]["Y"][0],

        0.01,

        "I1"

    )


    residuals = orbit3_visual_residuals(

        [observation],

        elements

    )


    np.testing.assert_allclose(

        residuals,

        np.zeros(2),

        atol=1e-12,

    )


# ==========================================================
# I2 residuals
# ==========================================================

def test_orbit3_i2_residuals_are_zero_for_model_data():

    elements = make_elements()


    epoch = 2004.0


    model = compute_orbit3_visual(

        np.array([
            epoch
        ]),

        elements

    )


    observation = make_visual_observation(

        epoch,

        model["POS2"]["X"][0],

        model["POS2"]["Y"][0],

        0.01,

        "I2"

    )


    residuals = orbit3_visual_residuals(

        [observation],

        elements

    )


    np.testing.assert_allclose(

        residuals,

        np.zeros(2),

        atol=1e-12,

    )


# ==========================================================
# Visual weighting
# ==========================================================

def test_orbit3_visual_residuals_are_weighted():

    elements = make_elements()


    epoch = 2004.0


    model = compute_orbit3_visual(

        np.array([
            epoch
        ]),

        elements

    )


    sigma = 0.1


    observation = make_visual_observation(

        epoch,

        model["POS1"]["X"][0] + 0.02,

        model["POS1"]["Y"][0] - 0.03,

        sigma,

        "I1"

    )


    residuals = orbit3_visual_residuals(

        [observation],

        elements

    )


    np.testing.assert_allclose(

        residuals,

        np.array([
            0.2,
            -0.3
        ]),

        atol=1e-12,

    )


# ==========================================================
# Va residual
# ==========================================================

def test_orbit3_va_residual_is_zero_for_model_data():

    elements = make_elements()


    epoch = 2004.0


    model = compute_orbit3_radial_velocities(

        np.array([
            epoch
        ]),

        elements

    )


    observation = make_rv_observation(

        epoch,

        model["Aa"][0],

        1.0,

        "Va"

    )


    residuals = orbit3_radial_velocity_residuals(

        [observation],

        elements

    )


    np.testing.assert_allclose(

        residuals,

        np.zeros(1),

        atol=1e-12,

    )


# ==========================================================
# Vb residual
# ==========================================================

def test_orbit3_vb_residual_is_zero_for_model_data():

    elements = make_elements()


    epoch = 2004.0


    model = compute_orbit3_radial_velocities(

        np.array([
            epoch
        ]),

        elements

    )


    observation = make_rv_observation(

        epoch,

        model["Ab"][0],

        1.0,

        "Vb"

    )


    residuals = orbit3_radial_velocity_residuals(

        [observation],

        elements

    )


    np.testing.assert_allclose(

        residuals,

        np.zeros(1),

        atol=1e-12,

    )


# ==========================================================
# Vc residual
# ==========================================================

def test_orbit3_vc_residual_is_zero_for_model_data():

    elements = make_elements()


    epoch = 2004.0


    model = compute_orbit3_radial_velocities(

        np.array([
            epoch
        ]),

        elements

    )


    observation = make_rv_observation(

        epoch,

        model["B"][0],

        1.0,

        "Vc"

    )


    residuals = orbit3_radial_velocity_residuals(

        [observation],

        elements

    )


    np.testing.assert_allclose(

        residuals,

        np.zeros(1),

        atol=1e-12,

    )


# ==========================================================
# RV weighting
# ==========================================================

def test_orbit3_rv_residual_is_weighted():

    elements = make_elements()


    epoch = 2004.0


    model = compute_orbit3_radial_velocities(

        np.array([
            epoch
        ]),

        elements

    )


    sigma = 2.0


    observation = make_rv_observation(

        epoch,

        model["Aa"][0] + 6.0,

        sigma,

        "Va"

    )


    residuals = orbit3_radial_velocity_residuals(

        [observation],

        elements

    )


    np.testing.assert_allclose(

        residuals,

        np.array([
            3.0
        ]),

        atol=1e-12,

    )


# ==========================================================
# Complete residual vector
# ==========================================================

def test_orbit3_complete_residual_vector():

    elements = make_elements()


    epochs = np.array([

        2000.0,

        2004.0,

    ])


    visual_model = compute_orbit3_visual(

        epochs,

        elements

    )


    rv_model = compute_orbit3_radial_velocities(

        epochs,

        elements

    )


    visual_observations = [

        make_visual_observation(

            epochs[0],

            visual_model["POS1"]["X"][0],

            visual_model["POS1"]["Y"][0],

            0.01,

            "I1"

        ),

        make_visual_observation(

            epochs[1],

            visual_model["POS2"]["X"][1],

            visual_model["POS2"]["Y"][1],

            0.01,

            "I2"

        ),

    ]


    rv_observations = [

        make_rv_observation(

            epochs[0],

            rv_model["Aa"][0],

            1.0,

            "Va"

        ),

        make_rv_observation(

            epochs[1],

            rv_model["Ab"][1],

            1.0,

            "Vb"

        ),

        make_rv_observation(

            epochs[0],

            rv_model["B"][0],

            1.0,

            "Vc"

        ),

    ]


    residuals = orbit3_residuals(

        visual_observations,

        rv_observations,

        elements

    )


    np.testing.assert_allclose(

        residuals,

        np.zeros(7),

        atol=1e-12,

    )


# ==========================================================
# Complete chi-square
# ==========================================================

def test_orbit3_chi_square_is_zero_for_model_data():

    elements = make_elements()


    epoch = 2004.0


    visual_model = compute_orbit3_visual(

        np.array([
            epoch
        ]),

        elements

    )


    rv_model = compute_orbit3_radial_velocities(

        np.array([
            epoch
        ]),

        elements

    )


    visual_observations = [

        make_visual_observation(

            epoch,

            visual_model["POS1"]["X"][0],

            visual_model["POS1"]["Y"][0],

            0.01,

            "I1"

        ),

        make_visual_observation(

            epoch,

            visual_model["POS2"]["X"][0],

            visual_model["POS2"]["Y"][0],

            0.01,

            "I2"

        ),

    ]


    rv_observations = [

        make_rv_observation(

            epoch,

            rv_model["Aa"][0],

            1.0,

            "Va"

        ),

        make_rv_observation(

            epoch,

            rv_model["Ab"][0],

            1.0,

            "Vb"

        ),

        make_rv_observation(

            epoch,

            rv_model["B"][0],

            1.0,

            "Vc"

        ),

    ]


    chi2 = orbit3_chi_square(

        visual_observations,

        rv_observations,

        elements

    )


    assert chi2 == pytest.approx(

        0.0,

        abs=1e-20,

    )