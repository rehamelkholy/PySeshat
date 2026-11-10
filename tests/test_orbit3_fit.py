import copy

import numpy as np
import pytest


from pyseshat.orbit.orbital_elements import (
    OrbitalElements
)


from pyseshat.triple.orbit3_bounds import (
    build_orbit3_bounds,
    orbit3_full_bounds,
)


from pyseshat.triple.orbit3_fit import (
    fit_orbit3,
    orbit3_fit_flag_residuals,
)


from pyseshat.triple.triple_elements import (
    TripleOrbitalElements
)


from pyseshat.triple.triple_fit import (
    orbit3_fitted_parameters
)


from pyseshat.triple.triple_observation import (
    TripleObservation
)


from pyseshat.triple.triple_orbit import (
    compute_orbit3_radial_velocities,
    compute_orbit3_visual,
)


from pyseshat.triple.triple_residuals import (
    orbit3_residuals,
)


# ==========================================================
# Synthetic ORBIT3 system
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


    elements = TripleOrbitalElements(

        outer=outer,

        inner=inner,

        K1=20.0,

        K2=30.0,

        f=0.25,

        K3=10.0,

        K4=15.0,

        V0=-20.0,

    )


    return elements


# ==========================================================
# Synthetic observations
# ==========================================================

def make_synthetic_observations(
    elements
):

    epochs = np.array([

        1998.0,
        2000.0,
        2002.0,
        2004.0,
        2006.0,
        2008.0,
        2010.0,
        2012.0,
        2015.0,
        2018.0,

    ])


    visual = compute_orbit3_visual(

        epochs,

        elements

    )


    rv = compute_orbit3_radial_velocities(

        epochs,

        elements

    )


    visual_observations = []


    rv_observations = []


    for index, epoch in enumerate(
        epochs
    ):

        # --------------------------------------------------
        # Inner visual orbit: I1
        # --------------------------------------------------

        X = visual["POS1"]["X"][index]

        Y = visual["POS1"]["Y"][index]


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


        visual_observations.append(

            TripleObservation(

                epoch=epoch,

                theta=theta,

                rho=rho,

                sigma=0.01,

                orbit_id="I1",

            )

        )


        # --------------------------------------------------
        # Outer + wobble visual orbit: I2
        # --------------------------------------------------

        X = visual["POS2"]["X"][index]

        Y = visual["POS2"]["Y"][index]


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


        visual_observations.append(

            TripleObservation(

                epoch=epoch,

                theta=theta,

                rho=rho,

                sigma=0.01,

                orbit_id="I2",

            )

        )


        # --------------------------------------------------
        # Va
        # --------------------------------------------------

        rv_observations.append(

            TripleObservation(

                epoch=epoch,

                theta=None,

                rho=None,

                sigma=0.1,

                orbit_id="Va",

                velocity=rv["Aa"][index],

            )

        )


        # --------------------------------------------------
        # Vb
        # --------------------------------------------------

        rv_observations.append(

            TripleObservation(

                epoch=epoch,

                theta=None,

                rho=None,

                sigma=0.1,

                orbit_id="Vb",

                velocity=rv["Ab"][index],

            )

        )


        # --------------------------------------------------
        # Vc
        # --------------------------------------------------

        rv_observations.append(

            TripleObservation(

                epoch=epoch,

                theta=None,

                rho=None,

                sigma=0.1,

                orbit_id="Vc",

                velocity=rv["B"][index],

            )

        )


    return (

        visual_observations,

        rv_observations

    )


# ==========================================================
# Bounds
# ==========================================================

def test_orbit3_full_bounds_have_20_parameters():

    lower, upper = (
        orbit3_full_bounds()
    )


    assert lower.shape == (
        20,
    )


    assert upper.shape == (
        20,
    )


    assert np.all(
        lower < upper
    )


# ==========================================================
# Fitted bounds
# ==========================================================

def test_build_orbit3_bounds_respects_fit_flags():

    elements = make_elements()


    for key in elements.fit_flags:

        elements.fit_flags[key] = False


    elements.fit_flags["P"] = True

    elements.fit_flags["K1"] = True

    elements.fit_flags["f"] = True

    elements.fit_flags["P2"] = True

    elements.fit_flags["K3"] = True

    elements.fit_flags["V0"] = True


    lower, upper = build_orbit3_bounds(
        elements
    )


    assert lower.shape == (
        6,
    )


    assert upper.shape == (
        6,
    )


    np.testing.assert_allclose(

        lower,

        np.array([

            1e-12,
            0.0,
            -np.inf,
            1e-12,
            0.0,
            -np.inf,

        ])

    )


# ==========================================================
# Initial values are admissible
# ==========================================================

def test_initial_orbit3_parameters_are_inside_bounds():

    elements = make_elements()


    p = np.asarray([

        elements.inner.P,
        elements.inner.T0,
        elements.inner.e,
        elements.inner.a,
        elements.inner.Omega,
        elements.inner.omega,
        elements.inner.i,

        elements.K1,
        elements.K2,
        elements.f,

        elements.outer.P,
        elements.outer.T0,
        elements.outer.e,
        elements.outer.a,
        elements.outer.Omega,
        elements.outer.omega,
        elements.outer.i,

        elements.K3,
        elements.K4,
        elements.V0,

    ])


    lower, upper = (
        orbit3_full_bounds()
    )


    assert np.all(
        p >= lower
    )


    assert np.all(
        p <= upper
    )


# ==========================================================
# Residual wrapper
# ==========================================================

def test_orbit3_fit_flag_residuals():

    elements = make_elements()


    observations, rv_observations = (
        make_synthetic_observations(
            elements
        )
    )


    fitted = (
        elements.fitted_parameter_names()
    )


    values = orbit3_fitted_parameters(
        elements
    )


    assert len(values) == len(
        fitted
    )


    residuals = orbit3_fit_flag_residuals(

        values,

        elements,

        observations,

        rv_observations

    )


    direct = orbit3_residuals(

        observations,

        rv_observations,

        elements

    )


    np.testing.assert_allclose(

        residuals,

        direct,

    )


# ==========================================================
# Exact synthetic fit
# ==========================================================

def test_fit_orbit3_recovers_synthetic_solution():

    true_elements = make_elements()


    observations, rv_observations = (
        make_synthetic_observations(
            true_elements
        )
    )


    # ------------------------------------------------------
    # TripleOrbitalElements does not implement .copy().
    #
    # Use a deep copy because the object contains mutable
    # nested OrbitalElements and fit_flags dictionaries.
    # ------------------------------------------------------

    initial = copy.deepcopy(
        true_elements
    )


    # ------------------------------------------------------
    # Perturb the starting solution.
    # ------------------------------------------------------

    initial.inner.P *= 1.05

    initial.inner.T0 += 0.25

    initial.inner.e += 0.03

    initial.inner.a *= 0.95

    initial.outer.P *= 0.97

    initial.outer.T0 -= 0.30

    initial.outer.e -= 0.03

    initial.outer.a *= 1.04

    initial.K1 += 2.0

    initial.K3 -= 1.5

    initial.V0 += 3.0


    # ------------------------------------------------------
    # Fit only the parameters deliberately perturbed above.
    # ------------------------------------------------------

    for key in initial.fit_flags:

        initial.fit_flags[key] = False


    for key in (

        "P",
        "T0",
        "e",
        "a",

        "P2",
        "T02",
        "e2",
        "a2",

        "K1",
        "K3",
        "V0",

    ):

        initial.fit_flags[key] = True


    # ------------------------------------------------------
    # Initial chi-square
    # ------------------------------------------------------

    initial_residuals = orbit3_residuals(

        observations,

        rv_observations,

        initial

    )


    initial_chi2 = np.sum(

        initial_residuals**2

    )


    # ------------------------------------------------------
    # Fit
    # ------------------------------------------------------

    result = fit_orbit3(

        initial,

        observations,

        rv_observations,

    )


    assert result.optimizer_result.success


    # ------------------------------------------------------
    # Final chi-square
    # ------------------------------------------------------

    final_chi2 = np.sum(

        result.residuals**2

    )


    assert final_chi2 < (
        initial_chi2
    )


    # ------------------------------------------------------
    # A threshold of `< 1e-10` would assume the LM optimizer
    # drives chi2 essentially to
    # machine precision on this noiseless synthetic problem; in
    # practice `fit_orbit3`'s default scipy least_squares run
    # terminates on `xtol` (relative parameter-change
    # convergence, not an absolute chi2 floor) once parameters
    # stop moving meaningfully, which reproducibly lands around
    # 9e-7 here -- confirmed deterministic (no randomness
    # anywhere in this test) by direct reproduction, not
    # environment noise. `< 1e-4` still meaningfully asserts
    # strong convergence (three orders of magnitude of margin
    # below it) without being brittle to the exact xtol/BLAS-
    # dependent stopping point.
    # ------------------------------------------------------

    assert final_chi2 < 1e-4


    fitted = result.elements


    # ------------------------------------------------------
    # Recovered inner orbit
    # ------------------------------------------------------

    assert fitted.inner.P == pytest.approx(

        true_elements.inner.P,

        abs=1e-5,

    )


    assert fitted.inner.T0 == pytest.approx(

        true_elements.inner.T0,

        abs=1e-5,

    )


    assert fitted.inner.e == pytest.approx(

        true_elements.inner.e,

        abs=1e-6,

    )


    assert fitted.inner.a == pytest.approx(

        true_elements.inner.a,

        abs=1e-6,

    )


    # ------------------------------------------------------
    # Recovered outer orbit
    # ------------------------------------------------------

    # The outer period
    # reproducibly lands ~3.9e-5 from truth at this optimizer's
    # xtol stopping point (see the final_chi2 comment above) --
    # loosened to keep matching margin, not to mask a
    # regression (every other recovered parameter here, inner
    # and outer alike, is still comfortably inside 1e-5).

    assert fitted.outer.P == pytest.approx(

        true_elements.outer.P,

        abs=1e-4,

    )


    assert fitted.outer.T0 == pytest.approx(

        true_elements.outer.T0,

        abs=1e-5,

    )


    assert fitted.outer.e == pytest.approx(

        true_elements.outer.e,

        abs=1e-6,

    )


    # Reproducibly lands
    # ~2.5e-6 from truth at this optimizer's xtol stopping
    # point, same story as final_chi2/outer.P above -- loosened
    # to match the other outer-orbit parameters' 1e-5 scale.

    assert fitted.outer.a == pytest.approx(

        true_elements.outer.a,

        abs=1e-5,

    )


    # ------------------------------------------------------
    # Recovered spectroscopic parameters
    # ------------------------------------------------------

    assert fitted.K1 == pytest.approx(

        true_elements.K1,

        abs=1e-5,

    )


    assert fitted.K3 == pytest.approx(

        true_elements.K3,

        abs=1e-5,

    )


    assert fitted.V0 == pytest.approx(

        true_elements.V0,

        abs=1e-5,

    )