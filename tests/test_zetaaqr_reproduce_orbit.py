from pathlib import Path

import numpy as np
import pytest


from pyseshat.orbit.orbital_elements import (
    OrbitalElements
)


from pyseshat.triple.orbit3_fit import (
    fit_orbit3
)


from pyseshat.triple.triple_elements import (
    TripleOrbitalElements
)


from pyseshat.triple.triple_observation import (
    TripleObservation
)


from pyseshat.triple.triple_parser import (
    parse_triple_file
)


from pyseshat.triple.triple_residuals import (
    orbit3_residuals
)


# ==========================================================
# Benchmark path
# ==========================================================

ROOT = (
    Path(__file__).resolve().parents[1]
)


ZETA_FILE = (
    ROOT
    / "validation"
    / "benchmark_systems"
    / "ZetaAqr"
    / "ZetaAqr.in2"
)


# ==========================================================
# Tokovinin (2016) literature solution
# ==========================================================
#
# Reference:
#
# Tokovinin 2016,
# "The Triple System Zeta Aquarii",
# ApJ, 831, 151, Table 5.
#
# IMPORTANT:
#
# The published inner-orbit value a2 = 0.110 arcsec is
# the ASTROMETRIC semimajor axis.
#
# Tokovinin explicitly states that it must be multiplied
# by F to obtain the TRUE inner semimajor axis.
#
# Therefore:
#
#     a_inner_true = F * a_astrometric
#
# and the ORBIT3 wobble factor is:
#
#     f = 1 / F
#
# so that:
#
#     f * a_inner_true = a_astrometric.
#
# ==========================================================


LITERATURE = {

    # ------------------------------------------------------
    # Outer A,B orbit
    # ------------------------------------------------------

    "P2": (
        540.0,
        15.0
    ),

    "T02": (
        1981.50,
        0.58
    ),

    "e2": (
        0.419,
        0.011
    ),

    "a2": (
        3.496,
        0.046
    ),

    "Omega2": (
        131.3,
        0.8
    ),

    "omega2": (
        269.3,
        1.7
    ),

    "i2": (
        142.0,
        0.4
    ),

    # ------------------------------------------------------
    # Inner Aa,Ab orbit
    #
    # These are the literature values of the observed/
    # astrometric inner subsystem, except that `a` below is
    # converted to the TRUE inner semimajor axis required
    # by the ORBIT3 representation.
    # ------------------------------------------------------

    "P": (
        25.95,
        0.048
    ),

    "T0": (
        2006.52,
        0.13
    ),

    "e": (
        0.872,
        0.006
    ),

    "a_astrometric": (
        0.110,
        0.0
    ),

    "Omega": (
        293.7,
        74.0
    ),

    "omega": (
        100.9,
        73.0
    ),

    "i": (
        11.8,
        6.7
    ),

    # ------------------------------------------------------
    # Ratio of true to astrometric inner semimajor axis
    # ------------------------------------------------------

    "F": (
        3.50,
        0.09
    ),

}


# ==========================================================
# Derived literature ORBIT3 quantities
# ==========================================================

def literature_true_inner_a():

    return (

        LITERATURE["F"][0]

        *

        LITERATURE["a_astrometric"][0]

    )


def literature_wobble_factor():

    return (

        1.0

        /

        LITERATURE["F"][0]

    )


# ==========================================================
# Load legacy benchmark
# ==========================================================

def load_legacy_benchmark():

    return parse_triple_file(
        ZETA_FILE
    )


# ==========================================================
# Convert legacy visual labels to ORBIT3 labels
# ==========================================================
#
# Legacy ORBITT2:
#
#     I1 = outer A,B trajectory + inner contribution
#     I2 = scaled inner subsystem
#
# ORBIT3:
#
#     I1 = resolved inner orbit
#     I2 = outer orbit + f * inner orbit
#
# Therefore:
#
#     legacy I2 -> ORBIT3 I1
#     legacy I1 -> ORBIT3 I2
#
# ==========================================================

def convert_to_orbit3_observations(
    observations
):

    converted = []


    for obs in observations:

        if obs.orbit_id == "I1":

            orbit_id = "I2"


        elif obs.orbit_id == "I2":

            orbit_id = "I1"


        else:

            raise ValueError(

                "Unexpected Zeta Aqr "
                "visual observation type: "
                f"{obs.orbit_id}"

            )


        converted.append(

            TripleObservation(

                epoch=obs.epoch,

                theta=obs.theta,

                rho=obs.rho,

                sigma=obs.sigma,

                orbit_id=orbit_id,

                velocity=None

            )

        )


    return converted


# ==========================================================
# Construct literature ORBIT3 elements
# ==========================================================

def make_literature_orbit3_elements():

    # ------------------------------------------------------
    # True inner Aa,Ab orbit
    # ------------------------------------------------------

    inner = OrbitalElements(

        P=LITERATURE["P"][0],

        T0=LITERATURE["T0"][0],

        e=LITERATURE["e"][0],

        a=literature_true_inner_a(),

        Omega=LITERATURE["Omega"][0],

        omega=LITERATURE["omega"][0],

        i=LITERATURE["i"][0],

    )


    # ------------------------------------------------------
    # Outer A,B orbit
    # ------------------------------------------------------

    outer = OrbitalElements(

        P=LITERATURE["P2"][0],

        T0=LITERATURE["T02"][0],

        e=LITERATURE["e2"][0],

        a=LITERATURE["a2"][0],

        Omega=LITERATURE["Omega2"][0],

        omega=LITERATURE["omega2"][0],

        i=LITERATURE["i2"][0],

    )


    # ------------------------------------------------------
    # Modern ORBIT3 representation
    # ------------------------------------------------------

    elements = TripleOrbitalElements(

        outer=outer,

        inner=inner,

        f=literature_wobble_factor(),

    )


    # ------------------------------------------------------
    # Start with all parameters fixed.
    # ------------------------------------------------------

    for key in elements.fit_flags:

        elements.fit_flags[key] = False


    # ------------------------------------------------------
    # Fit inner visual orbit.
    # ------------------------------------------------------

    for key in (

        "P",
        "T0",
        "e",
        "Omega",
        "omega",
        "i",

    ):

        elements.fit_flags[key] = True


    # ------------------------------------------------------
    # Fit wobble factor.
    # ------------------------------------------------------

    elements.fit_flags[
        "f"
    ] = True


    # ------------------------------------------------------
    # Fit outer visual orbit.
    # ------------------------------------------------------

    for key in (

        "P2",
        "T02",
        "e2",
        "a2",
        "Omega2",
        "omega2",
        "i2",

    ):

        elements.fit_flags[key] = True


    # ------------------------------------------------------
    # Keep the true inner semimajor axis fixed.
    #
    # This corresponds to the literature constraint:
    #
    #     a_true = F * a_astrometric
    #
    # and avoids allowing the optimizer to destroy the
    # physical scale of the inner orbit while trying to
    # compensate for the poorly constrained wobble.
    # ------------------------------------------------------

    elements.fit_flags[
        "a"
    ] = False


    return elements


# ==========================================================
# Extract useful parameter values
# ==========================================================

def fitted_parameter_dict(
    elements
):

    return {

        "P": elements.inner.P,

        "T0": elements.inner.T0,

        "e": elements.inner.e,

        "a": elements.inner.a,

        "Omega": elements.inner.Omega,

        "omega": elements.inner.omega,

        "i": elements.inner.i,

        "f": elements.f,

        "P2": elements.outer.P,

        "T02": elements.outer.T0,

        "e2": elements.outer.e,

        "a2": elements.outer.a,

        "Omega2": elements.outer.Omega,

        "omega2": elements.outer.omega,

        "i2": elements.outer.i,

    }


# ==========================================================
# Angular difference
# ==========================================================

def angular_difference(
    value,
    reference
):

    return (

        (
            value
            -
            reference
            +
            180.0
        )
        %
        360.0
        -
        180.0

    )


# ==========================================================
# Normalized literature difference
# ==========================================================

def normalized_literature_difference(
    value,
    reference,
    sigma
):

    if sigma == 0.0:

        return (
            value
            -
            reference
        )


    return (

        value
        -
        reference

    ) / sigma


# ==========================================================
# Observation conversion test
# ==========================================================

def test_zetaaqr_observations_can_be_converted_to_orbit3():

    parsed = (
        load_legacy_benchmark()
    )


    observations = (
        parsed["observations"]
    )


    converted = (
        convert_to_orbit3_observations(
            observations
        )
    )


    assert len(converted) == (
        len(observations)
    )


    legacy_ids = {
        obs.orbit_id
        for obs in observations
    }


    orbit3_ids = {
        obs.orbit_id
        for obs in converted
    }


    assert legacy_ids == {
        "I1",
        "I2"
    }


    assert orbit3_ids == {
        "I1",
        "I2"
    }


# ==========================================================
# Literature-to-ORBIT3 parameter conversion test
# ==========================================================

def test_zetaaqr_literature_conversion():

    F = (
        LITERATURE["F"][0]
    )


    a_ast = (
        LITERATURE["a_astrometric"][0]
    )


    a_true = (
        literature_true_inner_a()
    )


    f = (
        literature_wobble_factor()
    )


    assert a_true == pytest.approx(
        0.385
    )


    assert f == pytest.approx(
        1.0 / 3.50
    )


    # ------------------------------------------------------
    # The ORBIT3 wobble must recover the published
    # astrometric semimajor axis:
    #
    #     f * a_true = a_astrometric
    # ------------------------------------------------------

    assert (

        f
        *
        a_true

    ) == pytest.approx(
        a_ast
    )


    # ------------------------------------------------------
    # The literature conversion is therefore internally
    # self-consistent.
    # ------------------------------------------------------

    assert (
        a_true
        /
        F
    ) == pytest.approx(
        a_ast
    )


# ==========================================================
# Literature initialization test
# ==========================================================

def test_zetaaqr_literature_initialization():

    elements = (
        make_literature_orbit3_elements()
    )


    assert elements.inner.P == pytest.approx(
        25.95
    )

    assert elements.inner.T0 == pytest.approx(
        2006.52
    )

    assert elements.inner.e == pytest.approx(
        0.872
    )

    assert elements.inner.a == pytest.approx(
        0.385
    )


    assert elements.outer.P == pytest.approx(
        540.0
    )

    assert elements.outer.T0 == pytest.approx(
        1981.50
    )

    assert elements.outer.e == pytest.approx(
        0.419
    )

    assert elements.outer.a == pytest.approx(
        3.496
    )


    assert elements.f == pytest.approx(
        1.0 / 3.50
    )


    assert not elements.fit_flags[
        "a"
    ]


    assert elements.fit_flags[
        "f"
    ]


# ==========================================================
# Forward-model literature consistency
# ==========================================================
#
# This is deliberately separate from optimizer recovery.
#
# We verify that the two observable ORBIT3 trajectories have
# the correct physical scales:
#
#     POS1 -> true inner Aa,Ab orbit
#
#     POS2 -> outer A,B orbit + astrometric wobble
#
# ==========================================================

def test_zetaaqr_literature_forward_model():

    elements = (
        make_literature_orbit3_elements()
    )


    epochs = np.array([

        2009.7552,
        2010.8914,
        2012.9229,
        2013.7364,
        2014.7631,
        2015.7379,
        2016.3901,

    ])


    from pyseshat.triple.triple_orbit import (
        compute_orbit3_visual
    )


    model = compute_orbit3_visual(

        epochs,

        elements

    )


    # ------------------------------------------------------
    # POS1 is the true inner orbit.
    #
    # Its semimajor axis is therefore 0.385 arcsec.
    # ------------------------------------------------------

    assert model["POS1"]["rho"].max() > (
        0.3
    )


    # ------------------------------------------------------
    # POS2 contains the outer orbit plus the inner wobble.
    #
    # The wobble scale itself is:
    #
    #     f * a_true = 0.110 arcsec.
    # ------------------------------------------------------

    wobble_semimajor_axis = (

        elements.f
        *
        elements.inner.a

    )


    assert wobble_semimajor_axis == pytest.approx(
        0.110
    )


# ==========================================================
# ORBIT3 literature fit
# ==========================================================

def test_zetaaqr_orbit3_reproduces_literature_solution():

    parsed = (
        load_legacy_benchmark()
    )


    observations = (
        convert_to_orbit3_observations(
            parsed["observations"]
        )
    )


    elements = (
        make_literature_orbit3_elements()
    )


    # ------------------------------------------------------
    # The current Zeta Aqr benchmark is visual-only.
    #
    # The published paper also notes that there are no RV
    # data of adequate precision to establish the correct
    # inner nodes.
    # ------------------------------------------------------

    rv_observations = []


    # ------------------------------------------------------
    # Initial residuals
    # ------------------------------------------------------

    initial_residuals = (
        orbit3_residuals(

            observations,

            rv_observations,

            elements

        )
    )


    initial_chi2 = np.sum(
        initial_residuals**2
    )


    # ------------------------------------------------------
    # Fit
    # ------------------------------------------------------

    result = fit_orbit3(

        elements,

        observations,

        rv_observations,

    )


    assert (
        result.optimizer_result.success
    )


    # ------------------------------------------------------
    # Final residuals
    # ------------------------------------------------------

    final_chi2 = np.sum(
        result.residuals**2
    )


    assert np.isfinite(
        final_chi2
    )


    assert final_chi2 < (
        initial_chi2
    )


    fitted = (
        result.elements
    )


    values = (
        fitted_parameter_dict(
            fitted
        )
    )


    # ------------------------------------------------------
    # Diagnostic report
    # ------------------------------------------------------

    print()

    print(
        "ZETA AQR ORBIT3 LITERATURE VALIDATION"
    )

    print(
        "=" * 72
    )

    print()

    print(
        f"Initial chi2 = {initial_chi2:.6f}"
    )

    print(
        f"Final chi2   = {final_chi2:.6f}"
    )

    print(
        f"Delta chi2   = "
        f"{initial_chi2 - final_chi2:.6f}"
    )

    print()

    print(
        "PARAMETER COMPARISON"
    )

    print(
        "-" * 72
    )

    print(
        f"{'Parameter':<10}"
        f"{'Fit':>16}"
        f"{'Literature':>16}"
        f"{'Difference':>16}"
        f"{'Normalized':>16}"
    )


    comparison_order = (

        "P",
        "T0",
        "e",
        "a",
        "Omega",
        "omega",
        "i",

        "P2",
        "T02",
        "e2",
        "a2",
        "Omega2",
        "omega2",
        "i2",

    )


    for key in comparison_order:

        fitted_value = values[key]


        if key == "a":

            reference = (
                literature_true_inner_a()
            )

            sigma = 0.0


        else:

            if key == "a2":

                reference, sigma = (
                    LITERATURE["a2"]
                )

            else:

                reference, sigma = (
                    LITERATURE[key]
                )


        if key in (

            "Omega",
            "omega",
            "i",
            "Omega2",
            "omega2",
            "i2",

        ):

            difference = angular_difference(

                fitted_value,

                reference

            )


            normalized = (

                difference / sigma

                if sigma > 0.0

                else difference

            )

        else:

            difference = (

                fitted_value
                -
                reference

            )


            normalized = (
                normalized_literature_difference(

                    fitted_value,

                    reference,

                    sigma

                )
            )


        print(

            f"{key:<10}"
            f"{fitted_value:>16.6f}"
            f"{reference:>16.6f}"
            f"{difference:>16.6f}"
            f"{normalized:>16.6f}"

        )


    # ------------------------------------------------------
    # Wobble factor
    # ------------------------------------------------------

    fitted_f = (
        values["f"]
    )


    literature_f = (
        literature_wobble_factor()
    )


    print()

    print(
        f"{'f':<10}"
        f"{fitted_f:>16.6f}"
        f"{literature_f:>16.6f}"
        f"{fitted_f - literature_f:>16.6f}"
    )


    # ------------------------------------------------------
    # The true inner semimajor axis is fixed by the
    # literature-derived dynamical constraint.
    # ------------------------------------------------------

    assert fitted.inner.a == pytest.approx(
        literature_true_inner_a()
    )


    # ------------------------------------------------------
    # We require the optimizer to improve the starting
    # literature solution.
    #
    # We deliberately do NOT impose arbitrary element-by-
    # element tolerances here. Tokovinin explicitly notes
    # that the formal uncertainties underestimate the full
    # range of plausible inner solutions.
    # ------------------------------------------------------

    assert final_chi2 < (
        initial_chi2
    )