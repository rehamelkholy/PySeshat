from pathlib import Path

import numpy as np
import pytest


from pyseshat.triple.triple_fit import (
    fit_triple_orbit
)


from pyseshat.triple.triple_fit_result import (
    TripleFitResult
)


from pyseshat.triple.triple_parser import (
    parse_triple_file
)


from pyseshat.triple.triple_residuals import (
    triple_visual_residuals
)


# ==========================================================
# Shared benchmark
# ==========================================================

def load_benchmark():

    filename = (
        Path(__file__).resolve().parents[1]
        / "validation"
        / "benchmark_systems"
        / "ZetaAqr"
        / "ZetaAqr.in2"
    )


    parsed = parse_triple_file(
        filename
    )


    return (
        parsed["elements"],
        parsed["observations"],
    )


# ==========================================================
# Basic fit behavior
# ==========================================================

def test_fit_triple_orbit():

    elements, observations = load_benchmark()


    initial = np.sum(

        triple_visual_residuals(

            observations,

            elements

        ) ** 2

    )


    result = fit_triple_orbit(

        elements,

        observations,

    )


    assert isinstance(

        result,

        TripleFitResult,

    )


    final = np.sum(

        triple_visual_residuals(

            observations,

            result.elements

        ) ** 2

    )


    assert final < initial


# ==========================================================
# Optimizer success
# ==========================================================

def test_optimizer_success_flag():

    elements, observations = load_benchmark()


    result = fit_triple_orbit(

        elements,

        observations,

    )


    assert isinstance(

        result.optimizer_result.success,

        bool,

    )


    assert result.optimizer_result.success


# ==========================================================
# Summary
# ==========================================================

def test_summary_runs():

    elements, observations = load_benchmark()


    result = fit_triple_orbit(

        elements,

        observations,

    )


    summary = result.summary()


    assert isinstance(

        summary,

        dict,

    )


    assert "chi_square" in summary

    assert "reduced_chi_square" in summary

    assert "rms" in summary


# ==========================================================
# Report
# ==========================================================

def test_report_runs():

    elements, observations = load_benchmark()


    result = fit_triple_orbit(

        elements,

        observations,

    )


    report = result.report()


    assert isinstance(

        report,

        str,

    )


    assert "TRIPLE FIT REPORT" in report


# ==========================================================
# ORBIT fit-flag configuration
# ==========================================================

def test_zeta_aqr_fit_flags():

    elements, observations = load_benchmark()


    fitted = (
        elements.fitted_parameter_names()
    )


    expected = [

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
        "Omega2",
        "omega2",
        "i2",

    ]


    assert fitted == expected


    #
    # B is a top-level triple parameter rather than an
    # orbital element, so check its fit flag separately.
    #

    assert elements.fit_flags[
        "B"
    ]


    #
    # a2 is fixed in the ORBIT input.
    #

    assert not elements.fit_flags[
        "a2"
    ]


    #
    # a2 is fixed in the ORBIT input.
    #

    assert not elements.fit_flags[
        "a2"
    ]


# ==========================================================
# Zeta Aqr numerical validation
# ==========================================================

def test_zeta_aqr_published_orbit():

    elements, observations = load_benchmark()


    result = fit_triple_orbit(

        elements,

        observations,

    )


    assert result.optimizer_result.success


    fitted = result.elements


    # ------------------------------------------------------
    # Published reference solution
    #
    # Tokovinin (2016), Table 5:
    #
    # Outer A,B:
    #
    # P    = 540 yr
    # T0   = 1981.50 yr
    # e    = 0.419
    # a    = 3.496 arcsec
    # Omega= 131.3 deg
    # omega= 269.3 deg
    # i    = 142.0 deg
    #
    # Inner Aa,Ab:
    #
    # P2    = 25.95 yr
    # T02   = 2006.52 yr
    # e2    = 0.872
    # a2    = 0.110 arcsec (fixed)
    # Omega2= 293.7 deg
    # omega2= 100.9 deg
    # i2    = 11.8 deg
    # F     = 3.50
    #
    # ------------------------------------------------------

    # ------------------------------------------------------
    # Outer orbit
    #
    # These parameters are well constrained and should
    # reproduce the published solution to approximately
    # percent-level accuracy.
    # ------------------------------------------------------

    assert fitted.outer.P == pytest.approx(

        540.0,

        rel=0.02,

    )


    assert fitted.outer.T0 == pytest.approx(

        1981.50,

        abs=1.0,

    )


    assert fitted.outer.e == pytest.approx(

        0.419,

        rel=0.02,

    )


    assert fitted.outer.a == pytest.approx(

        3.496,

        rel=0.02,

    )


    assert fitted.outer.Omega == pytest.approx(

        131.3,

        abs=1.5,

    )


    assert fitted.outer.omega == pytest.approx(

        269.3,

        abs=2.5,

    )


    assert fitted.outer.i == pytest.approx(

        142.0,

        abs=1.0,

    )


    # ------------------------------------------------------
    # Inner orbit.
    #
    # P2, T02 and e2 are substantially better constrained
    # than the angular orientation. We therefore use
    # percent/absolute tolerances appropriate to the
    # published solution.
    # ------------------------------------------------------

    assert fitted.inner.P == pytest.approx(

        25.95,

        rel=0.02,

    )


    assert fitted.inner.T0 == pytest.approx(

        2006.52,

        abs=1.0,

    )


    assert fitted.inner.e == pytest.approx(

        0.872,

        rel=0.02,

    )


    # ------------------------------------------------------
    # a2 must remain fixed.
    # ------------------------------------------------------

    assert fitted.inner.a == pytest.approx(

        0.110,

        abs=1e-12,

    )


    # ------------------------------------------------------
    # Inner orientation.
    #
    # Tokovinin (2016) reports very large uncertainties:
    #
    # Omega2 = 293.7 +/- 74 deg
    # omega2 = 100.9 +/- 73 deg
    # i2     = 11.8  +/- 6.7 deg
    #
    # Therefore these are tested against the published
    # uncertainty scale rather than with a tight numerical
    # regression tolerance.
    # ------------------------------------------------------

    assert abs(

        fitted.inner.Omega
        -
        293.7

    ) <= 74.0


    assert abs(

        fitted.inner.omega
        -
        100.9

    ) <= 73.0


    assert abs(

        fitted.inner.i
        -
        11.8

    ) <= 6.7


    # ------------------------------------------------------
    # Wobble factor.
    #
    # Tokovinin denotes this quantity F.
    # The legacy PySeS̅hat representation calls it B.
    # ------------------------------------------------------

    assert fitted.B == pytest.approx(

        3.50,

        rel=0.03,

    )


# ==========================================================
# Zeta Aqr goodness-of-fit validation
# ==========================================================

def test_zeta_aqr_goodness_of_fit():

    elements, observations = load_benchmark()


    result = fit_triple_orbit(

        elements,

        observations,

    )


    summary = result.summary()


    # ------------------------------------------------------
    # The current validated solution gives approximately:
    #
    # chi_square          = 691.956
    # reduced_chi_square  = 1.066
    # RMS                  = 1.022
    #
    # We use modest tolerances so that this remains a
    # numerical regression test without unnecessarily
    # freezing implementation details.
    # ------------------------------------------------------

    assert summary[
        "chi_square"
    ] == pytest.approx(

        691.9561479812571,

        rel=0.02,

    )


    assert summary[
        "reduced_chi_square"
    ] == pytest.approx(

        1.0661882095242792,

        rel=0.02,

    )


    assert summary[
        "rms"
    ] == pytest.approx(

        1.02237516539852,

        rel=0.02,

    )


# ==========================================================
# Fixed a2 regression
# ==========================================================

def test_zeta_aqr_a2_remains_fixed():

    elements, observations = load_benchmark()


    initial_a2 = elements.inner.a


    result = fit_triple_orbit(

        elements,

        observations,

    )


    final_a2 = result.elements.inner.a


    assert initial_a2 == pytest.approx(

        0.110,

        abs=1e-12,

    )


    assert final_a2 == pytest.approx(

        initial_a2,

        abs=1e-12,

    )