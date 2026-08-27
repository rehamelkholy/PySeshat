from pathlib import Path

import numpy as np


from pyseshat.triple.orbit3_fit import (
    fit_orbit3
)


from pyseshat.triple.triple_parser import (
    parse_triple_file
)


# ==========================================================
# Benchmark path
# ==========================================================

ROOT = (
    Path(__file__).resolve().parents[1]
)


HIP111805_FILE = (
    ROOT
    / "validation"
    / "benchmark_systems"
    / "HIP111805"
    / "HIP111805.in3"
)


# ==========================================================
# Tokovinin & Latham (2017) published solution
# ==========================================================
#
# Reference:
#
# Tokovinin, A. & Latham, D. W. 2017,
# "Relative orbit orientation in several resolved
# multiple systems", ApJ, 838, 54.
#
# Table 2, HIP 111805.
#
# The parameter ordering is the same as the PySeshat
# ORBIT3 20-parameter representation.
# ==========================================================

TOKOVININ_VALUES = np.array([

    # ------------------------------------------------------
    # Inner orbit
    # ------------------------------------------------------

    1.5012,
    1986.093,
    0.022,
    0.0385,
    334.5,
    232.9,
    85.80,

    # ------------------------------------------------------
    # Inner spectroscopy / wobble
    # ------------------------------------------------------

    13.13,
    19.21,
    -0.330,

    # ------------------------------------------------------
    # Outer orbit
    # ------------------------------------------------------

    30.127,
    2010.179,
    0.324,
    0.3361,
    154.25,
    84.92,
    88.28,

    # ------------------------------------------------------
    # Outer spectroscopy
    # ------------------------------------------------------

    6.06,
    8.60,
    -22.58,

], dtype=float)


# ==========================================================
# Published 1-sigma uncertainties
# ==========================================================

TOKOVININ_SIGMA = np.array([

    # ------------------------------------------------------
    # Inner orbit
    # ------------------------------------------------------

    0.0004,
    0.093,
    0.011,
    0.0010,
    1.0,
    22.3,
    1.6,

    # ------------------------------------------------------
    # Inner spectroscopy / wobble
    # ------------------------------------------------------

    0.25,
    3.1,
    0.015,

    # ------------------------------------------------------
    # Outer orbit
    # ------------------------------------------------------

    0.031,
    0.073,
    0.004,
    0.0015,
    0.09,
    0.18,
    0.10,

    # ------------------------------------------------------
    # Outer spectroscopy
    # ------------------------------------------------------

    0.14,
    0.23,
    0.08,

], dtype=float)


# ==========================================================
# Parameter names
# ==========================================================

PARAMETER_NAMES = [

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
# Convert fitted elements to complete ORBIT3 vector
# ==========================================================

def fitted_parameter_vector(
    elements
):

    return np.array([

        # --------------------------------------------------
        # Inner orbit
        # --------------------------------------------------

        elements.inner.P,
        elements.inner.T0,
        elements.inner.e,
        elements.inner.a,
        elements.inner.Omega,
        elements.inner.omega,
        elements.inner.i,

        # --------------------------------------------------
        # Inner spectroscopy / wobble
        # --------------------------------------------------

        elements.K1,
        elements.K2,
        elements.f,

        # --------------------------------------------------
        # Outer orbit
        # --------------------------------------------------

        elements.outer.P,
        elements.outer.T0,
        elements.outer.e,
        elements.outer.a,
        elements.outer.Omega,
        elements.outer.omega,
        elements.outer.i,

        # --------------------------------------------------
        # Outer spectroscopy
        # --------------------------------------------------

        elements.K3,
        elements.K4,
        elements.V0,

    ], dtype=float)


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
    ) - 180.0


# ==========================================================
# Parameter differences
# ==========================================================

def parameter_difference(
    fitted,
    reference,
    name
):

    if name in {

        "Omega",
        "omega",
        "Omega2",
        "omega2",

    }:

        return angular_difference(

            fitted,

            reference

        )

    return (
        fitted
        -
        reference
    )


# ==========================================================
# Main validation diagnostic
# ==========================================================

def test_hip111805_tokovinin_validation():

    result = parse_triple_file(
        HIP111805_FILE
    )

    elements = result[
        "elements"
    ]

    observations = result[
        "observations"
    ]

    rv_observations = result[
        "rv_observations"
    ]

    # ------------------------------------------------------
    # Run the current production ORBIT3 fit.
    # ------------------------------------------------------

    fit = fit_orbit3(

        elements,

        observations,

        rv_observations,

    )

    assert (
        fit.optimizer_result.success
    )

    fitted = fitted_parameter_vector(
        fit.elements
    )

    # ------------------------------------------------------
    # Validate vector dimensions.
    # ------------------------------------------------------

    assert fitted.size == 20

    assert TOKOVININ_VALUES.size == 20

    assert TOKOVININ_SIGMA.size == 20

    assert len(
        PARAMETER_NAMES
    ) == 20

    # ------------------------------------------------------
    # Calculate differences.
    # ------------------------------------------------------

    differences = np.array([

        parameter_difference(

            fitted[index],

            TOKOVININ_VALUES[index],

            PARAMETER_NAMES[index]

        )

        for index in range(20)

    ])

    sigma_units = (

        differences
        /
        TOKOVININ_SIGMA

    )

    # ------------------------------------------------------
    # Identify largest discrepancies.
    # ------------------------------------------------------

    absolute_sigma_units = np.abs(
        sigma_units
    )

    largest_index = int(
        np.argmax(
            absolute_sigma_units
        )
    )

    # ------------------------------------------------------
    # Print validation report.
    # ------------------------------------------------------

    print()

    print(
        "HIP111805 PYSESHAT vs TOKOVININ & LATHAM (2017)"
    )

    print(
        "=" * 112
    )

    print()

    print(
        "CURRENT PYSESHAT FIT"
    )

    print(
        "-" * 112
    )

    print(
        f"Success       : "
        f"{fit.optimizer_result.success}"
    )

    print(
        f"Status        : "
        f"{fit.optimizer_result.status}"
    )

    print(
        f"Message       : "
        f"{fit.optimizer_result.message}"
    )

    print(
        f"Chi2          : "
        f"{2.0 * fit.optimizer_result.cost:.12f}"
    )

    print(
        f"Optimality    : "
        f"{fit.optimizer_result.optimality:.12e}"
    )

    print(
        f"nfev          : "
        f"{fit.optimizer_result.nfev}"
    )

    print()

    print(
        "PARAMETER COMPARISON"
    )

    print(
        "-" * 112
    )

    print(

        f"{'Parameter':<12}"
        f"{'Tokovinin':>18}"
        f"{'PySeshat':>18}"
        f"{'Difference':>18}"
        f"{'1-sigma':>18}"
        f"{'Delta / sigma':>18}"

    )

    for index, name in enumerate(
        PARAMETER_NAMES
    ):

        print(

            f"{name:<12}"
            f"{TOKOVININ_VALUES[index]:>18.10g}"
            f"{fitted[index]:>18.10g}"
            f"{differences[index]:>18.10g}"
            f"{TOKOVININ_SIGMA[index]:>18.10g}"
            f"{sigma_units[index]:>18.6f}"

        )

    print()

    print(
        "SUMMARY"
    )

    print(
        "-" * 112
    )

    print(

        f"Largest absolute "
        f"published-sigma displacement : "
        f"{absolute_sigma_units[largest_index]:.6f} sigma"

    )

    print(

        f"Parameter responsible           : "
        f"{PARAMETER_NAMES[largest_index]}"

    )

    print(

        f"Maximum absolute parameter "
        f"difference                     : "
        f"{abs(differences[largest_index]):.12g}"

    )

    print()

    print(
        "PARAMETERS WITH |DELTA| > 1 SIGMA"
    )

    print(
        "-" * 112
    )

    count_above_one_sigma = 0

    for index, name in enumerate(
        PARAMETER_NAMES
    ):

        if absolute_sigma_units[index] > 1.0:

            count_above_one_sigma += 1

            print(

                f"{name:<12}"
                f"{sigma_units[index]:>18.6f}"

            )

    if count_above_one_sigma == 0:

        print(
            "None"
        )

    print()

    print(
        "PARAMETERS WITH |DELTA| > 3 SIGMA"
    )

    print(
        "-" * 112
    )

    count_above_three_sigma = 0

    for index, name in enumerate(
        PARAMETER_NAMES
    ):

        if absolute_sigma_units[index] > 3.0:

            count_above_three_sigma += 1

            print(

                f"{name:<12}"
                f"{sigma_units[index]:>18.6f}"

            )

    if count_above_three_sigma == 0:

        print(
            "None"
        )

    print()

    print(
        "INTERPRETATION"
    )

    print(
        "-" * 112
    )

    print(
        "The published values are used here as an "
        "external scientific benchmark."
    )

    print(
        "No acceptance threshold is imposed by this "
        "diagnostic."
    )

    print(
        "The purpose of this test is to determine whether "
        "the current PySeshat solution differs materially "
        "from the published solution and, if so, which "
        "parameters account for the discrepancy."
    )

    # ------------------------------------------------------
    # Numerical sanity.
    # ------------------------------------------------------

    assert np.all(
        np.isfinite(
            fitted
        )
    )

    assert np.all(
        np.isfinite(
            differences
        )
    )

    assert np.all(
        np.isfinite(
            sigma_units
        )
    )