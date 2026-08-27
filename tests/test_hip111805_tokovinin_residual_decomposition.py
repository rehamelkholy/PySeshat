from copy import deepcopy
from pathlib import Path

import numpy as np


from pyseshat.triple.triple_parser import (
    parse_triple_file
)

from pyseshat.triple.triple_residuals import (
    orbit3_visual_residuals,
    orbit3_radial_velocity_residuals
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
# Tokovinin & Latham (2017)
# HIP 111805 published solution
# ==========================================================

TOKOVININ_VALUES = {

    "P": 1.5012,
    "T0": 1986.093,
    "e": 0.022,
    "a": 0.0385,
    "Omega": 334.5,
    "omega": 232.9,
    "i": 85.80,

    "K1": 13.13,
    "K2": 19.21,
    "f": -0.330,

    "P2": 30.127,
    "T02": 2010.179,
    "e2": 0.324,
    "a2": 0.3361,
    "Omega2": 154.25,
    "omega2": 84.92,
    "i2": 88.28,

    "K3": 6.06,
    "K4": 8.60,
    "V0": -22.58,

}


# ==========================================================
# Construct published solution from parsed benchmark
# ==========================================================

def tokovinin_elements(
    reference_elements
):

    elements = deepcopy(
        reference_elements
    )

    # ------------------------------------------------------
    # Inner orbit
    # ------------------------------------------------------

    elements.inner.P = (
        TOKOVININ_VALUES["P"]
    )

    elements.inner.T0 = (
        TOKOVININ_VALUES["T0"]
    )

    elements.inner.e = (
        TOKOVININ_VALUES["e"]
    )

    elements.inner.a = (
        TOKOVININ_VALUES["a"]
    )

    elements.inner.Omega = (
        TOKOVININ_VALUES["Omega"]
    )

    elements.inner.omega = (
        TOKOVININ_VALUES["omega"]
    )

    elements.inner.i = (
        TOKOVININ_VALUES["i"]
    )


    # ------------------------------------------------------
    # Inner spectroscopy
    # ------------------------------------------------------

    elements.K1 = (
        TOKOVININ_VALUES["K1"]
    )

    elements.K2 = (
        TOKOVININ_VALUES["K2"]
    )

    elements.f = (
        TOKOVININ_VALUES["f"]
    )


    # ------------------------------------------------------
    # Outer orbit
    # ------------------------------------------------------

    elements.outer.P = (
        TOKOVININ_VALUES["P2"]
    )

    elements.outer.T0 = (
        TOKOVININ_VALUES["T02"]
    )

    elements.outer.e = (
        TOKOVININ_VALUES["e2"]
    )

    elements.outer.a = (
        TOKOVININ_VALUES["a2"]
    )

    elements.outer.Omega = (
        TOKOVININ_VALUES["Omega2"]
    )

    elements.outer.omega = (
        TOKOVININ_VALUES["omega2"]
    )

    elements.outer.i = (
        TOKOVININ_VALUES["i2"]
    )


    # ------------------------------------------------------
    # Outer spectroscopy
    # ------------------------------------------------------

    elements.K3 = (
        TOKOVININ_VALUES["K3"]
    )

    elements.K4 = (
        TOKOVININ_VALUES["K4"]
    )

    elements.V0 = (
        TOKOVININ_VALUES["V0"]
    )


    return elements


# ==========================================================
# Residual block statistics
# ==========================================================

def block_statistics(
    residuals
):

    residuals = np.asarray(
        residuals,
        dtype=float
    )

    return {

        "N": len(residuals),

        "chi2": float(
            np.sum(
                residuals ** 2
            )
        ),

        "rms": float(
            np.sqrt(
                np.mean(
                    residuals ** 2
                )
            )
        ),

        "mean": float(
            np.mean(
                residuals
            )
        ),

        "max_abs": float(
            np.max(
                np.abs(
                    residuals
                )
            )
        ),

    }


# ==========================================================
# Visual blocks
# ==========================================================

def visual_blocks(
    observations,
    elements
):

    blocks = {}

    for orbit_id, x_name, y_name in (

        ("I1", "X1", "Y1"),
        ("I2", "X2", "Y2"),

    ):

        selected = [

            obs

            for obs in observations

            if obs.orbit_id == orbit_id

        ]

        residuals = np.asarray(

            orbit3_visual_residuals(

                selected,

                elements

            ),

            dtype=float

        ).reshape(

            -1,

            2

        )

        blocks[x_name] = (
            block_statistics(
                residuals[:, 0]
            )
        )

        blocks[y_name] = (
            block_statistics(
                residuals[:, 1]
            )
        )


    return blocks


# ==========================================================
# Radial-velocity blocks
# ==========================================================

def radial_velocity_blocks(
    rv_observations,
    elements
):

    blocks = {}

    for orbit_id, block_name in (

        ("Va", "RV1"),
        ("Vb", "RV2"),
        ("Vc", "RV3"),

    ):

        selected = [

            obs

            for obs in rv_observations

            if obs.orbit_id == orbit_id

        ]

        residuals = (
            orbit3_radial_velocity_residuals(

                selected,

                elements

            )
        )

        blocks[block_name] = (
            block_statistics(
                residuals
            )
        )


    return blocks


# ==========================================================
# Complete decomposition
# ==========================================================

def decompose_blocks(
    observations,
    rv_observations,
    elements
):

    blocks = {}

    blocks.update(
        visual_blocks(
            observations,
            elements
        )
    )

    blocks.update(
        radial_velocity_blocks(
            rv_observations,
            elements
        )
    )

    return blocks


# ==========================================================
# Diagnostic
# ==========================================================

def test_hip111805_tokovinin_residual_decomposition():

    result = parse_triple_file(
        HIP111805_FILE
    )

    observations = result[
        "observations"
    ]

    rv_observations = result[
        "rv_observations"
    ]

    reference_elements = result[
        "elements"
    ]


    # ------------------------------------------------------
    # Published solution
    # ------------------------------------------------------

    tokovinin = tokovinin_elements(
        reference_elements
    )


    # ------------------------------------------------------
    # Current PySeshat production fit
    # ------------------------------------------------------

    from pyseshat.triple.orbit3_fit import (
        fit_orbit3
    )

    fit = fit_orbit3(

        reference_elements,

        observations,

        rv_observations

    )

    assert (
        fit.optimizer_result.success
    )


    # ------------------------------------------------------
    # Decompose both solutions
    # ------------------------------------------------------

    published_blocks = (
        decompose_blocks(

            observations,

            rv_observations,

            tokovinin

        )
    )

    fitted_blocks = (
        decompose_blocks(

            observations,

            rv_observations,

            fit.elements

        )
    )


    block_order = (

        "X1",
        "Y1",
        "X2",
        "Y2",
        "RV1",
        "RV2",
        "RV3",

    )


    published_chi2 = sum(

        published_blocks[name]["chi2"]

        for name in block_order

    )


    fitted_chi2 = sum(

        fitted_blocks[name]["chi2"]

        for name in block_order

    )


    # ------------------------------------------------------
    # Internal consistency
    # ------------------------------------------------------

    assert np.isfinite(
        published_chi2
    )

    assert np.isfinite(
        fitted_chi2
    )

    assert fitted_chi2 < (
        published_chi2
    )


    # ------------------------------------------------------
    # Report
    # ------------------------------------------------------

    print()

    print(
        "HIP111805 TOKOVININ vs PYSESHAT "
        "RESIDUAL DECOMPOSITION"
    )

    print(
        "=" * 112
    )

    print()

    print(
        f"Tokovinin chi2 : "
        f"{published_chi2:.6f}"
    )

    print(
        f"PySeshat chi2  : "
        f"{fitted_chi2:.6f}"
    )

    print(
        f"Delta chi2     : "
        f"{published_chi2 - fitted_chi2:.6f}"
    )

    print()

    print(

        f"{'Block':<8}"
        f"{'N':>8}"
        f"{'Tokovinin chi2':>20}"
        f"{'PySeshat chi2':>20}"
        f"{'Difference':>18}"

    )

    for name in block_order:

        published = (
            published_blocks[name]
        )

        fitted = (
            fitted_blocks[name]
        )

        difference = (

            published["chi2"]
            -
            fitted["chi2"]

        )

        print(

            f"{name:<8}"
            f"{published['N']:>8}"
            f"{published['chi2']:>20.6f}"
            f"{fitted['chi2']:>20.6f}"
            f"{difference:>18.6f}"

        )

    print()

    print(
        "BLOCK CONTRIBUTION TO TOTAL "
        "CHI2 IMPROVEMENT"
    )

    print(
        "-" * 112
    )

    total_improvement = (

        published_chi2
        -
        fitted_chi2

    )

    for name in block_order:

        improvement = (

            published_blocks[name]["chi2"]
            -
            fitted_blocks[name]["chi2"]

        )

        fraction = (

            improvement
            /
            total_improvement

        )

        print(

            f"{name:<8}"
            f"Delta chi2 = "
            f"{improvement:>14.6f}"
            f"   Fraction = "
            f"{fraction:>12.6f}"

        )

    print()

    print(
        "FITTED BLOCK RMS"
    )

    print(
        "-" * 112
    )

    for name in block_order:

        print(

            f"{name:<8}"
            f"Tokovinin = "
            f"{published_blocks[name]['rms']:>12.6f}"
            f"   PySeshat = "
            f"{fitted_blocks[name]['rms']:>12.6f}"

        )