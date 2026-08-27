from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from dataclasses import replace

from pyseshat.triple.orbit3_fit import (
    fit_orbit3
)

from pyseshat.triple.triple_parser import (
    parse_triple_file
)

from pyseshat.triple.triple_residuals import (
    orbit3_residuals,
    orbit3_visual_residuals,
    orbit3_radial_velocity_residuals
)

from pyseshat.triple.triple_orbit import (
    compute_orbit3_radial_velocities
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
# Load benchmark
# ==========================================================

def load_hip111805():

    return parse_triple_file(
        HIP111805_FILE
    )


# ==========================================================
# Observation counts
# ==========================================================

def observation_counts(
    observations,
    rv_observations
):

    return {

        "I1": sum(
            obs.orbit_id == "I1"
            for obs in observations
        ),

        "I2": sum(
            obs.orbit_id == "I2"
            for obs in observations
        ),

        "Va": sum(
            obs.orbit_id == "Va"
            for obs in rv_observations
        ),

        "Vb": sum(
            obs.orbit_id == "Vb"
            for obs in rv_observations
        ),

        "Vc": sum(
            obs.orbit_id == "Vc"
            for obs in rv_observations
        )

    }


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

    if len(residuals) == 0:

        return {

            "N": 0,
            "chi2": 0.0,
            "rms": 0.0,
            "mean": 0.0,
            "max_abs": 0.0,

        }

    return {

        "N": len(residuals),

        "chi2": float(
            np.sum(
                residuals**2
            )
        ),

        "rms": float(
            np.sqrt(
                np.mean(
                    residuals**2
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
# Visual residual decomposition
# ==========================================================

def visual_residual_blocks(
    observations,
    elements
):

    blocks = {}

    # ------------------------------------------------------
    # I1
    # ------------------------------------------------------

    observations_I1 = [

        obs

        for obs in observations

        if obs.orbit_id == "I1"

    ]

    if observations_I1:

        residuals_I1 = (
            orbit3_visual_residuals(

                observations_I1,

                elements

            )
        )

        residuals_I1 = np.asarray(
            residuals_I1,
            dtype=float
        ).reshape(
            -1,
            2
        )

        blocks["X1"] = (
            residuals_I1[:, 0]
        )

        blocks["Y1"] = (
            residuals_I1[:, 1]
        )

    else:

        blocks["X1"] = np.array(
            [],
            dtype=float
        )

        blocks["Y1"] = np.array(
            [],
            dtype=float
        )

    # ------------------------------------------------------
    # I2
    # ------------------------------------------------------

    observations_I2 = [

        obs

        for obs in observations

        if obs.orbit_id == "I2"

    ]

    if observations_I2:

        residuals_I2 = (
            orbit3_visual_residuals(

                observations_I2,

                elements

            )
        )

        residuals_I2 = np.asarray(
            residuals_I2,
            dtype=float
        ).reshape(
            -1,
            2
        )

        blocks["X2"] = (
            residuals_I2[:, 0]
        )

        blocks["Y2"] = (
            residuals_I2[:, 1]
        )

    else:

        blocks["X2"] = np.array(
            [],
            dtype=float
        )

        blocks["Y2"] = np.array(
            [],
            dtype=float
        )

    return {

        key: block_statistics(
            value
        )

        for key, value in blocks.items()

    }


# ==========================================================
# Radial-velocity residual decomposition
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

        if selected:

            residuals = (
                orbit3_radial_velocity_residuals(

                    selected,

                    elements

                )
            )

        else:

            residuals = np.array(
                [],
                dtype=float
            )

        blocks[block_name] = (
            block_statistics(
                residuals
            )
        )

    return blocks


# ==========================================================
# Complete residual decomposition
# ==========================================================

def decompose_residuals(
    observations,
    rv_observations,
    elements
):

    visual_blocks = (
        visual_residual_blocks(

            observations,

            elements

        )
    )

    radial_blocks = (
        radial_velocity_blocks(

            rv_observations,

            elements

        )
    )

    return {

        **visual_blocks,
        **radial_blocks,

    }


# ==========================================================
# Toggle spectroscopic omega convention
# ==========================================================

def toggle_rv_omega_convention(
    elements
):

    toggled = deepcopy(
        elements
    )

    toggled.inner.omega = (
        toggled.inner.omega
        +
        180.0
    ) % 360.0

    toggled.outer.omega = (
        toggled.outer.omega
        +
        180.0
    ) % 360.0

    return toggled


# ==========================================================
# Isolate inner spectroscopic contribution
# ==========================================================

def inner_only_elements(
    elements
):

    inner_only = deepcopy(
        elements
    )

    inner_only.K3 = 0.0
    inner_only.K4 = 0.0
    inner_only.V0 = 0.0

    return inner_only


# ==========================================================
# Isolate outer spectroscopic contribution
# ==========================================================

def outer_only_elements(
    elements
):

    outer_only = deepcopy(
        elements
    )

    outer_only.K1 = 0.0
    outer_only.K2 = 0.0

    return outer_only


# ==========================================================
# RV model values in observation order
# ==========================================================

def model_values_for_observations(
    rv_observations,
    elements
):

    values = []

    for obs in rv_observations:

        model = (
            compute_orbit3_radial_velocities(

                np.array([
                    obs.epoch
                ]),

                elements

            )
        )

        if obs.orbit_id == "Va":

            values.append(
                model["Aa"][0]
            )

        elif obs.orbit_id == "Vb":

            values.append(
                model["Ab"][0]
            )

        elif obs.orbit_id == "Vc":

            values.append(
                model["B"][0]
            )

        else:

            raise ValueError(
                "Unexpected RV observation "
                f"type: {obs.orbit_id}"
            )

    return np.asarray(
        values,
        dtype=float
    )


# ==========================================================
# Weighted residual statistics
# ==========================================================

def rv_statistics(
    observations,
    model_values
):

    observed = np.asarray([

        obs.velocity

        for obs in observations

    ])

    sigma = np.asarray([

        obs.sigma

        for obs in observations

    ])

    residuals = (
        observed
        -
        model_values
    )

    normalized = (
        residuals
        /
        sigma
    )

    return {

        "N": len(
            residuals
        ),

        "chi2": float(
            np.sum(
                normalized**2
            )
        ),

        "rms": float(
            np.sqrt(
                np.mean(
                    residuals**2
                )
            )
        ),

        "weighted_rms": float(
            np.sqrt(
                np.mean(
                    normalized**2
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
# Basic ORBIT3 file validation
# ==========================================================

def test_hip111805_orbit3_file_is_parsed():

    result = load_hip111805()

    assert (
        result["metadata"]["format"]
        == "in3"
    )

    assert (
        result["metadata"]["object_name"]
        == "HIP111805"
    )

    elements = result[
        "elements"
    ]

    assert elements.inner.P == pytest.approx(
        1.50119
    )

    assert elements.inner.T0 == pytest.approx(
        1986.093
    )

    assert elements.inner.e == pytest.approx(
        0.0220
    )

    assert elements.inner.a == pytest.approx(
        0.0385
    )

    assert elements.inner.Omega == pytest.approx(
        334.5
    )

    assert elements.inner.omega == pytest.approx(
        232.9
    )

    assert elements.inner.i == pytest.approx(
        85.8
    )

    assert elements.K1 == pytest.approx(
        13.13
    )

    assert elements.K2 == pytest.approx(
        19.21
    )

    assert elements.f == pytest.approx(
        -0.33
    )

    assert elements.outer.P == pytest.approx(
        30.127
    )

    assert elements.outer.T0 == pytest.approx(
        2010.179
    )

    assert elements.outer.e == pytest.approx(
        0.324
    )

    assert elements.outer.a == pytest.approx(
        0.3361
    )

    assert elements.outer.Omega == pytest.approx(
        154.25
    )

    assert elements.outer.omega == pytest.approx(
        84.92
    )

    assert elements.outer.i == pytest.approx(
        88.28
    )

    assert elements.K3 == pytest.approx(
        6.057
    )

    assert elements.K4 == pytest.approx(
        8.60
    )

    assert elements.V0 == pytest.approx(
        -22.575
    )


# ==========================================================
# Observation-subset validation
# ==========================================================

def test_hip111805_orbit3_observation_counts():

    result = load_hip111805()

    counts = observation_counts(

        result["observations"],

        result["rv_observations"]

    )

    assert counts == {

        "I1": 10,
        "I2": 100,

        "Va": 62,
        "Vb": 2,
        "Vc": 29,

    }


# ==========================================================
# Fitted-parameter mask
# ==========================================================

def test_hip111805_orbit3_fit_mask():

    result = load_hip111805()

    elements = result[
        "elements"
    ]

    fitted_names = (
        elements.fitted_parameter_names()
    )

    assert len(
        fitted_names
    ) == 20

    assert set(
        fitted_names
    ) == {

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

    }


# ==========================================================
# Initial chi-square
# ==========================================================

def test_hip111805_orbit3_initial_chi_square_is_finite():

    result = load_hip111805()

    residuals = orbit3_residuals(

        result["observations"],

        result["rv_observations"],

        result["elements"]

    )

    chi2 = np.sum(
        residuals**2
    )

    assert residuals.ndim == 1

    assert len(
        residuals
    ) == 313

    assert np.all(
        np.isfinite(
            residuals
        )
    )

    assert np.isfinite(
        chi2
    )

    assert chi2 >= 0.0


# ==========================================================
# Numerical ORBIT3 fit
# ==========================================================

def test_hip111805_orbit3_residual_decomposition():

    result = load_hip111805()

    elements = result[
        "elements"
    ]

    observations = result[
        "observations"
    ]

    rv_observations = result[
        "rv_observations"
    ]

    initial_blocks = (
        decompose_residuals(

            observations,

            rv_observations,

            elements

        )
    )

    initial_chi2 = sum(

        block["chi2"]

        for block
        in initial_blocks.values()

    )

    fit = fit_orbit3(

        elements,

        observations,

        rv_observations,

    )

    assert (
        fit.optimizer_result.success
    )

    fitted_blocks = (
        decompose_residuals(

            observations,

            rv_observations,

            fit.elements

        )
    )

    fitted_chi2 = sum(

        block["chi2"]

        for block
        in fitted_blocks.values()

    )

    production_chi2 = np.sum(

        orbit3_residuals(

            observations,

            rv_observations,

            fit.elements

        )
        ** 2

    )

    np.testing.assert_allclose(

        fitted_chi2,

        production_chi2

    )

    assert fitted_chi2 < (
        initial_chi2
    )

    expected_counts = {

        "X1": 10,
        "Y1": 10,

        "X2": 100,
        "Y2": 100,

        "RV1": 62,
        "RV2": 2,
        "RV3": 29,

    }

    for block_name, expected_N in (
        expected_counts.items()
    ):

        assert (
            initial_blocks[
                block_name
            ]["N"]
            ==
            expected_N
        )

        assert (
            fitted_blocks[
                block_name
            ]["N"]
            ==
            expected_N
        )

    print()

    print(
        "HIP111805 ORBIT3 RESIDUAL DECOMPOSITION"
    )

    print(
        "=" * 80
    )

    print()

    print(
        f"Initial chi2 : "
        f"{initial_chi2:.6f}"
    )

    print(
        f"Fitted chi2  : "
        f"{fitted_chi2:.6f}"
    )

    print(
        f"Delta chi2   : "
        f"{initial_chi2 - fitted_chi2:.6f}"
    )

    print()

    print(
        f"{'Block':<8}"
        f"{'N':>8}"
        f"{'Initial chi2':>18}"
        f"{'Fitted chi2':>18}"
        f"{'Delta':>16}"
    )

    for block_name in (

        "X1",
        "Y1",
        "X2",
        "Y2",
        "RV1",
        "RV2",
        "RV3",

    ):

        initial = (
            initial_blocks[
                block_name
            ]
        )

        fitted = (
            fitted_blocks[
                block_name
            ]
        )

        print(

            f"{block_name:<8}"
            f"{initial['N']:>8}"
            f"{initial['chi2']:>18.6f}"
            f"{fitted['chi2']:>18.6f}"
            f"{initial['chi2'] - fitted['chi2']:>16.6f}"

        )


# ==========================================================
# RV convention diagnostic
# ==========================================================

def test_hip111805_orbit3_rv_convention_diagnostic():

    result = load_hip111805()

    elements = result[
        "elements"
    ]

    rv_observations = result[
        "rv_observations"
    ]

    current_model = (
        model_values_for_observations(

            rv_observations,

            elements

        )
    )

    toggled_elements = (
        toggle_rv_omega_convention(
            elements
        )
    )

    toggled_model = (
        model_values_for_observations(

            rv_observations,

            toggled_elements

        )
    )

    current_stats = rv_statistics(

        rv_observations,

        current_model

    )

    toggled_stats = rv_statistics(

        rv_observations,

        toggled_model

    )

    assert np.all(
        np.isfinite(
            current_model
        )
    )

    assert np.all(
        np.isfinite(
            toggled_model
        )
    )

    print()

    print(
        "HIP111805 ORBIT3 RV CONVENTION DIAGNOSTIC"
    )

    print(
        "=" * 80
    )

    print()

    print(
        f"Current RV chi2 : "
        f"{current_stats['chi2']:.6f}"
    )

    print(
        f"Toggled RV chi2 : "
        f"{toggled_stats['chi2']:.6f}"
    )

    print(
        f"Difference       : "
        f"{toggled_stats['chi2'] - current_stats['chi2']:.6f}"
    )


# ==========================================================
# Inner/outer RV contribution diagnostic
# ==========================================================

def test_hip111805_orbit3_inner_outer_rv_diagnostic():

    result = load_hip111805()

    elements = result[
        "elements"
    ]

    rv_observations = result[
        "rv_observations"
    ]

    # ------------------------------------------------------
    # Full production model.
    # ------------------------------------------------------

    full_model = (
        model_values_for_observations(

            rv_observations,

            elements

        )
    )

    # ------------------------------------------------------
    # Inner-only model.
    #
    # Remove K3, K4 and V0. The remaining RV signal is
    # generated exclusively by the inner spectroscopic orbit.
    # ------------------------------------------------------

    inner_elements = (
        inner_only_elements(
            elements
        )
    )

    inner_model = (
        model_values_for_observations(

            rv_observations,

            inner_elements

        )
    )

    # ------------------------------------------------------
    # Outer-only model.
    #
    # Remove K1 and K2. The remaining RV signal is generated
    # exclusively by the outer spectroscopic orbit plus V0.
    # ------------------------------------------------------

    outer_elements = (
        outer_only_elements(
            elements
        )
    )

    outer_model = (
        model_values_for_observations(

            rv_observations,

            outer_elements

        )
    )

    # ------------------------------------------------------
    # The production full model must equal the algebraic sum
    # of the isolated inner and outer contributions.
    #
    # This is an internal consistency check, not a physical
    # correctness assertion.
    # ------------------------------------------------------

    np.testing.assert_allclose(

        full_model,

        inner_model
        +
        outer_model,

        rtol=1e-12,

        atol=1e-12

    )

    # ------------------------------------------------------
    # Separate observations by ORBIT3 RV component.
    # ------------------------------------------------------

    observations_Va = [

        obs

        for obs in rv_observations

        if obs.orbit_id == "Va"

    ]

    observations_Vb = [

        obs

        for obs in rv_observations

        if obs.orbit_id == "Vb"

    ]

    observations_Vc = [

        obs

        for obs in rv_observations

        if obs.orbit_id == "Vc"

    ]

    # ------------------------------------------------------
    # Extract corresponding model values.
    # ------------------------------------------------------

    full_Va = (
        model_values_for_observations(

            observations_Va,

            elements

        )
    )

    inner_Va = (
        model_values_for_observations(

            observations_Va,

            inner_elements

        )
    )

    outer_Va = (
        model_values_for_observations(

            observations_Va,

            outer_elements

        )
    )

    full_Vb = (
        model_values_for_observations(

            observations_Vb,

            elements

        )
    )

    inner_Vb = (
        model_values_for_observations(

            observations_Vb,

            inner_elements

        )
    )

    outer_Vb = (
        model_values_for_observations(

            observations_Vb,

            outer_elements

        )
    )

    full_Vc = (
        model_values_for_observations(

            observations_Vc,

            elements

        )
    )

    inner_Vc = (
        model_values_for_observations(

            observations_Vc,

            inner_elements

        )
    )

    outer_Vc = (
        model_values_for_observations(

            observations_Vc,

            outer_elements

        )
    )

    # ------------------------------------------------------
    # Component-level statistics.
    # ------------------------------------------------------

    Va_full_stats = rv_statistics(
        observations_Va,
        full_Va
    )

    Va_inner_stats = rv_statistics(
        observations_Va,
        inner_Va
    )

    Va_outer_stats = rv_statistics(
        observations_Va,
        outer_Va
    )

    Vb_full_stats = rv_statistics(
        observations_Vb,
        full_Vb
    )

    Vb_inner_stats = rv_statistics(
        observations_Vb,
        inner_Vb
    )

    Vb_outer_stats = rv_statistics(
        observations_Vb,
        outer_Vb
    )

    Vc_full_stats = rv_statistics(
        observations_Vc,
        full_Vc
    )

    Vc_inner_stats = rv_statistics(
        observations_Vc,
        inner_Vc
    )

    Vc_outer_stats = rv_statistics(
        observations_Vc,
        outer_Vc
    )

    # ------------------------------------------------------
    # Report.
    # ------------------------------------------------------

    print()

    print(
        "HIP111805 ORBIT3 INNER/OUTER RV DIAGNOSTIC"
    )

    print(
        "=" * 80
    )

    print()

    print(
        "Va = Aa: inner primary + outer primary"
    )

    print(
        "-" * 80
    )

    print(
        f"Full   chi2 : "
        f"{Va_full_stats['chi2']:.6f}"
    )

    print(
        f"Inner  chi2 : "
        f"{Va_inner_stats['chi2']:.6f}"
    )

    print(
        f"Outer  chi2 : "
        f"{Va_outer_stats['chi2']:.6f}"
    )

    print(
        f"Full   RMS  : "
        f"{Va_full_stats['rms']:.6f}"
    )

    print(
        f"Inner  RMS  : "
        f"{Va_inner_stats['rms']:.6f}"
    )

    print(
        f"Outer  RMS  : "
        f"{Va_outer_stats['rms']:.6f}"
    )

    print()

    print(
        "Vb = Ab: inner secondary + outer primary"
    )

    print(
        "-" * 80
    )

    print(
        f"Full   chi2 : "
        f"{Vb_full_stats['chi2']:.6f}"
    )

    print(
        f"Inner  chi2 : "
        f"{Vb_inner_stats['chi2']:.6f}"
    )

    print(
        f"Outer  chi2 : "
        f"{Vb_outer_stats['chi2']:.6f}"
    )

    print(
        f"Full   RMS  : "
        f"{Vb_full_stats['rms']:.6f}"
    )

    print(
        f"Inner  RMS  : "
        f"{Vb_inner_stats['rms']:.6f}"
    )

    print(
        f"Outer  RMS  : "
        f"{Vb_outer_stats['rms']:.6f}"
    )

    print()

    print(
        "Vc = B: outer secondary"
    )

    print(
        "-" * 80
    )

    print(
        f"Full   chi2 : "
        f"{Vc_full_stats['chi2']:.6f}"
    )

    print(
        f"Inner  chi2 : "
        f"{Vc_inner_stats['chi2']:.6f}"
    )

    print(
        f"Outer  chi2 : "
        f"{Vc_outer_stats['chi2']:.6f}"
    )

    print(
        f"Full   RMS  : "
        f"{Vc_full_stats['rms']:.6f}"
    )

    print(
        f"Inner  RMS  : "
        f"{Vc_inner_stats['rms']:.6f}"
    )

    print(
        f"Outer  RMS  : "
        f"{Vc_outer_stats['rms']:.6f}"
    )

    print()

    print(
        "MODEL COMPONENT CONSISTENCY"
    )

    print(
        "-" * 80
    )

    print(
        "Full model = inner-only + outer-only: PASS"
    )

def test_hip111805_orbit3_precession_diagnostic():
    result = load_hip111805()

    observations = result["observations"]

    # ORBIT3 packed-sexagesimal parsing:
    # RA 22.3848 -> 22h 38m 48s
    # Dec 44.19  -> 44d 19m 00s
    ra_hours = 22.0 + 38.0 / 60.0 + 48.0 / 3600.0
    ra_deg = 15.0 * ra_hours

    dec_deg = 44.0 + 19.0 / 60.0

    pr = (
        0.0057
        * np.sin(np.deg2rad(ra_deg))
        / np.cos(np.deg2rad(dec_deg))
    )

    corrected = []

    for obs in observations:
        if obs.orbit_id in ("I1", "I2"):

            corrected.append(
                replace(
                    obs,
                    theta=(
                        obs.theta
                        + (2000.0 - obs.epoch) * pr
                    ),
                )
            )

        else:
            corrected.append(obs)

    raw_blocks = decompose_residuals(
        observations,
        result["rv_observations"],
        result["elements"],
    )

    corrected_blocks = decompose_residuals(
        corrected,
        result["rv_observations"],
        result["elements"],
    )

    print()
    print("HIP111805 ORBIT3 PRECESSION DIAGNOSTIC")
    print("=" * 80)
    print(f"ORBIT3 PR : {pr:.15f} deg/yr")
    print()

    for block in (
        "X1",
        "Y1",
        "X2",
        "Y2",
        "RV1",
        "RV2",
        "RV3",
    ):
        raw = raw_blocks[block]["chi2"]
        corrected_chi2 = corrected_blocks[block]["chi2"]

        print(
            f"{block:<6}"
            f" raw={raw:14.6f}"
            f" corrected={corrected_chi2:14.6f}"
            f" delta={raw - corrected_chi2:14.6f}"
        )

    raw_total = sum(
        raw_blocks[b]["chi2"]
        for b in raw_blocks
    )

    corrected_total = sum(
        corrected_blocks[b]["chi2"]
        for b in corrected_blocks
    )

    print()
    print(
        f"TOTAL  raw={raw_total:.6f}"
        f" corrected={corrected_total:.6f}"
        f" delta={raw_total - corrected_total:.6f}"
    )

    print()
    print("SELECTED I2 ANGLE CORRECTIONS")
    print("-" * 80)

    for obs, corr in zip(observations, corrected):

        if (
            obs.orbit_id == "I2"
            and obs.epoch in (
                2007.5991,
                2007.8225,
                2013.5704,
            )
        ):
            print(
                f"epoch={obs.epoch:.4f}"
                f" theta_raw={obs.theta:.6f}"
                f" theta_corrected={corr.theta:.6f}"
                f" delta={corr.theta - obs.theta:.6f}"
            )

def test_hip111805_orbit3_forward_model_diagnostic():

    import numpy as np

    from pyseshat.triple.triple_orbit import (
        compute_orbit3_visual,
        compute_orbit3_radial_velocities,
    )

    result = load_hip111805()

    elements = result["elements"]
    observations = result["observations"]
    rv_observations = result["rv_observations"]

    i1 = [
        obs for obs in observations
        if obs.orbit_id == "I1"
    ]

    i2 = [
        obs for obs in observations
        if obs.orbit_id == "I2"
    ]

    va = [
        obs for obs in rv_observations
        if obs.orbit_id == "Va"
    ]

    vb = [
        obs for obs in rv_observations
        if obs.orbit_id == "Vb"
    ]

    vc = [
        obs for obs in rv_observations
        if obs.orbit_id == "Vc"
    ]

    # ------------------------------------------------------
    # PySeshat ORBIT3 forward model
    # ------------------------------------------------------

    model_i1 = compute_orbit3_visual(
        np.array([obs.epoch for obs in i1]),
        elements
    )

    model_i2 = compute_orbit3_visual(
        np.array([obs.epoch for obs in i2]),
        elements
    )

    model_va = compute_orbit3_radial_velocities(
        np.array([obs.epoch for obs in va]),
        elements
    )

    model_vb = compute_orbit3_radial_velocities(
        np.array([obs.epoch for obs in vb]),
        elements
    )

    model_vc = compute_orbit3_radial_velocities(
        np.array([obs.epoch for obs in vc]),
        elements
    )

    # ------------------------------------------------------
    # Extract actual ORBIT3 observables
    # ------------------------------------------------------

    i1_x = model_i1["POS1"]["X"]
    i1_y = model_i1["POS1"]["Y"]

    i2_x = model_i2["POS2"]["X"]
    i2_y = model_i2["POS2"]["Y"]

    va_model = model_va["Aa"]
    vb_model = model_vb["Ab"]
    vc_model = model_vc["B"]

    # ------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------

    assert np.all(np.isfinite(i1_x))
    assert np.all(np.isfinite(i1_y))

    assert np.all(np.isfinite(i2_x))
    assert np.all(np.isfinite(i2_y))

    assert np.all(np.isfinite(va_model))
    assert np.all(np.isfinite(vb_model))
    assert np.all(np.isfinite(vc_model))

    # ------------------------------------------------------
    # Diagnostic output
    # ------------------------------------------------------

    print()
    print("=" * 110)
    print("HIP111805 ORBIT3 FORWARD-MODEL DIAGNOSTIC")
    print("=" * 110)

    print()
    print("INITIAL ORBIT3 PARAMETERS")
    print("-" * 110)

    print(f"P       = {elements.inner.P:.12g}")
    print(f"T0      = {elements.inner.T0:.12g}")
    print(f"e       = {elements.inner.e:.12g}")
    print(f"a       = {elements.inner.a:.12g}")
    print(f"Omega   = {elements.inner.Omega:.12g}")
    print(f"omega   = {elements.inner.omega:.12g}")
    print(f"i       = {elements.inner.i:.12g}")
    print(f"K1      = {elements.K1:.12g}")
    print(f"K2      = {elements.K2:.12g}")
    print(f"f       = {elements.f:.12g}")

    print(f"P2      = {elements.outer.P:.12g}")
    print(f"T02     = {elements.outer.T0:.12g}")
    print(f"e2      = {elements.outer.e:.12g}")
    print(f"a2      = {elements.outer.a:.12g}")
    print(f"Omega2  = {elements.outer.Omega:.12g}")
    print(f"omega2  = {elements.outer.omega:.12g}")
    print(f"i2      = {elements.outer.i:.12g}")
    print(f"K3      = {elements.K3:.12g}")
    print(f"K4      = {elements.K4:.12g}")
    print(f"V0      = {elements.V0:.12g}")

    print()
    print("FORWARD-MODEL OUTPUT")
    print("-" * 110)

    print()
    print("I1 / POS1")
    print(
        "epoch                 X                         Y"
    )

    for j in range(min(10, len(i1))):

        print(
            f"{i1[j].epoch:12.6f} "
            f"{i1_x[j]: .16e} "
            f"{i1_y[j]: .16e}"
        )

    print()
    print("I2 / POS2")
    print(
        "epoch                 X                         Y"
    )

    for j in range(min(10, len(i2))):

        print(
            f"{i2[j].epoch:12.6f} "
            f"{i2_x[j]: .16e} "
            f"{i2_y[j]: .16e}"
        )

    print()
    print("RADIAL VELOCITIES")
    print("-" * 110)

    print()
    print("Va / Aa")

    for j in range(min(10, len(va))):

        print(
            f"{va[j].epoch:12.6f} "
            f"{va_model[j]: .16e}"
        )

    print()
    print("Vb / Ab")

    for j in range(min(10, len(vb))):

        print(
            f"{vb[j].epoch:12.6f} "
            f"{vb_model[j]: .16e}"
        )

    print()
    print("Vc / B")

    for j in range(min(10, len(vc))):

        print(
            f"{vc[j].epoch:12.6f} "
            f"{vc_model[j]: .16e}"
        )

    # ------------------------------------------------------
    # Explicit ORBIT3 POS2 composition check
    # ------------------------------------------------------

    inner_i2 = model_i2["inner"]
    outer_i2 = model_i2["outer"]

    expected_x = (
        outer_i2["X"]
        +
        elements.f * inner_i2["X"]
    )

    expected_y = (
        outer_i2["Y"]
        +
        elements.f * inner_i2["Y"]
    )

    dx = np.max(
        np.abs(
            i2_x - expected_x
        )
    )

    dy = np.max(
        np.abs(
            i2_y - expected_y
        )
    )

    print()
    print("ORBIT3 POS2 COMPOSITION")
    print("-" * 110)

    print(
        "max |POS2 X - (outer X + f*inner X)| = "
        f"{dx:.16e}"
    )

    print(
        "max |POS2 Y - (outer Y + f*inner Y)| = "
        f"{dy:.16e}"
    )

    assert np.allclose(
        i2_x,
        expected_x,
        rtol=0.0,
        atol=1e-12
    )

    assert np.allclose(
        i2_y,
        expected_y,
        rtol=0.0,
        atol=1e-12
    )

    print()
    print("RESULT")
    print("-" * 110)
    print(
        "PySeshat POS1/POS2 forward-model composition "
        "is internally consistent."
    )


def test_hip111805_orbit3_equation_level_diagnostic():

    import numpy as np

    from pyseshat.orbit.orbit_model import (
        compute_visual_orbit,
    )

    from pyseshat.triple.triple_orbit import (
        compute_orbit3_visual,
        compute_orbit3_radial_velocities,
    )

    result = load_hip111805()

    elements = result["elements"]

    observations = result["observations"]
    rv_observations = result["rv_observations"]

    print()
    print("=" * 110)
    print("HIP111805 ORBIT3 EQUATION-LEVEL DIAGNOSTIC")
    print("=" * 110)

    # ------------------------------------------------------
    # Test the underlying Kepler/visual orbit calculation
    # separately for inner and outer orbits.
    # ------------------------------------------------------

    epochs = np.array(
        sorted(
            set(
                obs.epoch
                for obs in observations
            )
        ),
        dtype=float,
    )

    inner = compute_visual_orbit(
        epochs,
        elements.inner,
    )

    outer = compute_visual_orbit(
        epochs,
        elements.outer,
    )

    # ------------------------------------------------------
    # Independently reconstruct the ORBIT3 visual equations
    # from the quantities returned by compute_visual_orbit.
    # ------------------------------------------------------

    expected_pos1_x = inner["X"]
    expected_pos1_y = inner["Y"]

    expected_pos2_x = (
        outer["X"]
        +
        elements.f * inner["X"]
    )

    expected_pos2_y = (
        outer["Y"]
        +
        elements.f * inner["Y"]
    )

    production = compute_orbit3_visual(
        epochs,
        elements,
    )

    dx1 = np.max(
        np.abs(
            production["POS1"]["X"]
            -
            expected_pos1_x
        )
    )

    dy1 = np.max(
        np.abs(
            production["POS1"]["Y"]
            -
            expected_pos1_y
        )
    )

    dx2 = np.max(
        np.abs(
            production["POS2"]["X"]
            -
            expected_pos2_x
        )
    )

    dy2 = np.max(
        np.abs(
            production["POS2"]["Y"]
            -
            expected_pos2_y
        )
    )

    print()
    print("VISUAL EQUATIONS")
    print("-" * 110)

    print(
        f"POS1 X max difference : {dx1:.16e}"
    )

    print(
        f"POS1 Y max difference : {dy1:.16e}"
    )

    print(
        f"POS2 X max difference : {dx2:.16e}"
    )

    print(
        f"POS2 Y max difference : {dy2:.16e}"
    )

    # ------------------------------------------------------
    # Reconstruct ORBIT3 RV equations independently.
    #
    # A1 = e*cos(omega) + cos(nu + omega)
    #
    # Aa = K1*A1 + V0 + K3*A1_outer
    # Ab = -K2*A1 + V0 + K3*A1_outer
    # B  = V0 - K4*A1_outer
    # ------------------------------------------------------

    rv_epochs = {
        obs.orbit_id: np.array(
            [
                o.epoch
                for o in rv_observations
                if o.orbit_id == obs.orbit_id
            ],
            dtype=float,
        )
        for obs in rv_observations
    }

    rv_epochs = {
        key: value
        for key, value in rv_epochs.items()
    }

    rv_model = compute_orbit3_radial_velocities(
        np.concatenate(
            list(rv_epochs.values())
        ),
        elements,
    )

    # ------------------------------------------------------
    # Compare the individual algebraic components.
    # ------------------------------------------------------

    for orbit_id, label in (
        ("Va", "Aa"),
        ("Vb", "Ab"),
        ("Vc", "B"),
    ):

        times = rv_epochs[orbit_id]

        inner_rv = compute_visual_orbit(
            times,
            elements.inner,
        )

        outer_rv = compute_visual_orbit(
            times,
            elements.outer,
        )

        inner_omega = np.deg2rad(
            elements.inner.omega
        )

        outer_omega = np.deg2rad(
            elements.outer.omega
        )

        inner_a1 = (
            elements.inner.e
            *
            np.cos(inner_omega)
            +
            np.cos(
                inner_rv["nu"]
                +
                inner_omega
            )
        )

        outer_a1 = (
            elements.outer.e
            *
            np.cos(outer_omega)
            +
            np.cos(
                outer_rv["nu"]
                +
                outer_omega
            )
        )

        if label == "Aa":

            expected = (
                elements.K1 * inner_a1
                +
                elements.V0
                +
                elements.K3 * outer_a1
            )

        elif label == "Ab":

            expected = (
                -elements.K2 * inner_a1
                +
                elements.V0
                +
                elements.K3 * outer_a1
            )

        else:

            expected = (
                elements.V0
                -
                elements.K4 * outer_a1
            )

        actual = np.asarray(
            rv_model[label]
        )

        # rv_model was evaluated on concatenated epochs, so
        # compare against the corresponding block directly.
        #
        # Re-evaluate the production function on this block
        # to avoid any ambiguity in concatenation ordering.

        block_model = compute_orbit3_radial_velocities(
            times,
            elements,
        )[label]

        difference = np.max(
            np.abs(
                block_model
                -
                expected
            )
        )

        print()
        print(
            f"RV {orbit_id} / {label}"
        )

        print(
            f"maximum equation difference : "
            f"{difference:.16e}"
        )

        assert np.allclose(
            block_model,
            expected,
            rtol=0.0,
            atol=1e-12,
        )

    # ------------------------------------------------------
    # Final result
    # ------------------------------------------------------

    print()
    print("RESULT")
    print("-" * 110)
    print(
        "Current PySeshat ORBIT3 equations are "
        "internally reproduced independently."
    )


def test_hip111805_orbit3_kepler_visual_primitive_diagnostic():

    import numpy as np

    from pyseshat.orbit.orbit_model import compute_visual_orbit

    result = load_hip111805()
    elements = result["elements"]

    epochs = np.array(
        sorted(
            set(
                obs.epoch
                for obs in result["observations"]
                if obs.orbit_id in ("I1", "I2")
            )
        ),
        dtype=float,
    )

    print()
    print("=" * 110)
    print("HIP111805 ORBIT3 KEPLER/VISUAL PRIMITIVE DIAGNOSTIC")
    print("=" * 110)

    for name, element in (
        ("INNER", elements.inner),
        ("OUTER", elements.outer),
    ):

        orbit = compute_visual_orbit(
            epochs,
            element,
        )

        print()
        print(name)
        print("-" * 110)

        print(f"P       = {element.P:.12g}")
        print(f"T0      = {element.T0:.12g}")
        print(f"e       = {element.e:.12g}")
        print(f"a       = {element.a:.12g}")
        print(f"Omega   = {element.Omega:.12g}")
        print(f"omega   = {element.omega:.12g}")
        print(f"i       = {element.i:.12g}")

        print()
        print("Returned fields:")
        print(
            "    "
            + ", ".join(
                sorted(orbit.keys())
            )
        )

        for field in ("X", "Y"):

            values = np.asarray(
                orbit[field],
                dtype=float,
            )

            print(
                f"{field} range : "
                f"{np.min(values): .16e} "
                f"to "
                f"{np.max(values): .16e}"
            )

            assert np.all(
                np.isfinite(values)
            )

        # --------------------------------------------------
        # Independently reconstruct the orbital phase from
        # the public orbital parameters.
        #
        # This is a diagnostic only. It does not modify the
        # production implementation.
        # --------------------------------------------------

        M = (
            2.0
            * np.pi
            * (epochs - element.T0)
            / element.P
        )

        # ORBIT3-style mean-anomaly normalization.
        M = np.mod(
            M + np.pi,
            2.0 * np.pi,
        ) - np.pi

        # Newton solution of Kepler's equation.
        E = M.copy()

        for _ in range(100):

            correction = (
                E
                -
                element.e * np.sin(E)
                -
                M
            ) / (
                1.0
                -
                element.e * np.cos(E)
            )

            E -= correction

            if np.max(
                np.abs(correction)
            ) < 1e-14:

                break

        kepler_error = (
            E
            -
            element.e * np.sin(E)
            -
            M
        )

        nu = 2.0 * np.arctan2(
            np.sqrt(1.0 + element.e)
            * np.sin(E / 2.0),
            np.sqrt(1.0 - element.e)
            * np.cos(E / 2.0),
        )

        print()
        print(
            "Independent Kepler reconstruction"
        )
        print(
            "M range : "
            f"{np.min(M): .16e} "
            f"to "
            f"{np.max(M): .16e}"
        )
        print(
            "E range : "
            f"{np.min(E): .16e} "
            f"to "
            f"{np.max(E): .16e}"
        )
        print(
            "nu range: "
            f"{np.min(nu): .16e} "
            f"to "
            f"{np.max(nu): .16e}"
        )
        print(
            "max |E - e sin(E) - M| = "
            f"{np.max(np.abs(kepler_error)):.16e}"
        )

        assert np.max(
            np.abs(kepler_error)
        ) < 1e-12

    print()
    print("RESULT")
    print("-" * 110)
    print(
        "The production visual-orbit primitive returns "
        "finite X/Y values, and the independently "
        "reconstructed Kepler solution is numerically "
        "consistent."
    )


def test_hip111805_orbit3_jacobian_descent_diagnostic():

    import numpy as np

    from pyseshat.triple.orbit3_fit import (
        orbit3_jacobian,
        orbit3_fit_flag_residuals,
    )

    result = load_hip111805()

    elements = result["elements"]
    observations = result["observations"]
    rv_observations = result["rv_observations"]

    fitted = (
        np.asarray(
            result["elements"].fitted_mask(),
            dtype=bool
        )
    )

    full = np.array([
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
    ], dtype=float)

    p0 = full[fitted]

    r0 = orbit3_fit_flag_residuals(
        p0,
        elements,
        observations,
        rv_observations,
    )

    J = orbit3_jacobian(
        p0,
        elements,
        observations,
        rv_observations,
    )

    gradient = J.T @ r0

    print()
    print("=" * 110)
    print("HIP111805 ORBIT3 JACOBIAN DESCENT DIAGNOSTIC")
    print("=" * 110)

    print()
    print(f"Residual norm       : {np.linalg.norm(r0):.12e}")
    print(f"Jacobian shape      : {J.shape}")
    print(f"Jacobian norm       : {np.linalg.norm(J):.12e}")
    print(f"Gradient norm       : {np.linalg.norm(gradient):.12e}")

    print()
    print("PARAMETER             J-NORM              J^T r")
    print("-" * 110)

    names = elements.fitted_parameter_names()

    for j, name in enumerate(names):

        print(
            f"{name:<12}"
            f"{np.linalg.norm(J[:, j]):>20.12e}"
            f"{gradient[j]:>20.12e}"
        )

    print()
    print("SMALLEST SINGULAR VALUES")
    print("-" * 110)

    singular_values = np.linalg.svd(
        J,
        compute_uv=False,
    )

    for value in singular_values[-5:]:

        print(
            f"{value:.12e}"
        )

    print()
    print("RESULT")
    print("-" * 110)

    assert np.all(
        np.isfinite(J)
    )

    assert np.linalg.norm(J) > 0.0

    assert np.linalg.norm(gradient) > 0.0