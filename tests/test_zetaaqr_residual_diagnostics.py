from pathlib import Path


import numpy as np


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
    compute_orbit3_visual
)


from pyseshat.triple.triple_parser import (
    parse_triple_file
)


from pyseshat.triple.triple_residuals import (
    orbit3_visual_residuals
)


from pyseshat.triple.orbit3_fit import (
    fit_orbit3
)


# ==========================================================
# Paths
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
# Literature reference
# ==========================================================

LITERATURE = {

    "P": 25.95,
    "T0": 2006.52,
    "e": 0.872,

    "a_astrometric": 0.110,

    "Omega": 293.7,
    "omega": 100.9,
    "i": 11.8,

    "F": 3.50,

    "P2": 540.0,
    "T02": 1981.50,
    "e2": 0.419,
    "a2": 3.496,
    "Omega2": 131.3,
    "omega2": 269.3,
    "i2": 142.0,

}


# ==========================================================
# Literature -> ORBIT3 conversion
# ==========================================================

def literature_true_inner_a():

    return (

        LITERATURE["F"]
        *
        LITERATURE["a_astrometric"]

    )


def literature_wobble_factor():

    return (

        1.0
        /
        LITERATURE["F"]

    )


# ==========================================================
# Load observations
# ==========================================================

def load_observations():

    parsed = parse_triple_file(
        ZETA_FILE
    )


    observations = (
        parsed["observations"]
    )


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
# Literature ORBIT3 elements
# ==========================================================

def make_literature_elements():

    inner = OrbitalElements(

        P=LITERATURE["P"],

        T0=LITERATURE["T0"],

        e=LITERATURE["e"],

        a=literature_true_inner_a(),

        Omega=LITERATURE["Omega"],

        omega=LITERATURE["omega"],

        i=LITERATURE["i"],

    )


    outer = OrbitalElements(

        P=LITERATURE["P2"],

        T0=LITERATURE["T02"],

        e=LITERATURE["e2"],

        a=LITERATURE["a2"],

        Omega=LITERATURE["Omega2"],

        omega=LITERATURE["omega2"],

        i=LITERATURE["i2"],

    )


    elements = TripleOrbitalElements(

        outer=outer,

        inner=inner,

        f=literature_wobble_factor(),

    )


    # ------------------------------------------------------
    # Use exactly the same fitting mask as the literature
    # benchmark.
    # ------------------------------------------------------

    for key in elements.fit_flags:

        elements.fit_flags[key] = False


    for key in (

        "P",
        "T0",
        "e",
        "Omega",
        "omega",
        "i",
        "f",

        "P2",
        "T02",
        "e2",
        "a2",
        "Omega2",
        "omega2",
        "i2",

    ):

        elements.fit_flags[key] = True


    # The true inner semimajor axis is fixed.

    elements.fit_flags[
        "a"
    ] = False


    return elements


# ==========================================================
# Convert one observation to Cartesian coordinates
# ==========================================================

def observation_xy(
    obs
):

    theta = np.deg2rad(
        obs.theta
    )


    X = (
        obs.rho
        *
        np.cos(theta)
    )


    Y = (
        obs.rho
        *
        np.sin(theta)
    )


    return X, Y


# ==========================================================
# Direct diagnostic residual calculation
# ==========================================================
#
# We calculate the residuals ourselves here rather than
# relying on the concatenated residual vector. This gives
# us an explicit association between every residual and:
#
#     observation
#     ORBIT3 component
#     X/Y coordinate
#
# ==========================================================

def diagnostic_residuals(
    observations,
    elements
):

    rows = []


    for obs in observations:

        X_obs, Y_obs = (
            observation_xy(obs)
        )


        model = compute_orbit3_visual(

            np.array([
                obs.epoch
            ]),

            elements

        )


        if obs.orbit_id == "I1":

            X_model = (
                model["POS1"]["X"][0]
            )

            Y_model = (
                model["POS1"]["Y"][0]
            )


        elif obs.orbit_id == "I2":

            X_model = (
                model["POS2"]["X"][0]
            )

            Y_model = (
                model["POS2"]["Y"][0]
            )


        else:

            raise ValueError(
                obs.orbit_id
            )


        res_x = (
            X_obs
            -
            X_model
        ) / obs.sigma


        res_y = (
            Y_obs
            -
            Y_model
        ) / obs.sigma


        rows.append({

            "epoch": obs.epoch,

            "orbit_id": obs.orbit_id,

            "res_x": res_x,

            "res_y": res_y,

            "chi2": (
                res_x**2
                +
                res_y**2
            ),

        })


    return rows


# ==========================================================
# Summarize residuals
# ==========================================================

def summarize_residuals(
    rows
):

    summary = {}


    for orbit_id in (
        "I1",
        "I2",
    ):

        selected = [

            row
            for row in rows

            if row["orbit_id"] == orbit_id

        ]


        if not selected:

            continue


        res_x = np.array([

            row["res_x"]
            for row in selected

        ])


        res_y = np.array([

            row["res_y"]
            for row in selected

        ])


        chi2 = np.array([

            row["chi2"]
            for row in selected

        ])


        summary[orbit_id] = {

            "N": len(selected),

            "chi2": np.sum(
                chi2
            ),

            "rms_x": np.sqrt(
                np.mean(
                    res_x**2
                )
            ),

            "rms_y": np.sqrt(
                np.mean(
                    res_y**2
                )
            ),

            "mean_x": np.mean(
                res_x
            ),

            "mean_y": np.mean(
                res_y
            ),

            "max_abs_x": np.max(
                np.abs(res_x)
            ),

            "max_abs_y": np.max(
                np.abs(res_y)
            ),

        }


    return summary


# ==========================================================
# Fit starting from literature solution
# ==========================================================

def fit_from_literature(
    observations
):

    elements = (
        make_literature_elements()
    )


    result = fit_orbit3(

        elements,

        observations,

        [],

    )


    return result


# ==========================================================
# Main diagnostic test
# ==========================================================

def test_zetaaqr_residual_decomposition():

    observations = (
        load_observations()
    )


    literature_elements = (
        make_literature_elements()
    )


    # ------------------------------------------------------
    # Literature solution residuals
    # ------------------------------------------------------

    literature_rows = (
        diagnostic_residuals(

            observations,

            literature_elements

        )
    )


    literature_summary = (
        summarize_residuals(
            literature_rows
        )
    )


    literature_chi2 = sum(

        value["chi2"]

        for value
        in literature_summary.values()

    )


    # ------------------------------------------------------
    # Fit from literature solution
    # ------------------------------------------------------

    fit = fit_from_literature(
        observations
    )


    assert (
        fit.optimizer_result.success
    )


    fitted_elements = (
        fit.elements
    )


    # ------------------------------------------------------
    # Fitted residuals
    # ------------------------------------------------------

    fitted_rows = (
        diagnostic_residuals(

            observations,

            fitted_elements

        )
    )


    fitted_summary = (
        summarize_residuals(
            fitted_rows
        )
    )


    fitted_chi2 = sum(

        value["chi2"]

        for value
        in fitted_summary.values()

    )


    # ------------------------------------------------------
    # Report
    # ------------------------------------------------------

    print()

    print(
        "ZETA AQR RESIDUAL DECOMPOSITION"
    )

    print(
        "=" * 72
    )


    print()

    print(
        "TOTAL"
    )

    print(
        "-" * 72
    )

    print(
        f"Literature chi2 : "
        f"{literature_chi2:.6f}"
    )

    print(
        f"Fitted chi2     : "
        f"{fitted_chi2:.6f}"
    )

    print(
        f"Delta chi2      : "
        f"{literature_chi2 - fitted_chi2:.6f}"
    )


    print()

    print(
        "BY ORBIT"
    )

    print(
        "-" * 72
    )


    print(

        f"{'Orbit':<8}"
        f"{'N':>8}"
        f"{'Lit chi2':>16}"
        f"{'Fit chi2':>16}"
        f"{'Delta':>16}"

    )


    for orbit_id in (
        "I1",
        "I2",
    ):

        lit = (
            literature_summary[
                orbit_id
            ]
        )


        fit_summary = (
            fitted_summary[
                orbit_id
            ]
        )


        delta = (

            lit["chi2"]
            -
            fit_summary["chi2"]

        )


        print(

            f"{orbit_id:<8}"
            f"{lit['N']:>8}"
            f"{lit['chi2']:>16.6f}"
            f"{fit_summary['chi2']:>16.6f}"
            f"{delta:>16.6f}"

        )


    print()

    print(
        "RMS NORMALIZED RESIDUALS"
    )

    print(
        "-" * 72
    )


    print(

        f"{'Orbit':<8}"
        f"{'Lit RMS X':>16}"
        f"{'Fit RMS X':>16}"
        f"{'Lit RMS Y':>16}"
        f"{'Fit RMS Y':>16}"

    )


    for orbit_id in (
        "I1",
        "I2",
    ):

        lit = (
            literature_summary[
                orbit_id
            ]
        )


        fit_summary = (
            fitted_summary[
                orbit_id
            ]
        )


        print(

            f"{orbit_id:<8}"
            f"{lit['rms_x']:>16.6f}"
            f"{fit_summary['rms_x']:>16.6f}"
            f"{lit['rms_y']:>16.6f}"
            f"{fit_summary['rms_y']:>16.6f}"

        )


    print()

    print(
        "MEAN NORMALIZED RESIDUALS"
    )

    print(
        "-" * 72
    )


    print(

        f"{'Orbit':<8}"
        f"{'Lit mean X':>16}"
        f"{'Fit mean X':>16}"
        f"{'Lit mean Y':>16}"
        f"{'Fit mean Y':>16}"

    )


    for orbit_id in (
        "I1",
        "I2",
    ):

        lit = (
            literature_summary[
                orbit_id
            ]
        )


        fit_summary = (
            fitted_summary[
                orbit_id
            ]
        )


        print(

            f"{orbit_id:<8}"
            f"{lit['mean_x']:>16.6f}"
            f"{fit_summary['mean_x']:>16.6f}"
            f"{lit['mean_y']:>16.6f}"
            f"{fit_summary['mean_y']:>16.6f}"

        )


    print()

    print(
        "MAXIMUM ABSOLUTE NORMALIZED RESIDUAL"
    )

    print(
        "-" * 72
    )


    print(

        f"{'Orbit':<8}"
        f"{'Lit max X':>16}"
        f"{'Fit max X':>16}"
        f"{'Lit max Y':>16}"
        f"{'Fit max Y':>16}"

    )


    for orbit_id in (
        "I1",
        "I2",
    ):

        lit = (
            literature_summary[
                orbit_id
            ]
        )


        fit_summary = (
            fitted_summary[
                orbit_id
            ]
        )


        print(

            f"{orbit_id:<8}"
            f"{lit['max_abs_x']:>16.6f}"
            f"{fit_summary['max_abs_x']:>16.6f}"
            f"{lit['max_abs_y']:>16.6f}"
            f"{fit_summary['max_abs_y']:>16.6f}"

        )


    # ------------------------------------------------------
    # Consistency checks
    # ------------------------------------------------------

    # The diagnostic calculation must reproduce the same
    # total chi-square as the production residual function.

    production_literature = np.sum(

        orbit3_visual_residuals(

            observations,

            literature_elements

        ) ** 2

    )


    production_fitted = np.sum(

        orbit3_visual_residuals(

            observations,

            fitted_elements

        ) ** 2

    )


    np.testing.assert_allclose(

        literature_chi2,

        production_literature,

    )


    np.testing.assert_allclose(

        fitted_chi2,

        production_fitted,

    )


    assert fitted_chi2 < (
        literature_chi2
    )