import json
import numpy as np

from pyseshat.io.orbit_parser import (
    parse_orbit_file
)

from pyseshat.inference.optimizer import (
    fit_visual_orbit
)

from pyseshat.inference.likelihood import (
    visual_xy_residuals,
    rv_residuals
)

from pyseshat.analysis.statistics import (
    combined_fit_summary
)

from metrics import (
    parameter_relative_error
)


# ==========================================================
# Benchmark execution
# ==========================================================

def run_benchmark(
    system_name,
    config
):
    """
    Execute deterministic benchmark fit.
    """

    # ------------------------------------------------------
    # Load metadata
    # ------------------------------------------------------

    with open(
        config["expected_file"],
        "r"
    ) as f:

        expected = json.load(f)

    # ------------------------------------------------------
    # Parse dataset
    # ------------------------------------------------------

    parsed = parse_orbit_file(
        str(config["data_file"])
    )

    elements = parsed["elements"]

    visual_data = parsed["visual_data"]

    rv_data = parsed["rv_data"]

    # ------------------------------------------------------
    # Convert visual observations
    # ------------------------------------------------------

    t = visual_data[:,0]

    theta = np.deg2rad(
        visual_data[:,1]
    )

    rho = visual_data[:,2]

    sigma = visual_data[:,3]

    X = rho * np.cos(theta)

    Y = rho * np.sin(theta)

    sx = sigma

    sy = sigma

    # ------------------------------------------------------
    # Run fit
    # ------------------------------------------------------

    fit_result = fit_visual_orbit(
        elements,
        t,
        X,
        Y,
        sx,
        sy,
        rv_data=rv_data
    )

    fitted = fit_result.elements

    expected_elements = expected["initial_parameters"]

    # ------------------------------------------------------
    # Residual diagnostics
    # ------------------------------------------------------

    res_visual = visual_xy_residuals(
        fitted.fitted_parameters(),
        fitted,
        t,
        X,
        Y,
        sx,
        sy
    )

    if len(rv_data) > 0:

        res_rv = rv_residuals(
            fitted.fitted_parameters(),
            fitted,
            rv_data
        )

        stats = combined_fit_summary(
            visual_residuals=res_visual,
            rv_residuals=res_rv,
            rv_components=rv_data[:,3],
            n_params=len(
                fitted.fitted_parameters()
            )
        )

    else:

        stats = combined_fit_summary(
            visual_residuals=res_visual,
            rv_residuals=None,
            rv_components=None,
            n_params=len(
                fitted.fitted_parameters()
            )
        )

    parameter_errors = {}

    for name in expected_elements:

        expected_value = (
            expected_elements[name]
        )

        fitted_value = getattr(
            fitted,
            name
        )

        rel_error = (
            parameter_relative_error(
                name,
                expected_value,
                fitted_value
            )
        )

        parameter_errors[name] = {

            "expected": float(
                expected_value
            ),

            "fitted": float(
                fitted_value
            ),

            "relative_error": float(
                rel_error
            )
        }

    return {

        "system": system_name,

        "statistics": stats,

        "parameter_errors": parameter_errors,

        "fitted_elements": {
            name: float(
                getattr(fitted, name)
            )
            for name in expected_elements
        },

        "optimizer_success": fit_result.optimizer_result.success,
        
        "optimizer_message":
        str(
            fit_result.optimizer_result.message
        ),
    }