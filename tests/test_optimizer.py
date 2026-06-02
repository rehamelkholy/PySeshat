from pathlib import Path

import numpy as np

from pyseshat.io.orbit_parser import (
    parse_orbit_file
)

from pyseshat.inference.optimizer import (
    fit_visual_orbit,
    build_parameter_bounds
)

# ==========================================================
# Benchmark datasets
# ==========================================================

ROOT = Path("validation/benchmark_systems")

benchmark_files = [

    ROOT / "FIN379" / "FIN379.inp",

    ROOT / "GL765" / "GL765.inp",

    ROOT / "ZetaAqr" / "ZetaAqr.in2"
]

# ==========================================================
# Run tests
# ==========================================================

for filepath in benchmark_files:

    print()

    print("=" * 60)

    print(filepath.name)

    print("=" * 60)

    parsed = parse_orbit_file(
        str(filepath)
    )

    elements = parsed["elements"]

    visual_data = parsed["visual_data"]

    rv_data = parsed["rv_data"]

    # ------------------------------------------------------
    # Polar -> Cartesian conversion
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
    # Parameter bounds
    # ------------------------------------------------------

    lower, upper = (
        build_parameter_bounds(
            elements
        )
    )

    print()

    print("Bounds lengths:")

    print(len(lower), len(upper))

    print()

    print("Initial parameter vector:")

    print(
        elements.fitted_parameters()
    )

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

    # ------------------------------------------------------
    # Optimizer status
    # ------------------------------------------------------

    result = fit_result.optimizer_result

    print()

    print("Optimization success:")

    print(result.success)

    print()

    print("Optimizer message:")

    print(result.message)

    # ------------------------------------------------------
    # Residual diagnostics
    # ------------------------------------------------------

    residuals = fit_result.residuals

    print()

    print("Residual vector:")

    print(
        f"Length: {len(residuals)}"
    )

    print(
        f"Finite: "
        f"{np.all(np.isfinite(residuals))}"
    )

    # ------------------------------------------------------
    # Covariance
    # ------------------------------------------------------

    covariance = fit_result.covariance

    print()

    print("Covariance shape:")

    print(covariance.shape)

    print()

    print("Covariance finite:")

    print(
        np.all(np.isfinite(covariance))
    )

    # ------------------------------------------------------
    # Uncertainties
    # ------------------------------------------------------

    print()

    print("Parameter uncertainties:")

    for key, value in (
        fit_result.uncertainties.items()
    ):

        print(
            f"{key}: "
            f"{value:.6f}"
        )

    # ------------------------------------------------------
    # Final fitted elements
    # ------------------------------------------------------

    print()

    print("Fitted elements:")

    print(
        fit_result.elements
    )