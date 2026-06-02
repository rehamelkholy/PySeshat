from pathlib import Path

import numpy as np

from pyseshat.io.orbit_parser import (
    parse_orbit_file
)

from pyseshat.inference.likelihood import (
    visual_xy_residuals,
    rv_residuals,
    combined_residuals,
    chi_square,
    reduced_chi_square
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
    # Fitted parameters
    # ------------------------------------------------------

    fitted_values = (
        elements.fitted_parameters()
    )

    # ------------------------------------------------------
    # Visual residuals
    # ------------------------------------------------------

    res_visual = visual_xy_residuals(
        fitted_values,
        elements,
        t,
        X,
        Y,
        sx,
        sy
    )

    print()

    print("Visual residuals:")

    print(
        f"Length: {len(res_visual)}"
    )

    print(
        f"Finite: "
        f"{np.all(np.isfinite(res_visual))}"
    )

    # ------------------------------------------------------
    # RV residuals
    # ------------------------------------------------------

    res_rv = rv_residuals(
        fitted_values,
        elements,
        rv_data
    )

    print()

    print("RV residuals:")

    print(
        f"Length: {len(res_rv)}"
    )

    print(
        f"Finite: "
        f"{np.all(np.isfinite(res_rv))}"
    )

    # ------------------------------------------------------
    # Combined residuals
    # ------------------------------------------------------

    res_combined = combined_residuals(
        fitted_values,
        elements,
        t,
        X,
        Y,
        sx,
        sy,
        rv_data=rv_data
    )

    print()

    print("Combined residuals:")

    print(
        f"Length: {len(res_combined)}"
    )

    print(
        f"Finite: "
        f"{np.all(np.isfinite(res_combined))}"
    )

    # ------------------------------------------------------
    # Chi-square
    # ------------------------------------------------------

    chi2 = chi_square(
        fitted_values,
        elements,
        t,
        X,
        Y,
        sx,
        sy
    )

    redchi2 = reduced_chi_square(
        fitted_values,
        elements,
        t,
        X,
        Y,
        sx,
        sy
    )

    print()

    print(
        f"Chi-square: "
        f"{chi2:.6f}"
    )

    print(
        f"Reduced chi-square: "
        f"{redchi2:.6f}"
    )