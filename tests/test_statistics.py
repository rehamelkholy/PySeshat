from pathlib import Path

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
    chi_square,
    reduced_chi_square,
    rms_residual,
    fit_summary,
    combined_fit_summary,
    parameter_comparison
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
    # Cartesian conversion
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
    # Fit
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

    residuals = fit_result.residuals

    # ------------------------------------------------------
    # Basic statistics
    # ------------------------------------------------------

    print()

    print("Chi-square:")

    print(
        chi_square(residuals)
    )

    print()

    print("Reduced chi-square:")

    print(
        reduced_chi_square(
            residuals,
            len(
                fitted.fitted_parameters()
            )
        )
    )

    print()

    print("RMS residual:")

    print(
        rms_residual(residuals)
    )

    # ------------------------------------------------------
    # Fit summary
    # ------------------------------------------------------

    summary = fit_summary(
        residuals,
        fitted
    )

    print()

    print("Fit summary:")

    print(summary)

    # ------------------------------------------------------
    # Combined diagnostics
    # ------------------------------------------------------

    visual_res = visual_xy_residuals(
        fitted.fitted_parameters(),
        fitted,
        t,
        X,
        Y,
        sx,
        sy
    )

    if len(rv_data) > 0:

        rv_res = rv_residuals(
            fitted.fitted_parameters(),
            fitted,
            rv_data
        )

        combined = combined_fit_summary(
            visual_residuals=visual_res,
            rv_residuals=rv_res,
            rv_components=rv_data[:,3],
            n_params=len(
                fitted.fitted_parameters()
            )
        )

    else:

        combined = combined_fit_summary(
            visual_residuals=visual_res,
            rv_residuals=None,
            rv_components=None,
            n_params=len(
                fitted.fitted_parameters()
            )
        )

    print()

    print("Visual residual count:")

    print(len(visual_res))

    if len(rv_data) > 0:

        print()

        print("RV residual count:")

        print(len(rv_res))
    
    print("Combined summary:")

    print(combined)

    # ------------------------------------------------------
    # Parameter comparison
    # ------------------------------------------------------

    comparison = parameter_comparison(
        elements,
        fitted
    )

    print()

    print("Parameter comparison keys:")

    print(
        list(comparison.keys())
    )