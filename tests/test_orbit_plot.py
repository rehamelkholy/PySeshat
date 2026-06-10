from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from pyseshat.io.orbit_parser import (
    parse_orbit_file
)

from pyseshat.orbit.orbit_model import (
    sample_visual_orbit
)

from pyseshat.inference.optimizer import (
    fit_visual_orbit
)

from pyseshat.visualization.orbit_plot import (
    plot_visual_orbit,
    plot_residuals,
    plot_rv_orbit
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
    # Smooth orbit
    # ------------------------------------------------------

    orbit = sample_visual_orbit(
        fitted,
        n_points=1000
    )

    # ------------------------------------------------------
    # Orbit plot
    # ------------------------------------------------------

    fig1, ax1 = plot_visual_orbit(
        X,
        Y,
        orbit,
        title=filepath.stem,
        show=True
    )

    print()

    print("Orbit plot generated.")

    print(
        f"Axes aspect: "
        f"{ax1.get_aspect()}"
    )

    # ------------------------------------------------------
    # Residual plots
    # ------------------------------------------------------

    visual_residuals = residuals[:2*len(t)]

    fig2, axes = plot_residuals(
        t,
        visual_residuals,
        title=f"{filepath.stem} Residuals",
        show=True
    )

    print()

    print("Residual plot generated.")

    print(
        f"Residual panels: "
        f"{len(axes)}"
    )

    # ------------------------------------------------------
    # RV orbit plot
    # ------------------------------------------------------
    fig3, ax3 = plot_rv_orbit(

        parsed["rv_data"],

        parsed["elements"],

        title=f"{filepath.stem} RV Orbit",

        phase=True,

        show=True
    )

    print()

    print(
        "RV plot generated."
    )

    # ------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------

    plt.close(fig1)

    plt.close(fig2)

    plt.close(fig3)