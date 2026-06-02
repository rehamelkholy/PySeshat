from pathlib import Path

import numpy as np

from pyseshat.io.orbit_parser import (
    parse_orbit_file
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

    metadata = parsed["metadata"]

    # ------------------------------------------------------
    # Metadata
    # ------------------------------------------------------

    print()

    print("Metadata:")

    print(metadata)

    # ------------------------------------------------------
    # Orbital elements
    # ------------------------------------------------------

    print()

    print("Orbital elements:")

    print(elements)

    # ------------------------------------------------------
    # Visual data
    # ------------------------------------------------------

    print()

    print("Visual data shape:")

    print(visual_data.shape)

    print()

    print(
        "Visual finite:"
    )

    print(
        np.all(np.isfinite(visual_data))
    )

    # ------------------------------------------------------
    # RV data
    # ------------------------------------------------------

    print()

    print("RV data shape:")

    print(rv_data.shape)

    if len(rv_data) > 0:

        print()

        print("RV finite:")

        print(
            np.all(np.isfinite(rv_data))
        )

        print()

        print("RV components:")

        print(
            np.unique(rv_data[:,3])
        )

    # ------------------------------------------------------
    # Time normalization
    # ------------------------------------------------------

    print()

    print("Visual time range:")

    print(
        visual_data[:,0].min(),
        visual_data[:,0].max()
    )

    if len(rv_data) > 0:

        print()

        print("RV time range:")

        print(
            rv_data[:,0].min(),
            rv_data[:,0].max()
        )