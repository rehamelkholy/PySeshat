from pathlib import Path

import numpy as np

from pyseshat.io.orbit_parser import (
    parse_orbit_file
)

from pyseshat.orbit.orbit_model import (
    compute_visual_orbit,
    sample_visual_orbit
)

# ------------------------------------------------------
# Benchmark datasets
# ------------------------------------------------------

ROOT = Path("validation/benchmark_systems")

benchmark_files = [

    ROOT / "FIN379" / "FIN379.inp",

    ROOT / "GL765" / "GL765.inp",

    ROOT / "ZetaAqr" / "ZetaAqr.in2"
]

# ------------------------------------------------------
# Run tests
# ------------------------------------------------------

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

    t = visual_data[:,0]

    orbit = compute_visual_orbit(
        t,
        elements
    )

    print()

    print("Returned keys:")

    print(list(orbit.keys()))

    print()

    print("Array lengths:")

    for key, value in orbit.items():

        print(
            f"{key}: {len(value)}"
        )

    print()

    print("Finite checks:")

    for key, value in orbit.items():

        print(
            f"{key}: "
            f"{np.all(np.isfinite(value))}"
        )

    # --------------------------------------------------
    # Smooth orbit
    # --------------------------------------------------

    sampled = sample_visual_orbit(
        elements,
        n_points=500
    )

    print()

    print("Sampled orbit finite:")

    for key, value in sampled.items():

        print(
            f"{key}: "
            f"{np.all(np.isfinite(value))}"
        )