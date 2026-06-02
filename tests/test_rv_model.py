from pathlib import Path

import numpy as np

from pyseshat.io.orbit_parser import (
    parse_orbit_file
)

from pyseshat.orbit.rv_model import (
    compute_rv_orbit,
    compute_rv_model
)

# ------------------------------------------------------
# Benchmark datasets
# ------------------------------------------------------

ROOT = Path("validation/benchmark_systems")

benchmark_files = [

    ROOT / "FIN379" / "FIN379.inp",

    ROOT / "GL765" / "GL765.inp"
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

    rv_data = parsed["rv_data"]

    # --------------------------------------------------
    # Skip systems without RV data
    # --------------------------------------------------

    if len(rv_data) == 0:

        print("No RV data.")

        continue

    t = rv_data[:,0]

    # --------------------------------------------------
    # Primary RV
    # --------------------------------------------------

    rv1 = compute_rv_orbit(
        t,
        elements,
        component=0
    )

    print()

    print("Primary RV finite:")

    print(
        np.all(np.isfinite(rv1))
    )

    print(
        f"RV1 range: "
        f"{rv1.min():.3f} "
        f"to "
        f"{rv1.max():.3f}"
    )

    # --------------------------------------------------
    # Secondary RV
    # --------------------------------------------------

    rv2 = compute_rv_orbit(
        t,
        elements,
        component=1
    )

    print()

    print("Secondary RV finite:")

    print(
        np.all(np.isfinite(rv2))
    )

    print(
        f"RV2 range: "
        f"{rv2.min():.3f} "
        f"to "
        f"{rv2.max():.3f}"
    )

    # --------------------------------------------------
    # Mixed RV model
    # --------------------------------------------------

    rv_model = compute_rv_model(
        rv_data,
        elements
    )

    print()

    print("Mixed RV finite:")

    print(
        np.all(np.isfinite(rv_model))
    )

    print()

    print("Mixed RV shape:")

    print(rv_model.shape)

    # --------------------------------------------------
    # Component counts
    # --------------------------------------------------

    component = rv_data[:,3].astype(int)

    print()

    print(
        f"Primary observations: "
        f"{np.sum(component == 0)}"
    )

    print(
        f"Secondary observations: "
        f"{np.sum(component == 1)}"
    )