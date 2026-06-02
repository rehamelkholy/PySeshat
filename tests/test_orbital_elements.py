from pathlib import Path

from pyseshat.io.orbit_parser import (
    parse_orbit_file
)

from pyseshat.orbit.orbital_elements import (
    OrbitalElements
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

    # --------------------------------------------------
    # Array conversion
    # --------------------------------------------------

    arr = elements.to_array()

    print()

    print("Parameter array:")

    print(arr)

    # --------------------------------------------------
    # Fitted parameter names
    # --------------------------------------------------

    print()

    print("Fitted parameter names:")

    print(
        elements.fitted_parameter_names()
    )

    # --------------------------------------------------
    # Round-trip reconstruction
    # --------------------------------------------------

    reconstructed = (
        OrbitalElements.from_array(
            arr,
            fit_flags=elements.fit_flags
        )
    )

    print()

    print("Reconstructed object:")

    print(reconstructed)

    # --------------------------------------------------
    # Equality check
    # --------------------------------------------------

    print()

    print("Round-trip consistency:")

    print(
        reconstructed.to_array()
    )