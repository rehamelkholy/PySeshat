from pathlib import Path

import numpy as np

from pyseshat.triple.triple_parser import (
    parse_triple_file
)

from pyseshat.triple.triple_fit import (
    fit_triple_orbit
)

# ==========================================================
# Load benchmark
# ==========================================================

filename = (
    Path(__file__).resolve().parents[1]
    / "validation"
    / "benchmark_systems"
    / "ZetaAqr"
    / "ZetaAqr.in2"
)

result = parse_triple_file(
    filename
)

elements = result["elements"]

observations = result["observations"]

# ==========================================================
# Fit flags
# ==========================================================

elements.fit_flags["e2"] = True
elements.fit_flags["a2"] = True
elements.fit_flags["Omega2"] = True
elements.fit_flags["omega2"] = True

fit = fit_triple_orbit(
    elements,
    observations
)

# ==========================================================
# Eigen analysis
# ==========================================================

cov = fit.covariance

eigenvalues, eigenvectors = (
    np.linalg.eigh(cov)
)

names = (
    fit.elements
    .fitted_parameter_names()
)

print()

print("EIGENVALUES")
print("=" * 80)

for value in eigenvalues:

    print(value)

print()

print("PRINCIPAL DIRECTIONS")
print("=" * 80)

for i in range(
    len(eigenvalues)
):

    print()

    print(
        f"Mode {i+1}"
    )

    print(
        f"Variance = {eigenvalues[i]}"
    )

    print()

    for name, coeff in zip(
        names,
        eigenvectors[:, i]
    ):

        print(
            f"{name:<12}"
            f"{coeff: .6f}"
        )