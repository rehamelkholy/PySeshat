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

observations = result[
    "observations"
]

# ==========================================================
# Multi-parameter fit
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
# PCA
# ==========================================================

covariance = fit.covariance

eigenvalues, eigenvectors = (
    np.linalg.eigh(
        covariance
    )
)

names = (
    fit.elements
    .fitted_parameter_names()
)

# ==========================================================
# Print all modes
# ==========================================================

print()

print(
    "PARAMETER MODES"
)

print(
    "=" * 80
)

for mode_index in range(
    len(eigenvalues)
):

    print()

    print(
        f"Mode {mode_index + 1}"
    )

    print(
        f"Variance = "
        f"{eigenvalues[mode_index]}"
    )

    print()

    for name, coeff in zip(
        names,
        eigenvectors[:, mode_index]
    ):

        print(
            f"{name:<12}"
            f"{coeff: .6f}"
        )

# ==========================================================
# Best constrained mode
# ==========================================================

best_index = np.argmin(
    eigenvalues
)

print()
print(
    "=" * 80
)

print(
    "BEST CONSTRAINED MODE"
)

print(
    "=" * 80
)

print(
    f"Variance = "
    f"{eigenvalues[best_index]}"
)

print()

for name, coeff in zip(
    names,
    eigenvectors[:, best_index]
):

    print(
        f"{name:<12}"
        f"{coeff: .6f}"
    )

# ==========================================================
# Worst constrained mode
# ==========================================================

worst_index = np.argmax(
    eigenvalues
)

print()
print(
    "=" * 80
)

print(
    "WORST CONSTRAINED MODE"
)

print(
    "=" * 80
)

print(
    f"Variance = "
    f"{eigenvalues[worst_index]}"
)

print()

for name, coeff in zip(
    names,
    eigenvectors[:, worst_index]
):

    print(
        f"{name:<12}"
        f"{coeff: .6f}"
    )

# ==========================================================
# Variance ratio
# ==========================================================

variance_ratio = (
    eigenvalues[worst_index]
    /
    eigenvalues[best_index]
)

print()
print(
    "=" * 80
)

print(
    "VARIANCE RATIO"
)

print(
    "=" * 80
)

print(
    variance_ratio
)