from pathlib import Path

import numpy as np

from scipy.optimize import least_squares

from pyseshat.triple.triple_parser import (
    parse_triple_file
)

from pyseshat.triple.triple_fit import (
    triple_B_residuals
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
# Initial chi-square
# ==========================================================

initial_residuals = triple_B_residuals(
    [elements.B],
    elements,
    observations
)

initial_chi2 = np.sum(
    initial_residuals**2
)

# ==========================================================
# Fit B
# ==========================================================

fit = least_squares(
    triple_B_residuals,
    x0=[elements.B],
    args=(
        elements,
        observations
    ),
    bounds=(
        [0.0],
        [100.0]
    )
)

# ==========================================================
# Final chi-square
# ==========================================================

final_residuals = triple_B_residuals(
    fit.x,
    elements,
    observations
)

final_chi2 = np.sum(
    final_residuals**2
)

# ==========================================================
# Diagnostics
# ==========================================================

print()

print(
    "Initial B:"
)

print(
    elements.B
)

print()

print(
    "Fitted B:"
)

print(
    fit.x[0]
)

print()

print(
    "Initial chi2:"
)

print(
    initial_chi2
)

print()

print(
    "Final chi2:"
)

print(
    final_chi2
)

print()

print(
    "Delta chi2:"
)

print(
    initial_chi2
    -
    final_chi2
)

print()

print(
    "Optimizer success:"
)

print(
    fit.success
)

print()

print(
    fit.message
)