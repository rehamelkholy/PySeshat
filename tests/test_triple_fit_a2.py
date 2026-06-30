from pathlib import Path

import numpy as np

from scipy.optimize import least_squares

from pyseshat.triple.triple_parser import (
    parse_triple_file
)

from pyseshat.triple.triple_fit import (
    triple_a2_residuals
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
# Initial diagnostics
# ==========================================================

initial_a2 = elements.inner.a

initial_residuals = triple_a2_residuals(
    [initial_a2],
    elements,
    observations
)

initial_chi2 = np.sum(
    initial_residuals**2
)

# ==========================================================
# Fit
# ==========================================================

fit = least_squares(
    triple_a2_residuals,
    x0=[initial_a2],
    args=(
        elements,
        observations
    ),
    bounds=(
        [0.001],
        [10.0]
    ),
    method="trf"
)

# ==========================================================
# Final diagnostics
# ==========================================================

final_a2 = fit.x[0]

final_residuals = triple_a2_residuals(
    fit.x,
    elements,
    observations
)

final_chi2 = np.sum(
    final_residuals**2
)

# ==========================================================
# Output
# ==========================================================

print()

print(
    "Initial a2:"
)

print(
    initial_a2
)

print()

print(
    "Fitted a2:"
)

print(
    final_a2
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