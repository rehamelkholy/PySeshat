from pathlib import Path

import numpy as np

from pyseshat.triple.triple_parser import (
    parse_triple_file
)

from pyseshat.triple.triple_fit import (
    fit_triple_orbit
)

from pyseshat.triple.triple_residuals import (
    triple_visual_residuals
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

initial_residuals = (
    triple_visual_residuals(
        observations,
        elements
    )
)

initial_chi2 = np.sum(
    initial_residuals**2
)

# ==========================================================
# Fit
# ==========================================================

fitted_elements, optimizer = (
    fit_triple_orbit(
        elements,
        observations
    )
)

# ==========================================================
# Final chi-square
# ==========================================================

final_residuals = (
    triple_visual_residuals(
        observations,
        fitted_elements
    )
)

final_chi2 = np.sum(
    final_residuals**2
)

# ==========================================================
# Diagnostics
# ==========================================================

print()

print(
    "Fitted parameters:"
)

print(
    fitted_elements.fitted_parameter_names()
)

print()

print(
    "Initial a2:"
)

print(
    elements.inner.a
)

print()

print(
    "Final a2:"
)

print(
    fitted_elements.inner.a
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
    optimizer.success
)

print()

print(
    optimizer.message
)