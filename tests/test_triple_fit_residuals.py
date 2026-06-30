from pathlib import Path

import numpy as np

from pyseshat.triple.triple_parser import (
    parse_triple_file
)

from pyseshat.triple.triple_fit import (
    triple_to_array,
    triple_fit_residuals
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
# Parameter vector
# ==========================================================

p = triple_to_array(
    elements
)

# ==========================================================
# Residuals
# ==========================================================

residuals = triple_fit_residuals(
    p,
    elements,
    observations
)

# ==========================================================
# Diagnostics
# ==========================================================

print()

print(
    "Parameter vector shape:"
)

print(
    p.shape
)

print()

print(
    "Residual shape:"
)

print(
    residuals.shape
)

print()

print(
    "Chi-square:"
)

print(
    np.sum(
        residuals**2
    )
)

print()

print(
    "RMS:"
)

print(
    np.sqrt(
        np.mean(
            residuals**2
        )
    )
)