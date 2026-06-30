from pathlib import Path

import numpy as np

from pyseshat.triple.triple_parser import (
    parse_triple_file
)

from pyseshat.triple.triple_residuals import (
    triple_visual_residuals
)

# ==========================================================
# Load benchmark system
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
# Compute residual vector
# ==========================================================

residuals = triple_visual_residuals(
    observations,
    elements
)

# ==========================================================
# Diagnostics
# ==========================================================

print()

print(
    "Number of observations:"
)

print(
    len(observations)
)

print()

print(
    "Residual vector shape:"
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