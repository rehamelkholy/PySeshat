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
# Two-parameter fit
# ==========================================================

elements.fit_flags["B"] = True

fit = fit_triple_orbit(
    elements,
    observations
)

# ==========================================================
# Covariance diagnostics
# ==========================================================

cov = fit.covariance

eigenvalues = np.linalg.eigvals(
    cov
)

condition_number = np.linalg.cond(
    cov
)

print()

print("COVARIANCE")
print("=" * 60)

print(cov)

print()

print("EIGENVALUES")
print("=" * 60)

print(eigenvalues)

print()

print("CONDITION NUMBER")
print("=" * 60)

print(condition_number)