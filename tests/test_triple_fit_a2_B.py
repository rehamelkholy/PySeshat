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

observations = result["observations"]

# ==========================================================
# Enable additional parameter
# ==========================================================

elements.fit_flags["B"] = True

# ==========================================================
# Initial chi2
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

fit = fit_triple_orbit(
    elements,
    observations
)

# ==========================================================
# Final chi2
# ==========================================================

final_chi2 = np.sum(
    fit.residuals**2
)

# ==========================================================
# Report
# ==========================================================

print()

print("FITTED PARAMETERS")
print("=" * 60)

print(
    fit.elements.fitted_parameter_names()
)

print()

print("VALUES")
print("=" * 60)

print(
    f"a2 = {fit.elements.inner.a}"
)

print(
    f"B  = {fit.elements.B}"
)

print()

print("UNCERTAINTIES")
print("=" * 60)

for key, value in fit.uncertainties.items():

    print(
        f"{key:<10} {value}"
    )

print()

print("COVARIANCE SHAPE")
print("=" * 60)

print(
    fit.covariance.shape
)

print()

print("CHI2")
print("=" * 60)

print(
    f"Initial = {initial_chi2}"
)

print(
    f"Final   = {final_chi2}"
)

print(
    f"Delta   = {initial_chi2 - final_chi2}"
)

print()

print("OPTIMIZER")
print("=" * 60)

print(
    fit.optimizer_result.success
)

print(
    fit.optimizer_result.message
)