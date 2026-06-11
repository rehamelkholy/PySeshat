from pathlib import Path

import numpy as np

from pyseshat.io.orbit_parser import (
    parse_orbit_file
)

from pyseshat.inference.optimizer import (
    fit_visual_orbit
)

# ==========================================================
# Load benchmark system
# ==========================================================

filepath = Path(
    "validation/benchmark_systems/FIN379/FIN379.inp"
)

parsed = parse_orbit_file(
    str(filepath)
)

elements = parsed["elements"]

visual_data = parsed["visual_data"]

rv_data = parsed["rv_data"]

# ==========================================================
# Convert visual observations
# ==========================================================

t = visual_data[:,0]

theta = np.deg2rad(
    visual_data[:,1]
)

rho = visual_data[:,2]

sigma = visual_data[:,3]

X = rho * np.cos(theta)

Y = rho * np.sin(theta)

sx = sigma

sy = sigma

# ==========================================================
# Fit orbit
# ==========================================================

fit_result = fit_visual_orbit(
    elements,
    t,
    X,
    Y,
    sx,
    sy,
    rv_data=rv_data
)

# ==========================================================
# Inspect result structure
# ==========================================================

print()

print("Elements:")
print(type(fit_result.elements))

print()

print("Uncertainties:")
print(type(fit_result.uncertainties))

print()

print("Covariance:")
print(type(fit_result.covariance))

print()

print("Residuals:")
print(type(fit_result.residuals))

print()

print("Optimizer Result:")
print(type(fit_result.optimizer_result))

print()

print("Covariance shape:")
print(fit_result.covariance.shape)

print()

print("Residual vector shape:")
print(fit_result.residuals.shape)