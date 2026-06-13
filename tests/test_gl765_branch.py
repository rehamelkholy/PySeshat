from pathlib import Path

import numpy as np

from pyseshat.io.orbit_parser import (
    parse_orbit_file
)

from pyseshat.inference.optimizer import (
    fit_visual_orbit
)

# ==========================================================
# Load GL765
# ==========================================================

filepath = Path(
    "validation/benchmark_systems/GL765/GL765.inp"
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

fitted = fit_result.elements

# ==========================================================
# Compare orientations
# ==========================================================

print()
print("Initial Elements")
print("-" * 40)

print(
    f"Omega = {elements.Omega:.6f}"
)

print(
    f"omega = {elements.omega:.6f}"
)

print()

print("Fitted Elements")
print("-" * 40)

print(
    f"Omega = {fitted.Omega:.6f}"
)

print(
    f"omega = {fitted.omega:.6f}"
)

print()

# ==========================================================
# Raw differences
# ==========================================================

dOmega = (
    fitted.Omega
    -
    elements.Omega
)

domega = (
    fitted.omega
    -
    elements.omega
)

print("Raw Differences")
print("-" * 40)

print(
    f"Delta Omega = {dOmega:.6f}"
)

print(
    f"Delta omega = {domega:.6f}"
)

print()

# ==========================================================
# Wrapped differences
# ==========================================================

wrapped_dOmega = (
    (dOmega + 180.0)
    % 360.0
    - 180.0
)

wrapped_domega = (
    (domega + 180.0)
    % 360.0
    - 180.0
)

print("Wrapped Differences")
print("-" * 40)

print(
    f"Wrapped Delta Omega = {wrapped_dOmega:.6f}"
)

print(
    f"Wrapped Delta omega = {wrapped_domega:.6f}"
)

print()

# ==========================================================
# Full parameter comparison
# ==========================================================

print("Other Orbital Parameters")
print("-" * 40)

print(
    f"P  : {elements.P:.6f} -> {fitted.P:.6f}"
)

print(
    f"T0 : {elements.T0:.6f} -> {fitted.T0:.6f}"
)

print(
    f"e  : {elements.e:.6f} -> {fitted.e:.6f}"
)

print(
    f"a  : {elements.a:.6f} -> {fitted.a:.6f}"
)

print(
    f"i  : {elements.i:.6f} -> {fitted.i:.6f}"
)

print(
    f"K1 : {elements.K1:.6f} -> {fitted.K1:.6f}"
)

print(
    f"K2 : {elements.K2:.6f} -> {fitted.K2:.6f}"
)

print(
    f"V0 : {elements.V0:.6f} -> {fitted.V0:.6f}"
)

print()

# ==========================================================
# Optimizer summary
# ==========================================================

print("Optimizer")
print("-" * 40)

print(
    "Success:",
    fit_result.optimizer_result.success
)

print()

print(
    "Message:"
)

print(
    fit_result.optimizer_result.message
)