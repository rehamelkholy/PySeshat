import numpy as np

from scipy.optimize import least_squares

from pyseshat.io.orbit_parser import (
    parse_orbit_file
)

from pyseshat.inference.likelihood import (
    rv_residuals,
    update_elements
)

# ==========================================================
# Load GL765
# ==========================================================

parsed = parse_orbit_file(
    "validation/benchmark_systems/GL765/GL765.inp"
)

elements = parsed["elements"]

rv_data = parsed["rv_data"]

# ==========================================================
# Fit flags
# ==========================================================
#
# RV-only fit:
#
# Free:
#   P
#   T0
#   e
#   omega
#   K1
#   K2
#   V0
#
# Fixed:
#   a
#   Omega
#   i
#
# ==========================================================

elements.fit_flags = {
    "P": True,
    "T0": True,
    "e": True,
    "a": False,
    "Omega": False,
    "omega": True,
    "i": False,
    "K1": True,
    "K2": True,
    "V0": True
}

# ==========================================================
# Initial parameter vector
# ==========================================================

p0 = elements.fitted_parameters()

# ==========================================================
# Initial chi2
# ==========================================================

r0 = rv_residuals(
    p0,
    elements,
    rv_data
)

chi2_initial = np.sum(
    r0**2
)

# ==========================================================
# Bounds
# ==========================================================

lower = []
upper = []

for name in elements.fitted_parameter_names():

    if name == "P":
        lower.append(1e-6)
        upper.append(np.inf)

    elif name == "T0":
        lower.append(-np.inf)
        upper.append(np.inf)

    elif name == "e":
        lower.append(0.0)
        upper.append(0.99)

    elif name == "omega":
        lower.append(0.0)
        upper.append(360.0)

    elif name == "K1":
        lower.append(0.0)
        upper.append(np.inf)

    elif name == "K2":
        lower.append(0.0)
        upper.append(np.inf)

    elif name == "V0":
        lower.append(-np.inf)
        upper.append(np.inf)

bounds = (
    np.array(lower),
    np.array(upper)
)

# ==========================================================
# Optimize
# ==========================================================

result = least_squares(
    rv_residuals,
    p0,
    args=(
        elements,
        rv_data
    ),
    method="trf",
    bounds=bounds
)

# ==========================================================
# Final orbit
# ==========================================================

fitted = update_elements(
    elements,
    result.x
)

# ==========================================================
# Final chi2
# ==========================================================

r_final = rv_residuals(
    result.x,
    elements,
    rv_data
)

chi2_final = np.sum(
    r_final**2
)

# ==========================================================
# Report
# ==========================================================

print()

print("Initial RV-only Fit")
print("-" * 40)

print(
    f"chi2 = {chi2_initial:.6f}"
)

print()

print("Final RV-only Fit")
print("-" * 40)

print(
    f"chi2 = {chi2_final:.6f}"
)

print()

print("Improvement")
print("-" * 40)

print(
    f"Delta chi2 = {chi2_initial - chi2_final:.6f}"
)

print()

print("Orbital Elements")
print("-" * 40)

print(
    f"P      : {elements.P:.6f} -> {fitted.P:.6f}"
)

print(
    f"T0     : {elements.T0:.6f} -> {fitted.T0:.6f}"
)

print(
    f"e      : {elements.e:.6f} -> {fitted.e:.6f}"
)

print(
    f"omega  : {elements.omega:.6f} -> {fitted.omega:.6f}"
)

print(
    f"K1     : {elements.K1:.6f} -> {fitted.K1:.6f}"
)

print(
    f"K2     : {elements.K2:.6f} -> {fitted.K2:.6f}"
)

print(
    f"V0     : {elements.V0:.6f} -> {fitted.V0:.6f}"
)

print()

print("Fixed Parameters")
print("-" * 40)

print(
    f"a      = {fitted.a:.6f}"
)

print(
    f"Omega  = {fitted.Omega:.6f}"
)

print(
    f"i      = {fitted.i:.6f}"
)

print()

print("Optimizer")
print("-" * 40)

print(
    f"Success: {result.success}"
)

print()

print(result.message)