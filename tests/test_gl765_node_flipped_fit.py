import copy
import numpy as np

from pyseshat.io.orbit_parser import (
    parse_orbit_file
)

from pyseshat.inference.optimizer import (
    fit_visual_orbit
)

from pyseshat.inference.likelihood import (
    combined_residuals
)

# ==========================================================
# Load GL765
# ==========================================================

parsed = parse_orbit_file(
    "validation/benchmark_systems/GL765/GL765.inp"
)

elements = parsed["elements"]

visual = parsed["visual_data"]

rv_data = parsed["rv_data"]

# ==========================================================
# Visual observations
# ==========================================================

t = visual[:, 0]

theta_deg = visual[:, 1]

rho = visual[:, 2]

sigma = visual[:, 3]

theta = np.deg2rad(theta_deg)

# Current benchmark convention
X_obs = rho * np.cos(theta)

Y_obs = rho * np.sin(theta)

# ==========================================================
# Create node-flipped orbit
# ==========================================================

flipped = copy.deepcopy(
    elements
)

flipped.Omega = (
    flipped.Omega + 180.0
) % 360.0

flipped.omega = (
    flipped.omega + 180.0
) % 360.0

# ==========================================================
# Initial combined chi2
# ==========================================================

r0 = combined_residuals(
    flipped.fitted_parameters(),
    flipped,
    t,
    X_obs,
    Y_obs,
    sigma,
    sigma,
    rv_data
)

chi2_initial = np.sum(
    r0**2
)

# ==========================================================
# Fit
# ==========================================================

fit_result = fit_visual_orbit(
    flipped,
    t,
    X_obs,
    Y_obs,
    sigma,
    sigma,
    rv_data=rv_data
)

fitted = fit_result.elements

# ==========================================================
# Final combined chi2
# ==========================================================

r1 = combined_residuals(
    fitted.fitted_parameters(),
    fitted,
    t,
    X_obs,
    Y_obs,
    sigma,
    sigma,
    rv_data
)

chi2_final = np.sum(
    r1**2
)

# ==========================================================
# Report
# ==========================================================

print()
print("NODE-FLIPPED INITIAL ORBIT")
print("=" * 60)

print(
    f"P      = {flipped.P:.6f}"
)

print(
    f"T0     = {flipped.T0:.6f}"
)

print(
    f"e      = {flipped.e:.6f}"
)

print(
    f"a      = {flipped.a:.6f}"
)

print(
    f"Omega  = {flipped.Omega:.6f}"
)

print(
    f"omega  = {flipped.omega:.6f}"
)

print(
    f"i      = {flipped.i:.6f}"
)

print(
    f"K1     = {flipped.K1:.6f}"
)

print(
    f"K2     = {flipped.K2:.6f}"
)

print(
    f"V0     = {flipped.V0:.6f}"
)

print()

print(
    f"Initial combined chi2 = {chi2_initial:.6f}"
)

print()

print("FITTED ORBIT")
print("=" * 60)

print(
    f"P      = {fitted.P:.6f}"
)

print(
    f"T0     = {fitted.T0:.6f}"
)

print(
    f"e      = {fitted.e:.6f}"
)

print(
    f"a      = {fitted.a:.6f}"
)

print(
    f"Omega  = {fitted.Omega:.6f}"
)

print(
    f"omega  = {fitted.omega:.6f}"
)

print(
    f"i      = {fitted.i:.6f}"
)

print(
    f"K1     = {fitted.K1:.6f}"
)

print(
    f"K2     = {fitted.K2:.6f}"
)

print(
    f"V0     = {fitted.V0:.6f}"
)

print()

print(
    f"Final combined chi2 = {chi2_final:.6f}"
)

print()

print(
    f"Delta chi2 = {chi2_initial - chi2_final:.6f}"
)

print()

print("OPTIMIZER")
print("=" * 60)

print(
    f"Success: {fit_result.optimizer_result.success}"
)

print()

print(
    fit_result.optimizer_result.message
)