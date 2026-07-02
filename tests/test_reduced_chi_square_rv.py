import numpy as np

from pyseshat.io.orbit_parser import (
    parse_orbit_file
)

from pyseshat.inference.optimizer import (
    fit_visual_orbit
)

from pyseshat.inference.likelihood import (
    reduced_chi_square
)

# ==========================================================
# Load FIN379
# ==========================================================

parsed = parse_orbit_file(
    "validation/benchmark_systems/FIN379/FIN379.inp"
)

elements = parsed["elements"]

visual = parsed["visual_data"]

rv_data = parsed["rv_data"]

# ==========================================================
# Observations
# ==========================================================

t = visual[:, 0]

theta = np.deg2rad(
    visual[:, 1]
)

rho = visual[:, 2]

sigma = visual[:, 3]

X_obs = rho * np.cos(theta)

Y_obs = rho * np.sin(theta)

# ==========================================================
# Fit orbit
# ==========================================================

fit = fit_visual_orbit(
    elements,
    t,
    X_obs,
    Y_obs,
    sigma,
    sigma,
    rv_data=rv_data
)

params = fit.elements.fitted_parameters()

# ==========================================================
# Compute reduced chi-square
# ==========================================================

rchisq = reduced_chi_square(
    params,
    fit.elements,
    t,
    X_obs,
    Y_obs,
    sigma,
    sigma,
    rv_data=rv_data
)

# ==========================================================
# Expected degrees of freedom
# ==========================================================

n_visual = 2 * len(t)

n_rv = len(rv_data)

n_fit = np.sum(
    fit.elements.fitted_mask()
)

expected_dof = (
    n_visual
    +
    n_rv
    -
    n_fit
)

print()

print("VISUAL POINTS")
print("=" * 60)
print(n_visual)

print()

print("RV POINTS")
print("=" * 60)
print(n_rv)

print()

print("TOTAL POINTS")
print("=" * 60)
print(n_visual + n_rv)

print()

print("FITTED PARAMETERS")
print("=" * 60)
print(n_fit)

print()

print("DEGREES OF FREEDOM")
print("=" * 60)
print(expected_dof)

print()

print("REDUCED CHI-SQUARE")
print("=" * 60)
print(rchisq)