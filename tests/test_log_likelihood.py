import numpy as np

from pyseshat.io.orbit_parser import (
    parse_orbit_file
)

from pyseshat.inference.optimizer import (
    fit_visual_orbit
)

from pyseshat.inference.likelihood import (
    log_likelihood,
    chi_square
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
# Convert observations
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

fitted = fit.elements

params = fitted.fitted_parameters()

# ==========================================================
# Compare χ² and log L
# ==========================================================

chi2 = chi_square(
    params,
    fitted,
    t,
    X_obs,
    Y_obs,
    sigma,
    sigma,
    rv_data=rv_data
)

logL = log_likelihood(
    params,
    fitted,
    t,
    X_obs,
    Y_obs,
    sigma,
    sigma,
    rv_data=rv_data
)

print()

print("CHI-SQUARE")
print("=" * 60)

print(chi2)

print()

print("LOG-LIKELIHOOD")
print("=" * 60)

print(logL)

print()

print("EXPECTED")
print("=" * 60)

print(-0.5 * chi2)

print()

print("DIFFERENCE")
print("=" * 60)

print(logL + 0.5 * chi2)