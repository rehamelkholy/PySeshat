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
# Observations
# ==========================================================

t = visual[:,0]

theta = np.deg2rad(
    visual[:,1]
)

rho = visual[:,2]

sigma = visual[:,3]

X_obs = rho * np.cos(theta)

Y_obs = rho * np.sin(theta)

# ==========================================================
# Fit
# ==========================================================

fit_result = fit_visual_orbit(
    elements,
    t,
    X_obs,
    Y_obs,
    sigma,
    sigma,
    rv_data=rv_data
)

fitted = fit_result.elements

# ==========================================================
# Final chi2
# ==========================================================

r = combined_residuals(
    fitted.fitted_parameters(),
    fitted,
    t,
    X_obs,
    Y_obs,
    sigma,
    sigma,
    rv_data
)

chi2 = np.sum(
    r**2
)

# ==========================================================
# Report
# ==========================================================

print()

print(
    "Final combined chi2:"
)

print(
    chi2
)

print()

print(
    "Omega:"
)

print(
    fitted.Omega
)

print()

print(
    "omega:"
)

print(
    fitted.omega
)

print()

print(
    "Success:"
)

print(
    fit_result.optimizer_result.success
)