from pyseshat.io.orbit_parser import (
    parse_orbit_file
)

from pyseshat.inference.likelihood import (
    visual_xy_residuals,
    rv_residuals,
    combined_residuals
)

import numpy as np

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
# Convert observations
# ==========================================================

t = visual[:,0]

theta = np.deg2rad(
    visual[:,1]
)

rho = visual[:,2]

sigma = visual[:,3]

X = rho * np.cos(theta)

Y = rho * np.sin(theta)

# ==========================================================
# Apply spectroscopic convention first
# ==========================================================

elements_spec = (
    elements.to_spectroscopic_convention()
)

# ==========================================================
# Visual chi-square
# ==========================================================

res_visual = visual_xy_residuals(
    elements_spec.fitted_parameters(),
    elements_spec,
    t,
    X,
    Y,
    sigma,
    sigma
)

chi2_visual = np.sum(
    res_visual**2
)

# ==========================================================
# RV chi-square
# ==========================================================

res_rv = rv_residuals(
    elements_spec.fitted_parameters(),
    elements_spec,
    rv_data
)

chi2_rv = np.sum(
    res_rv**2
)

# ==========================================================
# Combined chi-square
# ==========================================================

res_combined = combined_residuals(
    elements_spec.fitted_parameters(),
    elements_spec,
    t,
    X,
    Y,
    sigma,
    sigma,
    rv_data
)

chi2_combined = np.sum(
    res_combined**2
)

# ==========================================================
# Output
# ==========================================================

print()

print(
    "Visual chi2:"
)

print(
    chi2_visual
)

print()

print(
    "RV chi2:"
)

print(
    chi2_rv
)

print()

print(
    "Combined chi2:"
)

print(
    chi2_combined
)

print()

print(
    "Reduced ratios:"
)

print(
    f"RV / Visual = {chi2_rv / chi2_visual}"
)

print(
    f"Combined / Visual = {chi2_combined / chi2_visual}"
)