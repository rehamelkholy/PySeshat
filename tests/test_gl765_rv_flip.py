import copy
import numpy as np

from pyseshat.io.orbit_parser import (
    parse_orbit_file
)

from pyseshat.inference.likelihood import (
    visual_xy_residuals,
    rv_residuals
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
# Convert visual observations
# ==========================================================

t = visual[:, 0]

theta_deg = visual[:, 1]

rho = visual[:, 2]

sigma = visual[:, 3]

theta = np.deg2rad(theta_deg)

# ----------------------------------------------------------
# IMPORTANT:
# Use the benchmark convention currently used throughout
# the validation suite.
# ----------------------------------------------------------

X_obs = rho * np.cos(theta)

Y_obs = rho * np.sin(theta)

# ==========================================================
# Catalog orbit
# ==========================================================

r_visual_catalog = visual_xy_residuals(
    elements.fitted_parameters(),
    elements,
    t,
    X_obs,
    Y_obs,
    sigma,
    sigma
)

r_rv_catalog = rv_residuals(
    elements.fitted_parameters(),
    elements,
    rv_data
)

chi2_visual_catalog = np.sum(
    r_visual_catalog ** 2
)

chi2_rv_catalog = np.sum(
    r_rv_catalog ** 2
)

# ==========================================================
# Flipped orbit (omega + 180 deg)
# ==========================================================

flipped = copy.deepcopy(
    elements
)

flipped.omega = (
    flipped.omega + 180.0
) % 360.0

r_visual_flip = visual_xy_residuals(
    flipped.fitted_parameters(),
    flipped,
    t,
    X_obs,
    Y_obs,
    sigma,
    sigma
)

r_rv_flip = rv_residuals(
    flipped.fitted_parameters(),
    flipped,
    rv_data
)

chi2_visual_flip = np.sum(
    r_visual_flip ** 2
)

chi2_rv_flip = np.sum(
    r_rv_flip ** 2
)

# ==========================================================
# Report
# ==========================================================

print()
print("CATALOG ORBIT")
print("=" * 60)

print(
    f"omega = {elements.omega:.6f}"
)

print(
    f"Visual chi2 = {chi2_visual_catalog:.6f}"
)

print(
    f"RV chi2 = {chi2_rv_catalog:.6f}"
)

print(
    f"Combined chi2 = "
    f"{chi2_visual_catalog + chi2_rv_catalog:.6f}"
)

print()

print("FLIPPED ORBIT (omega + 180 deg)")
print("=" * 60)

print(
    f"omega = {flipped.omega:.6f}"
)

print(
    f"Visual chi2 = {chi2_visual_flip:.6f}"
)

print(
    f"RV chi2 = {chi2_rv_flip:.6f}"
)

print(
    f"Combined chi2 = "
    f"{chi2_visual_flip + chi2_rv_flip:.6f}"
)

print()

print("CHANGES")
print("=" * 60)

print(
    f"Delta Visual chi2 = "
    f"{chi2_visual_flip - chi2_visual_catalog:.6f}"
)

print(
    f"Delta RV chi2 = "
    f"{chi2_rv_flip - chi2_rv_catalog:.6f}"
)

print(
    f"Delta Combined chi2 = "
    f"{(chi2_visual_flip + chi2_rv_flip) - (chi2_visual_catalog + chi2_rv_catalog):.6f}"
)