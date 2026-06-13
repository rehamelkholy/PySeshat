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

# Benchmark convention used throughout validation
X_obs = rho * np.cos(theta)
Y_obs = rho * np.sin(theta)

# ==========================================================
# Helper
# ==========================================================

def compute_statistics(elements, label):

    r_visual = visual_xy_residuals(
        elements.fitted_parameters(),
        elements,
        t,
        X_obs,
        Y_obs,
        sigma,
        sigma
    )

    r_rv = rv_residuals(
        elements.fitted_parameters(),
        elements,
        rv_data
    )

    chi2_visual = np.sum(
        r_visual**2
    )

    chi2_rv = np.sum(
        r_rv**2
    )

    chi2_total = (
        chi2_visual
        +
        chi2_rv
    )

    print()
    print(label)
    print("=" * 60)

    print(
        f"Omega = {elements.Omega:.6f}"
    )

    print(
        f"omega = {elements.omega:.6f}"
    )

    print()

    print(
        f"Visual chi2   = {chi2_visual:.6f}"
    )

    print(
        f"RV chi2       = {chi2_rv:.6f}"
    )

    print(
        f"Combined chi2 = {chi2_total:.6f}"
    )

    return (
        chi2_visual,
        chi2_rv,
        chi2_total
    )

# ==========================================================
# Case 1
# Original catalog orbit
# ==========================================================

catalog = copy.deepcopy(
    elements
)

# ==========================================================
# Case 2
# omega + 180
# ==========================================================

omega_flip = copy.deepcopy(
    elements
)

omega_flip.omega = (
    omega_flip.omega + 180.0
) % 360.0

# ==========================================================
# Case 3
# Omega + 180 and omega + 180
# ==========================================================

node_flip = copy.deepcopy(
    elements
)

node_flip.Omega = (
    node_flip.Omega + 180.0
) % 360.0

node_flip.omega = (
    node_flip.omega + 180.0
) % 360.0

# ==========================================================
# Evaluate all cases
# ==========================================================

v1, r1, c1 = compute_statistics(
    catalog,
    "CASE 1: Catalog Orbit"
)

v2, r2, c2 = compute_statistics(
    omega_flip,
    "CASE 2: omega + 180 deg"
)

v3, r3, c3 = compute_statistics(
    node_flip,
    "CASE 3: Omega + 180 deg, omega + 180 deg"
)

# ==========================================================
# Summary
# ==========================================================

print()
print("SUMMARY")
print("=" * 60)

print()
print("Catalog")
print(
    f"Visual={v1:.3f}, RV={r1:.3f}, Combined={c1:.3f}"
)

print()

print("omega + 180")
print(
    f"Visual={v2:.3f}, RV={r2:.3f}, Combined={c2:.3f}"
)

print()

print("Omega + 180, omega + 180")
print(
    f"Visual={v3:.3f}, RV={r3:.3f}, Combined={c3:.3f}"
)

print()

print("Relative to catalog")

print(
    f"Delta Combined (omega flip) = {c2 - c1:.3f}"
)

print(
    f"Delta Combined (node flip) = {c3 - c1:.3f}"
)