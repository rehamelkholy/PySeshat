import numpy as np

from pyseshat.io.orbit_parser import (
    parse_orbit_file
)

from pyseshat.inference.node_disambiguation import (
    node_assignment_chi2
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
# Compare node assignments
# ==========================================================

result = node_assignment_chi2(
    elements,
    t,
    X_obs,
    Y_obs,
    sigma,
    sigma,
    rv_data
)

print()

print(
    "Original chi2:"
)

print(
    result["chi2_original"]
)

print()

print(
    "Flipped chi2:"
)

print(
    result["chi2_flipped"]
)

print()

print(
    "Prefer flipped:"
)

print(
    result["prefer_flipped"]
)