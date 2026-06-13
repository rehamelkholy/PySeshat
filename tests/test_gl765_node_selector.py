import numpy as np

from pyseshat.io.orbit_parser import (
    parse_orbit_file
)

from pyseshat.inference.node_disambiguation import (
    select_best_node_assignment
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
# Select node assignment
# ==========================================================

selected = select_best_node_assignment(
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
    "Selected Omega:"
)

print(
    selected.Omega
)

print()

print(
    "Selected omega:"
)

print(
    selected.omega
)