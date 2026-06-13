from pyseshat.io.orbit_parser import parse_orbit_file
from pyseshat.inference.likelihood import (
    visual_xy_residuals,
    rv_residuals
)

import numpy as np

parsed = parse_orbit_file(
    "validation/benchmark_systems/GL765/GL765.inp"
)

elements = parsed["elements"]

visual = parsed["visual_data"]

rv_data = parsed["rv_data"]

t = visual[:,0]

theta = np.deg2rad(
    visual[:,1]
)

rho = visual[:,2]

sigma = visual[:,3]

X = rho * np.cos(theta)
Y = rho * np.sin(theta)

res_visual = visual_xy_residuals(
    elements.fitted_parameters(),
    elements,
    t,
    X,
    Y,
    sigma,
    sigma
)

res_rv = rv_residuals(
    elements.fitted_parameters(),
    elements,
    rv_data
)

print()

print(
    "Initial visual chi2:",
    np.sum(res_visual**2)
)

print()

print(
    "Initial RV chi2:",
    np.sum(res_rv**2)
)

print()

print(
    "Initial total chi2:",
    np.sum(res_visual**2)
    +
    np.sum(res_rv**2)
)