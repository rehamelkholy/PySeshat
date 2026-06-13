from pyseshat.io.orbit_parser import parse_orbit_file
from pyseshat.inference.optimizer import fit_visual_orbit
from pyseshat.inference.likelihood import combined_residuals

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

# ------------------------------------------
# Initial objective
# ------------------------------------------

r0 = combined_residuals(
    elements.fitted_parameters(),
    elements,
    t,
    X,
    Y,
    sigma,
    sigma,
    rv_data
)

chi2_initial = np.sum(
    r0**2
)

# ------------------------------------------
# Fit
# ------------------------------------------

fit_result = fit_visual_orbit(
    elements,
    t,
    X,
    Y,
    sigma,
    sigma,
    rv_data=rv_data
)

fitted = fit_result.elements

# ------------------------------------------
# Final objective
# ------------------------------------------

r1 = combined_residuals(
    fitted.fitted_parameters(),
    fitted,
    t,
    X,
    Y,
    sigma,
    sigma,
    rv_data
)

chi2_final = np.sum(
    r1**2
)

print()
print("Initial combined chi2:", chi2_initial)

print()
print("Final combined chi2:", chi2_final)

print()
print("Improvement:", chi2_initial - chi2_final)

print()
print("Ratio:", chi2_final / chi2_initial)