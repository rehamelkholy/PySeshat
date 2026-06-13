from pathlib import Path
import numpy as np

from pyseshat.io.orbit_parser import parse_orbit_file
from pyseshat.inference.optimizer import fit_visual_orbit
from pyseshat.inference.likelihood import visual_xy_residuals

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

# ----------------------------------
# Initial orbit
# ----------------------------------

res_initial = visual_xy_residuals(
    elements.fitted_parameters(),
    elements,
    t,
    X,
    Y,
    sigma,
    sigma
)

chi2_initial = np.sum(
    res_initial**2
)

# ----------------------------------
# Fitted orbit
# ----------------------------------

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

res_final = visual_xy_residuals(
    fitted.fitted_parameters(),
    fitted,
    t,
    X,
    Y,
    sigma,
    sigma
)

chi2_final = np.sum(
    res_final**2
)

print()
print("Initial chi2:", chi2_initial)

print()
print("Final chi2:", chi2_final)

print()
print(
    "Improvement:",
    chi2_initial - chi2_final
)

print()
print(
    "Ratio:",
    chi2_final / chi2_initial
)