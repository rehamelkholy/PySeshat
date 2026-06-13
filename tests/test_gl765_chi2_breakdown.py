from pathlib import Path
import numpy as np

from pyseshat.io.orbit_parser import parse_orbit_file
from pyseshat.inference.optimizer import fit_visual_orbit
from pyseshat.inference.likelihood import (
    visual_xy_residuals,
    rv_residuals
)

# --------------------------------------------------
# Load GL765
# --------------------------------------------------

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

# --------------------------------------------------
# Fit
# --------------------------------------------------

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

# --------------------------------------------------
# Residuals
# --------------------------------------------------

res_visual = visual_xy_residuals(
    fitted.fitted_parameters(),
    fitted,
    t,
    X,
    Y,
    sigma,
    sigma
)

res_rv = rv_residuals(
    fitted.fitted_parameters(),
    fitted,
    rv_data
)

# --------------------------------------------------
# Chi-square
# --------------------------------------------------

chi2_visual = np.sum(
    res_visual**2
)

chi2_rv = np.sum(
    res_rv**2
)

print()
print("Visual chi2:", chi2_visual)

print()
print("RV chi2:", chi2_rv)

print()
print(
    "RV / Visual ratio:",
    chi2_rv / chi2_visual
)