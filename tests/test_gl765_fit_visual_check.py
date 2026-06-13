from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from pyseshat.io.orbit_parser import (
    parse_orbit_file
)

from pyseshat.inference.optimizer import (
    fit_visual_orbit
)

from pyseshat.orbit.orbit_model import (
    compute_visual_orbit
)

# ==========================================================
# Load GL765
# ==========================================================

filepath = Path(
    "validation/benchmark_systems/GL765/GL765.inp"
)

parsed = parse_orbit_file(
    str(filepath)
)

elements = parsed["elements"]

visual_data = parsed["visual_data"]

rv_data = parsed["rv_data"]

# ==========================================================
# Convert observations
# ==========================================================

t = visual_data[:,0]

theta = np.deg2rad(
    visual_data[:,1]
)

rho = visual_data[:,2]

sigma = visual_data[:,3]

X_obs = rho * np.cos(theta)

Y_obs = rho * np.sin(theta)

# ==========================================================
# Fit orbit
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
# Evaluate fitted orbit at observation epochs
# ==========================================================

orbit = compute_visual_orbit(
    t,
    fitted
)

X_model = orbit["X"]

Y_model = orbit["Y"]

# ==========================================================
# Diagnostics
# ==========================================================

rms_x = np.sqrt(
    np.mean(
        (X_obs - X_model)**2
    )
)

rms_y = np.sqrt(
    np.mean(
        (Y_obs - Y_model)**2
    )
)

print()

print("Fitted Elements")
print("-" * 40)

print(
    f"P      = {fitted.P:.6f}"
)

print(
    f"T0     = {fitted.T0:.6f}"
)

print(
    f"e      = {fitted.e:.6f}"
)

print(
    f"a      = {fitted.a:.6f}"
)

print(
    f"Omega  = {fitted.Omega:.6f}"
)

print(
    f"omega  = {fitted.omega:.6f}"
)

print(
    f"i      = {fitted.i:.6f}"
)

print()

print("Fit Quality")
print("-" * 40)

print(
    f"RMS X = {rms_x:.6f}"
)

print(
    f"RMS Y = {rms_y:.6f}"
)

print()

# ==========================================================
# Plot observations vs fitted model
# ==========================================================

fig, ax = plt.subplots(
    figsize=(8,8)
)

ax.scatter(
    X_obs,
    Y_obs,
    s=80,
    label="Observed"
)

ax.scatter(
    X_model,
    Y_model,
    s=80,
    label="Model at observation epochs"
)

# ----------------------------------------------------------
# Connect matching points
# ----------------------------------------------------------

for xo, yo, xm, ym in zip(
    X_obs,
    Y_obs,
    X_model,
    Y_model
):

    ax.plot(
        [xo, xm],
        [yo, ym],
        linewidth=1
    )

# ----------------------------------------------------------
# Formatting
# ----------------------------------------------------------

ax.set_aspect(
    "equal"
)

ax.set_xlabel(
    "X (arcsec)"
)

ax.set_ylabel(
    "Y (arcsec)"
)

ax.set_title(
    "GL765: Observed vs Fitted Orbit Positions"
)

ax.grid(True)

ax.legend()

plt.show()