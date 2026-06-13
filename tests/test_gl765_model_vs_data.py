from pathlib import Path
import numpy as np

from pyseshat.io.orbit_parser import (
    parse_orbit_file
)

from pyseshat.orbit.orbit_model import (
    compute_visual_orbit
)

# --------------------------------------------------
# Load GL765
# --------------------------------------------------

parsed = parse_orbit_file(
    "validation/benchmark_systems/GL765/GL765.inp"
)

elements = parsed["elements"]

visual = parsed["visual_data"]

t = visual[:,0]

theta = np.deg2rad(
    visual[:,1]
)

rho = visual[:,2]

X_obs = rho * np.cos(theta)
Y_obs = rho * np.sin(theta)

# --------------------------------------------------
# Model from catalog orbit
# --------------------------------------------------

orbit = compute_visual_orbit(
    t,
    elements
)

X_model = orbit["X"]
Y_model = orbit["Y"]

# --------------------------------------------------
# RMS
# --------------------------------------------------

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

print("RMS X:", rms_x)

print("RMS Y:", rms_y)

print()

print("Semi-major axis:", elements.a)

import matplotlib.pyplot as plt

plt.figure(figsize=(6,6))

plt.scatter(
    X_obs,
    Y_obs,
    label="Observed"
)

plt.scatter(
    X_model,
    Y_model,
    label="Model"
)

plt.axis("equal")
plt.legend()
plt.show()