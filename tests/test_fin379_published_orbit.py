import numpy as np
import matplotlib.pyplot as plt

from pyseshat.io.orbit_parser import (
    parse_orbit_file
)

from pyseshat.orbit.orbit_model import (
    compute_visual_orbit
)

# ==========================================================
# Load FIN379
# ==========================================================

parsed = parse_orbit_file(
    "validation/benchmark_systems/FIN379/FIN379.inp"
)

elements = parsed["elements"]

visual = parsed["visual_data"]

# ==========================================================
# Observations
# ==========================================================

t = visual[:, 0]

theta_deg = visual[:, 1]

rho = visual[:, 2]

theta = np.deg2rad(theta_deg)

# ==========================================================
# Correct astronomical convention
# ==========================================================

X_obs = rho * np.cos(theta)

Y_obs = rho * np.sin(theta)

# ==========================================================
# Published orbit
# ==========================================================

orbit = compute_visual_orbit(
    t,
    elements
)

X_model = orbit["X"]

Y_model = orbit["Y"]

# ==========================================================
# RMS
# ==========================================================

rms_x = np.sqrt(
    np.mean(
        (X_obs - X_model) ** 2
    )
)

rms_y = np.sqrt(
    np.mean(
        (Y_obs - Y_model) ** 2
    )
)

print()
print("Published orbit diagnostics")
print("-" * 40)

print(
    f"RMS X = {rms_x:.6f}"
)

print(
    f"RMS Y = {rms_y:.6f}"
)

print()

# ==========================================================
# Plot
# ==========================================================

fig, ax = plt.subplots(
    figsize=(8, 8)
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
    label="Published Orbit"
)

for xo, yo, xm, ym in zip(
    X_obs,
    Y_obs,
    X_model,
    Y_model
):

    ax.plot(
        [xo, xm],
        [yo, ym]
    )

ax.set_aspect("equal")

ax.set_xlabel("X (arcsec)")
ax.set_ylabel("Y (arcsec)")

ax.set_title(
    "FIN379 Published Orbit Check"
)

ax.grid(True)

ax.legend()

plt.show()