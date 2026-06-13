import copy
import numpy as np

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
# Load FIN379
# ==========================================================

parsed = parse_orbit_file(
    "validation/benchmark_systems/FIN379/FIN379.inp"
)

elements = parsed["elements"]

visual = parsed["visual_data"]

rv_data = parsed["rv_data"]

# ==========================================================
# Observations
# ==========================================================

t = visual[:,0]

theta_deg = visual[:,1]

rho = visual[:,2]

sigma = visual[:,3]

theta = np.deg2rad(
    theta_deg
)

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
# Construct node-flipped orbit
# ==========================================================

flipped = copy.deepcopy(
    fitted
)

flipped.Omega = (
    flipped.Omega + 180.0
) % 360.0

flipped.omega = (
    flipped.omega + 180.0
) % 360.0

# ==========================================================
# Dense orbit sampling
# ==========================================================

t_dense = np.linspace(
    fitted.T0,
    fitted.T0 + fitted.P,
    2000
)

orbit_a = compute_visual_orbit(
    t_dense,
    fitted
)

orbit_b = compute_visual_orbit(
    t_dense,
    flipped
)

X_a = orbit_a["X"]
Y_a = orbit_a["Y"]

X_b = orbit_b["X"]
Y_b = orbit_b["Y"]

# ==========================================================
# Compare trajectories
# ==========================================================

max_dx = np.max(
    np.abs(X_a - X_b)
)

max_dy = np.max(
    np.abs(Y_a - Y_b)
)

rms_dx = np.sqrt(
    np.mean(
        (X_a - X_b)**2
    )
)

rms_dy = np.sqrt(
    np.mean(
        (Y_a - Y_b)**2
    )
)

# ==========================================================
# Report
# ==========================================================

print()

print("FITTED ORBIT")
print("=" * 60)

print(
    f"Omega = {fitted.Omega:.12f}"
)

print(
    f"omega = {fitted.omega:.12f}"
)

print()

print("NODE-FLIPPED ORBIT")
print("=" * 60)

print(
    f"Omega = {flipped.Omega:.12f}"
)

print(
    f"omega = {flipped.omega:.12f}"
)

print()

print("TRAJECTORY DIFFERENCES")
print("=" * 60)

print(
    f"Max |ΔX| = {max_dx:.15e}"
)

print(
    f"Max |ΔY| = {max_dy:.15e}"
)

print()

print(
    f"RMS ΔX = {rms_dx:.15e}"
)

print(
    f"RMS ΔY = {rms_dy:.15e}"
)

print()

if (
    max_dx < 1e-10
    and
    max_dy < 1e-10
):

    print(
        "RESULT: VISUAL ORBITS ARE IDENTICAL"
    )

else:

    print(
        "RESULT: VISUAL ORBITS DIFFER"
    )