import numpy as np

from pyseshat.io.orbit_parser import parse_orbit_file

parsed = parse_orbit_file(
    "validation/benchmark_systems/GL765/GL765.inp"
)

visual = parsed["visual_data"]

theta_deg = visual[:,1]
rho = visual[:,2]

theta_rad = np.deg2rad(theta_deg)

# ------------------------------------------
# Astronomical PA convention
# ------------------------------------------

X = rho * np.cos(theta_rad)
Y = rho * np.sin(theta_rad)

theta_recovered = np.rad2deg(
    np.arctan2(X, Y)
)

theta_recovered %= 360

print()

print("Original:")
print(theta_deg)

print()

print("Recovered:")
print(theta_recovered)

print()

print("Difference:")
print(theta_recovered - theta_deg)