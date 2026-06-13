from pyseshat.io.orbit_parser import parse_orbit_file

import numpy as np

parsed = parse_orbit_file(
    "validation/benchmark_systems/GL765/GL765.inp"
)

visual = parsed["visual_data"]

sigma = visual[:,3]

print()

print("Min sigma:")
print(np.min(sigma))

print()

print("Max sigma:")
print(np.max(sigma))

print()

print("Mean sigma:")
print(np.mean(sigma))

print()

print("All sigmas:")
print(sigma)