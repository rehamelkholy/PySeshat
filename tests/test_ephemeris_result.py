import numpy as np

from pyseshat.io.orbit_parser import (
    parse_orbit_file
)

from pyseshat.ephemeris.ephemeris import (
    generate_combined_ephemeris
)

# ==========================================================
# Load benchmark orbit
# ==========================================================

parsed = parse_orbit_file(
    "validation/benchmark_systems/FIN379/FIN379.inp"
)

elements = parsed["elements"]

# ==========================================================
# Generate ephemeris
# ==========================================================

epochs = np.arange(
    2025,
    2031,
    1
)

eph = generate_combined_ephemeris(
    elements,
    epochs
)

# ==========================================================
# DataFrame
# ==========================================================

print()

print(
    eph.to_dataframe()
)

# ==========================================================
# Text table
# ==========================================================

print()

print(
    eph.to_text()
)

# ==========================================================
# Save files
# ==========================================================

eph.save_csv(
    "fin379_ephemeris.csv"
)

eph.save_text(
    "fin379_ephemeris.txt"
)

print()

print(
    "Files written successfully."
)