from pathlib import Path
import numpy as np

from pyseshat.io.orbit_parser import (
    parse_orbit_file
)

from pyseshat.analysis.ephemeris import (
    generate_visual_ephemeris,
    visual_ephemeris_table,
    generate_rv_ephemeris,
    rv_ephemeris_table,
    combined_ephemeris_table
)

filepath = Path(
    "validation/benchmark_systems/FIN379/FIN379.inp"
)

parsed = parse_orbit_file(
    str(filepath)
)

elements = parsed["elements"]

epochs = np.linspace(
    2010,
    2020,
    10
)

eph = generate_visual_ephemeris(
    elements,
    epochs
)

print(eph.keys())

for key, value in eph.items():

    print(
        key,
        len(value)
    )

table = visual_ephemeris_table(
    elements,
    epochs
)

print()
print(table.head(10))

rv_eph = generate_rv_ephemeris(
    elements,
    epochs
)

print()

print(rv_eph.keys())

for key, value in rv_eph.items():

    print(
        key,
        len(value)
    )

print()

rv_table = rv_ephemeris_table(
    elements,
    epochs
)

print(
    rv_table.head(10)
)

combined = combined_ephemeris_table(
    elements,
    epochs
)

print()

print(combined.head(10))

print()

print(combined.columns)