from pathlib import Path

from pyseshat.io.orbit_parser import (
    parse_orbit_file
)

files = [

    Path(
        "validation/benchmark_systems/GL765/GL765.inp"
    )
]

for f in files:

    parsed = parse_orbit_file(
        str(f)
    )

    print()

    print(
        parsed["metadata"]
    )