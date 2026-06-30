from pathlib import Path

from pyseshat.triple.triple_parser import (
    parse_triple_file
)

from pyseshat.triple.triple_bounds import (
    build_triple_bounds
)

filename = (
    Path(__file__).resolve().parents[1]
    / "validation"
    / "benchmark_systems"
    / "ZetaAqr"
    / "ZetaAqr.in2"
)

result = parse_triple_file(
    filename
)

elements = result["elements"]

lower, upper = build_triple_bounds(
    elements
)

print()
print("Lower bounds")
print("=" * 60)
print(lower)

print()
print("Upper bounds")
print("=" * 60)
print(upper)