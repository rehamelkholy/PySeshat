from pathlib import Path

from pyseshat.triple.triple_parser import (
    parse_triple_file
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

triple = result["elements"]

obs = result["observations"]

print()
print("OUTER")
print("=" * 60)

print(
    triple.outer
)

print()

print("INNER")
print("=" * 60)

print(
    triple.inner
)

print()

print("B")
print("=" * 60)

print(
    triple.B
)

print()

print("OBSERVATIONS")
print("=" * 60)

print(
    len(obs)
)

print()

print(
    obs[0]
)

print()

print(
    obs[-1]
)