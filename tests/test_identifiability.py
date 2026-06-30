from pathlib import Path

from pyseshat.triple.triple_parser import (
    parse_triple_file
)

from pyseshat.triple.triple_fit import (
    fit_triple_orbit
)

# ==========================================================
# Load benchmark
# ==========================================================

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

observations = result[
    "observations"
]

# ==========================================================
# Two-parameter fit
# ==========================================================

elements.fit_flags["B"] = True

fit = fit_triple_orbit(
    elements,
    observations
)

# ==========================================================
# Diagnostics
# ==========================================================

print()

print(
    "Condition Number"
)

print(
    "=" * 60
)

print(
    fit.condition_number()
)

print()

print(
    "Rank"
)

print(
    "=" * 60
)

print(
    fit.rank()
)

print()

print(
    "Degenerate"
)

print(
    "=" * 60
)

print(
    fit.is_degenerate()
)

print()

print(
    fit.report()
)