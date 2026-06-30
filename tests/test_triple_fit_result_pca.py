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

elements.fit_flags["e2"] = True
elements.fit_flags["a2"] = True
elements.fit_flags["Omega2"] = True
elements.fit_flags["omega2"] = True

fit = fit_triple_orbit(
    elements,
    observations
)

best_variance, best_mode = (
    fit.best_mode()
)

worst_variance, worst_mode = (
    fit.worst_mode()
)

print()

print(
    "BEST MODE"
)

print(
    "=" * 80
)

print(
    best_variance
)

print(
    best_mode
)

print()

print(
    "WORST MODE"
)

print(
    "=" * 80
)

print(
    worst_variance
)

print(
    worst_mode
)

print()

print(
    "VARIANCE RATIO"
)

print(
    "=" * 80
)

print(
    fit.variance_ratio()
)