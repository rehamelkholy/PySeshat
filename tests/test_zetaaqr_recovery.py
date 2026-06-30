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

observations = result["observations"]

# ==========================================================
# Enable two-parameter fit
# ==========================================================

elements.fit_flags["a2"] = True
elements.fit_flags["B"] = True

# ==========================================================
# Perturb starting values
# ==========================================================

elements.inner.a = 0.15
elements.B = 2.50

print()
print("INITIAL GUESS")
print("=" * 60)

print(
    f"a2 = {elements.inner.a}"
)

print(
    f"B  = {elements.B}"
)

# ==========================================================
# Fit
# ==========================================================

fit = fit_triple_orbit(
    elements,
    observations
)

print()
print("RECOVERED SOLUTION")
print("=" * 60)

print(
    f"a2 = {fit.elements.inner.a}"
)

print(
    f"B  = {fit.elements.B}"
)

print()

print(
    fit.report()
)