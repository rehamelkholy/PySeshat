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
# Enable B
# ==========================================================

elements.fit_flags["B"] = True

# ==========================================================
# Initial product
# ==========================================================

initial_product = (
    elements.inner.a
    *
    elements.B
)

# ==========================================================
# Fit
# ==========================================================

fit = fit_triple_orbit(
    elements,
    observations
)

final_product = (
    fit.elements.inner.a
    *
    fit.elements.B
)

# ==========================================================
# Report
# ==========================================================

print()

print("INITIAL")

print(
    f"a2 = {elements.inner.a}"
)

print(
    f"B  = {elements.B}"
)

print(
    f"a2*B = {initial_product}"
)

print()

print("FINAL")

print(
    f"a2 = {fit.elements.inner.a}"
)

print(
    f"B  = {fit.elements.B}"
)

print(
    f"a2*B = {final_product}"
)

print()

print("Difference")

print(
    final_product
    -
    initial_product
)