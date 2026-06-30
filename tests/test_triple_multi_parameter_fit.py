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
# Enable multiple parameters
# ==========================================================

elements.fit_flags["a2"] = True
elements.fit_flags["e2"] = True
elements.fit_flags["omega2"] = True
elements.fit_flags["Omega2"] = True

# ==========================================================
# Perturb initial guess
# ==========================================================

elements.inner.a = 0.15

elements.inner.e = 0.75

elements.inner.omega = 130.0

elements.inner.Omega = 250.0

# ==========================================================
# Fit
# ==========================================================

fit = fit_triple_orbit(
    elements,
    observations
)

print()

print(
    fit.report()
)

print()

print(
    "CORRELATION MATRIX"
)

print(
    "=" * 80
)

print(
    fit.correlations()
)