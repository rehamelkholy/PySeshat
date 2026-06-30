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
# Fit
# ==========================================================

fit = fit_triple_orbit(
    elements,
    observations
)

# ==========================================================
# Diagnostics
# ==========================================================

print()

print("Fitted parameters")
print("=" * 60)

print(
    fit.elements.fitted_parameter_names()
)

print()

print("Uncertainties")
print("=" * 60)

for key, value in fit.uncertainties.items():

    print(
        f"{key:<10} {value}"
    )

print()

print("Covariance shape")
print("=" * 60)

print(
    fit.covariance.shape
)

print()

print("Optimizer success")
print("=" * 60)

print(
    fit.optimizer_result.success
)