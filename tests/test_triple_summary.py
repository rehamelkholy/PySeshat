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

elements.fit_flags["B"] = True

fit = fit_triple_orbit(
    elements,
    observations
)

summary = fit.summary()

print()

print("FIT SUMMARY")
print("=" * 60)

for key, value in summary.items():

    print(
        f"{key:<20} {value}"
    )