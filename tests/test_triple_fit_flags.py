from pathlib import Path

from pyseshat.triple.triple_parser import (
    parse_triple_file
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

triple = result["elements"]

# ==========================================================
# Print fit flags
# ==========================================================

print()

print("FIT FLAGS")
print("=" * 60)

for key, value in triple.fit_flags.items():

    print(
        f"{key:<10} {value}"
    )

print()

print(
    "Fitted parameter names:"
)

print(
    triple.fitted_parameter_names()
)

print()

print(
    "Fitted mask:"
)

print(
    triple.fitted_mask()
)

print()

print(
    "Number fitted:"
)

print(
    triple.fitted_mask().sum()
)