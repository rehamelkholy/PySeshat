from pyseshat.io.orbit_parser import (
    parse_orbit_file
)

from pyseshat.inference.priors import (
    log_prior
)

# ==========================================================
# Load FIN379
# ==========================================================

parsed = parse_orbit_file(
    "validation/benchmark_systems/FIN379/FIN379.inp"
)

elements = parsed["elements"]

# ==========================================================
# Evaluate prior
# ==========================================================

value = log_prior(
    elements
)

print()

print("LOG PRIOR")
print("=" * 60)

print(value)

print()

print("FINITE")
print("=" * 60)

print(value != float("-inf"))