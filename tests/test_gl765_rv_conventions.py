from pyseshat.io.orbit_parser import (
    parse_orbit_file
)

from pyseshat.inference.likelihood import (
    rv_residuals
)

import numpy as np

# ==========================================================
# Load GL765
# ==========================================================

parsed = parse_orbit_file(
    "validation/benchmark_systems/GL765/GL765.inp"
)

elements = parsed["elements"]

rv_data = parsed["rv_data"]

# ==========================================================
# Case 1
# Current convention
# ==========================================================

res1 = rv_residuals(
    elements.fitted_parameters(),
    elements,
    rv_data
)

chi2_case1 = np.sum(
    res1**2
)

# ==========================================================
# Case 2
# Manually apply spectroscopic convention
# ==========================================================

spec_elements = (
    elements.to_spectroscopic_convention()
)

res2 = rv_residuals(
    spec_elements.fitted_parameters(),
    spec_elements,
    rv_data
)

chi2_case2 = np.sum(
    res2**2
)

# ==========================================================
# Output
# ==========================================================

print()

print(
    "RV chi2 (current convention):"
)

print(
    chi2_case1
)

print()

print(
    "RV chi2 (spectroscopic convention applied first):"
)

print(
    chi2_case2
)

print()

print(
    "Ratio:"
)

print(
    chi2_case2 / chi2_case1
)