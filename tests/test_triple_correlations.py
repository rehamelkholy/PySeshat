from pathlib import Path

from pyseshat.triple.triple_parser import (
    parse_triple_file
)

from pyseshat.triple.triple_fit import (
    fit_triple_orbit
)

from pyseshat.triple.triple_statistics import (
    print_correlation_matrix
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
# Fit two parameters
# ==========================================================

elements.fit_flags["B"] = True

fit = fit_triple_orbit(
    elements,
    observations
)

# ==========================================================
# Correlation matrix
# ==========================================================

print_correlation_matrix(

    fit.covariance,

    fit.elements.fitted_parameter_names()
)