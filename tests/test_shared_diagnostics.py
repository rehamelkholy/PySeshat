import numpy as np

from pyseshat.inference.diagnostics import (
    chi_square,
    reduced_chi_square,
    rms_residual,
    fit_summary
)

# ==========================================================
# Synthetic residuals
# ==========================================================

residuals = np.array(
    [
        1.0,
        -2.0,
        3.0,
        -4.0
    ]
)

print()

print(
    "CHI-SQUARE"
)

print(
    "=" * 60
)

print(
    chi_square(
        residuals
    )
)

print()

print(
    "REDUCED CHI-SQUARE"
)

print(
    "=" * 60
)

print(
    reduced_chi_square(
        residuals,
        n_parameters=2
    )
)

print()

print(
    "RMS"
)

print(
    "=" * 60
)

print(
    rms_residual(
        residuals
    )
)

print()

print(
    "SUMMARY"
)

print(
    "=" * 60
)

print(
    fit_summary(
        residuals,
        n_parameters=2
    )
)