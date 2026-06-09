import numpy as np


# ==========================================================
# Angular difference
# ==========================================================
# NOTE:
# This metric does not account for
# visual-binary node degeneracies.
# Large angular differences may
# correspond to equivalent orbits.
def angular_difference(a, b):
    """
    Minimum angular separation in degrees.
    """

    diff = (a - b) % 360.0

    if diff > 180.0:
        diff -= 360.0

    return abs(diff)

# ==========================================================
# Relative parameter error
# ==========================================================

def parameter_relative_error(
    parameter,
    expected,
    fitted
):
    """
    Physics-aware parameter comparison.
    """

    angular_parameters = [
        "Omega",
        "omega"
    ]

    if parameter in angular_parameters:

        return (
            angular_difference(
                expected,
                fitted
            ) / 180.0
        )

    if expected == 0:
        return np.nan
    
    return abs(
        fitted - expected
    ) / abs(expected)