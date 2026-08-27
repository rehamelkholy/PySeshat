import numpy as np


# ==========================================================
# Wrapped angular difference
# ==========================================================

def angular_difference(a, b):
    """
    Minimum angular separation in degrees.
    """

    diff = (a - b) % 360.0

    if diff > 180.0:
        diff -= 360.0

    return abs(diff)


# ==========================================================
# Node-aware angular difference
# ==========================================================

def node_aware_angular_difference(
    expected,
    fitted
):
    """
    Angular difference allowing for
    visual-binary node ambiguity.

    The following are treated as
    equivalent:

        angle

    and

        angle + 180 deg

    Returns
    -------
    float
    """

    d1 = angular_difference(
        expected,
        fitted
    )

    d2 = angular_difference(
        expected,
        fitted + 180.0
    )

    return min(
        d1,
        d2
    )


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
        "omega",
        # Outer-orbit angles of a hierarchical triple in the
        # flat ORBIT3 parameter schema (see
        # triple.triple_elements.ORBIT3_PARAMETER_ORDER) carry
        # the same node-ambiguity as the inner/binary "Omega"
        # and "omega" above and must be compared the same way.
        "Omega2",
        "omega2"
    ]

    if parameter in angular_parameters:

        return (
            node_aware_angular_difference(
                expected,
                fitted
            ) / 180.0
        )

    if expected == 0:

        return np.nan

    return abs(
        fitted - expected
    ) / abs(expected)