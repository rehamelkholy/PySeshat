import numpy as np


# ==========================================================
# Correlation matrix
# ==========================================================

def correlation_matrix(
    covariance
):
    """
    Convert covariance matrix
    to correlation matrix.
    """

    covariance = np.asarray(
        covariance
    )

    sigma = np.sqrt(
        np.diag(covariance)
    )

    denom = np.outer(
        sigma,
        sigma
    )

    correlation = (
        covariance
        /
        denom
    )

    return correlation


# ==========================================================
# Print correlation matrix
# ==========================================================

def print_correlation_matrix(
    covariance,
    parameter_names
):
    """
    Nicely formatted correlation matrix.
    """

    corr = correlation_matrix(
        covariance
    )

    print()

    print(
        "Correlation Matrix"
    )

    print("=" * 80)

    header = (
        " " * 12
        +
        "".join(
            f"{p:>12}"
            for p in parameter_names
        )
    )

    print(header)

    for name, row in zip(
        parameter_names,
        corr
    ):

        line = (
            f"{name:<12}"
            +
            "".join(
                f"{v:12.4f}"
                for v in row
            )
        )

        print(line)