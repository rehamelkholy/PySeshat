import numpy as np


def parameter_bias(true_values, estimated_values):
    """
    Compute parameter bias.

    Returns
    -------
    ndarray
        estimated - true
    """

    true_values = np.asarray(true_values, dtype=float)
    estimated_values = np.asarray(estimated_values, dtype=float)

    return estimated_values - true_values


def absolute_error(true_values, estimated_values):
    """
    Absolute parameter error.
    """

    return np.abs(
        parameter_bias(
            true_values,
            estimated_values
        )
    )


def relative_error(true_values, estimated_values):
    """
    Relative parameter error.

    Returns
    -------
    ndarray
    """

    true_values = np.asarray(true_values, dtype=float)

    return np.divide(
        parameter_bias(
            true_values,
            estimated_values
        ),
        true_values,
        out=np.zeros_like(true_values),
        where=true_values != 0
    )


def normalized_error(
    true_values,
    estimated_values,
    uncertainties
):
    """
    Error measured in units of uncertainty.

    Returns
    -------
    ndarray
    """

    uncertainties = np.asarray(
        uncertainties,
        dtype=float
    )

    return np.divide(

        parameter_bias(
            true_values,
            estimated_values
        ),

        uncertainties,

        out=np.zeros_like(
            uncertainties
        ),

        where=uncertainties > 0
    )