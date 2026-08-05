"""
Analytical uncertainty propagation utilities.

The functions in this module propagate independent
one-sigma uncertainties through common logarithmic
stellar quantities used throughout PySeshat.
"""

from __future__ import annotations

import numpy as np


LOG10 = np.log(10.0)


def sigma_log10(
    value: float,
    sigma: float,
) -> float:
    """
    Propagate the uncertainty of

        log10(x).

    Parameters
    ----------
    value : float
        Physical quantity.

    sigma : float
        One-sigma uncertainty of ``value``.

    Returns
    -------
    float
        One-sigma uncertainty of ``log10(value)``.
    """

    value = float(value)

    sigma = float(sigma)

    if value <= 0.0:

        raise ValueError(

            "value must be positive."

        )

    if sigma < 0.0:

        raise ValueError(

            "sigma must be non-negative."

        )

    return sigma / (

        value * LOG10

    )


def sigma_log_temperature(
    temperature,
    sigma_temperature,
):
    """
    Uncertainty of

        log10(T).
    """

    return sigma_log10(

        temperature,

        sigma_temperature,

    )


def sigma_log_radius(
    radius,
    sigma_radius,
):
    """
    Uncertainty of

        log10(R).
    """

    return sigma_log10(

        radius,

        sigma_radius,

    )


def sigma_log_luminosity(
    *,
    temperature,
    sigma_temperature,
    radius,
    sigma_radius,
):
    """
    Propagate the uncertainty of

        log10(L/Lsun)

    using

        L ∝ R² T⁴

    assuming independent uncertainties.

    Returns
    -------
    float
    """

    sigma_log_t = sigma_log_temperature(

        temperature,

        sigma_temperature,

    )

    sigma_log_r = sigma_log_radius(

        radius,

        sigma_radius,

    )

    return np.sqrt(

        4.0 * sigma_log_r**2

        +

        16.0 * sigma_log_t**2

    )