"""
Analytical uncertainty propagation utilities.

The functions in this module propagate independent
one-sigma uncertainties through common logarithmic
stellar quantities and derived physical quantities
used throughout PySeshat.
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
        One-sigma uncertainty of
        log10(L/Lsun).
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


def sigma_total_mass(
    *,
    semi_major_axis,
    sigma_semi_major_axis,
    period,
    sigma_period,
    parallax,
    sigma_parallax,
):
    """
    Propagate independent uncertainties in the
    visual-binary dynamical mass.

    The total mass is

        M ∝ a³ / (P² π³),

    where ``a`` and ``π`` are expressed in the same
    angular units and ``P`` is the orbital period.

    Equivalently,

        M = (a / π)³ / P².

    The propagated fractional uncertainty is

        (σ_M / M)²
            =
        9 (σ_a / a)²
        +
        4 (σ_P / P)²
        +
        9 (σ_π / π)².

    Parameters
    ----------
    semi_major_axis : float
        Angular semimajor axis.

    sigma_semi_major_axis : float
        One-sigma uncertainty in the angular
        semimajor axis.

    period : float
        Orbital period.

    sigma_period : float
        One-sigma uncertainty in the orbital period.

    parallax : float
        Parallax in the same angular units as
        ``semi_major_axis``.

    sigma_parallax : float
        One-sigma uncertainty in the parallax.

    Returns
    -------
    float
        One-sigma uncertainty of the dynamical mass
        in the corresponding mass units.

    Notes
    -----
    This function assumes independent uncertainties
    and therefore does not include covariance terms.
    """

    semi_major_axis = float(
        semi_major_axis
    )

    sigma_semi_major_axis = float(
        sigma_semi_major_axis
    )

    period = float(period)

    sigma_period = float(
        sigma_period
    )

    parallax = float(
        parallax
    )

    sigma_parallax = float(
        sigma_parallax
    )

    if semi_major_axis <= 0.0:

        raise ValueError(
            "semi_major_axis must be positive."
        )

    if period <= 0.0:

        raise ValueError(
            "period must be positive."
        )

    if parallax <= 0.0:

        raise ValueError(
            "parallax must be positive."
        )

    if sigma_semi_major_axis < 0.0:

        raise ValueError(
            "sigma_semi_major_axis must be "
            "non-negative."
        )

    if sigma_period < 0.0:

        raise ValueError(
            "sigma_period must be "
            "non-negative."
        )

    if sigma_parallax < 0.0:

        raise ValueError(
            "sigma_parallax must be "
            "non-negative."
        )

    mass = (
        semi_major_axis
        / parallax
    ) ** 3 / period**2

    fractional_variance = (

        9.0
        * (
            sigma_semi_major_axis
            / semi_major_axis
        ) ** 2

        +

        4.0
        * (
            sigma_period
            / period
        ) ** 2

        +

        9.0
        * (
            sigma_parallax
            / parallax
        ) ** 2
    )

    return mass * np.sqrt(
        fractional_variance
    )