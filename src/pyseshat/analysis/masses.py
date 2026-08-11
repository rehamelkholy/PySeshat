import numpy as np

from pyseshat.analysis.error_propagation import (
    sigma_total_mass,
)

from pyseshat.orbit.orbital_elements import (
    OrbitalElements
)


# ==========================================================
# Visual-binary total mass
# ==========================================================

def compute_total_mass(
    period_years,
    semimajor_axis_arcsec,
    parallax_mas
):
    """
    Compute total system mass using
    Kepler's Third Law.

    Parameters
    ----------
    period_years : float

    semimajor_axis_arcsec : float

    parallax_mas : float

    Returns
    -------
    mass : float
        Total system mass in solar masses.
    """

    parallax_arcsec = (
        parallax_mas / 1000.0
    )

    a_au = (
        semimajor_axis_arcsec
        /
        parallax_arcsec
    )

    mass = (
        a_au**3
        /
        period_years**2
    )

    return mass


# ==========================================================
# OrbitalElements wrapper
# ==========================================================

def total_mass_from_elements(
    elements: OrbitalElements,
    parallax_mas
):
    """
    Compute total mass directly from
    OrbitalElements.
    """

    return compute_total_mass(

        period_years=elements.P,

        semimajor_axis_arcsec=elements.a,

        parallax_mas=parallax_mas
    )


def total_mass_uncertainty_from_elements(
    elements: OrbitalElements,
    uncertainties: dict,
    parallax_mas: float,
    sigma_parallax_mas: float,
):
    """
    Propagate independent uncertainties in the
    dynamical total mass derived from OrbitalElements.

    The dynamical mass is

        M = (a / pi)^3 / P^2,

    where ``a`` and ``pi`` are expressed in the same
    angular units and ``P`` is in years.

    Parameters
    ----------
    elements : OrbitalElements
        Orbital solution.

    uncertainties : dict
        One-sigma uncertainties of the fitted orbital
        parameters. The dictionary must contain
        ``"P"`` and ``"a"``.

    parallax_mas : float
        Parallax in milliarcseconds.

    sigma_parallax_mas : float
        One-sigma parallax uncertainty in
        milliarcseconds.

    Returns
    -------
    float
        One-sigma uncertainty of the dynamical
        total mass in solar masses.

    Notes
    -----
    This uses the same independent-error,
    first-order analytical propagation convention
    as ``pyseshat.analysis.error_propagation``.
    """

    if not isinstance(
        uncertainties,
        dict
    ):

        raise TypeError(
            "uncertainties must be a dictionary."
        )

    required = (
        "P",
        "a",
    )

    missing = [
        name
        for name in required
        if name not in uncertainties
    ]

    if missing:

        raise KeyError(
            "Missing orbital uncertainties: "
            + ", ".join(missing)
        )

    parallax_arcsec = (
        float(parallax_mas) / 1000.0
    )

    sigma_parallax_arcsec = (
        float(sigma_parallax_mas) / 1000.0
    )

    return sigma_total_mass(

        semi_major_axis=elements.a,

        sigma_semi_major_axis=(
            uncertainties["a"]
        ),

        period=elements.P,

        sigma_period=(
            uncertainties["P"]
        ),

        parallax=parallax_arcsec,

        sigma_parallax=(
            sigma_parallax_arcsec
        ),
    )


def compute_mass_function(
    period_years,
    eccentricity,
    K1
):
    """
    Spectroscopic mass function.

    Parameters
    ----------
    period_years : float

    eccentricity : float

    K1 : float
        km/s

    Returns
    -------
    float
        Mass function in solar masses.
    """

    period_days = (
        period_years * 365.25
    )

    return (

        1.036149e-7

        *

        (1.0 - eccentricity**2)**1.5

        *

        K1**3

        *

        period_days
    )


def compute_component_masses(
    period_years,
    eccentricity,
    inclination_deg,
    K1,
    K2
):
    """
    Compute SB2 component masses.

    Parameters
    ----------
    period_years : float

    eccentricity : float

    inclination_deg : float

    K1, K2 : float
        km/s

    Returns
    -------
    dict
    """

    period_days = (
        period_years * 365.25
    )

    sini = np.sin(
        np.deg2rad(
            inclination_deg
        )
    )

    factor = (

        1.036149e-7

        *

        (1.0 - eccentricity**2)**1.5

        *

        period_days

    )

    M1_sin3i = (
        factor
        *
        (K1 + K2)**2
        *
        K2
    )

    M2_sin3i = (
        factor
        *
        (K1 + K2)**2
        *
        K1
    )

    sin3i = sini**3

    M1 = (
        M1_sin3i / sin3i
    )

    M2 = (
        M2_sin3i / sin3i
    )

    return {

        "M1": M1,

        "M2": M2,

        "M_total": (
            M1 + M2
        ),

        "M1_sin3i":
            M1_sin3i,

        "M2_sin3i":
            M2_sin3i
    }


def component_masses_from_elements(
    elements
):
    """
    Compute component masses from
    OrbitalElements.
    """

    return compute_component_masses(

        period_years=elements.P,

        eccentricity=elements.e,

        inclination_deg=elements.i,

        K1=elements.K1,

        K2=elements.K2
    )


def minimum_secondary_mass(
    mass_function,
    primary_mass,
    tol=1e-10,
    max_iter=100
):
    """
    Compute minimum companion mass
    for an SB1 system.

    Parameters
    ----------
    mass_function : float

    primary_mass : float
        Solar masses

    Returns
    -------
    float
        Minimum companion mass
        in solar masses.
    """

    M2 = max(
        mass_function,
        0.01
    )

    for _ in range(max_iter):

        f = (
            M2**3
            /
            (primary_mass + M2)**2
            -
            mass_function
        )

        df = (
            (
                3 * M2**2
                *
                (primary_mass + M2)**2
            )
            -
            (
                2 * M2**3
                *
                (primary_mass + M2)
            )
        ) / (
            primary_mass + M2
        )**4

        step = f / df

        M2 -= step

        if abs(step) < tol:
            break

    return M2


def minimum_secondary_mass_from_elements(
    elements,
    primary_mass
):
    """
    SB1 minimum companion mass.
    """

    fm = compute_mass_function(

        period_years=elements.P,

        eccentricity=elements.e,

        K1=elements.K1
    )

    return minimum_secondary_mass(

        mass_function=fm,

        primary_mass=primary_mass
    )