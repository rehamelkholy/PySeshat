import numpy as np

from pyseshat.orbit.orbit_model import (
    compute_visual_orbit
)

from pyseshat.orbit.orbital_elements import (
    OrbitalElements
)


# ==========================================================
# RV model for binary system
# ==========================================================

def compute_rv_orbit(
    t,
    elements: OrbitalElements,
    component=0
):
    """
    Compute radial velocity orbit.

    Parameters
    ----------
    t : ndarray
        Observation epochs

    elements : OrbitalElements

    component : int
        0 = primary
        1 = secondary

    Returns
    -------
    rv : ndarray
    """

    # ------------------------------------------------------
    # Use spectroscopic convention
    # ------------------------------------------------------

    spec_elements = (
        elements.to_spectroscopic_convention()
    )

    # ------------------------------------------------------
    # Obtain orbital anomalies
    # ------------------------------------------------------

    orbit = compute_visual_orbit(
        t,
        spec_elements
    )

    nu = orbit["nu"]

    # ------------------------------------------------------
    # Convert omega to radians
    # ------------------------------------------------------

    omega = np.deg2rad(
        spec_elements.omega
    )

    # ------------------------------------------------------
    # Common RV term
    # ------------------------------------------------------

    common = (
        np.cos(omega + nu)
        +
        spec_elements.e * np.cos(omega)
    )

    # ------------------------------------------------------
    # Primary RV
    # ------------------------------------------------------

    if component == 0:

        rv = (
            elements.V0
            +
            elements.K1 * common
        )

    # ------------------------------------------------------
    # Secondary RV
    # ------------------------------------------------------

    elif component == 1:

        rv = (
            elements.V0
            -
            elements.K2 * common
        )

    else:

        raise ValueError(
            "component must be 0 or 1"
        )

    return rv


# ==========================================================
# Mixed RV model matching parser structure
# ==========================================================

def compute_rv_model(
    rv_data,
    elements: OrbitalElements
):
    """
    Compute RV model matching observational
    RV array structure.

    Parameters
    ----------
    rv_data : ndarray

        Columns:
            0 : epoch
            1 : observed RV
            2 : uncertainty
            3 : component flag

    elements : OrbitalElements

    Returns
    -------
    rv_model : ndarray
    """

    t = rv_data[:,0]

    component = rv_data[:,3].astype(int)

    rv_model = np.zeros_like(t)

    # ------------------------------------------------------
    # Primary observations
    # ------------------------------------------------------

    mask_primary = (component == 0)

    if np.any(mask_primary):

        rv_model[mask_primary] = compute_rv_orbit(
            t[mask_primary],
            elements,
            component=0
        )

    # ------------------------------------------------------
    # Secondary observations
    # ------------------------------------------------------

    mask_secondary = (component == 1)

    if np.any(mask_secondary):

        rv_model[mask_secondary] = compute_rv_orbit(
            t[mask_secondary],
            elements,
            component=1
        )

    return rv_model