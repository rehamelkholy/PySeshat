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
    Compute the radial-velocity orbit.

    Parameters
    ----------
    t : array_like
        Observation epochs.

    elements : OrbitalElements
        Orbital elements.

    component : int
        Internal component convention:

            0 = primary
            1 = secondary

    Returns
    -------
    numpy.ndarray
        Model radial velocities.
    """

    t = np.asarray(
        t,
        dtype=float
    )


    orbit = compute_visual_orbit(
        t,
        elements
    )


    nu = orbit["nu"]


    omega = np.deg2rad(
        elements.omega
    )


    # ------------------------------------------------------
    # Common spectroscopic factor
    #
    # A = e cos(omega) + cos(nu + omega)
    # ------------------------------------------------------

    common = (

        elements.e
        *
        np.cos(
            omega
        )

        +

        np.cos(
            nu
            +
            omega
        )

    )


    # ------------------------------------------------------
    # Primary
    #
    # RV1 = V0 + K1 A
    # ------------------------------------------------------

    if component == 0:

        return (

            elements.V0

            +

            elements.K1
            *
            common

        )


    # ------------------------------------------------------
    # Secondary
    #
    # RV2 = V0 - K2 A
    # ------------------------------------------------------

    if component == 1:

        return (

            elements.V0

            -

            elements.K2
            *
            common

        )


    raise ValueError(
        "component must be 0 or 1."
    )


# ==========================================================
# Modular binary RV model
# ==========================================================

def compute_rv_model(
    rv_data,
    elements: OrbitalElements
):
    """
    Compute model radial velocities for the modular binary
    inference interface.

    Parameters
    ----------
    rv_data : ndarray
        Array with columns:

            0 : epoch
            1 : observed RV
            2 : uncertainty
            3 : component identifier

    Supported component encodings
    ------------------------------

    Legacy binary encoding:

        0 = primary
        1 = secondary

    Alternate modular encoding:

        1 = primary
        2 = secondary

    The function determines which convention is being used
    from the component identifiers present in the data.

    If component 0 is present, the legacy 0/1 convention is
    used.

    Otherwise, if component 2 is present, the 1/2 convention
    is used.

    A dataset containing only component 1 is therefore
    interpreted as primary under the 1/2 convention.
    """

    rv_data = np.asarray(
        rv_data
    )


    if rv_data.ndim != 2:

        raise ValueError(
            "rv_data must be a two-dimensional array."
        )


    if rv_data.shape[1] < 4:

        raise ValueError(
            "rv_data must contain at least four columns."
        )


    t = rv_data[:, 0]


    component = rv_data[:, 3]


    rv_model = np.zeros_like(
        t,
        dtype=float
    )


    # ------------------------------------------------------
    # Determine component convention.
    #
    # Legacy:
    #
    #     0 = primary
    #     1 = secondary
    #
    # Alternate:
    #
    #     1 = primary
    #     2 = secondary
    #
    # The presence of 0 unambiguously selects the legacy
    # convention.
    #
    # Otherwise the presence of 2 selects the 1/2 convention.
    # ------------------------------------------------------

    has_zero = np.any(
        component == 0
    )


    has_two = np.any(
        component == 2
    )


    if has_zero and has_two:

        raise ValueError(
            "rv_data mixes incompatible component encodings: "
            "both 0 and 2 are present."
        )


    # ------------------------------------------------------
    # Legacy 0/1 convention
    # ------------------------------------------------------

    if has_zero:

        mask_primary = (
            component == 0
        )

        mask_secondary = (
            component == 1
        )


    # ------------------------------------------------------
    # Alternate 1/2 convention
    # ------------------------------------------------------

    elif has_two:

        mask_primary = (
            component == 1
        )

        mask_secondary = (
            component == 2
        )


    # ------------------------------------------------------
    # Component 1 only.
    #
    # This is inherently ambiguous. The modular convention
    # used by the dedicated RV test defines 1 as primary,
    # so use that interpretation.
    # ------------------------------------------------------

    else:

        mask_primary = (
            component == 1
        )

        mask_secondary = np.zeros(
            component.shape,
            dtype=bool
        )


    # ------------------------------------------------------
    # Primary observations
    # ------------------------------------------------------

    if np.any(
        mask_primary
    ):

        rv_model[
            mask_primary
        ] = compute_rv_orbit(

            t[
                mask_primary
            ],

            elements,

            component=0

        )


    # ------------------------------------------------------
    # Secondary observations
    # ------------------------------------------------------

    if np.any(
        mask_secondary
    ):

        rv_model[
            mask_secondary
        ] = compute_rv_orbit(

            t[
                mask_secondary
            ],

            elements,

            component=1

        )


    # ------------------------------------------------------
    # Validate component identifiers.
    # ------------------------------------------------------

    recognised = (

        mask_primary
        |
        mask_secondary

    )


    unknown = ~recognised


    if np.any(
        unknown
    ):

        invalid = np.unique(
            component[
                unknown
            ]
        )

        raise ValueError(
            "rv_data contains invalid component flags: "
            f"{invalid}. "
            "Supported encodings are "
            "0/1 (primary/secondary) or "
            "1/2 (primary/secondary)."
        )


    return rv_model