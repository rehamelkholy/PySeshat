from __future__ import annotations

import numpy as np

from pyseshat.triple.triple_elements import (
    TripleOrbitalElements
)


# ==========================================================
# Legacy ORBITT2 parameter order
# ==========================================================

LEGACY_TRIPLE_PARAMETER_ORDER = [

    # Outer orbit

    "P",
    "T0",
    "e",
    "a",
    "Omega",
    "omega",
    "i",

    # Inner orbit

    "P2",
    "T02",
    "e2",
    "a2",
    "Omega2",
    "omega2",
    "i2",

    # Legacy scale parameter

    "B"
]


# ==========================================================
# Modern ORBIT3 parameter order
# ==========================================================

ORBIT3_PARAMETER_ORDER = [

    # Inner orbit

    "P",
    "T0",
    "e",
    "a",
    "Omega",
    "omega",
    "i",

    # Inner spectroscopic orbit / wobble

    "K1",
    "K2",
    "f",

    # Outer orbit

    "P2",
    "T02",
    "e2",
    "a2",
    "Omega2",
    "omega2",
    "i2",

    # Outer spectroscopic orbit

    "K3",
    "K4",
    "V0"
]


# ==========================================================
# Legacy Triple -> array
# ==========================================================

def triple_to_array(
    elements: TripleOrbitalElements
):
    """
    Convert TripleOrbitalElements into the legacy
    15-parameter ORBITT2-style parameter vector.

    The ordering is:

        P, T0, e, a, Omega, omega, i,
        P2, T02, e2, a2, Omega2, omega2, i2,
        B

    This function is retained for compatibility with the
    existing .in2 / orbitT2 workflow.

    Parameters
    ----------
    elements
        Triple orbital elements.

    Returns
    -------
    numpy.ndarray
        Fifteen-element legacy parameter vector.
    """

    return np.array([

        # Outer orbit

        elements.outer.P,
        elements.outer.T0,
        elements.outer.e,
        elements.outer.a,
        elements.outer.Omega,
        elements.outer.omega,
        elements.outer.i,

        # Inner orbit

        elements.inner.P,
        elements.inner.T0,
        elements.inner.e,
        elements.inner.a,
        elements.inner.Omega,
        elements.inner.omega,
        elements.inner.i,

        # Legacy scale

        elements.B

    ], dtype=float)


# ==========================================================
# Legacy array -> Triple
# ==========================================================

def array_to_triple(
    parameter_vector,
    template: TripleOrbitalElements
):
    """
    Build TripleOrbitalElements from a legacy
    15-parameter ORBITT2-style vector.

    Parameters
    ----------
    parameter_vector
        Fifteen-element legacy parameter vector.

    template
        Template containing the current triple
        configuration and fit flags.

    Returns
    -------
    TripleOrbitalElements
        Updated triple elements.

    Raises
    ------
    ValueError
        If the supplied vector does not contain exactly
        fifteen parameters.
    """

    p = np.asarray(
        parameter_vector,
        dtype=float
    )

    if p.shape != (15,):

        raise ValueError(
            "Legacy triple parameter vector must "
            "contain exactly 15 elements."
        )

    outer = template.outer.copy()

    inner = template.inner.copy()

    # ------------------------------------------------------
    # Outer orbit
    # ------------------------------------------------------

    outer.P = p[0]
    outer.T0 = p[1]
    outer.e = p[2]
    outer.a = p[3]
    outer.Omega = p[4]
    outer.omega = p[5]
    outer.i = p[6]

    # ------------------------------------------------------
    # Inner orbit
    # ------------------------------------------------------

    inner.P = p[7]
    inner.T0 = p[8]
    inner.e = p[9]
    inner.a = p[10]
    inner.Omega = p[11]
    inner.omega = p[12]
    inner.i = p[13]

    # ------------------------------------------------------
    # Legacy scale
    # ------------------------------------------------------

    B = p[14]

    outer.normalize_angles()

    inner.normalize_angles()

    return TripleOrbitalElements(

        outer=outer,

        inner=inner,

        K1=template.K1,

        K2=template.K2,

        f=template.f,

        K3=template.K3,

        K4=template.K4,

        V0=template.V0,

        B=B,

        fit_flags=template.fit_flags.copy()
    )


# ==========================================================
# Modern ORBIT3 Triple -> array
# ==========================================================

def triple_to_orbit3_array(
    elements: TripleOrbitalElements
):
    """
    Convert TripleOrbitalElements into the modern
    20-parameter ORBIT3 vector.

    The ordering follows Tokovinin's ORBIT3 format:

        P, T0, e, a, Omega, omega, i,
        K1, K2, f,
        P2, T02, e2, a2, Omega2, omega2, i2,
        K3, K4, V0

    Parameters
    ----------
    elements
        Triple orbital elements.

    Returns
    -------
    numpy.ndarray
        Twenty-element ORBIT3 parameter vector.
    """

    return np.array([

        # --------------------------------------------------
        # Inner orbit
        # --------------------------------------------------

        elements.inner.P,
        elements.inner.T0,
        elements.inner.e,
        elements.inner.a,
        elements.inner.Omega,
        elements.inner.omega,
        elements.inner.i,

        # --------------------------------------------------
        # Inner spectroscopic orbit / wobble
        # --------------------------------------------------

        elements.K1,
        elements.K2,
        elements.f,

        # --------------------------------------------------
        # Outer orbit
        # --------------------------------------------------

        elements.outer.P,
        elements.outer.T0,
        elements.outer.e,
        elements.outer.a,
        elements.outer.Omega,
        elements.outer.omega,
        elements.outer.i,

        # --------------------------------------------------
        # Outer spectroscopic orbit
        # --------------------------------------------------

        elements.K3,
        elements.K4,
        elements.V0

    ], dtype=float)


# ==========================================================
# Modern ORBIT3 array -> Triple
# ==========================================================

def orbit3_array_to_triple(
    parameter_vector,
    template: TripleOrbitalElements
):
    """
    Build TripleOrbitalElements from a modern
    20-parameter ORBIT3 vector.

    Parameters
    ----------
    parameter_vector
        Twenty-element ORBIT3 parameter vector.

    template
        Template containing the current triple
        configuration and fit flags.

    Returns
    -------
    TripleOrbitalElements
        Updated triple elements.

    Raises
    ------
    ValueError
        If the supplied vector does not contain exactly
        twenty parameters.
    """

    p = np.asarray(
        parameter_vector,
        dtype=float
    )

    if p.shape != (20,):

        raise ValueError(
            "ORBIT3 parameter vector must "
            "contain exactly 20 elements."
        )

    outer = template.outer.copy()

    inner = template.inner.copy()

    # ------------------------------------------------------
    # Inner orbit
    # ------------------------------------------------------

    inner.P = p[0]
    inner.T0 = p[1]
    inner.e = p[2]
    inner.a = p[3]
    inner.Omega = p[4]
    inner.omega = p[5]
    inner.i = p[6]

    # ------------------------------------------------------
    # Inner spectroscopic orbit / wobble
    # ------------------------------------------------------

    K1 = p[7]
    K2 = p[8]
    f = p[9]

    # ------------------------------------------------------
    # Outer orbit
    # ------------------------------------------------------

    outer.P = p[10]
    outer.T0 = p[11]
    outer.e = p[12]
    outer.a = p[13]
    outer.Omega = p[14]
    outer.omega = p[15]
    outer.i = p[16]

    # ------------------------------------------------------
    # Outer spectroscopic orbit
    # ------------------------------------------------------

    K3 = p[17]
    K4 = p[18]
    V0 = p[19]

    inner.normalize_angles()

    outer.normalize_angles()

    return TripleOrbitalElements(

        outer=outer,

        inner=inner,

        K1=K1,

        K2=K2,

        f=f,

        K3=K3,

        K4=K4,

        V0=V0,

        # Preserve the legacy value because the
        # modern ORBIT3 vector does not contain B.

        B=template.B,

        fit_flags=template.fit_flags.copy()
    )


# ==========================================================
# Modern fitted-parameter extraction
# ==========================================================

def orbit3_fitted_parameters(
    elements: TripleOrbitalElements
):
    """
    Return the fitted parameters from the modern
    ORBIT3 representation.

    Parameters
    ----------
    elements
        Triple orbital elements.

    Returns
    -------
    numpy.ndarray
        Fitted ORBIT3 parameter vector.
    """

    full = triple_to_orbit3_array(
        elements
    )

    names = ORBIT3_PARAMETER_ORDER

    values = []

    for name, value in zip(
        names,
        full
    ):

        if elements.fit_flags.get(
            name,
            False
        ):

            values.append(
                value
            )

    return np.asarray(
        values,
        dtype=float
    )


# ==========================================================
# Modern fitted-parameter update
# ==========================================================

def update_orbit3_fitted_parameters(
    elements: TripleOrbitalElements,
    fitted_values
):
    """
    Update only parameters marked as fitted in the
    modern ORBIT3 representation.

    Parameters
    ----------
    elements
        Template triple orbital elements.

    fitted_values
        Values corresponding to the parameters whose
        ORBIT3 fit flags are True.

    Returns
    -------
    TripleOrbitalElements
        Updated triple orbital elements.
    """

    full = triple_to_orbit3_array(
        elements
    ).copy()

    values = np.asarray(
        fitted_values,
        dtype=float
    )

    mask = np.array([

        elements.fit_flags.get(
            name,
            False
        )

        for name
        in ORBIT3_PARAMETER_ORDER

    ], dtype=bool)

    if values.shape != (
        int(np.count_nonzero(mask)),
    ):

        raise ValueError(
            "Number of fitted ORBIT3 values does "
            "not match the number of fitted "
            "parameters."
        )

    full[mask] = values

    return orbit3_array_to_triple(

        full,

        elements
    )


# ==========================================================
# Parameter-name helpers
# ==========================================================

def legacy_triple_parameter_names():
    """
    Return the legacy 15-parameter ordering.
    """

    return list(
        LEGACY_TRIPLE_PARAMETER_ORDER
    )


def orbit3_parameter_names():
    """
    Return the modern ORBIT3 20-parameter ordering.
    """

    return list(
        ORBIT3_PARAMETER_ORDER
    )