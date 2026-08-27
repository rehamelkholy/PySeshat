"""
Triple-orbit fitting.

This module contains the legacy ORBITT2 15-parameter fitting
path and the modern ORBIT3 20-parameter parameter-vector
utilities.

The two parameterizations deliberately use independent
parameter masks.
"""

import numpy as np


from pyseshat.inference.optimizer_engine import (
    run_least_squares
)


from pyseshat.triple.triple_bounds import (
    build_triple_bounds
)


from pyseshat.triple.triple_elements import (
    LEGACY_TRIPLE_PARAMETER_ORDER,
    ORBIT3_PARAMETER_ORDER,
    TripleOrbitalElements
)


from pyseshat.triple.triple_fit_result import (
    TripleFitResult
)


from pyseshat.triple.triple_residuals import (
    triple_visual_residuals
)


# ==========================================================
# Parameter orders
# ==========================================================
#
# LEGACY_TRIPLE_PARAMETER_ORDER and ORBIT3_PARAMETER_ORDER
# are imported from triple_elements.py, the single canonical
# source of both orderings. Both are re-exported here for
# backward compatibility with existing imports.


# ORBIT3_PARAMETER_ORDER is imported from triple_elements.py,
# which is the single canonical source of this ordering. It
# is re-exported here (and from triple_parameters.py) for
# backward compatibility with existing imports.


# ==========================================================
# Legacy Triple -> array
# ==========================================================

def triple_to_array(
    elements: TripleOrbitalElements
):
    """
    Convert TripleOrbitalElements to the legacy 15-parameter
    ORBITT2 vector.

    Ordering
    --------
    P, T0, e, a, Omega, omega, i,
    P2, T02, e2, a2, Omega2, omega2, i2,
    B
    """

    return np.array([

        elements.outer.P,
        elements.outer.T0,
        elements.outer.e,
        elements.outer.a,
        elements.outer.Omega,
        elements.outer.omega,
        elements.outer.i,

        elements.inner.P,
        elements.inner.T0,
        elements.inner.e,
        elements.inner.a,
        elements.inner.Omega,
        elements.inner.omega,
        elements.inner.i,

        elements.B,

    ], dtype=float)


# ==========================================================
# Legacy array -> Triple
# ==========================================================

def array_to_triple(
    p,
    template: TripleOrbitalElements
):
    """
    Build a TripleOrbitalElements object from the legacy
    15-parameter ORBITT2 vector.
    """

    p = np.asarray(
        p,
        dtype=float
    )


    if p.size != 15:

        raise ValueError(

            "Legacy triple parameter vector must "
            "contain exactly 15 parameters; "
            f"received {p.size}."

        )


    outer = template.outer.copy()

    inner = template.inner.copy()


    outer.P = p[0]

    outer.T0 = p[1]

    outer.e = p[2]

    outer.a = p[3]

    outer.Omega = p[4]

    outer.omega = p[5]

    outer.i = p[6]


    inner.P = p[7]

    inner.T0 = p[8]

    inner.e = p[9]

    inner.a = p[10]

    inner.Omega = p[11]

    inner.omega = p[12]

    inner.i = p[13]


    outer.normalize_angles()

    inner.normalize_angles()


    return TripleOrbitalElements(

        outer=outer,

        inner=inner,

        # The legacy 15-parameter ORBITT2 vector does not
        # carry K1, K2, f, K3, K4, or V0 -- those belong to
        # the modern ORBIT3 spectroscopic/wobble formalism.
        # They must be carried over from the template rather
        # than left at the TripleOrbitalElements defaults,
        # otherwise a legacy-only fit silently zeroes out any
        # ORBIT3-relevant state the object was carrying.

        K1=template.K1,

        K2=template.K2,

        f=template.f,

        K3=template.K3,

        K4=template.K4,

        V0=template.V0,

        B=p[14],

        fit_flags=template.fit_flags.copy()

    )


# ==========================================================
# Legacy update
# ==========================================================

def update_triple_elements(
    template,
    parameter_vector
):
    """
    Update a legacy 15-parameter triple model.
    """

    return array_to_triple(

        parameter_vector,

        template

    )


# ==========================================================
# Legacy fit mask
# ==========================================================

def legacy_triple_fitted_mask(
    elements: TripleOrbitalElements
):
    """
    Return the 15-element fit mask used by the legacy
    ORBITT2 parameterization.

    This must remain separate from
    TripleOrbitalElements.fitted_mask(), which represents
    the complete 20-parameter ORBIT3 vector.
    """

    mask = np.array(

        [

            elements.fit_flags[name]

            for name
            in LEGACY_TRIPLE_PARAMETER_ORDER

        ],

        dtype=bool

    )


    if mask.size != 15:

        raise RuntimeError(

            "Internal legacy ORBITT2 fit mask must "
            f"contain 15 entries; received {mask.size}."

        )


    return mask


# ==========================================================
# Legacy fitted parameter names
# ==========================================================

def legacy_triple_fitted_parameter_names(
    elements: TripleOrbitalElements
):
    """
    Return the fitted parameter names for the legacy
    ORBITT2 representation.
    """

    mask = legacy_triple_fitted_mask(
        elements
    )


    return [

        name

        for name, fitted
        in zip(

            LEGACY_TRIPLE_PARAMETER_ORDER,

            mask

        )

        if fitted

    ]


# ==========================================================
# Legacy fitted parameters
# ==========================================================

def triple_fitted_parameters(
    elements
):
    """
    Return the fitted parameters of the legacy 15-element
    ORBITT2 representation.
    """

    full = triple_to_array(
        elements
    )


    mask = legacy_triple_fitted_mask(
        elements
    )


    if mask.size != full.size:

        raise RuntimeError(

            "Internal error: legacy ORBITT2 fit mask "
            f"contains {mask.size} entries, but the "
            f"legacy parameter vector contains "
            f"{full.size} entries."

        )


    return full[
        mask
    ]


# ==========================================================
# Legacy update fitted parameters
# ==========================================================

def update_triple_fitted_parameters(
    elements,
    fitted_values
):
    """
    Update only parameters marked as fitted in the legacy
    ORBITT2 representation.
    """

    full = triple_to_array(
        elements
    ).copy()


    mask = legacy_triple_fitted_mask(
        elements
    )


    fitted_values = np.asarray(

        fitted_values,

        dtype=float

    )


    expected = int(
        np.count_nonzero(mask)
    )


    if fitted_values.size != expected:

        raise ValueError(

            "The number of fitted values does not match "
            "the number of fitted legacy ORBITT2 "
            "parameters: "
            f"{fitted_values.size} != {expected}."

        )


    full[
        mask
    ] = fitted_values


    return array_to_triple(

        full,

        elements

    )


# ==========================================================
# Legacy generic fit residuals
# ==========================================================

def triple_fit_flag_residuals(
    fitted_values,
    elements,
    observations
):
    """
    Residual function using legacy ORBITT2 fit flags.
    """

    updated = (

        update_triple_fitted_parameters(

            elements,

            fitted_values

        )

    )


    return triple_visual_residuals(

        observations,

        updated

    )


# ==========================================================
# Legacy full residuals
# ==========================================================

def triple_fit_residuals(
    parameter_vector,
    template,
    observations
):
    """
    SciPy-compatible residual function for the legacy
    15-parameter triple model.
    """

    elements = update_triple_elements(

        template,

        parameter_vector

    )


    return triple_visual_residuals(

        observations,

        elements

    )


# ==========================================================
# Legacy B-only residuals
# ==========================================================

def triple_B_residuals(
    B,
    template,
    observations
):
    """
    Residuals when fitting only the legacy B parameter.
    """

    trial = TripleOrbitalElements(

        outer=template.outer.copy(),

        inner=template.inner.copy(),

        B=float(B[0]),

        fit_flags=template.fit_flags.copy()

    )


    return triple_visual_residuals(

        observations,

        trial

    )


# ==========================================================
# Legacy a2-only residuals
# ==========================================================

def triple_a2_residuals(
    a2,
    template,
    observations
):
    """
    Residuals when fitting only the legacy inner
    semi-major axis.
    """

    outer = template.outer.copy()

    inner = template.inner.copy()


    inner.a = float(
        a2[0]
    )


    trial = TripleOrbitalElements(

        outer=outer,

        inner=inner,

        B=template.B,

        fit_flags=template.fit_flags.copy()

    )


    return triple_visual_residuals(

        observations,

        trial

    )


# ==========================================================
# ORBIT3 -> array
# ==========================================================

def orbit3_to_array(
    elements: TripleOrbitalElements
):
    """
    Convert TripleOrbitalElements into the complete
    20-parameter ORBIT3 vector.

    Ordering
    --------
    P, T0, e, a, Omega, omega, i,
    K1, K2, f,
    P2, T02, e2, a2, Omega2, omega2, i2,
    K3, K4, V0
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
        # Inner spectroscopy / wobble
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
        # Outer spectroscopy
        # --------------------------------------------------

        elements.K3,
        elements.K4,
        elements.V0,

    ], dtype=float)


# ==========================================================
# ORBIT3 array -> Triple
# ==========================================================

def array_to_orbit3(
    p,
    template: TripleOrbitalElements
):
    """
    Build a TripleOrbitalElements object from the complete
    20-parameter ORBIT3 vector.
    """

    p = np.asarray(

        p,

        dtype=float

    )


    if p.size != 20:

        raise ValueError(

            "ORBIT3 parameter vector must contain "
            "exactly 20 parameters; "
            f"received {p.size}."

        )


    inner = template.inner.copy()

    outer = template.outer.copy()


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
    # Inner spectroscopy
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
    # Outer spectroscopy
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

        B=template.B,

        fit_flags=template.fit_flags.copy()

    )


# ==========================================================
# ORBIT3 fitted parameters
# ==========================================================

def orbit3_fitted_parameters(
    elements: TripleOrbitalElements
):
    """
    Return the fitted parameters of the complete
    20-parameter ORBIT3 representation.
    """

    full = orbit3_to_array(
        elements
    )


    mask = elements.fitted_mask()


    if mask.size != full.size:

        raise ValueError(

            "The ORBIT3 fit mask must contain exactly "
            f"{full.size} entries; "
            f"received {mask.size}."

        )


    return full[
        mask
    ]


# ==========================================================
# ORBIT3 update fitted parameters
# ==========================================================

def update_orbit3_fitted_parameters(
    elements: TripleOrbitalElements,
    fitted_values
):
    """
    Update only parameters marked as fitted in the modern
    ORBIT3 representation.
    """

    full = orbit3_to_array(
        elements
    ).copy()


    mask = elements.fitted_mask()


    if mask.size != full.size:

        raise ValueError(

            "The ORBIT3 fit mask must contain exactly "
            f"{full.size} entries; "
            f"received {mask.size}."

        )


    fitted_values = np.asarray(

        fitted_values,

        dtype=float

    )


    expected = int(
        np.count_nonzero(mask)
    )


    if fitted_values.size != expected:

        raise ValueError(

            "The number of fitted values does not match "
            "the number of fitted ORBIT3 parameters: "
            f"{fitted_values.size} != {expected}."

        )


    full[
        mask
    ] = fitted_values


    return array_to_orbit3(

        full,

        elements

    )


# ==========================================================
# ORBIT3 parameter names
# ==========================================================

def orbit3_fitted_parameter_names(
    elements: TripleOrbitalElements
):
    """
    Return fitted ORBIT3 parameter names.
    """

    return [

        name

        for name, fitted
        in zip(

            ORBIT3_PARAMETER_ORDER,

            elements.fitted_mask()

        )

        if fitted

    ]


# ==========================================================
# ORBIT3 residual update
# ==========================================================

def orbit3_fit_flag_elements(
    fitted_values,
    template
):
    """
    Construct an updated TripleOrbitalElements object from
    fitted ORBIT3 values.
    """

    return update_orbit3_fitted_parameters(

        template,

        fitted_values

    )


# ==========================================================
# Legacy triple orbit fitting
# ==========================================================

def fit_triple_orbit(
    elements,
    observations,
    method="trf"
):
    """
    Fit the legacy ORBITT2 hierarchical triple orbit.

    This function intentionally remains separate from the
    modern ORBIT3 fitting path.
    """

    p0 = triple_fitted_parameters(
        elements
    )


    bounds = build_triple_bounds(
        elements
    )


    return run_least_squares(

        initial_parameters=p0,

        residual_function=triple_fit_flag_residuals,

        residual_args=(

            elements,

            observations

        ),

        solution_builder=lambda p:

            update_triple_fitted_parameters(

                elements,

                p

            ),

        parameter_names=(

            legacy_triple_fitted_parameter_names(

                elements

            )

        ),

        fit_result_class=TripleFitResult,

        method=method,

        bounds=bounds

    )