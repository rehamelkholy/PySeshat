import numpy as np


from pyseshat.triple.triple_elements import (
    TripleOrbitalElements
)


from pyseshat.triple.triple_residuals import (
    triple_visual_residuals
)


from pyseshat.triple.triple_bounds import (
    build_triple_bounds
)


from pyseshat.triple.triple_fit_result import (
    TripleFitResult
)


from pyseshat.inference.optimizer_engine import (
    run_least_squares
)


# ==========================================================
# Legacy ORBITT2 parameter order
# ==========================================================

TRIPLE_PARAMETER_ORDER = [

    # outer orbit

    "P",
    "T0",
    "e",
    "a",
    "Omega",
    "omega",
    "i",

    # inner orbit

    "P2",
    "T02",
    "e2",
    "a2",
    "Omega2",
    "omega2",
    "i2",

    # scale

    "B"

]


# ==========================================================
# Legacy fitted-parameter mask
# ==========================================================

def _legacy_fitted_mask(
    elements: TripleOrbitalElements
):
    """
    Return the fitted-parameter mask used by the legacy
    ORBITT2 triple fitter.

    The legacy fitter operates on the 15-parameter vector
    defined by TRIPLE_PARAMETER_ORDER.

    This is intentionally independent of
    TripleOrbitalElements.fitted_mask(), because the latter
    now represents the expanded ORBIT3 parameter space.
    """

    return np.array([

        # --------------------------------------------------
        # Outer orbit
        # --------------------------------------------------

        elements.fit_flags["P"],
        elements.fit_flags["T0"],
        elements.fit_flags["e"],
        elements.fit_flags["a"],
        elements.fit_flags["Omega"],
        elements.fit_flags["omega"],
        elements.fit_flags["i"],

        # --------------------------------------------------
        # Inner orbit
        # --------------------------------------------------

        elements.fit_flags["P2"],
        elements.fit_flags["T02"],
        elements.fit_flags["e2"],
        elements.fit_flags["a2"],
        elements.fit_flags["Omega2"],
        elements.fit_flags["omega2"],
        elements.fit_flags["i2"],

        # --------------------------------------------------
        # Legacy scale
        # --------------------------------------------------

        elements.fit_flags["B"]

    ], dtype=bool)


# ==========================================================
# Legacy fitted-parameter names
# ==========================================================

def _legacy_fitted_parameter_names(
    elements: TripleOrbitalElements
):
    """
    Return fitted parameter names in legacy ORBITT2 order.
    """

    mask = _legacy_fitted_mask(
        elements
    )


    return [

        name

        for name, fitted

        in zip(
            TRIPLE_PARAMETER_ORDER,
            mask
        )

        if fitted

    ]


# ==========================================================
# Triple -> array
# ==========================================================

def triple_to_array(
    elements: TripleOrbitalElements
):
    """
    Convert TripleOrbitalElements into the legacy
    15-parameter ORBITT2 vector.

    The ordering is:

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

        elements.B

    ], dtype=float)


# ==========================================================
# Array -> triple
# ==========================================================

def array_to_triple(
    p,
    template: TripleOrbitalElements
):
    """
    Build TripleOrbitalElements from the legacy
    15-parameter ORBITT2 vector.

    Parameters
    ----------
    p : array-like
        Fifteen legacy triple parameters.

    template : TripleOrbitalElements
        Template providing the non-legacy parameters
        introduced for ORBIT3.

    Returns
    -------
    TripleOrbitalElements
    """

    p = np.asarray(
        p,
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
    # Normalize angular elements.
    # ------------------------------------------------------

    outer.normalize_angles()

    inner.normalize_angles()


    # ------------------------------------------------------
    # Preserve the complete fit-flag dictionary.
    #
    # The legacy fitter simply ignores the new ORBIT3 flags.
    # ------------------------------------------------------

    return TripleOrbitalElements(

        outer=outer,

        inner=inner,

        B=p[14],

        fit_flags=template.fit_flags.copy()

    )


# ==========================================================
# Update triple elements
# ==========================================================

def update_triple_elements(
    template,
    parameter_vector
):
    """
    Alias matching the binary API style.

    This remains a legacy 15-parameter operation.
    """

    return array_to_triple(

        parameter_vector,

        template

    )


# ==========================================================
# Extract fitted parameters
# ==========================================================

def triple_fitted_parameters(
    elements
):
    """
    Return fitted parameters for the legacy ORBITT2
    15-parameter representation.
    """

    full = triple_to_array(
        elements
    )


    mask = _legacy_fitted_mask(
        elements
    )


    return full[
        mask
    ]


# ==========================================================
# Update fitted parameters only
# ==========================================================

def update_triple_fitted_parameters(
    elements,
    fitted_values
):
    """
    Update only parameters marked as fitted in the legacy
    ORBITT2 parameter representation.
    """

    full = triple_to_array(
        elements
    ).copy()


    mask = _legacy_fitted_mask(
        elements
    )


    fitted_values = np.asarray(
        fitted_values,
        dtype=float
    )


    expected = int(
        np.count_nonzero(
            mask
        )
    )


    if fitted_values.shape != (
        expected,
    ):

        raise ValueError(

            "Number of fitted triple parameters "
            "does not match the number of parameters "
            "marked as fitted."

        )


    full[
        mask
    ] = fitted_values


    return array_to_triple(

        full,

        elements

    )


# ==========================================================
# Generic triple fit residuals
# ==========================================================

def triple_fit_flag_residuals(
    fitted_values,
    elements,
    observations
):
    """
    Residual function using the legacy fit flags.
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
# Triple fit residuals
# ==========================================================

def triple_fit_residuals(
    parameter_vector,
    template,
    observations
):
    """
    SciPy-compatible residual function.

    Parameters
    ----------
    parameter_vector : ndarray
        Complete legacy 15-parameter vector.

    template : TripleOrbitalElements
        Template triple system.

    observations : list
        Astrometric triple observations.

    Returns
    -------
    ndarray
        Visual-orbit residuals.
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
# B-only residuals
# ==========================================================

def triple_B_residuals(
    B,
    template,
    observations
):
    """
    Residuals when fitting only B.

    Parameters
    ----------
    B : ndarray
        Length-1 parameter vector.

    template : TripleOrbitalElements

    observations : list

    Returns
    -------
    ndarray
    """

    B = np.asarray(
        B,
        dtype=float
    )


    if B.shape != (1,):

        raise ValueError(
            "B must contain exactly one value."
        )


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
# a2-only residuals
# ==========================================================

def triple_a2_residuals(
    a2,
    template,
    observations
):
    """
    Residuals when fitting only the inner semi-major axis.

    Parameters
    ----------
    a2 : ndarray
        Length-1 parameter vector.

    template : TripleOrbitalElements

    observations : list

    Returns
    -------
    ndarray
    """

    a2 = np.asarray(
        a2,
        dtype=float
    )


    if a2.shape != (1,):

        raise ValueError(
            "a2 must contain exactly one value."
        )


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
# Triple orbit fitting
# ==========================================================

def fit_triple_orbit(
    elements,
    observations,
    method="trf"
):
    """
    Fit a hierarchical triple orbit using the legacy
    ORBITT2 astrometric parameterization.

    This function deliberately remains a 15-parameter
    visual-orbit fitter. ORBIT3 spectroscopic parameters
    are not fitted here yet.
    """

    # ------------------------------------------------------
    # Initial fitted parameter vector.
    # ------------------------------------------------------

    p0 = triple_fitted_parameters(
        elements
    )


    # ------------------------------------------------------
    # Bounds.
    #
    # build_triple_bounds() is part of the existing legacy
    # triple-fitting pathway. We pass the same elements but
    # ensure that the parameter vector used by this function
    # remains the 15-parameter legacy vector.
    # ------------------------------------------------------

    bounds = build_triple_bounds(
        elements
    )


    # ------------------------------------------------------
    # Parameter names.
    #
    # Do not use elements.fitted_parameter_names() here:
    # that method now represents the expanded ORBIT3 model.
    # ------------------------------------------------------

    parameter_names = (
        _legacy_fitted_parameter_names(
            elements
        )
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

        parameter_names=parameter_names,

        fit_result_class=TripleFitResult,

        method=method,

        bounds=bounds

    )