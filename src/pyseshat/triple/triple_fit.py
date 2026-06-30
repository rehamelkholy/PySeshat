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
# Triple parameter order
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
# Triple -> array
# ==========================================================

def triple_to_array(
    elements: TripleOrbitalElements
):
    """
    Convert TripleOrbitalElements
    into parameter vector.
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
    Build TripleOrbitalElements
    from parameter vector.
    """

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
    Alias matching binary API style.
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
    Return fitted parameter vector.
    """

    full = triple_to_array(
        elements
    )

    return full[
        elements.fitted_mask()
    ]

# ==========================================================
# Update fitted parameters only
# ==========================================================

def update_triple_fitted_parameters(
    elements,
    fitted_values
):
    """
    Update only parameters marked
    as fitted.
    """

    full = triple_to_array(
        elements
    ).copy()

    full[
        elements.fitted_mask()
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
    Residual function using fit_flags.
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

    template : TripleOrbitalElements

    observations : list[TripleObservation]

    Returns
    -------
    ndarray
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
        Length-1 parameter vector

    template : TripleOrbitalElements

    observations : list

    Returns
    -------
    ndarray
    """

    trial = TripleOrbitalElements(
        outer=template.outer.copy(),
        inner=template.inner.copy(),
        B=float(B[0])
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
    Residuals when fitting only inner semi-major axis.

    Parameters
    ----------
    a2 : ndarray
        Length-1 parameter vector

    template : TripleOrbitalElements

    observations : list[TripleObservation]

    Returns
    -------
    ndarray
    """

    outer = template.outer.copy()

    inner = template.inner.copy()

    inner.a = float(a2[0])

    trial = TripleOrbitalElements(
        outer=outer,
        inner=inner,
        B=template.B
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
    Fit hierarchical triple orbit.
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
            elements.fitted_parameter_names()
        ),

        fit_result_class=TripleFitResult,

        method=method,

        bounds=bounds
    )