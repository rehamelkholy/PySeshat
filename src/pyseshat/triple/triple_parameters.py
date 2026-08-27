"""
Backward-compatible parameter-vector API for triple orbits.

This module used to contain an independent re-implementation of
the legacy ORBITT2 and modern ORBIT3 parameter-vector conversion
logic, duplicating (and, in one case -- the legacy
``array_to_triple`` in ``triple_fit.py`` before it was fixed to
match this module's behavior -- silently diverging from)
``triple_fit.py``, which is the module actually wired into the
live fitting path (``orbit3_fit.py``, ``fit_triple_orbit``).

To remove that duplication risk, every function here is now a
thin, name-preserving wrapper delegating to the canonical
implementation in ``triple_fit.py`` (parameter ordering itself is
canonical in ``triple_elements.py``). Existing code importing from
``pyseshat.triple.triple_parameters`` keeps working unchanged; new
code should prefer importing directly from ``triple_fit.py``.
"""

from __future__ import annotations

from pyseshat.triple.triple_elements import (
    TripleOrbitalElements
)

from pyseshat.triple import triple_fit as _triple_fit

from pyseshat.triple.triple_fit import (
    LEGACY_TRIPLE_PARAMETER_ORDER,
    ORBIT3_PARAMETER_ORDER,
)


# ==========================================================
# Legacy Triple <-> array
# ==========================================================

def triple_to_array(
    elements: TripleOrbitalElements
):
    """
    Convert TripleOrbitalElements into the legacy
    15-parameter ORBITT2-style parameter vector.

    Delegates to ``triple_fit.triple_to_array`` -- see that
    function for the canonical implementation and ordering.
    """

    return _triple_fit.triple_to_array(
        elements
    )


def array_to_triple(
    parameter_vector,
    template: TripleOrbitalElements
):
    """
    Build TripleOrbitalElements from a legacy
    15-parameter ORBITT2-style vector.

    Delegates to ``triple_fit.array_to_triple`` -- see that
    function for the canonical implementation. K1, K2, f, K3,
    K4, and V0 (which are outside the legacy 15-parameter
    scope) are carried over from ``template``.
    """

    return _triple_fit.array_to_triple(
        parameter_vector,
        template
    )


# ==========================================================
# Modern ORBIT3 Triple <-> array
# ==========================================================

def triple_to_orbit3_array(
    elements: TripleOrbitalElements
):
    """
    Convert TripleOrbitalElements into the modern
    20-parameter ORBIT3 vector.

    Delegates to ``triple_fit.orbit3_to_array``.
    """

    return _triple_fit.orbit3_to_array(
        elements
    )


def orbit3_array_to_triple(
    parameter_vector,
    template: TripleOrbitalElements
):
    """
    Build TripleOrbitalElements from a modern
    20-parameter ORBIT3 vector.

    Delegates to ``triple_fit.array_to_orbit3``.
    """

    return _triple_fit.array_to_orbit3(
        parameter_vector,
        template
    )


# ==========================================================
# Modern fitted-parameter extraction / update
# ==========================================================

def orbit3_fitted_parameters(
    elements: TripleOrbitalElements
):
    """
    Return the fitted parameters from the modern
    ORBIT3 representation.

    Delegates to ``triple_fit.orbit3_fitted_parameters``.
    """

    return _triple_fit.orbit3_fitted_parameters(
        elements
    )


def update_orbit3_fitted_parameters(
    elements: TripleOrbitalElements,
    fitted_values
):
    """
    Update only parameters marked as fitted in the
    modern ORBIT3 representation.

    Delegates to ``triple_fit.update_orbit3_fitted_parameters``.
    """

    return _triple_fit.update_orbit3_fitted_parameters(
        elements,
        fitted_values
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
