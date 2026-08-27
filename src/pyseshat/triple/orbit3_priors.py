"""
Prior probability for Bayesian ORBIT3 hierarchical-triple
fitting.

This mirrors ``pyseshat.inference.priors.log_prior`` (the
binary/single-orbit default Keplerian prior), applied to both
the inner and outer ``OrbitalElements`` of a
``TripleOrbitalElements`` object, plus the ORBIT3-specific
wobble factor ``f`` and the four spectroscopic semi-amplitudes.

The bound choices below are deliberately identical to
``triple.orbit3_bounds.orbit3_full_bounds()`` -- the same
physical/numerical safeguards used by the deterministic ORBIT3
optimizer's bounded methods -- so that the MCMC prior support
and the least-squares optimizer bounds never disagree about
what counts as a physically admissible ORBIT3 solution.
"""

import numpy as np

from pyseshat.inference.priors import (
    log_uniform_prior,
    log_jeffreys_prior,
    log_isotropic_prior,
)


# ==========================================================
# Single-orbit (inner or outer) Keplerian prior
# ==========================================================

def _log_prior_single_orbit(
    elements,
    fit_flags,
    suffix
):
    """
    Keplerian prior contribution for one of ORBIT3's two
    orbits (inner or outer).

    Parameters
    ----------
    elements : OrbitalElements
        Either ``triple_elements.inner`` or
        ``triple_elements.outer``.

    fit_flags : dict
        The parent TripleOrbitalElements' fit_flags.

    suffix : str
        "" for the inner orbit, "2" for the outer orbit --
        matches the ORBIT3_PARAMETER_ORDER naming convention
        (e.g. fit_flags["a2"] for the outer semimajor axis).

    Returns
    -------
    float
    """

    logp = 0.0

    # ------------------------------------------------------
    # Period (Jeffreys / log-uniform scale prior)
    # ------------------------------------------------------

    logp += log_jeffreys_prior(
        elements.P,
        1e-12,
        1e8
    )

    # ------------------------------------------------------
    # Epoch of periastron
    # ------------------------------------------------------

    logp += log_uniform_prior(
        elements.T0,
        -1e9,
        1e9
    )

    # ------------------------------------------------------
    # Eccentricity
    #
    # Matches orbit3_bounds.orbit3_full_bounds()'s upper
    # limit of 0.999999, not the binary path's 0.99 -- the
    # two are independent physical-bounds conventions and
    # ORBIT3's own is authoritative here.
    # ------------------------------------------------------

    logp += log_uniform_prior(
        elements.e,
        0.0,
        0.999999
    )

    # ------------------------------------------------------
    # Semimajor axis (Jeffreys / log-uniform scale prior)
    # ------------------------------------------------------

    logp += log_jeffreys_prior(
        elements.a,
        1e-12,
        1e6
    )

    # ------------------------------------------------------
    # Inclination (isotropic orbital-plane prior)
    # ------------------------------------------------------

    logp += log_isotropic_prior(
        elements.i
    )

    # ------------------------------------------------------
    # Longitude of ascending node
    # ------------------------------------------------------

    logp += log_uniform_prior(
        elements.Omega,
        0.0,
        360.0
    )

    # ------------------------------------------------------
    # Argument of periastron
    # ------------------------------------------------------

    logp += log_uniform_prior(
        elements.omega,
        0.0,
        360.0
    )

    return logp


# ==========================================================
# Default ORBIT3 prior
# ==========================================================

def log_prior_orbit3(
    elements
):
    """
    Default Bayesian prior for a hierarchical-triple ORBIT3
    solution.

    Parameters
    ----------
    elements : TripleOrbitalElements

    Returns
    -------
    float
        Sum of the log-priors for the inner orbit, the outer
        orbit, and the ORBIT3-specific wobble/spectroscopic
        parameters participating in the current fit.

    Notes
    -----
    The wobble factor ``f`` is deliberately left unrestricted
    (flat/improper prior, contributing 0.0): the reference
    ORBIT3 implementation does not bound it, and the published
    HIP111805 solution has ``f = -0.330`` -- see
    ``orbit3_bounds.orbit3_full_bounds()`` for the identical
    reasoning on the deterministic-optimizer side.

    K1, K2, K3, and K4 are subject to positive support only
    when they are fitted parameters, matching
    ``inference.priors.log_prior``'s treatment of K1/K2 for
    the binary case. Fixed spectroscopic values (e.g.
    ZetaAqr's K1=K2=K3=K4=0.0, all unfitted, since it has no
    usable RV data) do not contribute to the prior.

    V0 is a signed location parameter and is unrestricted.
    """

    logp = 0.0

    # ------------------------------------------------------
    # Inner orbit
    # ------------------------------------------------------

    logp += _log_prior_single_orbit(
        elements.inner,
        elements.fit_flags,
        suffix=""
    )

    # ------------------------------------------------------
    # Outer orbit
    # ------------------------------------------------------

    logp += _log_prior_single_orbit(
        elements.outer,
        elements.fit_flags,
        suffix="2"
    )

    if not np.isfinite(logp):

        return -np.inf

    # ------------------------------------------------------
    # Spectroscopic semi-amplitudes
    # ------------------------------------------------------

    for name, value in (
        ("K1", elements.K1),
        ("K2", elements.K2),
        ("K3", elements.K3),
        ("K4", elements.K4),
    ):

        if elements.fit_flags.get(name, False):

            if value <= 0.0:

                return -np.inf

    # ------------------------------------------------------
    # Wobble factor f and systemic velocity V0
    #
    # Both unrestricted; see the docstring above.
    # ------------------------------------------------------

    return logp
