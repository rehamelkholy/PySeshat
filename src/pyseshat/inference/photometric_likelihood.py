import numpy as np

from pyseshat.photometry.model import (
    synthetic_binary_magnitudes
)

from pyseshat.photometry.residuals import (
    photometric_residual
)

from pyseshat.inference.photometric_problem import (
    PhotometricInferenceProblem
)

from pyseshat.inference.photometric_context import (
    PhotometricContext
)

from pyseshat.photometry.parameters import (
    PhotometricParameters,
    update_parameters,
)

def photometric_residuals_from_problem(
    problem
):
    """
    Calculate the standardized photometric residual vector
    from a canonical photometric inference problem.

    Parameters
    ----------
    problem : PhotometricInferenceProblem
        Canonical photometric inference problem.

    Returns
    -------
    numpy.ndarray
        Standardized residual vector

            (model - observed) / sigma.
    """

    if not isinstance(

        problem,

        PhotometricInferenceProblem

    ):

        raise TypeError(

            "problem must be a "
            "PhotometricInferenceProblem instance."

        )

    # ------------------------------------------------------
    # Read the FULL current parameter state directly off the
    # PhotometricParameters instance, not
    # ``problem.parameters.fitted_parameters()``.
    #
    # ``fitted_parameters()`` returns only the currently-FITTED
    # subset (as of the fit-flags mechanism added 2026-08-27,
    # see photometry/parameters.py) -- it exists purely to talk
    # to the optimizer/sampler (initial guess, trial-vector
    # shape), the same way OrbitalElements.fitted_parameters()
    # is never used to build the physical orbit model directly
    # (the residual functions in inference/likelihood.py always
    # read elements.<name> attributes instead). A FIXED
    # parameter still has a real, current value that must
    # participate in the synthetic-photometry model -- it's just
    # not part of the vector the optimizer is exploring -- so
    # building the model from fitted_parameters() alone would
    # silently drop every fixed parameter's contribution the
    # moment fewer than all 5 are fitted.
    # ------------------------------------------------------

    p = problem.parameters

    primary_temperature = p.primary_temperature

    secondary_temperature = p.secondary_temperature

    primary_radius = p.primary_radius

    secondary_radius = p.secondary_radius

    ebv = p.ebv

    context = problem.context

    model_magnitudes = synthetic_binary_magnitudes(

        atmosphere_grid=context.atmosphere_grid,

        passbands=context.passbands,

        zero_points=context.zero_points,

        primary_temperature=primary_temperature,

        secondary_temperature=secondary_temperature,

        primary_logg=context.primary_logg,

        secondary_logg=context.secondary_logg,

        metallicity=context.metallicity,

        primary_radius=primary_radius,

        secondary_radius=secondary_radius,

        distance=context.distance,

        ebv=ebv,

        rv=context.rv

    )

    residuals = []

    used_passbands = set()

    for measurement in context.measurements:

        if measurement.passband in used_passbands:

            raise ValueError(

                "Duplicate photometric measurement "
                f"for passband: "
                f"{measurement.passband}"

            )

        used_passbands.add(

            measurement.passband

        )

        if measurement.passband not in model_magnitudes:

            raise ValueError(

                "Missing model magnitude for "
                f"passband: "
                f"{measurement.passband}"

            )

        residuals.append(

            photometric_residual(

                model_magnitude=(

                    model_magnitudes[
                        measurement.passband
                    ]

                ),

                measurement=measurement

            )

        )

    return np.asarray(

        residuals,

        dtype=float

    )


# ==========================================================
# Gaussian-prior pseudo-residuals (least-squares path)
# ==========================================================
#
# scipy.optimize.least_squares only ever minimizes a sum of
# squared residuals -- it has no separate concept of a Bayesian
# prior the way the MCMC path's log-posterior does (see
# inference.priors.log_photometric_prior, which already picks up
# ``parameters.priors`` directly). To give a Gaussian prior
# (mean, sigma) the same *(value - mean) / sigma)^2* contribution
# to the least-squares objective that it would contribute to
# -2*log-posterior, this appends one extra "pseudo-residual"
# per parameter with a prior, standard practice for expressing
# Tikhonov/Gaussian regularization inside a nonlinear
# least-squares solve. These are NOT real photometric data --
# see photometric_residuals_with_priors_from_problem()'s
# docstring for how callers should keep the two apart when
# reporting chi-square/degrees-of-freedom.

def photometric_prior_residuals_from_problem(
    problem
):
    """
    Calculate the Gaussian-prior pseudo-residual vector for a
    canonical photometric inference problem's
    ``parameters.priors``.

    Parameters
    ----------
    problem : PhotometricInferenceProblem

    Returns
    -------
    numpy.ndarray
        One entry per ``(name, (mean, sigma))`` in
        ``problem.parameters.priors``, each
        ``(value - mean) / sigma``. Empty when there are no
        priors.
    """

    parameters = problem.parameters

    residuals = []

    for name, (mean, sigma) in parameters.priors.items():

        if sigma <= 0.0:

            raise ValueError(

                f"Prior sigma for '{name}' must be positive "
                f"(got {sigma})."

            )

        value = getattr(
            parameters,
            name
        )

        residuals.append(
            (value - mean) / sigma
        )

    return np.asarray(

        residuals,

        dtype=float

    )


def photometric_residuals_with_priors_from_problem(
    problem
):
    """
    Calculate the combined residual vector actually minimized by
    the least-squares photometric optimizer: the real
    photometric measurement residuals followed by any Gaussian-
    prior pseudo-residuals.

    This is deliberately a SEPARATE function from
    ``photometric_residuals_from_problem()`` (which stays
    measurement-only) rather than folding prior residuals into
    it directly, so that callers who need a clean, literature-
    comparable data chi-square/RMS/degrees-of-freedom (e.g.
    ``validation/benchmark_runner.py``'s per-system statistics)
    can always recompute pure measurement residuals fresh via
    ``photometric_residuals_from_problem(fitted_problem)``,
    independent of whatever this function's optimizer-facing
    vector happened to include.

    Parameters
    ----------
    problem : PhotometricInferenceProblem

    Returns
    -------
    numpy.ndarray
    """

    return np.concatenate(
        (
            photometric_residuals_from_problem(
                problem
            ),
            photometric_prior_residuals_from_problem(
                problem
            ),
        )
    )


def photometric_residuals(
    parameters,
    atmosphere_grid,
    passbands,
    zero_points,
    measurements,
    primary_logg,
    secondary_logg,
    metallicity,
    distance,
    rv=3.1
):
    """
    Calculate the standardized residual vector for
    binary-star photometric fitting.

    Parameters
    ----------
    parameters : array_like
        Model parameter vector

            [
                primary_temperature,
                secondary_temperature,
                primary_radius,
                secondary_radius,
                ebv
            ]

    atmosphere_grid
        Stellar atmosphere grid.

    passbands : mapping
        Mapping of passband names to
        PhotometricPassband objects.

    zero_points : mapping
        Mapping of passband names to
        photometric zero points.

    measurements : iterable of PhotometricMeasurement
        Observed photometric measurements.

    primary_logg : float
        Fixed primary surface gravity.

    secondary_logg : float
        Fixed secondary surface gravity.

    metallicity : float
        Fixed metallicity.

    distance : float
        Distance in parsecs.

    rv : float, default=3.1
        Ratio of total-to-selective extinction.

    Returns
    -------
    numpy.ndarray
        Vector of standardized residuals

            (model - observed) / sigma.
    """

    parameters = np.asarray(

        parameters,

        dtype=float

    )

    if parameters.ndim != 1 or parameters.size != 5:

        raise ValueError(

            "parameters must contain exactly "
            "five values."

        )

    problem = PhotometricInferenceProblem(

        parameters=update_parameters(

            PhotometricParameters(

                primary_temperature=float(parameters[0]),

                secondary_temperature=float(parameters[1]),

                primary_radius=float(parameters[2]),

                secondary_radius=float(parameters[3]),

                ebv=float(parameters[4])

            ),

            parameters

        ),

        context=PhotometricContext(

            atmosphere_grid=atmosphere_grid,

            passbands=passbands,

            zero_points=zero_points,

            measurements=measurements,

            primary_logg=primary_logg,

            secondary_logg=secondary_logg,

            metallicity=metallicity,

            distance=distance,

            rv=rv

        )

    )

    return photometric_residuals_from_problem(

        problem

    )


def chi_square(
    residuals
):
    """
    Calculate chi-square from a residual vector.
    """

    residuals = np.asarray(
        residuals,
        dtype=float
    )

    return float(
        np.sum(
            residuals ** 2
        )
    )


def reduced_chi_square(
    residuals,
    n_parameters
):
    """
    Calculate the reduced chi-square statistic.
    """

    residuals = np.asarray(
        residuals,
        dtype=float
    )

    degrees_of_freedom = (
        residuals.size
        -
        n_parameters
    )

    if degrees_of_freedom <= 0:

        raise ValueError(
            "The number of degrees of freedom "
            "must be positive."
        )

    return (

        chi_square(
            residuals
        )

        /

        degrees_of_freedom

    )

def photometric_log_likelihood_from_problem(
    problem
):
    """
    Evaluate the Gaussian photometric log-likelihood
    for a canonical photometric inference problem.

    Parameters
    ----------
    problem : PhotometricInferenceProblem
        Canonical photometric inference problem.

    Returns
    -------
    float
        Gaussian log-likelihood up to an additive
        normalization constant.
    """

    if not isinstance(

        problem,

        PhotometricInferenceProblem

    ):

        raise TypeError(

            "problem must be a "
            "PhotometricInferenceProblem instance."

        )

    residuals = (

        photometric_residuals_from_problem(

            problem

        )

    )

    return log_likelihood(

        residuals

    )

def log_likelihood(
    residuals
):
    """
    Gaussian log-likelihood.

    Parameters
    ----------
    residuals : array_like
        Standardized residual vector.

    Returns
    -------
    float
        Gaussian log likelihood up to an additive
        normalization constant.
    """

    return (

        -0.5

        *

        chi_square(
            residuals
        )

    )