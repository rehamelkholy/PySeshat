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

    parameters = (

        problem
        .parameters
        .fitted_parameters()

    )

    (

        primary_temperature,

        secondary_temperature,

        primary_radius,

        secondary_radius,

        ebv

    ) = parameters

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