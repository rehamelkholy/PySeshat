import numpy as np
import emcee


# ==========================================================
# Walker initialization
# ==========================================================

def initialize_walkers(
    initial_parameters,
    covariance=None,
    n_walkers=32,
    random_scale=1e-4,
    rng=None
):
    """
    Initialize MCMC walkers.

    Walker initialization uses the best-fit parameter vector
    and, when available, its covariance matrix.

    The preferred initialization is a multivariate Gaussian
    defined by the full covariance matrix.

    If the covariance matrix cannot be factorized as a
    positive-definite matrix, its diagonal variances are used
    to retain parameter-specific uncertainty scales.

    If no usable covariance information is available, a
    relative perturbation scale is used:

        sigma_j = random_scale * max(abs(theta_j), 1)

    This avoids applying the same absolute perturbation to
    orbital parameters with very different physical scales.

    Parameters
    ----------
    initial_parameters : ndarray
        Best-fit parameter vector.

    covariance : ndarray or None
        Parameter covariance matrix.

    n_walkers : int

    random_scale : float
        Relative standard deviation used when covariance
        information is unavailable.

    rng : numpy.random.Generator, optional

    Returns
    -------
    ndarray
        Initial walker positions with shape

        (n_walkers, n_parameters)

    Raises
    ------
    ValueError
        If the initial parameter vector contains non-finite
        values or if n_walkers is invalid.
    """

    if rng is None:

        rng = np.random.default_rng()

    initial_parameters = np.asarray(
        initial_parameters,
        dtype=float
    )

    if initial_parameters.ndim != 1:

        raise ValueError(
            "Initial parameters must be a one-dimensional array."
        )

    if not np.all(
        np.isfinite(
            initial_parameters
        )
    ):

        raise ValueError(
            "Initial parameters must contain only finite values."
        )

    if n_walkers < 2:

        raise ValueError(
            "n_walkers must be at least 2."
        )

    if random_scale <= 0.0:

        raise ValueError(
            "random_scale must be positive."
        )

    n_parameters = len(
        initial_parameters
    )

    # ------------------------------------------------------
    # Covariance initialization
    # ------------------------------------------------------

    if covariance is not None:

        covariance = np.asarray(
            covariance,
            dtype=float
        )

        if (
            covariance.shape
            ==
            (n_parameters, n_parameters)
            and
            np.all(
                np.isfinite(
                    covariance
                )
            )
        ):

            # --------------------------------------------------
            # Enforce numerical symmetry.
            #
            # A covariance matrix is mathematically symmetric,
            # but numerical operations can introduce tiny
            # asymmetries.
            # --------------------------------------------------

            covariance = (
                0.5
                *
                (
                    covariance
                    +
                    covariance.T
                )
            )

            # --------------------------------------------------
            # Preferred full-covariance initialization.
            # --------------------------------------------------

            try:

                np.linalg.cholesky(
                    covariance
                )

                walkers = (
                    rng.multivariate_normal(

                        mean=initial_parameters,

                        cov=covariance,

                        size=n_walkers,

                        check_valid="raise"

                    )
                )

                if np.all(
                    np.isfinite(
                        walkers
                    )
                ):

                    return walkers

            except np.linalg.LinAlgError:

                pass

            # --------------------------------------------------
            # Diagonal covariance fallback.
            #
            # If the full covariance is singular or otherwise
            # unsuitable for Cholesky factorization, retain
            # the individual parameter uncertainties whenever
            # they are available.
            # --------------------------------------------------

            diagonal = np.diag(
                covariance
            )

            valid_diagonal = (
                np.isfinite(
                    diagonal
                )
                &
                (
                    diagonal > 0.0
                )
            )

            if np.all(
                valid_diagonal
            ):

                standard_deviations = np.sqrt(
                    diagonal
                )

                perturbation = (
                    rng.normal(
                        loc=0.0,
                        scale=standard_deviations,
                        size=(
                            n_walkers,
                            n_parameters
                        )
                    )
                )

                walkers = (
                    initial_parameters
                    +
                    perturbation
                )

                if np.all(
                    np.isfinite(
                        walkers
                    )
                ):

                    return walkers

    # ------------------------------------------------------
    # Relative isotropic fallback
    # ------------------------------------------------------
    #
    # The previous implementation used one absolute scale
    # for every parameter. That is inappropriate for orbital
    # parameters with very different numerical scales.
    #
    # The fallback therefore uses a parameter-scaled
    # perturbation:
    #
    #     sigma_j =
    #         random_scale * max(abs(theta_j), 1)
    #
    # This preserves the existing random_scale parameter while
    # making it dimensionally sensible as a relative scale.
    # ------------------------------------------------------

    parameter_scales = (
        random_scale
        *
        np.maximum(
            np.abs(
                initial_parameters
            ),
            1.0
        )
    )

    perturbation = (
        rng.normal(
            loc=0.0,
            scale=parameter_scales,
            size=(
                n_walkers,
                n_parameters
            )
        )
    )

    walkers = (
        initial_parameters
        +
        perturbation
    )

    return walkers


# ==========================================================
# Generic MCMC engine
# ==========================================================

def run_emcee(
    *,
    log_probability_function,
    log_probability_args,
    initial_parameters,
    covariance=None,
    n_walkers=32,
    n_steps=5000,
    burn_in=1000,
    thin=1,
    random_scale=1e-4,
    rng=None
):
    """
    Generic emcee sampling engine.

    Parameters
    ----------
    log_probability_function : callable

    log_probability_args : tuple

    initial_parameters : ndarray

    covariance : ndarray or None

    n_walkers : int

    n_steps : int

    burn_in : int

    thin : int

    random_scale : float
        Relative fallback scale used when covariance
        information is unavailable.

    rng : numpy.random.Generator, optional

    Returns
    -------
    dict
        Dictionary containing the sampler, flattened
        posterior samples, log probabilities, acceptance
        fraction, and autocorrelation time.
    """

    initial_state = initialize_walkers(

        initial_parameters,

        covariance=covariance,

        n_walkers=n_walkers,

        random_scale=random_scale,

        rng=rng
    )

    n_dim = len(
        initial_parameters
    )

    sampler = emcee.EnsembleSampler(

        n_walkers,

        n_dim,

        log_probability_function,

        args=log_probability_args
    )

    sampler.run_mcmc(

        initial_state,

        n_steps,

        progress=False
    )

    samples = sampler.get_chain(

        discard=burn_in,

        thin=thin,

        flat=True
    )

    log_probability = sampler.get_log_prob(

        discard=burn_in,

        thin=thin,

        flat=True
    )

    try:

        tau = sampler.get_autocorr_time()

    except Exception:

        tau = None

    return {

        "sampler": sampler,

        "samples": samples,

        "log_probability": log_probability,

        "acceptance_fraction": float(

            sampler.acceptance_fraction.mean()

        ),

        "autocorrelation_time": tau
    }