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

    Parameters
    ----------
    initial_parameters : ndarray
        Best-fit parameter vector.

    covariance : ndarray or None
        Parameter covariance matrix.

    n_walkers : int

    random_scale : float
        Standard deviation used when covariance is
        unavailable.

    rng : numpy.random.Generator, optional

    Returns
    -------
    ndarray
        Initial walker positions with shape

        (n_walkers, n_parameters)
    """

    if rng is None:

        rng = np.random.default_rng()

    initial_parameters = np.asarray(
        initial_parameters,
        dtype=float
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

        # --------------------------------------------------
        # Require a finite, positive-definite covariance
        # --------------------------------------------------

        if (
            covariance.shape
            ==
            (n_parameters, n_parameters)
        ):

            try:

                np.linalg.cholesky(
                    covariance
                )

                walkers = rng.multivariate_normal(

                    mean=initial_parameters,

                    cov=covariance,

                    size=n_walkers
                )

                return walkers

            except np.linalg.LinAlgError:

                pass

    # ------------------------------------------------------
    # Isotropic fallback
    # ------------------------------------------------------

    perturbation = rng.normal(

        scale=random_scale,

        size=(
            n_walkers,
            n_parameters
        )
    )

    return (

        initial_parameters

        +

        perturbation
    )

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

    rng : numpy.random.Generator, optional

    Returns
    -------
    dict
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