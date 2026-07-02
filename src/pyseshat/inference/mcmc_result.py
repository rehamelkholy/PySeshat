from dataclasses import dataclass

import numpy as np

from pyseshat.inference.base_fit_result import (
    BaseFitResult
)


@dataclass
class MCMCResult(BaseFitResult):
    """
    Container for Bayesian orbit-fitting results.

    In addition to the best-fit solution inherited from
    BaseFitResult, this class stores the posterior samples
    and MCMC diagnostics.
    """

    # ------------------------------------------------------
    # Posterior samples
    # ------------------------------------------------------

    samples: np.ndarray

    log_probability: np.ndarray

    parameter_names: list[str]

    # ------------------------------------------------------
    # Sampling diagnostics
    # ------------------------------------------------------

    acceptance_fraction: float | None = None

    autocorrelation_time: np.ndarray | None = None

    effective_sample_size: float | None = None

    burn_in: int = 0

    thin: int = 1

    # ------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------

    @property
    def n_samples(self):
        """
        Number of posterior samples.
        """

        return self.samples.shape[0]

    @property
    def n_parameters(self):
        """
        Number of fitted parameters.
        """

        return self.samples.shape[1]
    
    @property
    def posterior_median(self):
        """
        Posterior median for every fitted parameter.
        """

        return np.median(
            self.samples,
            axis=0
        )


    @property
    def credible_interval_16(self):
        """
        Lower 68% credible limit.
        """

        return np.percentile(
            self.samples,
            16,
            axis=0
        )


    @property
    def credible_interval_84(self):
        """
        Upper 68% credible limit.
        """

        return np.percentile(
            self.samples,
            84,
            axis=0
        )


    @property
    def posterior_std(self):
        """
        Posterior standard deviation.
        """

        return np.std(
            self.samples,
            axis=0,
            ddof=1
        )


    @property
    def maximum_posterior_index(self):
        """
        Index of the highest-posterior sample.
        """

        return np.argmax(
            self.log_probability
        )


    @property
    def maximum_posterior_sample(self):
        """
        Maximum-posterior sample.
        """

        return self.samples[
            self.maximum_posterior_index
        ]