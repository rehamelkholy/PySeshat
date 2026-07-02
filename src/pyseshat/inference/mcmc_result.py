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