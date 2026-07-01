from dataclasses import dataclass

import numpy as np


@dataclass
class ExperimentResult:
    """
    Base container for experiment outputs.

    This class stores the fitted parameter vectors obtained
    during an experiment together with simple statistical
    summaries.

    Parameters
    ----------
    parameter_names : list[str]

    samples : ndarray
        Shape = (n_trials, n_parameters)
    """

    parameter_names: list

    samples: np.ndarray

    fit_results: list | None = None

    # ------------------------------------------------------
    # Dimensions
    # ------------------------------------------------------

    @property
    def n_trials(self):
        """
        Number of experiment realizations.
        """

        return self.samples.shape[0]

    @property
    def n_parameters(self):
        """
        Number of fitted parameters.
        """

        return self.samples.shape[1]
    
        # ------------------------------------------------------
    # Successful fits
    # ------------------------------------------------------

    @property
    def successful_fits(self):
        """
        Number of successful optimization runs.
        """

        if self.fit_results is None:
            return None

        return sum(
            fit.success
            for fit in self.fit_results
        )

    # ------------------------------------------------------
    # Statistics
    # ------------------------------------------------------

    def mean(self):
        """
        Mean fitted parameters.
        """

        return np.mean(
            self.samples,
            axis=0
        )

    def std(self):
        """
        Standard deviation of fitted parameters.
        """

        return np.std(
            self.samples,
            axis=0,
            ddof=1
        )

    def covariance(self):
        """
        Sample covariance matrix.
        """

        return np.cov(
            self.samples,
            rowvar=False
        )

    def correlation(self):
        """
        Sample correlation matrix.
        """

        cov = self.covariance()

        sigma = np.sqrt(
            np.diag(cov)
        )

        return cov / np.outer(
            sigma,
            sigma
        )