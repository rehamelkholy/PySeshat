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

    n_walkers: int = 0

    n_steps: int = 0

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
    
    @property
    def posterior_summary(self):
        """
        Posterior summary statistics.

        Returns
        -------
        list[dict]
            One dictionary per fitted parameter containing:

            - name
            - median
            - lower_68
            - upper_68
            - std
            - map
        """

        summary = []

        medians = self.posterior_median

        lower = self.credible_interval_16

        upper = self.credible_interval_84

        std = self.posterior_std

        map_sample = self.maximum_posterior_sample

        for i, name in enumerate(
            self.parameter_names
        ):

            summary.append({

                "name": name,

                "median": medians[i],

                "lower_68": lower[i],

                "upper_68": upper[i],

                "std": std[i],

                "map": map_sample[i]

            })

        return summary
    
    # ======================================================
    # Full text report
    # ======================================================

    def report(self):
        """
        Generate a formatted Bayesian fit report.

        Returns
        -------
        str
        """

        lines = []

        lines.append("")
        lines.append(
            "BAYESIAN FIT REPORT"
        )

        lines.append(
            "=" * 80
        )

        # --------------------------------------------------
        # Posterior summary
        # --------------------------------------------------

        lines.append("")
        lines.append(
            "Posterior Summary"
        )

        lines.append(
            "-" * 80
        )

        lines.append(

            f"{'Parameter':<12}"
            f"{'Median':>15}"
            f"{'-1σ':>15}"
            f"{'+1σ':>15}"

        )

        for item in self.posterior_summary:

            lower = (
                item["median"]
                -
                item["lower_68"]
            )

            upper = (
                item["upper_68"]
                -
                item["median"]
            )

            lines.append(

                f"{item['name']:<12}"

                f"{item['median']:15.8f}"

                f"{lower:15.8f}"

                f"{upper:15.8f}"

            )

        # --------------------------------------------------
        # Diagnostics
        # --------------------------------------------------

        lines.append("")
        lines.append(
            "Sampling Diagnostics"
        )

        lines.append(
            "-" * 80
        )

        lines.append(

            f"Acceptance Fraction : "
            f"{self.acceptance_fraction:.4f}"

        )

        if self.has_autocorrelation_time:

            lines.append(

                f"Minimum ESS         : "
                f"{self.minimum_effective_sample_size:.1f}"

            )

            lines.append(

                f"Converged          : "
                f"{self.converged}"

            )

        else:

            lines.append(
                "Autocorrelation    : Not available"
            )

            lines.append(
                "Minimum ESS        : Not available"
            )

            lines.append(
                "Converged          : False"
            )

        # --------------------------------------------------
        # MAP solution
        # --------------------------------------------------

        lines.append("")
        lines.append(
            "Maximum A Posteriori Solution"
        )

        lines.append(
            "-" * 80
        )

        for item in self.posterior_summary:

            lines.append(

                f"{item['name']:<12}"
                f"{item['map']:15.8f}"

            )

        return "\n".join(
            lines
        )
    
    @property
    def has_autocorrelation_time(self):
        """
        Whether an autocorrelation estimate is available.
        """

        return self.autocorrelation_time is not None


    @property
    def post_burn_in_steps(self):
        """
        Number of retained MCMC steps per walker
        before thinning.
        """

        return max(
            self.n_steps - self.burn_in,
            0
        )


    @property
    def retained_steps(self):
        """
        Number of retained steps per walker
        after thinning.
        """

        if self.post_burn_in_steps == 0:
            return 0

        return (
            self.post_burn_in_steps
            //
            self.thin
        )


    @property
    def effective_sample_size(self):
        """
        Effective sample size for each parameter.
        """

        if not self.has_autocorrelation_time:
            return None

        tau = np.asarray(
            self.autocorrelation_time,
            dtype=float
        )

        return (
            self.n_walkers
            *
            self.retained_steps
            /
            tau
        )


    @property
    def minimum_effective_sample_size(self):
        """
        Smallest effective sample size.
        """

        ess = self.effective_sample_size

        if ess is None:
            return None

        return float(
            np.min(ess)
        )


    @property
    def converged(self):
        """
        Simple convergence criterion.

        emcee recommends chains longer than
        approximately 50 autocorrelation times.
        """

        if not self.has_autocorrelation_time:
            return False

        tau_max = np.max(
            self.autocorrelation_time
        )

        return (
            self.post_burn_in_steps
            >=
            50.0 * tau_max
        )