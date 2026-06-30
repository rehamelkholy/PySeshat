from dataclasses import dataclass

from pyseshat.inference.diagnostics import (
    fit_summary
)


@dataclass
class BaseFitResult:
    """
    Base class shared by all orbit fit results.
    """

    elements: object

    optimizer_result: object

    residuals: object

    covariance: object

    uncertainties: dict

    # ======================================================
    # Standard diagnostics
    # ======================================================

    def summary(
        self,
        n_parameters
    ):
        """
        Compute standard fit diagnostics.

        Parameters
        ----------
        n_parameters : int

        Returns
        -------
        dict
        """

        return fit_summary(
            self.residuals,
            n_parameters
        )

    # ======================================================
    # Convenience properties
    # ======================================================

    @property
    def success(self):
        """
        Optimizer success flag.
        """

        return self.optimizer_result.success

    @property
    def message(self):
        """
        Optimizer termination message.
        """

        return self.optimizer_result.message