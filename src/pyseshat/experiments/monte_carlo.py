"""
Monte Carlo experiment.
"""

from dataclasses import dataclass

from pyseshat.experiments.experiment import (
    Experiment


)


@dataclass
class MonteCarloExperiment(Experiment):
    """
    Monte Carlo uncertainty propagation.

    Parameters
    ----------
    observations : object

    perturb : callable

    trial_function : callable
    """

    observations: object

    perturb: callable

    trial_function: callable

    def trial(self):
        """
        Execute one Monte Carlo realization.
        """

        perturbed = self.perturb(
            self.observations
        )

        return self.trial_function(
            perturbed
        )