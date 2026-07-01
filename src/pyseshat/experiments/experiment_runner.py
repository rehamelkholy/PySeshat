"""
Generic experiment runner.
"""

import numpy as np

from pyseshat.experiments.experiment_result import (
    ExperimentResult
)


class ExperimentRunner:
    """
    Execute repeated realizations of an Experiment.
    """

    def __init__(
        self,
        experiment,
        n_trials=100
    ):

        self.experiment = experiment

        self.n_trials = n_trials

    def run(self):
        """
        Execute the experiment repeatedly.

        Returns
        -------
        ExperimentResult
        """

        fit_results = []

        samples = []

        parameter_names = None

        for _ in range(self.n_trials):

            fit = self.experiment.trial()

            fit_results.append(fit)

            if fit.success:

                samples.append(
                    fit.elements.fitted_parameters()
                )

                if parameter_names is None:

                    parameter_names = (
                        fit.elements.fitted_parameter_names()
                    )

        if samples:

            samples = np.asarray(samples)

        else:

            samples = np.empty((0, 0))

        return ExperimentResult(

            parameter_names=parameter_names,

            samples=samples,

            fit_results=fit_results
        )