"""
Base class for experiment workflows.

Experiments generate repeated FitResult objects that are later
collected by an ExperimentRunner.
"""

from abc import ABC, abstractmethod


class Experiment(ABC):
    """
    Abstract base class for experiments.
    """

    @abstractmethod
    def trial(self):
        """
        Execute one realization of the experiment.

        Returns
        -------
        FitResult
        """
        raise NotImplementedError