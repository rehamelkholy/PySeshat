import numpy as np

from pyseshat.experiments.monte_carlo import (
    MonteCarloExperiment,
)
from pyseshat.experiments.experiment_runner import (
    ExperimentRunner,
)


class DummyElements:

    def fitted_parameters(self):
        return np.array([1.0, 2.0])

    def fitted_parameter_names(self):
        return ["A", "B"]


class DummyFit:

    def __init__(self):
        self.success = True
        self.elements = DummyElements()


def test_experiment_runner():

    observations = np.array([10.0, 20.0])

    def perturb(obs):
        return obs + 1.0

    def trial_function(obs):

        assert np.all(obs > observations)

        return DummyFit()

    experiment = MonteCarloExperiment(
        observations=observations,
        perturb=perturb,
        trial_function=trial_function,
    )

    runner = ExperimentRunner(
        experiment,
        n_trials=5,
    )

    result = runner.run()

    assert result.n_trials == 5
    assert result.successful_fits == 5
    assert result.samples.shape == (5, 2)