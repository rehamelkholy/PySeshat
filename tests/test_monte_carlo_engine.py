import numpy as np

from pyseshat.experiments.experiment_runner import ExperimentRunner
from pyseshat.experiments.monte_carlo import (
    MonteCarloExperiment
)


class DummyElements:

    def fitted_parameters(self):

        return np.array(
            [1.0, 2.0]
        )

    def fitted_parameter_names(self):

        return [
            "A",
            "B"
        ]


class DummyFit:

    def __init__(self):

        self.success = True

        self.elements = DummyElements()


observations = np.array(
    [10.0, 20.0]
)


def perturb(obs):

    return obs + 1.0


def trial(obs):

    assert np.all(obs > observations)

    return DummyFit()


mc = MonteCarloExperiment(

    observations=observations,

    perturb=perturb,

    trial_function=trial
)

runner = ExperimentRunner(
    experiment=mc,
    n_trials=5,
)

result = runner.run()

print()

print("TRIALS")
print("=" * 60)

print(result.n_trials)

print()

print("SUCCESSFUL FITS")
print("=" * 60)

print(result.successful_fits)

print()

print("SAMPLE MATRIX")
print("=" * 60)

print(result.samples)