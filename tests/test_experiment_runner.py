import numpy as np

from pyseshat.experiments.experiment import (
    Experiment
)

from pyseshat.experiments.experiment_runner import (
    ExperimentRunner
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


class DummyExperiment(Experiment):

    def trial(self):

        return DummyFit()


runner = ExperimentRunner(

    DummyExperiment(),

    n_trials=5
)

result = runner.run()

print()

print("TRIALS")
print("=" * 60)

print(result.n_trials)

print()

print("PARAMETERS")
print("=" * 60)

print(result.parameter_names)

print()

print("SAMPLES")
print("=" * 60)

print(result.samples)

print()

print("SUCCESSFUL FITS")
print("=" * 60)

print(result.successful_fits)