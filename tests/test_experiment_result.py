import numpy as np

from pyseshat.experiments.experiment_result import (
    ExperimentResult
)

samples = np.array(

    [

        [1.0, 2.0],

        [2.0, 4.0],

        [3.0, 6.0],

        [4.0, 8.0]

    ]

)

class DummyFit:

    def __init__(self, success):

        self.success = success


fits = [

    DummyFit(True),

    DummyFit(True),

    DummyFit(False),

    DummyFit(True)

]


result = ExperimentResult(

    parameter_names=[

        "A",

        "B"

    ],

    samples=samples,

    fit_results=fits

)

print()

print("TRIALS")
print("=" * 60)

print(result.n_trials)

print()

print("PARAMETERS")
print("=" * 60)

print(result.n_parameters)

print()

print("MEAN")
print("=" * 60)

print(result.mean())

print()

print("STD")
print("=" * 60)

print(result.std())

print()

print("COVARIANCE")
print("=" * 60)

print(result.covariance())

print()

print("CORRELATION")
print("=" * 60)

print(result.correlation())

print()

print("SUCCESSFUL FITS")
print("=" * 60)

print(result.successful_fits)