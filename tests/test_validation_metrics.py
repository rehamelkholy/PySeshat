import numpy as np

from pyseshat.validation.metrics import (
    parameter_bias,
    absolute_error,
    relative_error,
    normalized_error
)

truth = np.array([
    10.0,
    20.0,
    30.0
])

estimate = np.array([
    10.2,
    19.5,
    31.5
])

sigma = np.array([
    0.2,
    0.5,
    1.5
])

print()

print("BIAS")
print("=" * 60)
print(
    parameter_bias(
        truth,
        estimate
    )
)

print()

print("ABSOLUTE ERROR")
print("=" * 60)
print(
    absolute_error(
        truth,
        estimate
    )
)

print()

print("RELATIVE ERROR")
print("=" * 60)
print(
    relative_error(
        truth,
        estimate
    )
)

print()

print("NORMALIZED ERROR")
print("=" * 60)
print(
    normalized_error(
        truth,
        estimate,
        sigma
    )
)