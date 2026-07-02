import numpy as np

from pyseshat.inference.priors import (
    log_uniform_prior,
    log_jeffreys_prior,
    log_isotropic_prior
)

print()

print("UNIFORM")
print("=" * 60)

print(
    log_uniform_prior(
        5.0,
        0.0,
        10.0
    )
)

print()

print("JEFFREYS")
print("=" * 60)

print(
    log_jeffreys_prior(
        10.0,
        1.0,
        100.0
    )
)

print()

print("ISOTROPIC")
print("=" * 60)

angles = [

    0,

    30,

    60,

    90,

    120,

    150,

    180

]

for angle in angles:

    print(

        angle,

        log_isotropic_prior(
            angle
        )

    )