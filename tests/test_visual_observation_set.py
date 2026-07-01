import numpy as np

from pyseshat.experiments.observation_set import (
    VisualObservationSet
)

from pyseshat.experiments.synthetic import (
    perturb_xy
)

obs = VisualObservationSet(

    epoch=np.array([

        2000.0,

        2001.0

    ]),

    x=np.array([

        1.0,

        2.0

    ]),

    y=np.array([

        3.0,

        4.0

    ]),

    sigma_x=np.array([

        0.01,

        0.01

    ]),

    sigma_y=np.array([

        0.02,

        0.02

    ])
)

new_obs = perturb_xy(

    obs,

    rng=np.random.default_rng(42)
)

print()

print("TYPE")
print("=" * 60)

print(type(new_obs).__name__)

print()

print("EPOCH")
print("=" * 60)

print(new_obs.epoch)

print()

print("X")
print("=" * 60)

print(new_obs.x)

print()

print("Y")
print("=" * 60)

print(new_obs.y)

print()

print("SIGMA X")
print("=" * 60)

print(new_obs.sigma_x)

print()

print("SIGMA Y")
print("=" * 60)

print(new_obs.sigma_y)