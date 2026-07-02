import numpy as np

from pyseshat.inference.mcmc_engine import (
    initialize_walkers
)

rng = np.random.default_rng(42)

initial = np.array([

    10.0,

    20.0,

    30.0

])

covariance = np.array([

    [1.0, 0.2, 0.0],

    [0.2, 2.0, 0.1],

    [0.0, 0.1, 3.0]

])

# ==========================================================
# Covariance initialization
# ==========================================================

walkers = initialize_walkers(

    initial,

    covariance=covariance,

    n_walkers=32,

    rng=rng
)

print()

print("WITH COVARIANCE")
print("=" * 60)

print("Shape")

print(walkers.shape)

print()

print("Mean")

print(np.mean(

    walkers,

    axis=0

))

# ==========================================================
# Fallback initialization
# ==========================================================

bad_covariance = np.array([
    [1.0, 2.0],
    [2.0, 4.0]
])

initial = np.array([
    1.0,
    2.0
])

walkers = initialize_walkers(
    initial,
    covariance=bad_covariance,
    n_walkers=16,
    random_scale=1e-4,
    rng=rng
)

print()

print("FALLBACK")
print("=" * 60)

print("Shape")
print(walkers.shape)

print()

print("Maximum Perturbation")
print(
    np.max(
        np.abs(
            walkers - initial
        )
    )
)