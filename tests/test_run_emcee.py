import numpy as np

from pyseshat.inference.mcmc_engine import (
    run_emcee
)


# ==========================================================
# Standard Gaussian target
# ==========================================================

def log_probability(
    x
):
    return -0.5 * np.sum(
        x**2
    )


result = run_emcee(

    log_probability_function=log_probability,

    log_probability_args=(),

    initial_parameters=np.array([

        0.0,

        0.0

    ]),

    covariance=np.eye(2),

    n_walkers=20,

    n_steps=500,

    burn_in=100,

    thin=2,

    rng=np.random.default_rng(42)
)

print()

print("SAMPLES")
print("=" * 60)

print(result["samples"].shape)

print()

print("LOG PROBABILITY")
print("=" * 60)

print(result["log_probability"].shape)

print()

print("ACCEPTANCE FRACTION")
print("=" * 60)

print(result["acceptance_fraction"])

print()

print("AUTOCORRELATION")
print("=" * 60)

print(result["autocorrelation_time"])