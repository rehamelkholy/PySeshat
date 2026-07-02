import numpy as np

from pyseshat.io.orbit_parser import (
    parse_orbit_file
)

from pyseshat.inference.optimizer import (
    fit_visual_orbit_mcmc
)

# ==========================================================
# Load benchmark
# ==========================================================

parsed = parse_orbit_file(
    "validation/benchmark_systems/FIN379/FIN379.inp"
)

elements = parsed["elements"]

visual = parsed["visual_data"]

rv_data = parsed["rv_data"]

# ==========================================================
# Observations
# ==========================================================

t = visual[:,0]

theta = np.deg2rad(
    visual[:,1]
)

rho = visual[:,2]

sigma = visual[:,3]

X_obs = rho*np.cos(theta)

Y_obs = rho*np.sin(theta)

# ==========================================================
# Bayesian fit
# ==========================================================

posterior = fit_visual_orbit_mcmc(

    elements,

    t,

    X_obs,

    Y_obs,

    sigma,

    sigma,

    rv_data=rv_data,

    n_walkers=20,

    n_steps=500,

    burn_in=100,

    thin=2,

    rng=np.random.default_rng(42)
)

# ==========================================================
# Report
# ==========================================================

print()

print("POSTERIOR MEDIANS")
print("=" * 70)

for name, value in zip(

    posterior.parameter_names,

    posterior.posterior_median

):

    print(
        f"{name:10s} {value:.8f}"
    )

print()

print("68% CREDIBLE INTERVALS")
print("=" * 70)

for name, lo, hi in zip(

    posterior.parameter_names,

    posterior.credible_interval_16,

    posterior.credible_interval_84

):

    print(
        f"{name:10s} [{lo:.8f}, {hi:.8f}]"
    )

print()

print("POSTERIOR STANDARD DEVIATION")
print("=" * 70)

for name, std in zip(

    posterior.parameter_names,

    posterior.posterior_std

):

    print(
        f"{name:10s} {std:.8f}"
    )

print()

print("MAP SAMPLE")
print("=" * 70)

for name, value in zip(

    posterior.parameter_names,

    posterior.maximum_posterior_sample

):

    print(
        f"{name:10s} {value:.8f}"
    )