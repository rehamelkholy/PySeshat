import numpy as np

from pyseshat.io.orbit_parser import (
    parse_orbit_file
)

from pyseshat.inference.optimizer import (
    fit_visual_orbit,
    _sample_single_solution
)

# ==========================================================
# Load FIN379
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

t = visual[:, 0]

theta = np.deg2rad(
    visual[:, 1]
)

rho = visual[:, 2]

sigma = visual[:, 3]

X_obs = rho * np.cos(theta)

Y_obs = rho * np.sin(theta)

# ==========================================================
# Least-squares fit
# ==========================================================

fit = fit_visual_orbit(

    elements,

    t,

    X_obs,

    Y_obs,

    sigma,

    sigma,

    rv_data=rv_data
)

# ==========================================================
# Bayesian sampling
# ==========================================================

posterior = _sample_single_solution(

    fit,

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

print("POSTERIOR SAMPLES")
print("=" * 60)
print(posterior.samples.shape)

print()

print("PARAMETERS")
print("=" * 60)
print(posterior.parameter_names)

print()

print("ACCEPTANCE FRACTION")
print("=" * 60)
print(posterior.acceptance_fraction)

print()

print("AUTOCORRELATION")
print("=" * 60)
print(posterior.autocorrelation_time)