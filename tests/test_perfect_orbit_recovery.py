import copy
import numpy as np

from pyseshat.io.orbit_parser import (
    parse_orbit_file
)

from pyseshat.orbit.orbit_model import (
    compute_visual_orbit
)

from pyseshat.inference.optimizer import (
    fit_visual_orbit_mcmc
)

# ==========================================================
# Load FIN379
# ==========================================================

parsed = parse_orbit_file(
    "validation/benchmark_systems/FIN379/FIN379.inp"
)

true_elements = copy.deepcopy(
    parsed["elements"]
)

# ==========================================================
# Observation epochs
# ==========================================================

visual = parsed["visual_data"]

t = visual[:, 0]

# ==========================================================
# Generate perfect observations
# ==========================================================

orbit = compute_visual_orbit(
    t,
    true_elements
)

X_obs = orbit["X"]

Y_obs = orbit["Y"]

# ==========================================================
# Tiny uncertainties
# ==========================================================

sigma = np.full(
    len(t),
    1e-4
)

# ==========================================================
# Bayesian fit
# ==========================================================

posterior = fit_visual_orbit_mcmc(

    true_elements,

    t,

    X_obs,

    Y_obs,

    sigma,

    sigma,

    rv_data=None,

    n_walkers=20,

    n_steps=500,

    burn_in=100,

    thin=2,

    rng=np.random.default_rng(42)
)

# ==========================================================
# Compare with truth
# ==========================================================

median = posterior.posterior_median

names = posterior.parameter_names

print()

print("PERFECT ORBIT RECOVERY")
print("=" * 110)

print(
    f"{'Parameter':<12}"
    f"{'Difference':>18}"
    f"{'Posterior σ':>18}"
    f"{'|Δ| / σ':>15}"
)

print("-" * 110)

median = posterior.posterior_median
std = posterior.posterior_std

for i, name in enumerate(names):

    truth = getattr(true_elements, name)

    delta = median[i] - truth

    sigma = std[i]

    z = abs(delta) / sigma

    print(
        f"{name:<12}"
        f"{delta:18.8e}"
        f"{sigma:18.8e}"
        f"{z:15.3f}"
    )

    assert z < 1.0, (
        f"{name} not recovered within one posterior sigma."
    )

print()

print("SUCCESS: Synthetic orbit recovered within posterior uncertainty.")