import numpy as np

from pyseshat.io.orbit_parser import (
    parse_orbit_file
)

from pyseshat.inference.optimizer import (
    fit_visual_orbit,
    fit_visual_orbit_mcmc
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

ls = fit_visual_orbit(

    elements,

    t,

    X_obs,

    Y_obs,

    sigma,

    sigma,

    rv_data=rv_data
)

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
# Compare solutions
# ==========================================================

print()

print("LEAST-SQUARES VS POSTERIOR")
print("=" * 100)

print(
    f"{'Parameter':<12}"
    f"{'Δ':>15}"
    f"{'Posterior σ':>18}"
    f"{'|Δ| / σ':>15}"
)

print("-" * 100)

median = posterior.posterior_median
std = posterior.posterior_std

for i, name in enumerate(posterior.parameter_names):

    ls_value = getattr(ls.elements, name)

    delta = median[i] - ls_value

    sigma = std[i]

    z = abs(delta) / sigma

    print(
        f"{name:<12}"
        f"{delta:15.8e}"
        f"{sigma:18.8e}"
        f"{z:15.3f}"
    )

    assert z < 1.0, (
        f"{name} differs by more than one posterior sigma."
    )

print()

print("SUCCESS: All least-squares estimates lie within one posterior standard deviation.")