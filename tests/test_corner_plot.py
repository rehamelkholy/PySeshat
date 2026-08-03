import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from pyseshat.io.orbit_parser import (
    parse_orbit_file
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

elements = parsed["elements"]

visual = parsed["visual_data"]

rv_data = parsed["rv_data"]

t = visual[:, 0]

theta = np.deg2rad(
    visual[:, 1]
)

rho = visual[:, 2]

sigma = visual[:, 3]

X_obs = rho * np.cos(theta)

Y_obs = rho * np.sin(theta)

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
# Plot
# ==========================================================

fig = posterior.corner_plot()

print()

print("NUMBER OF AXES")
print("=" * 60)

print(
    len(fig.axes)
)

plt.show()