import numpy as np

from pyseshat.io.orbit_parser import (
    parse_orbit_file
)

from pyseshat.inference.optimizer import (
    fit_visual_orbit
)

from pyseshat.inference.priors import (
    log_prior
)

from pyseshat.inference.likelihood import (
    log_likelihood
)

from pyseshat.inference.posterior import (
    log_posterior
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
# Best fit
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

params = fit.elements.fitted_parameters()

# ==========================================================
# Evaluate
# ==========================================================

lp = log_prior(
    fit.elements
)

ll = log_likelihood(
    params,
    fit.elements,
    t,
    X_obs,
    Y_obs,
    sigma,
    sigma,
    rv_data=rv_data
)

post = log_posterior(
    params,
    fit.elements,
    t,
    X_obs,
    Y_obs,
    sigma,
    sigma,
    rv_data=rv_data
)

print()

print("LOG PRIOR")
print("=" * 60)
print(lp)

print()

print("LOG LIKELIHOOD")
print("=" * 60)
print(ll)

print()

print("LOG POSTERIOR")
print("=" * 60)
print(post)

print()

print("CHECK")
print("=" * 60)
print(post - (lp + ll))