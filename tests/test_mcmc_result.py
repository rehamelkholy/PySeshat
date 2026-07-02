import numpy as np

from scipy.optimize import OptimizeResult

from pyseshat.io.orbit_parser import (
    parse_orbit_file
)

from pyseshat.inference.mcmc_result import (
    MCMCResult
)

# ==========================================================
# Load benchmark orbit
# ==========================================================

parsed = parse_orbit_file(
    "validation/benchmark_systems/FIN379/FIN379.inp"
)

elements = parsed["elements"]

# ==========================================================
# Dummy optimizer
# ==========================================================

optimizer = OptimizeResult()

optimizer.success = True

optimizer.message = "Dummy optimizer."

# ==========================================================
# Fake posterior
# ==========================================================

rng = np.random.default_rng(42)

samples = rng.normal(
    size=(500, 4)
)

log_probability = rng.normal(
    size=500
)

# ==========================================================
# Construct result
# ==========================================================

result = MCMCResult(

    elements=elements,

    optimizer_result=optimizer,

    residuals=np.zeros(10),

    covariance=np.eye(4),

    uncertainties={},

    samples=samples,

    log_probability=log_probability,

    parameter_names=[
        "P",
        "e",
        "a",
        "i"
    ],

    acceptance_fraction=0.31,

    autocorrelation_time=None,

    effective_sample_size=None,

    burn_in=100,

    thin=5
)

# ==========================================================
# Report
# ==========================================================

print()

print("NUMBER OF SAMPLES")
print("=" * 60)
print(result.n_samples)

print()

print("NUMBER OF PARAMETERS")
print("=" * 60)
print(result.n_parameters)

print()

print("SUCCESS")
print("=" * 60)
print(result.success)

print()

print("MESSAGE")
print("=" * 60)
print(result.message)

print()

print("ACCEPTANCE FRACTION")
print("=" * 60)
print(result.acceptance_fraction)

print()

print("BURN-IN")
print("=" * 60)
print(result.burn_in)

print()

print("THINNING")
print("=" * 60)
print(result.thin)

print()

print("FIRST PARAMETER NAME")
print("=" * 60)
print(result.parameter_names[0])