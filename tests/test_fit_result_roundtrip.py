from pyseshat.io.fit_result_io import load_fit_result

loaded = load_fit_result(
    "FIN379_fit.json"
)

print()

print(
    loaded.elements
)

print()

print(
    loaded.uncertainties
)

print()

print(
    loaded.optimizer_result.success
)

print()

print(
    loaded.optimizer_result.message
)

print()

print(
    loaded.covariance.shape
)

print()

print(
    loaded.residuals.shape
)

print()

print(
    loaded.metadata
)