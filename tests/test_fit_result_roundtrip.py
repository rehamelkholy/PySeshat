from pathlib import Path

from pyseshat.io.fit_result_io import load_fit_result

# This file has no `def test_...` functions and no assertions --
# a demo script, not a real pytest test (see test_fit_result_io.py
# for the full explanation). It reads back FIN379_fit.json from
# the artifacts directory that test_fit_result_io.py writes to
# (these two files are coupled -- test_fit_result_io.py must run
# first, which it does alphabetically).

ARTIFACTS_DIR = (
    Path(__file__).resolve().parent / "artifacts"
)

loaded = load_fit_result(
    str(ARTIFACTS_DIR / "FIN379_fit.json")
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