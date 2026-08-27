from pathlib import Path
import json
import numpy as np

from pyseshat.io.orbit_parser import (
    parse_orbit_file
)

from pyseshat.inference.optimizer import (
    fit_visual_orbit
)

from pyseshat.io.fit_result_io import (
    fit_result_to_dict,
    save_fit_result
)

# This file has no `def test_...` functions and no assertions --
# it is a demo script, not a real pytest test (pytest reports 0
# tests collected from it, but still executes this module-level
# code on collection). It used to write FIN379_fit.json with a
# bare relative path, which landed wherever pytest's cwd happened
# to be (the repo root, in normal use) -- see
# docs/project_decisions.md, "FIN379 files stranded at repo root"
# (2026-08-27). Anchored to this file's own location instead, so
# the output lands in the same place regardless of invocation cwd.
# test_fit_result_roundtrip.py reads back from this same
# directory, so the two stay coupled (test_fit_result_io.py must
# still run first, which it does alphabetically).

ARTIFACTS_DIR = (
    Path(__file__).resolve().parent / "artifacts"
)

ARTIFACTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ------------------------------------------------------
# FIN379
# ------------------------------------------------------

filepath = Path(
    "validation/benchmark_systems/FIN379/FIN379.inp"
)

parsed = parse_orbit_file(
    str(filepath)
)

visual = parsed["visual_data"]

rv = parsed["rv_data"]

t = visual[:,0]

theta = np.deg2rad(
    visual[:,1]
)

rho = visual[:,2]

sigma = visual[:,3]

X = rho * np.cos(theta)

Y = rho * np.sin(theta)

fit_result = fit_visual_orbit(
    parsed["elements"],
    t,
    X,
    Y,
    sigma,
    sigma,
    rv_data=rv
)

payload = fit_result_to_dict(
    fit_result
)

print()

print(
    payload.keys()
)

print()

print(
    type(
        payload["covariance"]
    )
)

print()

print(
    type(
        payload["residuals"]
    )
)

print()

json.dumps(payload)

print(
    "JSON serialization successful."
)

outfile = save_fit_result(
    fit_result,
    str(ARTIFACTS_DIR / "FIN379_fit.json"),
    system_name="FIN379"
)

print()

print(
    "Saved:",
    outfile
)