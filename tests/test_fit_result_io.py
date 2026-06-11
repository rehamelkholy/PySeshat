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
    "FIN379_fit.json",
    system_name="FIN379"
)

print()

print(
    "Saved:",
    outfile
)