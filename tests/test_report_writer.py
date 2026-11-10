from pathlib import Path
import numpy as np

from pyseshat.io.orbit_parser import (
    parse_orbit_file
)

from pyseshat.inference.optimizer import (
    fit_visual_orbit
)

from pyseshat.analysis.report import (
    build_fit_report,
    report_to_text
)

from pyseshat.analysis.masses import (
    component_masses_from_elements
)

from pyseshat.io.report_writer import (
    write_text_report
)

# This file has no `def test_...` functions and no assertions --
# a demo script, not a real pytest test (see test_fit_result_io.py
# for the full explanation). It used to write FIN379_report.txt
# with a bare relative path, which landed at the repo root. Anchored to this file's own location instead.

ARTIFACTS_DIR = (
    Path(__file__).resolve().parent / "artifacts"
)

ARTIFACTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

filepath = Path(
    "validation/benchmark_systems/FIN379/FIN379.inp"
)

parsed = parse_orbit_file(
    str(filepath)
)

elements = parsed["elements"]

visual_data = parsed["visual_data"]

rv_data = parsed["rv_data"]

t = visual_data[:,0]

theta = visual_data[:,1]

rho = visual_data[:,2]

sigma = visual_data[:,3]

theta_rad = np.deg2rad(theta)

X = rho * np.cos(theta_rad)

Y = rho * np.sin(theta_rad)

sx = sigma

sy = sigma

fit_result = fit_visual_orbit(
    elements,
    t,
    X,
    Y,
    sx,
    sy,
    rv_data=rv_data
)

# ------------------------------------------------------
# Statistics
# ------------------------------------------------------

statistics = {

    "rms":
        np.sqrt(
            np.mean(
                fit_result.residuals**2
            )
        )
}

# ------------------------------------------------------
# Masses
# ------------------------------------------------------

masses = component_masses_from_elements(
    fit_result.elements
)

# ------------------------------------------------------
# Build report
# ------------------------------------------------------

report = build_fit_report(

    fit_result,

    statistics,

    parameter_uncertainties=
        fit_result.uncertainties,

    masses=masses
)

# ------------------------------------------------------
# Text report
# ------------------------------------------------------

text = report_to_text(
    report
)

outfile = write_text_report(
    text,
    str(ARTIFACTS_DIR / "FIN379_report.txt")
)

print()

print(
    "Written:",
    outfile
)