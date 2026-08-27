"""
Example 1: Visual + RV binary fit from scratch.

The "hello world" PySeshat workflow: parse a real orbit data file,
fit it with least squares, plot the orbit and residuals, and write
a text report. Uses FIN379, one of the four core validation
benchmark systems already checked into this repo
(validation/benchmark_systems/FIN379/FIN379.inp), so this script
runs out of the box with no external data.

Run

    python examples/01_visual_rv_binary_fit.py

See docs/project_decisions.md for what FIN379 is (a software/
inference regression benchmark, not the project's primary ORBIT3
physics benchmark) and CLAUDE.md's "Layering" section for how the
pieces used below (io -> inference.optimizer -> orbit.orbit_model
-> visualization / io.report_writer) fit into the rest of the
package.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pyseshat.io.orbit_parser import (
    parse_orbit_file,
)

from pyseshat.inference.optimizer import (
    fit_visual_orbit,
)

from pyseshat.orbit.orbit_model import (
    compute_visual_orbit,
)

from pyseshat.analysis.report import (
    build_fit_report,
    report_to_text,
)

from pyseshat.analysis.masses import (
    component_masses_from_elements,
)

from pyseshat.io.report_writer import (
    write_text_report,
)

from pyseshat.visualization.orbit_plot import (
    plot_visual_orbit,
    plot_residuals,
)


REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    REPO_ROOT
    / "validation"
    / "benchmark_systems"
    / "FIN379"
    / "FIN379.inp"
)

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # 1. Parse the .inp file.
    #
    # parse_orbit_file() returns a dict with the header-
    # supplied starting OrbitalElements plus the raw
    # visual astrometry (t, position angle, separation,
    # sigma) and, when present, radial-velocity rows.
    # --------------------------------------------------

    parsed = parse_orbit_file(
        str(DATA_FILE)
    )

    elements = parsed["elements"]

    visual = parsed["visual_data"]

    rv = parsed["rv_data"]

    print(
        f"Parsed {DATA_FILE.name}: "
        f"{len(visual)} visual observations, "
        f"{len(rv)} RV observations."
    )

    # --------------------------------------------------
    # 2. Convert (position angle, separation) to the
    # package's Cartesian convention:
    #
    #     X = rho * cos(theta)
    #     Y = rho * sin(theta)
    #
    # theta measured from North through East, matching
    # every residual/plotting function in the package
    # (see orbit/orbit_model.py's compute_visual_orbit
    # docstring).
    # --------------------------------------------------

    t = visual[:, 0]

    theta = np.deg2rad(
        visual[:, 1]
    )

    rho = visual[:, 2]

    sigma = visual[:, 3]

    X_obs = rho * np.cos(theta)

    Y_obs = rho * np.sin(theta)

    # --------------------------------------------------
    # 3. Fit. fit_visual_orbit() runs scipy least_squares
    # under the hood, automatically resolving the
    # visual-binary node ambiguity whenever RV data are
    # supplied alongside the astrometry.
    # --------------------------------------------------

    fit_result = fit_visual_orbit(
        elements,
        t,
        X_obs,
        Y_obs,
        sigma,
        sigma,
        rv_data=rv,
    )

    fitted = fit_result.elements

    print()

    print("Fitted elements:")

    for name in fitted.fitted_parameter_names():

        print(
            f"  {name:>6} = {getattr(fitted, name):.4f}"
        )

    # --------------------------------------------------
    # 4. Diagnostics. BaseFitResult.summary() computes
    # chi-square/reduced chi-square/RMS from the fit's own
    # residual vector and covariance.
    # --------------------------------------------------

    summary = fit_result.summary(
        n_parameters=len(
            fitted.fitted_parameter_names()
        )
    )

    print()

    print("Fit summary:")

    for key, value in summary.items():

        print(f"  {key}: {value}")

    # --------------------------------------------------
    # 5. Plot the orbit and residuals. plot_visual_orbit()
    # needs the smooth model curve from
    # compute_visual_orbit(), evaluated at a dense time
    # grid spanning one period.
    # --------------------------------------------------

    t_model = np.linspace(
        fitted.T0,
        fitted.T0 + fitted.P,
        500,
    )

    orbit_model = compute_visual_orbit(
        t_model,
        fitted,
    )

    fig_orbit, _ = plot_visual_orbit(
        X_obs,
        Y_obs,
        orbit_model,
        title="FIN379 -- Fitted Visual Orbit",
        show=False,
    )

    fig_orbit.savefig(
        OUTPUT_DIR / "fin379_orbit.png",
        dpi=150,
    )

    fig_res, _ = plot_residuals(
        t,
        fit_result.residuals[: 2 * len(t)],
        title="FIN379 -- Visual Residuals",
        show=False,
    )

    fig_res.savefig(
        OUTPUT_DIR / "fin379_residuals.png",
        dpi=150,
    )

    print()

    print(
        "Saved plots to "
        f"{OUTPUT_DIR / 'fin379_orbit.png'} and "
        f"{OUTPUT_DIR / 'fin379_residuals.png'}"
    )

    # --------------------------------------------------
    # 6. Write a text report. build_fit_report() bundles
    # the fit result with statistics, uncertainties, and
    # (for a system with RV data) component masses into
    # one structured report object; report_to_text()
    # renders it.
    # --------------------------------------------------

    masses = component_masses_from_elements(
        fitted
    )

    report = build_fit_report(
        fit_result,
        summary,
        parameter_uncertainties=fit_result.uncertainties,
        masses=masses,
    )

    report_path = write_text_report(
        report_to_text(report),
        str(OUTPUT_DIR / "fin379_report.txt"),
    )

    print(
        f"Saved report to {report_path}"
    )


if __name__ == "__main__":

    main()
