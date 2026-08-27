"""
Example 4: Al-Wardat unresolved-binary SED fit.

Fits HD214222's combined-light photometry to a two-component
Castelli-Kurucz (CK04) synthetic spectral model, recovering each
component's effective temperature, radius, and the system's
reddening E(B-V) -- the "Al-Wardat method" this project's SED
benchmarks are named for (Abushattal, Georgakarakos, Al-Wardat et
al. 2025, AJ, arXiv:2512.23652). Unlike Examples 1-3, there is no
orbital fit here at all: this validates stellar *modelling* given
a known orbital solution, using only plain (passband, magnitude,
uncertainty) measurements.

Run

    python examples/04_alwardat_sed_fit.py

Uses validation/expected_results/HD214222.json and this repo's
local CK04 atmosphere grid + passband transmission curves (all
checked in), so this script runs out of the box. See
docs/project_decisions.md, "PhotometricParameters fit-flags/prior
mechanism, Al-Wardat measurement-uncertainty reassessment, and
Stromgren u disposition (2026-08-27)" for the full derivation of
the choices made below (why radius is freed WITH a Gaussian prior
rather than fixed outright, why the assumed per-measurement
uncertainty is 0.30 mag rather than a naive 0.02 mag, and why
Stromgren u is excluded).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pyseshat.stellar.atmospheres import (
    load_ck04_grid,
)

from pyseshat.photometry.filters import (
    load_passband,
)

from pyseshat.photometry.observations import (
    PhotometricMeasurement,
)

from pyseshat.photometry.parameters import (
    PhotometricParameters,
)

from pyseshat.inference.photometric_optimizer import (
    fit_photometry,
    build_parameter_bounds,
)

from pyseshat.inference.photometric_likelihood import (
    photometric_residuals_from_problem,
)


REPO_ROOT = Path(__file__).resolve().parent.parent

VALIDATION_DIR = REPO_ROOT / "validation"

SYSTEM = "HD214222"

EXPECTED_FILE = (
    VALIDATION_DIR / "expected_results" / f"{SYSTEM}.json"
)

# SVO Filter Profile Service "Vega system" zero points, matched to
# each band's own transmission curve below -- see
# validation/benchmark_runner.py's _ALWARDAT_ZERO_POINTS for the
# full derivation and cross-check against a Sun-at-10pc pipeline
# calibration test.

ZERO_POINTS = {
    "Johnson U": 3.49719e-9,
    "Johnson B": 6.72553e-9,
    "Johnson V": 3.5833e-9,
    "Johnson R": 1.87529e-9,
    "Stromgren v": 7.27101e-9,
    "Stromgren b": 5.78663e-9,
    "Stromgren y": 3.6203e-9,
    "Tycho BT": 6.50910e-9,
    "Tycho VT": 3.99504e-9,
}

PASSBAND_FILES = {
    "Johnson U": "johnson_u.dat",
    "Johnson B": "johnson_b.dat",
    "Johnson V": "johnson_v.dat",
    "Johnson R": "johnson_r.dat",
    "Stromgren v": "stromgren_v.dat",
    "Stromgren b": "stromgren_b.dat",
    "Stromgren y": "stromgren_y.dat",
    "Tycho BT": "tycho_bt.dat",
    "Tycho VT": "tycho_vt.dat",
}

RADIUS_PRIOR_RELATIVE_SIGMA = 0.10


def main():

    with open(EXPECTED_FILE) as f:
        expected = json.load(f)

    # --------------------------------------------------
    # 1. Load the CK04 atmosphere grid and passbands this
    # system needs. atmosphere_files must bracket BOTH
    # components' literature Teff, not just one -- see
    # docs/project_decisions.md for a real bug this caused
    # for two of the four registered systems.
    # --------------------------------------------------

    atmosphere_grid = load_ck04_grid([
        VALIDATION_DIR / "atmosphere_models" / relative_path
        for relative_path in expected["atmosphere_files"]
    ])

    measurements = [
        PhotometricMeasurement(
            passband=entry["passband"],
            magnitude=entry["magnitude"],
            uncertainty=entry["uncertainty"],
        )
        for entry in expected["measurements"]
        # Stromgren u is present in Table 4 but excluded here
        # (see expected["excluded_measurements"] and
        # docs/project_decisions.md for why).
    ]

    passband_names = sorted({
        measurement.passband for measurement in measurements
    })

    passbands = {
        name: load_passband(
            filename=(
                VALIDATION_DIR
                / "photometry"
                / "passbands"
                / PASSBAND_FILES[name]
            ),
            name=name,
        )
        for name in passband_names
    }

    zero_points = {
        name: ZERO_POINTS[name] for name in passband_names
    }

    print(
        f"Fitting {SYSTEM} against {len(measurements)} real "
        f"photometric measurements in {len(passband_names)} bands."
    )

    # --------------------------------------------------
    # 2. Build the starting parameters, fit flags, and
    # radius priors. All 5 photometric parameters are
    # freed; radius carries a Gaussian prior tying it to
    # the literature value (unresolved photometry alone
    # cannot separate each component's (Teff, R) -- see
    # module docstring above).
    # --------------------------------------------------

    initial = expected["initial_parameters"]

    literature = expected["literature_component_parameters"]

    fit_flags = {
        name: True
        for name in (
            "primary_temperature",
            "secondary_temperature",
            "primary_radius",
            "secondary_radius",
            "ebv",
        )
    }

    priors = {
        "primary_radius": (
            literature["primary"]["radius_Rsun"],
            literature["primary"]["radius_Rsun"]
            * RADIUS_PRIOR_RELATIVE_SIGMA,
        ),
        "secondary_radius": (
            literature["secondary"]["radius_Rsun"],
            literature["secondary"]["radius_Rsun"]
            * RADIUS_PRIOR_RELATIVE_SIGMA,
        ),
    }

    parameters = PhotometricParameters(
        primary_temperature=initial["primary_temperature"],
        secondary_temperature=initial["secondary_temperature"],
        primary_radius=initial["primary_radius"],
        secondary_radius=initial["secondary_radius"],
        ebv=initial["ebv"],
        fit_flags=fit_flags,
        priors=priors,
    )

    bounds = build_parameter_bounds(
        parameters,
        atmosphere_grid=atmosphere_grid,
    )

    # --------------------------------------------------
    # 3. Fit.
    # --------------------------------------------------

    fit_result = fit_photometry(
        parameters=parameters,
        atmosphere_grid=atmosphere_grid,
        passbands=passbands,
        zero_points=zero_points,
        measurements=measurements,
        primary_logg=expected["primary_logg"]["value"],
        secondary_logg=expected["secondary_logg"]["value"],
        metallicity=expected["metallicity"]["value"],
        distance=expected["distance_pc"]["value"],
        bounds=bounds,
    )

    fitted = fit_result.elements.parameters

    # --------------------------------------------------
    # 4. Compare against the literature, using PURE
    # measurement residuals (not fit_result.residuals,
    # which also carries the radius prior's pseudo-
    # residuals -- see photometric_residuals_from_problem's
    # docstring).
    # --------------------------------------------------

    residuals = photometric_residuals_from_problem(
        fit_result.elements
    )

    n_params = sum(fit_flags.values())

    n_data = int(residuals.size)

    dof = max(n_data - n_params, 1)

    reduced_chi_square = float(
        np.sum(residuals ** 2) / dof
    )

    print()

    print(
        f"{'Parameter':>20}{'Fitted':>12}{'Literature':>14}"
        f"{'Rel. err.':>12}"
    )

    comparisons = (
        ("primary_temperature", fitted.primary_temperature,
         literature["primary"]["teff_K"]),
        ("secondary_temperature", fitted.secondary_temperature,
         literature["secondary"]["teff_K"]),
        ("primary_radius", fitted.primary_radius,
         literature["primary"]["radius_Rsun"]),
        ("secondary_radius", fitted.secondary_radius,
         literature["secondary"]["radius_Rsun"]),
    )

    for name, fitted_value, literature_value in comparisons:

        rel_err = (
            abs(fitted_value - literature_value)
            / literature_value
        )

        print(
            f"{name:>20}{fitted_value:>12.2f}"
            f"{literature_value:>14.2f}{rel_err:>11.2%}"
        )

    print()

    print(
        f"E(B-V) = {fitted.ebv:.4f}  "
        f"reduced chi-square = {reduced_chi_square:.4f}"
    )


if __name__ == "__main__":

    main()
