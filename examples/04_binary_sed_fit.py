"""
Example 4: unresolved-binary SED fit.

Fits HD214222's combined-light photometry to a two-component
Castelli-Kurucz (CK04) synthetic spectral model, recovering each
component's effective temperature and radius -- the "binary
SED-fitting method" this project's SED
benchmarks are named for (Abushattal, Georgakarakos, Al-Wardat et
al. 2025, AJ, arXiv:2512.23652). Unlike Examples 1-3, there is no
orbital fit here at all: this validates stellar *modelling* given
a known orbital solution, using only plain (passband, magnitude,
uncertainty) measurements.

Run

    python examples/04_binary_sed_fit.py

Uses validation/expected_results/HD214222.json and this repo's
local CK04 atmosphere grid + passband transmission curves (all
checked in), so this script runs out of the box. It follows the
benchmark configuration in that file (as
validation/benchmark_runner.py does): the free parameters listed
under "free_parameters" (E(B-V) is held at 0, since the combined
light alone cannot separate temperature from reddening), each radius
freed WITH a 10% Gaussian prior on its literature value, the
component magnitude difference under "magnitude_differences" as an
extra constraint, and a 0.30 mag tolerance per band (the
measurements are the source paper's own synthetic photometry).
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
    MagnitudeDifference,
    PhotometricMeasurement,
)

from pyseshat.photometry.calibration import (
    vega_zero_points,
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

PASSBAND_FILES = {
    "Johnson U": "johnson_u.dat",
    "Johnson B": "johnson_b.dat",
    "Johnson V": "johnson_v.dat",
    "Cousins R": "cousins_r.dat",
    "Stromgren u": "stromgren_u.dat",
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
    # components' literature Teff, not just one.
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
    ]

    magnitude_differences = [
        MagnitudeDifference(
            passband=entry["passband"],
            value=entry["value"],
            uncertainty=entry["uncertainty"],
        )
        for entry in expected.get("magnitude_differences", [])
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

    # Zero points on the standard photometric systems: the CALSPEC Vega
    # spectrum shipped with PySeshat, integrated through these same
    # passbands, with Vega's standard-system magnitudes (Maiz Apellaniz
    # 2007; see pyseshat.photometry.calibration).
    zero_points = vega_zero_points(passbands)

    print(
        f"Fitting {SYSTEM} against {len(measurements)} photometric "
        f"measurements in {len(passband_names)} bands and "
        f"{len(magnitude_differences)} magnitude difference(s)."
    )

    # --------------------------------------------------
    # 2. Build the starting parameters, fit flags, and
    # radius priors. The parameters in
    # expected["free_parameters"] are freed (E(B-V) is not);
    # radius carries a Gaussian prior tying it to the
    # literature value (unresolved photometry alone cannot
    # separate each component's (Teff, R) -- see module
    # docstring above).
    # --------------------------------------------------

    initial = expected["initial_parameters"]

    literature = expected["literature_component_parameters"]

    free = set(expected["free_parameters"])

    fit_flags = {
        name: name in free
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
        magnitude_differences=magnitude_differences,
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
