"""
Example 3: Hierarchical triple (ORBIT3) fit.

Fits Zeta Aquarii's inner+outer orbit through the real 20-parameter
ORBIT3 formalism (Tokovinin's convention, as implemented in
triple.orbit3_fit.fit_orbit3) against Tokovinin (2016)'s published
solution, and shows why a hierarchical triple cannot be fit through
the plain two-body path used in Example 1: fitting only the outer
pair, or fitting the inner residuals from a separate outer-only fit,
both throw away the coupling between the two orbits.

Run

    python examples/03_hierarchical_triple_orbit3.py

Uses validation/benchmark_systems/ZetaAqr/ZetaAqr.in2 (checked into
this repo) and validation/expected_results/ZetaAqr.json for the
literature starting solution and fit mask. See "A legacy
data-format wrinkle" below for the legacy-vs-ORBIT3 formalism split
this example's data file sits on the wrong side of.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pyseshat.orbit.orbital_elements import (
    OrbitalElements,
)

from pyseshat.triple.triple_elements import (
    TripleOrbitalElements,
)

from pyseshat.triple.triple_observation import (
    TripleObservation,
)

from pyseshat.triple.triple_parser import (
    parse_triple_file,
)

from pyseshat.triple.triple_residuals import (
    orbit3_residuals,
)

from pyseshat.triple.orbit3_fit import (
    fit_orbit3,
)


REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    REPO_ROOT
    / "validation"
    / "benchmark_systems"
    / "ZetaAqr"
    / "ZetaAqr.in2"
)

LITERATURE_FILE = (
    REPO_ROOT
    / "validation"
    / "expected_results"
    / "ZetaAqr.json"
)


def _convert_legacy_orbit3_observations(observations):
    """
    Swap the I1/I2 astrometric tags of a legacy ORBITT2-formatted
    (.in2) file into the modern ORBIT3 convention.

    Legacy .in2 files (as produced for ZetaAqr) tag the OUTER pair
    "I1" and the resolved INNER subsystem "I2" -- the opposite of
    ORBIT3's own convention (I1 = inner, I2 = outer). A genuinely
    modern .in3 file (see HIP111805 in validation/benchmark_systems/)
    needs no such conversion. This is the same conversion
    validation/benchmark_runner.py applies internally, duplicated
    here (rather than imported across the validation/ package
    boundary) so this example stays self-contained.
    """

    converted = []

    for obs in observations:

        if obs.orbit_id == "I1":
            orbit_id = "I2"

        elif obs.orbit_id == "I2":
            orbit_id = "I1"

        else:
            raise ValueError(
                f"Unexpected legacy triple observation tag: "
                f"{obs.orbit_id!r}"
            )

        converted.append(
            TripleObservation(
                epoch=obs.epoch,
                theta=obs.theta,
                rho=obs.rho,
                sigma=obs.sigma,
                orbit_id=orbit_id,
                velocity=obs.velocity,
            )
        )

    return converted


def _build_orbit3_elements(initial_parameters, fitted_flags):
    """
    Build a TripleOrbitalElements from the flat 15-key ORBIT3
    schema (unsuffixed keys = inner orbit, "2"-suffixed keys =
    outer orbit) used throughout validation/expected_results/.
    """

    inner = OrbitalElements(
        P=initial_parameters["P"],
        T0=initial_parameters["T0"],
        e=initial_parameters["e"],
        a=initial_parameters["a"],
        Omega=initial_parameters["Omega"],
        omega=initial_parameters["omega"],
        i=initial_parameters["i"],
    )

    outer = OrbitalElements(
        P=initial_parameters["P2"],
        T0=initial_parameters["T02"],
        e=initial_parameters["e2"],
        a=initial_parameters["a2"],
        Omega=initial_parameters["Omega2"],
        omega=initial_parameters["omega2"],
        i=initial_parameters["i2"],
    )

    elements = TripleOrbitalElements(
        outer=outer,
        inner=inner,
        f=initial_parameters.get("f", 0.0),
    )

    for name, flag in fitted_flags.items():

        elements.fit_flags[name] = bool(flag)

    return elements


def main():

    # --------------------------------------------------
    # 1. Parse the data file.
    # --------------------------------------------------

    parsed = parse_triple_file(
        str(DATA_FILE)
    )

    observations = parsed["observations"]

    rv_observations = parsed["rv_observations"]

    # --------------------------------------------------
    # A legacy data-format wrinkle: ZetaAqr.in2 is an
    # older ORBITT2-format file, whose I1/I2 astrometric
    # tags mean the opposite of ORBIT3's own convention.
    # --------------------------------------------------

    if parsed["metadata"].get("format") == "in2":

        observations = _convert_legacy_orbit3_observations(
            observations
        )

    print(
        f"Parsed {DATA_FILE.name}: "
        f"{len(observations)} astrometric observations, "
        f"{len(rv_observations)} RV observations."
    )

    # --------------------------------------------------
    # 2. Build the starting solution from Tokovinin
    # (2016)'s published inner+outer elements, rather than
    # from whatever parse_triple_file's header happened to
    # carry -- expected_results/ZetaAqr.json is this
    # project's single authoritative source for that
    # literature solution and its fit mask.
    # --------------------------------------------------

    with open(LITERATURE_FILE) as f:
        expected = json.load(f)

    elements = _build_orbit3_elements(
        expected["initial_parameters"],
        expected["fitted_flags"],
    )

    initial_chi2 = float(
        np.sum(
            orbit3_residuals(
                observations,
                rv_observations,
                elements,
            )
            ** 2
        )
    )

    print(
        f"Chi-square at the literature starting solution: "
        f"{initial_chi2:.2f}"
    )

    # --------------------------------------------------
    # 3. Fit through the real ORBIT3 formalism.
    # fit_orbit3() jointly fits inner+outer elements plus
    # the wobble factor f, coupling the two orbits the way
    # a genuine hierarchical triple requires -- this is
    # NOT equivalent to two independent two-body fits.
    # --------------------------------------------------

    fit_result = fit_orbit3(
        elements,
        observations,
        rv_observations,
    )

    fitted = fit_result.elements

    final_chi2 = float(
        np.sum(
            np.asarray(fit_result.residuals, dtype=float) ** 2
        )
    )

    print(
        f"Chi-square after fitting: {final_chi2:.2f}"
    )

    print()

    print("Fitted inner orbit:")

    for name in (
        "P", "T0", "e", "a", "Omega", "omega", "i"
    ):

        print(
            f"  {name:>6} = "
            f"{getattr(fitted.inner, name):.4f}"
        )

    print()

    print("Fitted outer orbit:")

    for name in (
        "P", "T0", "e", "a", "Omega", "omega", "i"
    ):

        print(
            f"  {name:>6} = "
            f"{getattr(fitted.outer, name):.4f}"
        )

    print()

    print(
        f"Wobble factor f = {fitted.f:.4f} "
        "(the fractional astrometric amplitude of the inner "
        "pair's motion around the outer pair's photocenter -- "
        "deliberately fit unbounded, not restricted to [0,1], since "
        "HIP111805's own literature value is f=-0.330)."
    )


if __name__ == "__main__":

    main()
