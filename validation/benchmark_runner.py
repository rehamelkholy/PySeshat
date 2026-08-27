import csv
import json
import sys
from pathlib import Path

import numpy as np

from pyseshat.io.orbit_parser import (
    parse_orbit_file
)

from pyseshat.inference.optimizer import (
    fit_visual_orbit
)

from pyseshat.inference.likelihood import (
    visual_xy_residuals,
    rv_residuals
)

from pyseshat.analysis.statistics import (
    combined_fit_summary
)

from pyseshat.analysis.masses import (
    total_mass_from_elements,
    total_mass_uncertainty_from_elements
)

from pyseshat.orbit.orbital_elements import (
    OrbitalElements
)

from pyseshat.triple.triple_elements import (
    TripleOrbitalElements
)

from pyseshat.triple.triple_observation import (
    TripleObservation
)

from pyseshat.triple.triple_parser import (
    parse_triple_file
)

from pyseshat.triple.triple_residuals import (
    orbit3_residuals
)

from pyseshat.triple.orbit3_fit import (
    fit_orbit3
)

from metrics import (
    parameter_relative_error
)


# ==========================================================
# Cross-path statistics normalization
# ==========================================================
#
# The binary path's combined_fit_summary() (pyseshat.analysis.
# statistics) and the ORBIT3 path below independently compute
# chi-square/reduced-chi-square/RMS using different key names
# ("chi2_total"/"reduced_chi2" vs "chi_square"/
# "reduced_chi_square" -- the latter matching the unrelated
# fit_summary() elsewhere in the same module). Rather than
# changing either library function's public return schema
# (other callers/tests depend on them), run_benchmark() and
# run_orbit3_benchmark() both normalize their "statistics"
# dict through this helper before returning, so that every
# validation/results/*.json file exposes the same core keys
# (chi_square, reduced_chi_square, rms, n_data, n_params)
# regardless of which fitting path produced it -- this is what
# makes it possible to compare benchmark results across
# FIN379/GL765/ZetaAqr/HIP111805 without special-casing each
# system's statistics schema.

def _normalize_statistics(
    stats,
    chi_square,
    reduced_chi_square,
    rms,
    n_data,
    n_params
):
    """
    Return a copy of ``stats`` with the common cross-system
    statistics keys added (existing path-specific keys, e.g.
    combined_fit_summary()'s "chi2_visual"/"chi2_rv1"/
    "chi2_rv2", are preserved alongside them).
    """

    normalized = dict(stats)

    normalized["chi_square"] = float(chi_square)

    normalized["reduced_chi_square"] = float(reduced_chi_square)

    normalized["rms"] = float(rms)

    normalized["n_data"] = int(n_data)

    normalized["n_params"] = int(n_params)

    return normalized


# ==========================================================
# ORBIT3 helpers
#
# These mirror the conventions established and validated in
# tests/test_zetaaqr_reproduce_orbit.py -- see that file (and
# validation/expected_results/ZetaAqr.json's "notes") for the
# full derivation and literature citations.
# ==========================================================

def _convert_legacy_orbit3_observations(
    observations
):
    """
    Swap the I1/I2 astrometric observation tags of a legacy
    ORBITT2-formatted (.in2) input file into the ORBIT3
    convention used by orbit3_residuals().

    Legacy .in2 files (as produced for e.g. ZetaAqr) tag the
    OUTER-pair observations as "I1" and the resolved INNER
    subsystem observations as "I2" -- the opposite of the
    modern ORBIT3 convention, where I1 is the inner visual
    orbit (POS1) and I2 is the outer visual orbit (POS2). This
    conversion is required before such observations can be
    passed to fit_orbit3()/orbit3_residuals(); it is a no-op
    for genuinely modern .in3 files, which are not routed
    through this function.
    """

    converted = []

    for obs in observations:

        if obs.orbit_id == "I1":
            orbit_id = "I2"

        elif obs.orbit_id == "I2":
            orbit_id = "I1"

        else:
            raise ValueError(
                "Unexpected legacy triple observation tag "
                f"for ORBIT3 conversion: {obs.orbit_id!r}"
            )

        converted.append(
            TripleObservation(
                epoch=obs.epoch,
                theta=obs.theta,
                rho=obs.rho,
                sigma=obs.sigma,
                orbit_id=orbit_id,
                velocity=obs.velocity
            )
        )

    return converted


def _build_orbit3_elements(
    initial_parameters,
    fitted_flags
):
    """
    Build a TripleOrbitalElements object from an
    expected_results-style flat ORBIT3 parameter dict.

    ``initial_parameters`` and ``fitted_flags`` both use the
    flat 15-key schema:

        P, T0, e, a, Omega, omega, i, f,
        P2, T02, e2, a2, Omega2, omega2, i2

    where the unsuffixed keys describe the INNER orbit and
    the "2"-suffixed keys describe the OUTER orbit, matching
    ORBIT3_PARAMETER_ORDER's inner-first convention.
    """

    inner = OrbitalElements(
        P=initial_parameters["P"],
        T0=initial_parameters["T0"],
        e=initial_parameters["e"],
        a=initial_parameters["a"],
        Omega=initial_parameters["Omega"],
        omega=initial_parameters["omega"],
        i=initial_parameters["i"]
    )

    outer = OrbitalElements(
        P=initial_parameters["P2"],
        T0=initial_parameters["T02"],
        e=initial_parameters["e2"],
        a=initial_parameters["a2"],
        Omega=initial_parameters["Omega2"],
        omega=initial_parameters["omega2"],
        i=initial_parameters["i2"]
    )

    elements = TripleOrbitalElements(
        outer=outer,
        inner=inner,
        K1=initial_parameters.get("K1", 0.0),
        K2=initial_parameters.get("K2", 0.0),
        f=initial_parameters.get("f", 0.0),
        K3=initial_parameters.get("K3", 0.0),
        K4=initial_parameters.get("K4", 0.0),
        V0=initial_parameters.get("V0", 0.0)
    )

    for name, flag in fitted_flags.items():

        elements.fit_flags[name] = bool(flag)

    return elements


def _flatten_orbit3_elements(
    elements: TripleOrbitalElements
):
    """
    Flatten a TripleOrbitalElements object into the flat
    schema used by the expected_results ORBIT3 files: the
    15 orbital keys shared by every hierarchical-triple
    benchmark, plus the 5 spectroscopic keys (K1, K2, K3, K4,
    V0) present only for systems with usable RV data (e.g.
    HIP111805; ZetaAqr's expected_results file omits them
    since it has none).
    """

    return {

        "P": elements.inner.P,
        "T0": elements.inner.T0,
        "e": elements.inner.e,
        "a": elements.inner.a,
        "Omega": elements.inner.Omega,
        "omega": elements.inner.omega,
        "i": elements.inner.i,

        "K1": elements.K1,
        "K2": elements.K2,
        "f": elements.f,

        "P2": elements.outer.P,
        "T02": elements.outer.T0,
        "e2": elements.outer.e,
        "a2": elements.outer.a,
        "Omega2": elements.outer.Omega,
        "omega2": elements.outer.omega,
        "i2": elements.outer.i,

        "K3": elements.K3,
        "K4": elements.K4,
        "V0": elements.V0,
    }


def run_orbit3_benchmark(
    system_name,
    config,
    expected
):
    """
    Execute a deterministic ORBIT3 hierarchical-triple
    benchmark fit.

    Unlike run_benchmark()'s plain binary path, this routes
    the system through the real 20-parameter ORBIT3
    formalism (triple.orbit3_fit.fit_orbit3), matching how
    ZetaAqr (and any future hierarchical-triple benchmark,
    e.g. HIP111805) must actually be fit -- fitting the outer
    orbit alone, or fitting the inner residuals of a separate
    outer-only fit, are both explicitly wrong for a
    hierarchical triple (see docs/project_decisions.md).
    """

    # ------------------------------------------------------
    # Parse dataset
    # ------------------------------------------------------

    parsed = parse_triple_file(
        str(config["data_file"])
    )

    observations = parsed["observations"]

    rv_observations = parsed["rv_observations"]

    if parsed["metadata"].get("format") == "in2":

        observations = (
            _convert_legacy_orbit3_observations(
                observations
            )
        )

    # ------------------------------------------------------
    # Build the ORBIT3 initial elements from the
    # expected_results file rather than from whatever
    # parse_triple_file happened to read out of the raw
    # input file's header -- the expected_results file is
    # the single authoritative, literature-cited source of
    # the initial parameters and fit mask (including the
    # a_astrometric -> a_true conversion for ZetaAqr).
    # ------------------------------------------------------

    elements = _build_orbit3_elements(
        expected["initial_parameters"],
        expected["fitted_flags"]
    )

    initial_residuals = orbit3_residuals(
        observations,
        rv_observations,
        elements
    )

    initial_chi2 = float(
        np.sum(initial_residuals**2)
    )

    # ------------------------------------------------------
    # Run fit
    # ------------------------------------------------------

    fit_result = fit_orbit3(
        elements,
        observations,
        rv_observations
    )

    fitted = fit_result.elements

    final_residuals = np.asarray(
        fit_result.residuals,
        dtype=float
    )

    final_chi2 = float(
        np.sum(final_residuals**2)
    )

    n_params = len(
        fitted.fitted_parameter_names()
    )

    n_data = int(
        final_residuals.size
    )

    dof = max(
        n_data - n_params,
        1
    )

    stats = _normalize_statistics(
        {
            "initial_chi_square": initial_chi2,
        },
        chi_square=final_chi2,
        reduced_chi_square=final_chi2 / dof,
        rms=np.sqrt(np.mean(final_residuals**2)),
        n_data=n_data,
        n_params=n_params
    )

    # ------------------------------------------------------
    # Parameter errors (flat 15-key schema)
    # ------------------------------------------------------

    expected_elements = expected["initial_parameters"]

    fitted_flat = _flatten_orbit3_elements(
        fitted
    )

    parameter_errors = {}

    for name, expected_value in expected_elements.items():

        fitted_value = fitted_flat[name]

        rel_error = parameter_relative_error(
            name,
            expected_value,
            fitted_value
        )

        parameter_errors[name] = {

            "expected": float(
                expected_value
            ),

            "fitted": float(
                fitted_value
            ),

            "relative_error": float(
                rel_error
            )
        }

    # ------------------------------------------------------
    # Dynamical mass
    #
    # Reported for whichever orbit expected["paper_table_row"]
    # names ("outer" for ZetaAqr, since no RV data constrain
    # the inner components' individual masses). Requires a
    # parallax in the expected_results file; skipped otherwise.
    # ------------------------------------------------------

    mass_dynamical_Msun = None

    mass_sigma_Msun = None

    mass_relative_sigma = None

    parallax = expected.get(
        "parallax_mas"
    )

    table_row = expected.get(
        "paper_table_row"
    )

    if parallax is not None and table_row in (
        "outer",
        "inner"
    ):

        row_elements = (
            fitted.outer
            if table_row == "outer"
            else fitted.inner
        )

        row_suffix = (
            "2"
            if table_row == "outer"
            else ""
        )

        mass_dynamical_Msun = float(
            total_mass_from_elements(
                row_elements,
                parallax_mas=parallax["value"]
            )
        )

        sigma_P = fit_result.uncertainties.get(
            f"P{row_suffix}"
        )

        sigma_a = fit_result.uncertainties.get(
            f"a{row_suffix}"
        )

        if sigma_P is not None and sigma_a is not None:

            mass_sigma_Msun = float(
                total_mass_uncertainty_from_elements(
                    row_elements,
                    uncertainties={
                        "P": sigma_P,
                        "a": sigma_a
                    },
                    parallax_mas=parallax["value"],
                    sigma_parallax_mas=parallax["sigma"]
                )
            )

            if mass_dynamical_Msun:

                mass_relative_sigma = (
                    mass_sigma_Msun
                    /
                    mass_dynamical_Msun
                )

    return {

        "system": system_name,

        "orbit_type": expected.get(
            "orbit_type",
            "hierarchical_triple"
        ),

        "fitting_path": "orbit3",

        "paper_table_row": table_row,

        "statistics": stats,

        "parameter_errors": parameter_errors,

        "fitted_elements": {
            name: float(value)
            for name, value in fitted_flat.items()
        },

        "optimizer_success": bool(
            fit_result.optimizer_result.success
        ),

        "optimizer_message": str(
            fit_result.optimizer_result.message
        ),

        "mass_dynamical_Msun": mass_dynamical_Msun,

        "mass_sigma_Msun": mass_sigma_Msun,

        "mass_relative_sigma": mass_relative_sigma,
    }


# ==========================================================
# Benchmark execution
# ==========================================================

def run_benchmark(
    system_name,
    config
):
    """
    Execute deterministic benchmark fit.

    Systems whose expected_results file declares
    ``"fitting_path": "orbit3"`` are routed through the real
    ORBIT3 hierarchical-triple fitting path
    (run_orbit3_benchmark()) instead of the plain binary path
    below, which assumes a flat OrbitalElements model and is
    not physically correct for a hierarchical triple.
    """

    # ------------------------------------------------------
    # Load metadata
    # ------------------------------------------------------

    with open(
        config["expected_file"],
        "r"
    ) as f:

        expected = json.load(f)

    if expected.get("fitting_path") == "orbit3":

        return run_orbit3_benchmark(
            system_name,
            config,
            expected
        )

    # ------------------------------------------------------
    # Parse dataset
    # ------------------------------------------------------

    parsed = parse_orbit_file(
        str(config["data_file"])
    )

    elements = parsed["elements"]

    visual_data = parsed["visual_data"]

    rv_data = parsed["rv_data"]

    # ------------------------------------------------------
    # Convert visual observations
    # ------------------------------------------------------

    t = visual_data[:,0]

    theta = np.deg2rad(
        visual_data[:,1]
    )

    rho = visual_data[:,2]

    sigma = visual_data[:,3]

    X = rho * np.cos(theta)

    Y = rho * np.sin(theta)

    sx = sigma

    sy = sigma

    # ------------------------------------------------------
    # Run fit
    # ------------------------------------------------------

    fit_result = fit_visual_orbit(
        elements,
        t,
        X,
        Y,
        sx,
        sy,
        rv_data=rv_data
    )

    fitted = fit_result.elements

    expected_elements = expected["initial_parameters"]

    # ------------------------------------------------------
    # Residual diagnostics
    # ------------------------------------------------------

    res_visual = visual_xy_residuals(
        fitted.fitted_parameters(),
        fitted,
        t,
        X,
        Y,
        sx,
        sy
    )

    if len(rv_data) > 0:

        res_rv = rv_residuals(
            fitted.fitted_parameters(),
            fitted,
            rv_data
        )

        stats = combined_fit_summary(
            visual_residuals=res_visual,
            rv_residuals=res_rv,
            rv_components=rv_data[:,3],
            n_params=len(
                fitted.fitted_parameters()
            )
        )

    else:

        stats = combined_fit_summary(
            visual_residuals=res_visual,
            rv_residuals=None,
            rv_components=None,
            n_params=len(
                fitted.fitted_parameters()
            )
        )

    n_params = len(
        fitted.fitted_parameters()
    )

    n_data = (
        len(res_visual)
        +
        (
            len(res_rv)
            if len(rv_data) > 0
            else 0
        )
    )

    stats = _normalize_statistics(
        stats,
        chi_square=stats["chi2_total"],
        reduced_chi_square=stats["reduced_chi2"],
        rms=stats["rms"],
        n_data=n_data,
        n_params=n_params
    )

    parameter_errors = {}

    for name in expected_elements:

        expected_value = (
            expected_elements[name]
        )

        fitted_value = getattr(
            fitted,
            name
        )

        rel_error = (
            parameter_relative_error(
                name,
                expected_value,
                fitted_value
            )
        )

        parameter_errors[name] = {

            "expected": float(
                expected_value
            ),

            "fitted": float(
                fitted_value
            ),

            "relative_error": float(
                rel_error
            )
        }

    return {

        "system": system_name,

        "orbit_type": expected.get(
            "orbit_type",
            "binary"
        ),

        "fitting_path": "binary",

        "paper_table_row": expected.get(
            "paper_table_row"
        ),

        "statistics": stats,

        "parameter_errors": parameter_errors,

        "fitted_elements": {
            name: float(
                getattr(fitted, name)
            )
            for name in expected_elements
        },

        "optimizer_success": bool(
            fit_result.optimizer_result.success
        ),

        "optimizer_message":
        str(
            fit_result.optimizer_result.message
        ),

        "mass_dynamical_Msun": None,

        "mass_sigma_Msun": None,

        "mass_relative_sigma": None,
    }


# ==========================================================
# Registry-driven cross-system validation
# ==========================================================
#
# The functions below implement the "writes validation/results/"
# behavior CLAUDE.md documents for `python validation/benchmark_runner.py`:
# run every supported system listed in benchmark_registry.csv
# through run_benchmark() (which itself dispatches each system
# to either the plain binary path or, for hierarchical triples
# such as ZetaAqr and HIP111805, run_orbit3_benchmark()) and
# write one JSON result file per system, plus a console summary
# table -- the single place all benchmark results are produced
# together, so that "are the benchmark results consistent"
# is something this script itself checks on every run rather
# than something inspected by hand per-system.

_VALIDATION_DIR = Path(__file__).resolve().parent

_REGISTRY_PATH = _VALIDATION_DIR / "benchmark_registry.csv"

_SYSTEMS_DIR = _VALIDATION_DIR / "benchmark_systems"

_EXPECTED_DIR = _VALIDATION_DIR / "expected_results"

_RESULTS_DIR = _VALIDATION_DIR / "results"


def _load_registry(
    registry_path=_REGISTRY_PATH
):
    """
    Load benchmark_registry.csv.

    Blank leading/trailing whitespace in field values (the
    registry's "notes" column, in particular) is stripped.
    Rows with no filename (e.g. a stray blank line) are
    skipped.
    """

    rows = []

    with open(
        registry_path,
        "r",
        newline=""
    ) as f:

        # A leading blank line (as in the current
        # benchmark_registry.csv) would otherwise be read by
        # csv.DictReader as the header row, silently
        # producing zero usable rows -- skip blank lines
        # before the real header instead.

        lines = [
            line
            for line in f
            if line.strip()
        ]

        reader = csv.DictReader(
            lines
        )

        for row in reader:

            filename = (
                row.get("filename") or ""
            ).strip()

            if not filename:

                continue

            rows.append({

                key: (
                    value.strip()
                    if isinstance(value, str)
                    else value
                )
                for key, value in row.items()
            })

    return rows


def _is_supported(
    row
):
    """
    A registry row is supported when its "supported" column
    is "yes" (case-insensitive). "partial" and "no" are both
    treated as not-yet-ready for the full benchmark run.
    """

    return (
        row.get("supported", "")
        .strip()
        .lower()
        == "yes"
    )


def run_all_benchmarks(
    registry_path=_REGISTRY_PATH,
    systems_dir=_SYSTEMS_DIR,
    expected_dir=_EXPECTED_DIR,
    results_dir=_RESULTS_DIR,
    write_results=True
):
    """
    Run every supported benchmark system in the registry and
    (by default) write one JSON result file per system into
    ``results_dir``.

    The data file's stem (e.g. "ZetaAqr" from "ZetaAqr.in2")
    is the single naming key shared by the benchmark_systems/
    subdirectory, the expected_results/ file, and the
    results/ file -- e.g. filename "ZetaAqr.in2" resolves to
    benchmark_systems/ZetaAqr/ZetaAqr.in2,
    expected_results/ZetaAqr.json, and results/ZetaAqr.json.

    Returns
    -------
    dict
        Mapping of system stem -> either the run_benchmark()
        result dict, or {"error": "..."} if that system's fit
        raised an exception (a failure in one system must not
        abort the rest of the registry run).
    """

    results = {}

    for row in _load_registry(
        registry_path
    ):

        if not _is_supported(row):

            continue

        filename = row["filename"]

        stem = Path(filename).stem

        data_file = (
            systems_dir / stem / filename
        )

        expected_file = (
            expected_dir / f"{stem}.json"
        )

        config = {

            "data_file": data_file,

            "expected_file": expected_file,
        }

        try:

            result = run_benchmark(
                stem,
                config
            )

        except Exception as exc:

            result = {

                "system": stem,

                "error": f"{type(exc).__name__}: {exc}",
            }

        results[stem] = result

        if write_results:

            results_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            with open(
                results_dir / f"{stem}.json",
                "w"
            ) as f:

                json.dump(
                    result,
                    f,
                    indent=2
                )

                f.write(
                    "\n"
                )

    return results


def _print_summary(
    results
):
    """
    Print a one-line-per-system console summary of a
    run_all_benchmarks() result, including the worst
    (maximum) parameter relative error per system so
    cross-system inconsistencies are visible at a glance.
    """

    print()

    print(
        f"{'System':<12}"
        f"{'Path':<8}"
        f"{'Success':<9}"
        f"{'RedChi2':>10}"
        f"{'WorstParam':>14}"
        f"{'WorstRelErr':>13}"
    )

    print(
        "-" * 68
    )

    for stem, result in results.items():

        if "error" in result:

            print(
                f"{stem:<12}"
                f"{'ERROR':<8}"
                f"{'':<9}"
                f"{'':>10}"
                f"{'':>14}"
                f"  {result['error']}"
            )

            continue

        worst_name = ""

        worst_value = float("-inf")

        for name, errors in (
            result["parameter_errors"].items()
        ):

            rel_error = errors["relative_error"]

            if (
                rel_error == rel_error
                and rel_error > worst_value
            ):

                worst_value = rel_error

                worst_name = name

        print(
            f"{stem:<12}"
            f"{result.get('fitting_path', 'binary'):<8}"
            f"{str(result['optimizer_success']):<9}"
            f"{result['statistics']['reduced_chi_square']:>10.4f}"
            f"{worst_name:>14}"
            f"{worst_value:>13.4%}"
        )

    print()


if __name__ == "__main__":

    all_results = run_all_benchmarks()

    _print_summary(
        all_results
    )

    if any(
        "error" in result
        or not result["optimizer_success"]
        for result in all_results.values()
    ):

        sys.exit(1)