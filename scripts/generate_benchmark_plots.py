"""
Generate the full available plot set for the benchmark systems
that have a matching visualization function, and save the paper's
figures under validation/results/plots/<system>/ for a one-time
visual pass.

This is the paper's actual figure-generation pipeline. Every figure this script writes is saved as vector
PDF with grid lines disabled (``output_format="pdf"``,
``show_grid=False`` -- see the ``PAPER_OUTPUT_FORMAT``/
``PAPER_SHOW_GRID`` constants below), per A&A's specific request
for this paper's submitted figures. This is a PIPELINE-level
default only: every plotting function in
``pyseshat.visualization`` (and ``MCMCResult.corner_plot()``)
keeps its own general-purpose default (PNG, grid on -- except
``corner_plot()``, which defaults to grid off for its own
already-documented reason)
unchanged for interactive/general use elsewhere in the software.

Scope, and why it is 2 of 8 systems rather than 8 of 8
--------------------------------------------------------
`src/pyseshat/visualization/` has exactly five plotting
functions: ``plot_visual_orbit``, ``plot_residuals``,
``plot_rv_orbit`` (all operating on a flat, two-body
``OrbitalElements`` model), ``plot_separation_vs_time`` (same),
and ``plot_binary_hrdiagram`` (operating on evolutionary-track
data, ``HRDiagramData``, entirely unrelated to an orbital fit).

Of the 8 registered benchmark systems:

- FIN379, GL765 are plain visual+RV binaries -- a flat
  ``OrbitalElements`` model is exactly correct for them, so all
  four orbit/dynamics plotting functions apply directly. This
  script fits both (mirroring
  ``validation/benchmark_runner.py``'s ``run_benchmark()``) and
  generates the full set.
- ZetaAqr, HIP111805 are hierarchical triples fit through the
  20-parameter ORBIT3 formalism
  (``triple.orbit3_fit.fit_orbit3``). ``TripleOrbitalElements``
  does carry a real ``outer: OrbitalElements`` sub-object, so
  ``plot_visual_orbit``/``plot_separation_vs_time`` COULD be
  pointed at it -- but the outer-orbit astrometric observations
  these systems are fit against include the unresolved inner
  subsystem's astrometric wobble (the ``f`` parameter;
  a negative ``f`` is possible). A flat two-body curve plotted
  directly against those observations would show a real,
  wobble-driven residual pattern that a reader could easily
  mistake for a fit-quality problem rather than an intentional,
  already-modeled physical effect.
  A dedicated triple-aware plotting
  function, ``visualization.triple_orbit_plot.plot_triple_orbit``,
  handles these systems -- it draws BOTH ORBIT3 visual observables side by
  side (POS1, the resolved inner orbit; POS2 = outer + f*inner,
  the wobble-corrected wide/outer orbit), so neither curve is ever
  plotted against observations it does not actually model.
  ``generate_triple_orbit_plots()`` below (wrapping
  ``scripts/generate_triple_orbit_plots.py``, which fits both
  systems via the exact same steps ``validation/benchmark_runner.py``'s
  ``run_orbit3_benchmark()`` uses -- see that script's own module
  docstring) generates and saves both systems' figures here
  too.
- HD214222, HD191854, HD130669, HD184467 are
  unresolved-binary SED fits -- there is no orbital solution
  being fit at all (see examples/04_binary_sed_fit.py's
  docstring: "Unlike Examples 1-3, there is no orbital fit here
  at all"), so none of the orbit-plotting functions apply.
  ``plot_binary_hrdiagram`` could in principle be used, but
  requires a full evolutionary-track cross-check
  (``evolution/binary.py``) to be run per system first.
  Real Girardi (2000) track/isochrone data
  (VizieR catalog J/A+AS/141/371, Z=0.019, Overshoot=1 -- see
  ``evolution/parsers/girardi2000.py``'s module docstring) is
  used as ``evolution/grids.py``'s default grid, and
  ``generate_hrdiagrams()`` below generates a real HR-diagram
  figure for exactly these six systems (the four binary SED
  systems, plus Capella and AlphaCenAB's photometry rows) using
  ``pyseshat.evolution.benchmark_checks`` (the same pipeline
  ``scripts/run_evolutionary_checks.py`` uses for the numeric
  age/mass/evolutionary-stage report, so the plotted points and
  the reported numbers cannot silently drift apart). This is
  separate from -- and does not require -- the four flat-orbit
  systems' plots above; it runs against literature (Teff, R)
  values, not a fitted orbit.
- Capella additionally gets a corner plot of its MCMC posterior
  (``generate_capella_corner_plot()`` below, wrapping
  ``scripts/generate_capella_corner_plot.py``) -- the paper's
  representative MCMC-posterior figure, demonstrating the
  near-circular-orbit eccentricity-argument-of-periastron
  degeneracy. See that script's module docstring for details.

Run

    MPLBACKEND=Agg python validation/../scripts/generate_benchmark_plots.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np

# validation/ is not a package (see validation/test_benchmark_runner.py's
# own "from benchmark_runner import ..." style) -- add it to sys.path so
# generate_triple_orbit_plots (imported below) can import benchmark_runner
# the same way.
_REPO_ROOT_FOR_IMPORTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT_FOR_IMPORTS / "validation"))

from pyseshat.io.orbit_parser import parse_orbit_file
from pyseshat.inference.optimizer import fit_visual_orbit
from pyseshat.visualization.orbit_plot import (
    plot_visual_orbit,
    plot_residuals,
    plot_rv_orbit,
)
from pyseshat.visualization.dynamics_plot import plot_separation_vs_time
from pyseshat.analysis.dynamics import separation_vs_time
from pyseshat.inference.likelihood import visual_xy_residuals
from pyseshat.evolution.benchmark_checks import run_all_benchmark_checks
from pyseshat.evolution.hrdiagram import build_binary_hrdiagram_data
from pyseshat.visualization.hrdiagram_plot import plot_binary_hrdiagram
from pyseshat.stellar.atmospheres import load_ck04_grid
from pyseshat.photometry.filters import load_passband
from pyseshat.photometry.model import synthetic_binary_sed
from pyseshat.visualization.sed_plot import plot_sed_fit

import generate_triple_orbit_plots as _triple_orbit_plots
import generate_capella_corner_plot as _capella_corner_plot
import benchmark_runner as _benchmark_runner


REPO_ROOT = Path(__file__).resolve().parent.parent

SYSTEMS_DIR = REPO_ROOT / "validation" / "benchmark_systems"

PLOTS_DIR = REPO_ROOT / "validation" / "results" / "plots"

# ------------------------------------------------------------
# Paper figure-generation pipeline defaults.
# See the module docstring above.
# ------------------------------------------------------------

PAPER_OUTPUT_FORMAT = "pdf"
PAPER_SHOW_GRID = False

BINARY_SYSTEMS = {
    "FIN379": SYSTEMS_DIR / "FIN379" / "FIN379.inp",
    "GL765": SYSTEMS_DIR / "GL765" / "GL765.inp",
    # Joint orbit+photometry named case studies (see
    # scripts/build_joint_benchmarks.py) -- both are plain flat
    # OrbitalElements binaries, same as FIN379/GL765 above, so the
    # same fitting/plotting pipeline applies directly.
    # AlphaCenAB_orbit has no RV data (see its expected_results
    # JSON's data_provenance note) -- generate_for_system()'s
    # `if len(rv) > 0` guard already skips the RV plot for it.
    "Capella_orbit": SYSTEMS_DIR / "Capella_orbit" / "Capella_orbit.inp",
    "AlphaCenAB_orbit": SYSTEMS_DIR / "AlphaCenAB_orbit" / "AlphaCenAB_orbit.inp",
}

# The four Abushattal et al. 2025 binary SED benchmark systems
# (see generate_sed_fit_plots() below). Capella_photometry
# and AlphaCenAB_photometry are also binary_sed-path systems but are
# deliberately NOT included here: both are entirely SYNTHETIC
# benchmarks (see docs/guide/binary_sed.md's "CK04 grid coverage"
# section), so an "observed vs. model" SED-fit figure for them would
# be comparing PySeshat's model to data PySeshat itself generated --
# not a meaningful validation figure the way it is for the four real
# (if paper-synthetic-origin) Abushattal systems. This matches this
# script's own existing scoping precedent (e.g. generate_for_system()
# above only fits real orbit-observation systems).
SED_BENCHMARK_SYSTEMS = (
    "HD214222",
    "HD191854",
    "HD130669",
    "HD184467",
)


def _fit(data_file):

    parsed = parse_orbit_file(str(data_file))

    elements = parsed["elements"]

    visual = parsed["visual_data"]

    rv = parsed["rv_data"]

    t = visual[:, 0]
    theta = np.deg2rad(visual[:, 1])
    rho = visual[:, 2]
    sigma = visual[:, 3]

    X = rho * np.cos(theta)
    Y = rho * np.sin(theta)

    fit_result = fit_visual_orbit(
        elements, t, X, Y, sigma, sigma, rv_data=rv,
    )

    return fit_result.elements, t, X, Y, sigma, rv


def generate_for_system(
    name,
    data_file,
    output_format=PAPER_OUTPUT_FORMAT,
    show_grid=PAPER_SHOW_GRID,
):

    out_dir = PLOTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    fitted, t, X_obs, Y_obs, sigma, rv = _fit(data_file)

    # Dense model curve over the observed time span for the
    # visual-orbit overlay.
    from pyseshat.orbit.orbit_model import compute_visual_orbit

    t_model = np.linspace(t.min(), t.max(), 500)
    orbit = compute_visual_orbit(t_model, fitted)

    fig, ax = plot_visual_orbit(
        X_obs, Y_obs, orbit, show=False,
        show_grid=show_grid, output_format=output_format,
        save_path=out_dir / f"visual_orbit.{output_format}",
    )
    matplotlib.pyplot.close(fig)

    res = visual_xy_residuals(
        fitted.fitted_parameters(), fitted, t, X_obs, Y_obs, sigma, sigma,
    )
    fig, axes = plot_residuals(
        t, res, show=False,
        show_grid=show_grid, output_format=output_format,
        save_path=out_dir / f"residuals.{output_format}",
    )
    matplotlib.pyplot.close(fig)

    if len(rv) > 0:
        fig, ax = plot_rv_orbit(
            rv, fitted, phase=True, show=False,
            show_grid=show_grid, output_format=output_format,
            save_path=out_dir / f"rv_orbit_phased.{output_format}",
        )
        matplotlib.pyplot.close(fig)

    epochs = np.linspace(t.min(), t.max(), 500)
    dyn = separation_vs_time(fitted, epochs)
    fig, ax = plot_separation_vs_time(
        dyn["epoch"], dyn["rho"], show=False,
        show_grid=show_grid, output_format=output_format,
        save_path=out_dir / f"separation_vs_time.{output_format}",
    )
    matplotlib.pyplot.close(fig)

    print(f"{name}: wrote 4 plots to {out_dir}")


def generate_hrdiagrams(
    output_format=PAPER_OUTPUT_FORMAT,
    show_grid=PAPER_SHOW_GRID,
):
    """
    Generate one HR-diagram figure per benchmark system with an
    binary SED or joint-photometry component: HD214222,
    HD191854, HD130669, HD184467, Capella, AlphaCenAB.

    Uses pyseshat.evolution.benchmark_checks.run_all_benchmark_checks()
    -- the exact same nearest-track/independent-fit/common-age-fit
    pipeline scripts/run_evolutionary_checks.py uses for the numeric
    report -- so the points plotted here and the ages/masses reported
    there cannot silently drift apart.

    Saved to validation/results/plots/<system>/hrdiagram.<output_format>
    (a system-name-keyed subdirectory distinct from the Capella_orbit/
    AlphaCenAB_orbit orbit-fit directories above, since these are
    photometry+evolution figures, not orbit-fit figures).
    """

    results = run_all_benchmark_checks()

    for name, result in results.items():

        if result.solution is None:
            print(
                f"{name}: SKIPPED HR-diagram -- common-age fit "
                f"failed ({result.common_age_error})"
            )
            continue

        data = build_binary_hrdiagram_data(
            primary_track=result.primary.track,
            secondary_track=result.secondary.track,
            primary_observables=result.primary.observables,
            secondary_observables=result.secondary.observables,
            solution=result.solution,
        )

        out_dir = PLOTS_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = out_dir / f"hrdiagram.{output_format}"

        ax = plot_binary_hrdiagram(
            data, save_path=save_path,
            show_grid=show_grid, output_format=output_format,
        )
        matplotlib.pyplot.close(ax.figure)

        print(f"{name}: wrote 1 HR-diagram plot to {out_dir}")


def _magnitude_to_flux(magnitude, uncertainty, zero_point):
    """
    Convert a photometric magnitude (+ its uncertainty) to a flux
    density point for the SED-fit plot, inverting the same Pogson
    relation ``photometry.synthetic.pogson_magnitude()`` uses:

        flux = zero_point * 10**(-0.4 * magnitude)

    Uncertainty is propagated via standard linear error propagation
    of that (monotonic, smooth) relation,
    ``sigma_flux = flux * ln(10)/2.5 * sigma_magnitude`` -- accurate
    for this dataset's assumed sigma_mag=0.30 mag (a ~30% relative
    flux uncertainty; the linearization error at this magnitude of
    uncertainty is a few percent of the error bar itself, well
    within this figure's illustrative purpose).
    """

    flux = zero_point * 10.0 ** (-0.4 * magnitude)

    flux_uncertainty = flux * (np.log(10.0) / 2.5) * uncertainty

    return flux, flux_uncertainty


def generate_sed_fit_plots(
    output_format=PAPER_OUTPUT_FORMAT,
    show_grid=PAPER_SHOW_GRID,
):
    """
    Generate the binary SED-fit figure (essential figure 4,
    ``Paper_Roadmap.pdf``) for each of the four Abushattal et al.
    2025 benchmark systems: HD214222, HD191854, HD130669, HD184467.

    Self-consistency with validation/benchmark_registry.csv's
    already-reported fit-quality numbers is structural, not just
    checked after the fact: this function calls
    ``benchmark_runner.run_sed_benchmark()`` directly (the
    EXACT function ``validation/benchmark_runner.py``'s own
    ``run_benchmark()`` dispatches to, and that
    the reported reduced-chi2/worst-
    parameter numbers all come from) to obtain each system's fitted
    solution, rather than re-fitting independently or hand-copying
    parameter values from a JSON file -- so the curve this figure
    draws and the number the paper cites cannot silently drift
    apart. The plotted model SED is then built from that SAME
    fitted solution via ``photometry.model.synthetic_binary_sed()``,
    a sibling of the tested ``synthetic_binary_magnitudes()`` that
    ``run_sed_benchmark()`` itself uses internally (see that
    function's own docstring for the direct numerical
    cross-check confirming the two never disagree).

    Saved to
    ``validation/results/plots/<system>/sed_fit.<output_format>``.
    """

    for name in SED_BENCHMARK_SYSTEMS:

        expected_path = (
            REPO_ROOT / "validation" / "expected_results" / f"{name}.json"
        )

        with open(expected_path) as f:
            expected = json.load(f)

        # ------------------------------------------------------
        # Authoritative fitted solution -- the same call
        # validation/benchmark_runner.py's own run_benchmark()
        # makes for this system.
        # ------------------------------------------------------

        result = _benchmark_runner.run_sed_benchmark(
            name, dict(expected)
        )

        fitted = result["fitted_elements"]

        # ------------------------------------------------------
        # Rebuild the same atmosphere grid / passbands / zero
        # points run_sed_benchmark() builds internally (it
        # returns fit statistics, not these intermediate objects)
        # to compute the plotted model curves from the SAME
        # fitted solution.
        # ------------------------------------------------------

        atmosphere_grid = load_ck04_grid([
            _benchmark_runner._VALIDATION_DIR
            / "atmosphere_models"
            / relative_path
            for relative_path in expected["atmosphere_files"]
        ])

        fitted_passband_names = sorted({
            entry["passband"] for entry in expected["measurements"]
        })

        passbands = {
            pb_name: load_passband(
                filename=(
                    _benchmark_runner._VALIDATION_DIR
                    / "photometry"
                    / "passbands"
                    / _benchmark_runner._SED_PASSBAND_FILES[pb_name]
                ),
                name=pb_name,
            )
            for pb_name in fitted_passband_names
        }

        zero_points = {
            pb_name: _benchmark_runner._SED_ZERO_POINTS[pb_name]
            for pb_name in fitted_passband_names
        }

        sed = synthetic_binary_sed(
            atmosphere_grid=atmosphere_grid,
            passbands=passbands,
            primary_temperature=fitted["primary_temperature"],
            secondary_temperature=fitted["secondary_temperature"],
            primary_logg=expected["primary_logg"]["value"],
            secondary_logg=expected["secondary_logg"]["value"],
            metallicity=expected["metallicity"]["value"],
            primary_radius=fitted["primary_radius"],
            secondary_radius=fitted["secondary_radius"],
            distance=expected["distance_pc"]["value"],
            ebv=fitted["ebv"],
        )

        # ------------------------------------------------------
        # Observed (fitted) points, converted to flux.
        # ------------------------------------------------------

        observed_wavelength = []
        observed_flux = []
        observed_flux_uncertainty = []
        observed_labels = []

        for entry in expected["measurements"]:

            pb = passbands[entry["passband"]]
            zp = zero_points[entry["passband"]]

            flux, flux_unc = _magnitude_to_flux(
                entry["magnitude"], entry["uncertainty"], zp
            )

            observed_wavelength.append(pb.pivot_wavelength)
            observed_flux.append(flux)
            observed_flux_uncertainty.append(flux_unc)
            observed_labels.append(entry["passband"])

        # ------------------------------------------------------
        # Excluded (disclosed, not fit) points -- e.g. Stromgren
        # u, see docs/guide/binary_sed.md's "Stromgren u"
        # section. Loaded from the same lookup tables
        # run_sed_benchmark() itself uses, since an excluded
        # passband is (by definition) not in `passbands`/
        # `zero_points` above.
        # ------------------------------------------------------

        excluded_wavelength = []
        excluded_flux = []
        excluded_flux_uncertainty = []
        excluded_labels = []

        for entry in expected.get("excluded_measurements", []):

            pb_name = entry["passband"]

            excluded_pb = load_passband(
                filename=(
                    _benchmark_runner._VALIDATION_DIR
                    / "photometry"
                    / "passbands"
                    / _benchmark_runner._SED_PASSBAND_FILES[pb_name]
                ),
                name=pb_name,
            )

            excluded_zp = _benchmark_runner._SED_ZERO_POINTS[pb_name]

            flux, flux_unc = _magnitude_to_flux(
                entry["magnitude"], entry["uncertainty"], excluded_zp
            )

            excluded_wavelength.append(excluded_pb.pivot_wavelength)
            excluded_flux.append(flux)
            excluded_flux_uncertainty.append(flux_unc)
            excluded_labels.append(pb_name)

        # ------------------------------------------------------
        # Save.
        # ------------------------------------------------------

        out_dir = PLOTS_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = out_dir / f"sed_fit.{output_format}"

        stats = result["statistics"]

        fig, ax = plot_sed_fit(
            wavelength=sed["wavelength"],
            composite_flux=sed["composite_flux"],
            primary_flux=sed["primary_flux"],
            secondary_flux=sed["secondary_flux"],
            observed_wavelength=observed_wavelength,
            observed_flux=observed_flux,
            observed_flux_uncertainty=observed_flux_uncertainty,
            observed_labels=observed_labels,
            excluded_wavelength=excluded_wavelength,
            excluded_flux=excluded_flux,
            excluded_flux_uncertainty=excluded_flux_uncertainty,
            excluded_labels=excluded_labels,
            title=(
                f"{name} -- binary SED fit "
                f"(reduced $\\chi^2$={stats['reduced_chi_square']:.3f})"
            ),
            show=False,
            show_grid=show_grid,
            output_format=output_format,
            save_path=save_path,
        )
        matplotlib.pyplot.close(fig)

        print(
            f"{name}: wrote 1 SED-fit plot to {out_dir} "
            f"(reduced chi2={stats['reduced_chi_square']:.4f}, "
            "from benchmark_runner.run_sed_benchmark)"
        )


def generate_triple_orbit_plots(
    output_format=PAPER_OUTPUT_FORMAT,
    show_grid=PAPER_SHOW_GRID,
):
    """
    Generate the wobble-corrected ORBIT3 triple-orbit figures for
    ZetaAqr and HIP111805 (see module docstring).

    Delegates to scripts/generate_triple_orbit_plots.py's own
    generate_for_system() -- the canonical implementation, which
    also cross-checks the plotted fit's chi_square against
    validation/benchmark_runner.py's run_orbit3_benchmark() before
    saving anything, so this wiring cannot silently plot a stale or
    mismatched solution. See that script's module docstring for the
    full reasoning.
    """

    _triple_orbit_plots.PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    _triple_orbit_plots.generate_for_system(
        "ZetaAqr",
        _triple_orbit_plots.SYSTEMS_DIR / "ZetaAqr" / "ZetaAqr.in2",
        title=(
            "ζ Aquarii -- ORBIT3 hierarchical-triple solution "
            "(χ²=694.51)"
        ),
        out_name=f"triple_orbit.{output_format}",
        anomaly_epoch=_triple_orbit_plots.ZETAAQR_ANOMALY_EPOCH,
        anomaly_tolerance=_triple_orbit_plots.ZETAAQR_ANOMALY_TOLERANCE,
        output_format=output_format,
        show_grid=show_grid,
    )

    _triple_orbit_plots.generate_for_system(
        "HIP111805",
        _triple_orbit_plots.SYSTEMS_DIR / "HIP111805" / "HIP111805.in3",
        title="HIP 111805 -- ORBIT3 hierarchical-triple solution",
        out_name=f"triple_orbit.{output_format}",
        output_format=output_format,
        show_grid=show_grid,
    )


def generate_capella_corner_plot_figure(
    output_format=PAPER_OUTPUT_FORMAT,
    show_grid=PAPER_SHOW_GRID,
):
    """
    Generate the Capella MCMC posterior corner plot (the paper's
    representative MCMC-posterior figure -- see module docstring and
    scripts/generate_capella_corner_plot.py's own module docstring
    for the full investigation/reasoning).

    Delegates to that script's run_capella_mcmc()/save_samples()/
    generate_corner_plot() -- the canonical implementation -- rather
    than re-fitting here, so there is exactly one place the Capella
    MCMC run is actually performed.
    """

    posterior = _capella_corner_plot.run_capella_mcmc()

    _capella_corner_plot.save_samples(posterior)

    _capella_corner_plot.generate_corner_plot(
        posterior, output_format=output_format, show_grid=show_grid,
    )


def main():

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    for name, data_file in BINARY_SYSTEMS.items():
        generate_for_system(name, data_file)

    print()
    generate_triple_orbit_plots()

    print()
    generate_hrdiagrams()

    print()
    generate_sed_fit_plots()

    print()
    generate_capella_corner_plot_figure()


if __name__ == "__main__":
    main()
