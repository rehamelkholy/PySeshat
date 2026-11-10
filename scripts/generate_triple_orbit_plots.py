"""
Generate wobble-corrected ORBIT3 orbit plots for the two
hierarchical-triple benchmark systems (ZetaAqr, HIP111805).

Uses the SAME fitted solution validation/benchmark_runner.py's
run_orbit3_benchmark() produces for the paper's Section 3.2 benchmark
table (ZetaAqr: chi_square~=686.84 since the 0.2.0 fit fix) -- NOT
a separately re-fit solution -- so the plotted curve and the reported
numeric benchmark result cannot silently drift apart. This script
prints run_orbit3_benchmark()'s own chi_square for each system so that
correspondence is checked every time the plots are regenerated, not
just assumed.

No new fitting code: run_orbit3_benchmark() itself does not return the
fitted TripleOrbitalElements object or the observation arrays needed
for plotting (only a flattened numeric summary), so this script
recovers them by re-invoking the exact same, already-existing,
already-tested steps run_orbit3_benchmark() itself performs internally
(parse_triple_file -> _convert_legacy_orbit3_observations [.in2 only]
-> _build_orbit3_elements -> fit_orbit3), reusing benchmark_runner's
own helper functions verbatim rather than reimplementing any of them.

Uses pyseshat.visualization.triple_orbit_plot.plot_triple_orbit() (see
that module's docstring) to draw both ORBIT3 visual
observables side by side: POS1 (the resolved inner orbit) and POS2
(the wobble-corrected wide/outer orbit, POS2 = r_outer + f*r_inner) --
never a raw, un-wobble-corrected outer orbit against observations that
actually contain the wobble term (see docs/guide/visualization.md /
docs/guide/triple_systems.md).

ZetaAqr's one known residual-anomaly epoch (2009.7552) is drawn with a distinct
marker/color on the inner-orbit panel.

Run with:
    MPLBACKEND=Agg PYTHONPATH=src:validation python3 \
        scripts/generate_triple_orbit_plots.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "src", _ROOT / "validation"):   # works without PYTHONPATH (also on Windows)
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyseshat.triple.triple_parser import parse_triple_file
from pyseshat.triple.triple_orbit import compute_orbit3_visual
from pyseshat.visualization.triple_orbit_plot import plot_triple_orbit

from benchmark_runner import (  # noqa: E402  (validation/ on sys.path)
    _convert_legacy_orbit3_observations,
    _build_orbit3_elements,
    run_orbit3_benchmark,
)
from pyseshat.triple.orbit3_fit import fit_orbit3
from pyseshat.triple.triple_residuals import orbit3_residuals


REPO_ROOT = Path(__file__).resolve().parent.parent
VALIDATION_DIR = REPO_ROOT / "validation"
SYSTEMS_DIR = VALIDATION_DIR / "benchmark_systems"
EXPECTED_DIR = VALIDATION_DIR / "expected_results"
PLOTS_DIR = VALIDATION_DIR / "results" / "plots"

ZETAAQR_ANOMALY_EPOCH = 2009.7552
ZETAAQR_ANOMALY_TOLERANCE = 1e-3

# ------------------------------------------------------------
# Paper figure-generation pipeline defaults.
#
# A&A has specifically requested vector PDF without grid lines
# for this paper's submitted figures. This
# is a PIPELINE-level default, not a plotting-function-level
# one: plot_triple_orbit() itself still defaults to
# output_format="png", show_grid=True (unchanged, for general/
# interactive use elsewhere in the software) -- this script
# explicitly overrides both defaults for every figure it writes.
# ------------------------------------------------------------

PAPER_OUTPUT_FORMAT = "pdf"
PAPER_SHOW_GRID = False


def _fit_via_canonical_orbit3_path(stem, data_file):
    """
    Reproduce run_orbit3_benchmark()'s own parse/convert/build/fit
    sequence exactly, using its own helper functions, purely to get
    back the objects (fitted elements, observations, rv_observations)
    it computes internally but does not return. See module docstring.
    """

    expected_file = EXPECTED_DIR / f"{stem}.json"

    with open(expected_file, "r") as f:
        expected = json.load(f)

    parsed = parse_triple_file(str(data_file))

    observations = parsed["observations"]
    rv_observations = parsed["rv_observations"]

    if parsed["metadata"].get("format") == "in2":
        observations = _convert_legacy_orbit3_observations(observations)

    elements = _build_orbit3_elements(
        expected["initial_parameters"], expected["fitted_flags"]
    )

    fit_result = fit_orbit3(elements, observations, rv_observations)

    return fit_result.elements, observations, rv_observations, expected


def _split_by_tag(observations):
    """
    Split a list of ORBIT3-tagged TripleObservation into (I1, I2)
    sub-lists (I1 = POS1/inner, I2 = POS2/outer+wobble).
    """

    i1 = [o for o in observations if o.orbit_id == "I1"]
    i2 = [o for o in observations if o.orbit_id == "I2"]

    return i1, i2


def _obs_XY(observations):
    theta = np.deg2rad(np.array([o.theta for o in observations]))
    rho = np.array([o.rho for o in observations])
    X = rho * np.cos(theta)
    Y = rho * np.sin(theta)
    return X, Y


def generate_for_system(
    stem,
    data_file,
    title,
    out_name,
    anomaly_epoch=None,
    anomaly_tolerance=1e-3,
    output_format=PAPER_OUTPUT_FORMAT,
    show_grid=PAPER_SHOW_GRID,
):

    print(f"=== {stem} ===")

    # ------------------------------------------------------
    # 1. Confirm against the canonical benchmark numeric
    #    result (the one the paper actually cites), so a
    #    stale/mismatched fit is never silently plotted.
    # ------------------------------------------------------

    config = {
        "data_file": data_file,
        "expected_file": EXPECTED_DIR / f"{stem}.json",
    }

    with open(config["expected_file"], "r") as f:
        expected = json.load(f)

    benchmark_result = run_orbit3_benchmark(stem, config, expected)

    print(
        "  benchmark_runner.run_orbit3_benchmark(): "
        f"chi_square={benchmark_result['statistics']['chi_square']:.4f} "
        "reduced_chi_square="
        f"{benchmark_result['statistics']['reduced_chi_square']:.4f} "
        f"optimizer_success={benchmark_result['optimizer_success']}"
    )

    # ------------------------------------------------------
    # 2. Recover the fitted elements + observations for
    #    plotting, via the identical steps (see module
    #    docstring -- no new fitting code).
    # ------------------------------------------------------

    fitted, observations, rv_observations, _ = (
        _fit_via_canonical_orbit3_path(stem, data_file)
    )

    final_residuals = orbit3_residuals(observations, rv_observations, fitted)
    final_chi2 = float(np.sum(final_residuals**2))

    print(
        f"  plotting script's own re-derivation: chi_square={final_chi2:.4f}"
        " (must match the line above)"
    )

    if not np.isclose(
        final_chi2, benchmark_result["statistics"]["chi_square"], rtol=1e-6
    ):
        raise RuntimeError(
            f"{stem}: plotted fit's chi_square ({final_chi2}) does not "
            "match run_orbit3_benchmark()'s own reported chi_square "
            f"({benchmark_result['statistics']['chi_square']}) -- refusing "
            "to plot a solution that isn't the one the paper actually "
            "cites."
        )

    # ------------------------------------------------------
    # 3. Dense ORBIT3 model curves (POS1, POS2) over the
    #    observed epoch span, following the same
    #    "linspace over the observed time span" convention
    #    scripts/generate_benchmark_plots.py already uses
    #    for the flat-orbit systems.
    # ------------------------------------------------------

    all_epochs = np.array(
        [o.epoch for o in observations]
        + [o.epoch for o in rv_observations]
    )

    epoch_model = np.linspace(all_epochs.min(), all_epochs.max(), 1000)

    orbit3_model = compute_orbit3_visual(epoch_model, fitted)

    # ------------------------------------------------------
    # 4. Split observations by ORBIT3 tag and build the
    #    optional anomaly-epoch highlight mask.
    # ------------------------------------------------------

    i1_obs, i2_obs = _split_by_tag(observations)

    inner_X, inner_Y = _obs_XY(i1_obs)
    outer_X, outer_Y = _obs_XY(i2_obs)

    inner_highlight = None

    if anomaly_epoch is not None:
        inner_epochs = np.array([o.epoch for o in i1_obs])
        inner_highlight = (
            np.abs(inner_epochs - anomaly_epoch) < anomaly_tolerance
        )
        n_flagged = int(inner_highlight.sum())
        print(f"  anomaly-epoch highlight: {n_flagged} point(s) flagged")

    # ------------------------------------------------------
    # 5. Draw and save.
    # ------------------------------------------------------

    out_dir = PLOTS_DIR / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / out_name

    fig, (ax_inner, ax_outer) = plot_triple_orbit(
        inner_X,
        inner_Y,
        outer_X,
        outer_Y,
        orbit3_model,
        title=title,
        inner_highlight_mask=inner_highlight,
        highlight_label=(
            f"Epoch {anomaly_epoch} (residual anomaly, Sect. 3.2)"
            if anomaly_epoch is not None
            else "Flagged observation"
        ),
        show=False,
        show_grid=show_grid,
        output_format=output_format,
        save_path=out_path,
    )

    plt.close(fig)

    print(f"  wrote {out_path}")
    print()

    return {
        "path": out_path.relative_to(REPO_ROOT).as_posix(),
        "chi_square": final_chi2,
        "n_inner_obs": len(i1_obs),
        "n_outer_obs": len(i2_obs),
        "n_rv_obs": len(rv_observations),
    }


def main():

    manifest = {}

    manifest["ZetaAqr"] = generate_for_system(
        "ZetaAqr",
        SYSTEMS_DIR / "ZetaAqr" / "ZetaAqr.in2",
        title="ζ Aquarii -- ORBIT3 hierarchical-triple solution",
        out_name="triple_orbit.pdf",
        anomaly_epoch=ZETAAQR_ANOMALY_EPOCH,
        anomaly_tolerance=ZETAAQR_ANOMALY_TOLERANCE,
    )

    manifest["HIP111805"] = generate_for_system(
        "HIP111805",
        SYSTEMS_DIR / "HIP111805" / "HIP111805.in3",
        title="HIP 111805 -- ORBIT3 hierarchical-triple solution",
        out_name="triple_orbit.pdf",
    )

    manifest_path = PLOTS_DIR / "triple_orbit_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
