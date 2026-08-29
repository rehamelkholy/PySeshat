"""
Run the evolutionary-track consistency check
(pyseshat.evolution.benchmark_checks, built on evolution/binary.py,
evolution/age.py, evolution/search.py) against the real Girardi
(2000) grid (Z=0.019, Overshoot=1 -- see
evolution/parsers/girardi2000.py's module docstring) for this
project's six benchmark systems with an Al-Wardat SED or joint
photometry component: HD214222, HD191854, HD130669, HD184467,
Capella, alpha Cen AB.

This is a thin CLI wrapper: the actual pipeline (which literature
Teff/R/mass to use per system, nearest-track selection, independent
and common-age fits, flagging) lives in
pyseshat.evolution.benchmark_checks so that this script and
scripts/generate_benchmark_plots.py (which needs the same fitted
tracks/solution objects to build each system's HR-diagram figure)
share one implementation rather than two that could drift apart.

For each system, prints:

- Each component's requested vs. nearest-available track mass (and
  the mismatch, with an explicit flag if it exceeds 5% or lands on
  an exact grid-spacing tie -- see HD184467 below).
- Each component's independent best-fit age/chi2/evolutionary state
  (dwarf/subgiant/giant, from log g) and the nearest raw "Stage"
  label in the source track (Stage is a sparse descriptive label,
  not this codebase's numeric EvolutionSample.phase -- see
  evolution/parsers/girardi2000.py).
- The age_difference between the two components' independent
  best-fit ages, and the joint common-age solution's age/chi2.

Writes validation/results/evolutionary_checks.json with the full
numeric results (used to update validation/expected_results/*.json's
evolutionary_track_source fields).

Run with: MPLBACKEND=Agg python scripts/run_evolutionary_checks.py
"""

from __future__ import annotations

import json
from pathlib import Path

from pyseshat.evolution.benchmark_checks import (
    BENCHMARK_SYSTEM_NAMES,
    load_benchmark_component_inputs,
    load_model_and_raw_tracks,
    run_system_evolutionary_check,
)
from pyseshat.evolution.parsers.girardi2000 import CITATION

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "validation" / "results"


def _serialize(result):

    def _component(c):
        return dict(
            requested_mass_Msun=c.requested_mass_Msun,
            nearest_track_mass_Msun=c.track.initial_mass,
            mass_mismatch_pct=c.mass_mismatch_pct,
            n_mass_ties=c.n_mass_ties,
            literature_teff_K=c.literature_teff_K,
            literature_radius_Rsun=c.literature_radius_Rsun,
            literature_log_luminosity=round(
                c.literature_log_luminosity, 4
            ),
            independent_best_fit_age_yr=c.fit.sample.age,
            independent_best_fit_chi2=round(c.fit.chi2, 4),
            independent_best_fit_log_teff=round(
                c.fit.sample.log_teff, 4
            ),
            independent_best_fit_log_luminosity=round(
                c.fit.sample.log_luminosity, 4
            ),
            independent_best_fit_log_g=round(
                c.fit.sample.log_g, 4
            ),
            evolutionary_state=c.evolutionary_state,
            nearest_raw_stage_label=c.nearest_raw_stage_label,
            nearest_raw_stage_label_age_yr=(
                c.nearest_raw_stage_label_age_yr
            ),
            flags=c.flags,
        )

    if result.common_age_fit is not None:
        from pyseshat.evolution.benchmark_checks import (
            evolutionary_state_from_logg,
        )
        common = dict(
            age_yr=result.common_age_fit.age,
            chi2=round(result.common_age_fit.chi2, 4),
            primary_log_g=round(
                result.common_age_fit.primary.log_g, 4
            ),
            secondary_log_g=round(
                result.common_age_fit.secondary.log_g, 4
            ),
            primary_evolutionary_state=evolutionary_state_from_logg(
                result.common_age_fit.primary.log_g
            ),
            secondary_evolutionary_state=(
                evolutionary_state_from_logg(
                    result.common_age_fit.secondary.log_g
                )
            ),
            error=None,
        )
    else:
        common = dict(error=result.common_age_error)

    return dict(
        grid="Girardi et al. (2000), Z=0.019, Overshoot=1",
        mass_source=result.mass_source,
        components=dict(
            primary=_component(result.primary),
            secondary=_component(result.secondary),
        ),
        age_difference_yr=result.independent_fit.age_difference,
        age_difference_pct_of_older_age=(
            result.age_difference_pct_of_older_age
        ),
        common_age_fit=common,
    )


def main():

    model, raw_tracks = load_model_and_raw_tracks()
    systems = load_benchmark_component_inputs()

    print(f"Girardi (2000) grid loaded: {model.name}, "
          f"{len(model.tracks)} tracks, {len(model.isochrones)} "
          f"isochrones, Z={model.available_metallicities}")
    print(f"Citation: {CITATION}")
    print()

    all_results = {}

    for name in BENCHMARK_SYSTEM_NAMES:

        payload = systems[name]

        result = run_system_evolutionary_check(
            name, payload, model, raw_tracks,
        )

        print(f"=== {name} ===")

        for c in (result.primary, result.secondary):
            print(f"  {c.role}: requested M={c.requested_mass_Msun} "
                  f"Msun -> nearest track M={c.track.initial_mass} "
                  f"Msun ({c.mass_mismatch_pct:.1f}% off); "
                  f"independent best-fit age={c.fit.sample.age:.3e} "
                  f"yr, chi2={c.fit.chi2:.3f}, "
                  f"state={c.evolutionary_state}, "
                  f"nearest Stage label={c.nearest_raw_stage_label!r}")
            for flag in c.flags:
                print(f"    FLAG: {flag}")

        print(f"  independent-fit age_difference: "
              f"{result.independent_fit.age_difference:.3e} yr "
              f"({result.age_difference_pct_of_older_age:.1f}% of "
              "the older independent age)")

        if result.common_age_fit is not None:
            from pyseshat.evolution.benchmark_checks import (
                evolutionary_state_from_logg,
            )
            print(f"  common-age fit: "
                  f"age={result.common_age_fit.age:.3e} yr, "
                  f"chi2={result.common_age_fit.chi2:.3f}, "
                  "primary=" + evolutionary_state_from_logg(
                      result.common_age_fit.primary.log_g
                  ) + ", secondary=" + evolutionary_state_from_logg(
                      result.common_age_fit.secondary.log_g
                  ))
        else:
            print(f"  common-age fit FAILED: "
                  f"{result.common_age_error}")

        print()

        all_results[name] = _serialize(result)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "evolutionary_checks.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
