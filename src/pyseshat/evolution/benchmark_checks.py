"""
Reusable evolutionary-consistency-check pipeline for this project's
six SED/joint-photometry benchmark systems: HD214222,
HD191854, HD130669, HD184467, Capella, alpha Cen AB.

This module exists so that the two scripts that need this exact
same per-system pipeline -- ``scripts/run_evolutionary_checks.py``
(the numeric age/mass/evolutionary-stage report) and
``scripts/generate_benchmark_plots.py`` (the HR-diagram figures) --
share one implementation rather than two independently-maintained
copies that could silently drift apart (e.g. a plotted point that
doesn't match the reported number because one script's nearest-track
selection changed and the other's didn't).

See ``pyseshat.evolution.parsers.girardi2000``'s module docstring
for the Girardi (2000) real-data handling this builds on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pyseshat.analysis.luminosity import compute_log_luminosity
from pyseshat.analysis.masses import component_masses_from_elements
from pyseshat.evolution.age import (
    CommonAgeFitResult,
    TrackFitResult,
    fit_common_age,
    fit_track,
)
from pyseshat.evolution.binary import BinaryTrackFitResult, fit_binary_tracks
from pyseshat.evolution.grids import (
    VIZIER_TRACKS_FILENAME,
    _default_data_dir,
    load_default_evolution_model,
)
from pyseshat.evolution.models import EvolutionModel, EvolutionTrack
from pyseshat.evolution.observables import StellarObservables
from pyseshat.evolution.parsers.girardi2000 import (
    DEFAULT_OVERSHOOT,
    SOLAR_METALLICITY,
    _read_vizier_asu_tsv,
)
from pyseshat.evolution.search import nearest_track
from pyseshat.evolution.solution import BinaryEvolutionSolution
from pyseshat.orbit.orbital_elements import OrbitalElements

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPECTED_RESULTS = REPO_ROOT / "validation" / "expected_results"
TRACKS_TSV = _default_data_dir() / VIZIER_TRACKS_FILENAME

MASS_MISMATCH_WARN_PCT = 5.0

#: Age of the universe (Planck 2018, ~13.787 Gyr), used only as a
#: sanity bound to flag an independent single-track age fit that is
#: physically implausible taken alone.
HUBBLE_TIME_YR = 13.787e9

#: If the tabulated-age grid spacing around a track's best-fit
#: point exceeds this fraction of the age itself, flag the age as
#: only loosely constrained by the grid's own sampling there,
#: independent of how good the chi2 looks.
SPARSE_AGE_GRID_WARN_PCT = 20.0

#: Capella's literature SB2 orbital elements (Torres et al. 2015
#: Table 1), used only to derive component masses via
#: component_masses_from_elements() -- no literature mass is
#: separately recorded anywhere else in this repository for Capella.
CAPELLA_ELEMENTS = dict(
    P=104.02128 / 365.25,
    e=0.00089,
    i=137.156,
    K1=25.9611,
    K2=26.860,
)

BENCHMARK_SYSTEM_NAMES = (
    "HD214222", "HD191854", "HD130669", "HD184467",
    "Capella", "AlphaCenAB",
)


@dataclass
class ComponentEvolutionResult:

    role: str
    requested_mass_Msun: float
    literature_teff_K: float
    literature_radius_Rsun: float
    literature_log_luminosity: float
    track: EvolutionTrack
    observables: StellarObservables
    fit: TrackFitResult
    mass_mismatch_pct: float
    n_mass_ties: int
    nearest_raw_stage_label: str | None
    nearest_raw_stage_label_age_yr: float | None
    evolutionary_state: str
    flags: list[str]


@dataclass
class SystemEvolutionResult:

    system_name: str
    mass_source: str
    primary: ComponentEvolutionResult
    secondary: ComponentEvolutionResult
    independent_fit: BinaryTrackFitResult
    age_difference_pct_of_older_age: float
    common_age_fit: CommonAgeFitResult | None
    common_age_error: str | None
    solution: BinaryEvolutionSolution | None


def _log_luminosity(teff_k, radius_rsun):

    return float(
        compute_log_luminosity(
            radius_solar=radius_rsun,
            temperature_kelvin=teff_k,
        )
    )


def load_benchmark_component_inputs():
    """
    Returns {system_name: {"components": {"primary": {...},
    "secondary": {...}}, "mass_source": str}} for all six systems,
    read from validation/expected_results/*.json (literature
    Teff/R/mass directly for the four Abushattal et al. (2025) systems and
    AlphaCenAB; Capella's mass is derived from its literature SB2
    orbital elements -- see this module's docstring).
    """

    systems = {}

    for name in ("HD214222", "HD191854", "HD130669", "HD184467"):

        with open(EXPECTED_RESULTS / f"{name}.json") as f:
            d = json.load(f)

        lit = d["literature_component_parameters"]

        comp = {}

        for role in ("primary", "secondary"):

            teff = lit[role]["teff_K"]
            radius = lit[role]["radius_Rsun"]
            mass = lit[role]["mass_Msun"]

            comp[role] = dict(
                teff_K=teff,
                radius_Rsun=radius,
                mass_Msun=mass,
                log_teff=float(np.log10(teff)),
                log_luminosity=_log_luminosity(teff, radius),
            )

        systems[name] = dict(
            components=comp,
            mass_source="literature_component_parameters "
                        "(validation/expected_results, direct)",
        )

    with open(EXPECTED_RESULTS / "AlphaCenAB_photometry.json") as f:
        aca_phot = json.load(f)

    with open(EXPECTED_RESULTS / "AlphaCenAB_orbit.json") as f:
        aca_orbit = json.load(f)

    lit = aca_phot["literature_component_parameters"]
    masses = aca_orbit["component_masses_Msun"]

    comp = {}
    for role in ("primary", "secondary"):
        teff = lit[role]["teff_K"]
        radius = lit[role]["radius_Rsun"]
        mass = masses[role]
        comp[role] = dict(
            teff_K=teff,
            radius_Rsun=radius,
            mass_Msun=mass,
            log_teff=float(np.log10(teff)),
            log_luminosity=_log_luminosity(teff, radius),
        )

    systems["AlphaCenAB"] = dict(
        components=comp,
        mass_source=f"AlphaCenAB_orbit.json component_masses_Msun "
                    f"({masses['source']})",
    )

    with open(EXPECTED_RESULTS / "Capella_photometry.json") as f:
        cap_phot = json.load(f)

    lit = cap_phot["literature_component_parameters"]

    elements = OrbitalElements(
        P=CAPELLA_ELEMENTS["P"], T0=0.0, e=CAPELLA_ELEMENTS["e"],
        a=1.0, Omega=0.0, omega=0.0, i=CAPELLA_ELEMENTS["i"],
        K1=CAPELLA_ELEMENTS["K1"], K2=CAPELLA_ELEMENTS["K2"], V0=0.0,
    )
    cap_masses = component_masses_from_elements(elements)

    comp = {}
    for role, mass_key in (("primary", "M1"), ("secondary", "M2")):
        teff = lit[role]["teff_K"]
        radius = lit[role]["radius_Rsun"]
        mass = cap_masses[mass_key]
        comp[role] = dict(
            teff_K=teff,
            radius_Rsun=radius,
            mass_Msun=mass,
            log_teff=float(np.log10(teff)),
            log_luminosity=_log_luminosity(teff, radius),
        )

    systems["Capella"] = dict(
        components=comp,
        mass_source="derived from literature SB2 orbital elements "
                    "(Torres et al. 2015 Table 1, P/e/i/K1/K2) via "
                    "analysis.masses.component_masses_from_elements "
                    "-- no separately-stated literature mass exists "
                    f"in this repository. M1={cap_masses['M1']:.4f}, "
                    f"M2={cap_masses['M2']:.4f} Msun.",
    )

    return systems


def _nearest_stage_label(raw_tracks_df, mass, age, metallicity, overshoot):

    sub = raw_tracks_df[
        (raw_tracks_df["Overshoot"] == overshoot)
        & np.isclose(raw_tracks_df["Z"], metallicity, rtol=1e-6, atol=1e-8)
        & np.isclose(raw_tracks_df["Mass"], mass, rtol=1e-9)
    ]

    if sub.empty:
        return None, None

    idx = (sub["Age"] - age).abs().idxmin()
    row = sub.loc[idx]
    stage = row["Stage"]

    if isinstance(stage, float) and np.isnan(stage):
        stage = None

    return stage, float(row["Age"])


def _local_age_spacing_pct(age_array, index):
    """
    Local relative spacing of the tabulated age grid around
    ``index``, as a percentage of the age at that point -- e.g. 5%
    means neighboring tabulated ages differ by only 5% of the age
    itself (well-sampled); 50% means the nearest tabulated
    neighbors are half the age apart (coarse). Returns None at an
    array edge where only one neighbor exists, or for a
    single-point track.
    """

    if len(age_array) < 2:
        return None

    age = age_array[index]

    if age <= 0:
        return None

    if index == 0:
        gap = age_array[1] - age_array[0]
    elif index == len(age_array) - 1:
        gap = age_array[-1] - age_array[-2]
    else:
        gap = 0.5 * (
            age_array[index + 1] - age_array[index - 1]
        )

    return 100.0 * gap / age


def evolutionary_state_from_logg(log_g):

    if log_g >= 4.0:
        return "dwarf/main-sequence (log g >= 4.0)"
    if log_g >= 3.0:
        return "subgiant (3.0 <= log g < 4.0)"
    return "giant (log g < 3.0)"


def _run_component(
    role, c, model, raw_tracks, available_masses,
):

    requested_mass = c["mass_Msun"]

    track = nearest_track(
        model, initial_mass=requested_mass,
        metallicity=SOLAR_METALLICITY,
    )

    distances = np.abs(available_masses - requested_mass)
    min_dist = distances.min()
    n_ties = int(np.sum(np.isclose(distances, min_dist)))

    mismatch_pct = (
        abs(track.initial_mass - requested_mass)
        / requested_mass * 100.0
    )

    obs = StellarObservables(
        log_teff=c["log_teff"], sigma_log_teff=0.01,
        log_luminosity=c["log_luminosity"],
        sigma_log_luminosity=0.05,
    )

    fit = fit_track(track, observables=obs)

    stage, stage_age = _nearest_stage_label(
        raw_tracks, track.initial_mass, fit.sample.age,
        SOLAR_METALLICITY, DEFAULT_OVERSHOOT,
    )

    evol_state = evolutionary_state_from_logg(fit.sample.log_g)

    flags = []
    if mismatch_pct > MASS_MISMATCH_WARN_PCT:
        flags.append(
            f"nearest available track mass ({track.initial_mass}) "
            f"differs from requested ({requested_mass}) by "
            f"{mismatch_pct:.1f}% (> {MASS_MISMATCH_WARN_PCT}% "
            "threshold)"
        )
    if n_ties > 1:
        flags.append(
            f"exact tie in nearest_track mass-distance "
            f"minimization ({n_ties} tracks equidistant from "
            f"{requested_mass} Msun) -- nearest_track picked the "
            f"first in iteration order ({track.initial_mass} "
            "Msun); this is a genuine grid-spacing coincidence, "
            "not a bug"
        )
    if requested_mass < 0.15 or requested_mass > 7.0:
        flags.append(
            "requested mass falls outside the grid's 0.15-7.0 "
            "Msun coverage"
        )
    if fit.sample.age > HUBBLE_TIME_YR:
        flags.append(
            f"independent best-fit age ({fit.sample.age:.3e} yr) "
            f"exceeds a Hubble time ({HUBBLE_TIME_YR:.3e} yr) -- "
            "physically implausible for this component read "
            "alone; the joint common-age fit below may still be "
            "well-constrained by the other component"
        )

    local_spacing_pct = _local_age_spacing_pct(
        track.age, fit.index
    )
    if (
        local_spacing_pct is not None
        and local_spacing_pct > SPARSE_AGE_GRID_WARN_PCT
    ):
        flags.append(
            f"best-fit age point falls in a sparsely-sampled part "
            f"of this track's age grid (local spacing "
            f"{local_spacing_pct:.0f}% of the age there, > "
            f"{SPARSE_AGE_GRID_WARN_PCT:.0f}% threshold) -- the "
            "age is only as well-constrained as this spacing "
            "allows, independent of the fit's chi2"
        )

    return ComponentEvolutionResult(
        role=role,
        requested_mass_Msun=requested_mass,
        literature_teff_K=c["teff_K"],
        literature_radius_Rsun=c["radius_Rsun"],
        literature_log_luminosity=c["log_luminosity"],
        track=track,
        observables=obs,
        fit=fit,
        mass_mismatch_pct=round(mismatch_pct, 3),
        n_mass_ties=n_ties,
        nearest_raw_stage_label=stage,
        nearest_raw_stage_label_age_yr=stage_age,
        evolutionary_state=evol_state,
        flags=flags,
    )


def run_system_evolutionary_check(
    system_name, payload, model, raw_tracks,
):
    """
    Run the full nearest-track / independent-fit / common-age-fit
    pipeline for one system.

    Parameters
    ----------
    system_name : str
    payload : dict
        One value from load_benchmark_component_inputs().
    model : EvolutionModel
        Typically load_default_evolution_model()'s return value.
    raw_tracks : pandas.DataFrame
        Typically _read_vizier_asu_tsv(TRACKS_TSV)'s return value
        (passed in rather than re-read per system, since it is
        the same for all six systems and re-parsing a 13,723-row
        file six times is wasted work).

    Returns
    -------
    SystemEvolutionResult
    """

    comp = payload["components"]
    available_masses = np.array(sorted(model.available_track_masses))

    primary = _run_component(
        "primary", comp["primary"], model, raw_tracks,
        available_masses,
    )
    secondary = _run_component(
        "secondary", comp["secondary"], model, raw_tracks,
        available_masses,
    )

    independent_fit = fit_binary_tracks(
        primary_track=primary.track,
        secondary_track=secondary.track,
        primary_log_teff=comp["primary"]["log_teff"],
        primary_log_luminosity=comp["primary"]["log_luminosity"],
        primary_sigma_log_teff=0.01,
        primary_sigma_log_luminosity=0.05,
        secondary_log_teff=comp["secondary"]["log_teff"],
        secondary_log_luminosity=comp["secondary"]["log_luminosity"],
        secondary_sigma_log_teff=0.01,
        secondary_sigma_log_luminosity=0.05,
    )

    age_diff_pct = (
        100.0 * independent_fit.age_difference
        / max(
            primary.fit.sample.age,
            secondary.fit.sample.age,
            1.0,
        )
    )

    common_age_fit = None
    common_age_error = None
    solution = None

    try:
        common_age_fit = fit_common_age(
            primary_track=primary.track,
            secondary_track=secondary.track,
            primary_observables=primary.observables,
            secondary_observables=secondary.observables,
        )
        solution = BinaryEvolutionSolution.from_common_age_fit(
            common_age_fit,
            metallicity=SOLAR_METALLICITY,
            degrees_of_freedom=4,
        )
    except ValueError as exc:
        common_age_error = str(exc)

    return SystemEvolutionResult(
        system_name=system_name,
        mass_source=payload["mass_source"],
        primary=primary,
        secondary=secondary,
        independent_fit=independent_fit,
        age_difference_pct_of_older_age=round(age_diff_pct, 2),
        common_age_fit=common_age_fit,
        common_age_error=common_age_error,
        solution=solution,
    )


def load_model_and_raw_tracks():

    model = load_default_evolution_model()
    raw_tracks = _read_vizier_asu_tsv(TRACKS_TSV)
    return model, raw_tracks


def run_all_benchmark_checks():
    """
    Convenience: load the default (Girardi 2000, Z=0.019) model,
    load all six systems' component inputs, and run the check for
    each. Returns {system_name: SystemEvolutionResult}.
    """

    model, raw_tracks = load_model_and_raw_tracks()
    systems = load_benchmark_component_inputs()

    return {
        name: run_system_evolutionary_check(
            name, payload, model, raw_tracks,
        )
        for name, payload in systems.items()
    }
