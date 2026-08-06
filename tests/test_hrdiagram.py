import numpy as np

from pyseshat.evolution.hrdiagram import (
    HRDiagramData,
    build_binary_hrdiagram_data,
)

from pyseshat.evolution.models import (
    EvolutionTrack,
)

from pyseshat.evolution.interpolate import (
    EvolutionSample,
)

from pyseshat.evolution.observables import (
    StellarObservables,
)

from pyseshat.evolution.solution import (
    BinaryEvolutionSolution,
)


def make_track(
    initial_mass,
    teff,
    luminosity,
):

    return EvolutionTrack(
        initial_mass=initial_mass,
        metallicity=0.0,
        age=np.array(
            [
                1.0,
                2.0,
            ]
        ),
        log_teff=np.array(teff),
        log_luminosity=np.array(luminosity),
        log_radius=np.array(
            [
                0.00,
                0.10,
            ]
        ),
        log_g=np.array(
            [
                4.40,
                4.30,
            ]
        ),
    )


def make_sample(
    initial_mass,
    log_teff,
    log_luminosity,
):

    return EvolutionSample(
        age=2.0,
        metallicity=0.0,
        initial_mass=initial_mass,
        log_teff=log_teff,
        log_luminosity=log_luminosity,
        log_radius=0.10,
        log_g=4.30,
    )


def make_observables(
    log_teff,
    log_luminosity,
):

    return StellarObservables(
        log_teff=log_teff,
        sigma_log_teff=0.01,
        log_luminosity=log_luminosity,
        sigma_log_luminosity=0.02,
    )


def make_solution():

    return BinaryEvolutionSolution(

        primary=make_sample(
            1.20,
            3.78,
            0.35,
        ),

        secondary=make_sample(
            0.90,
            3.70,
            0.02,
        ),

        age=2.0,
        metallicity=0.0,
        chi2=1.5,
    )


def test_hrdiagram_data_storage():

    primary_track = make_track(
        1.20,
        [
            3.80,
            3.78,
        ],
        [
            0.20,
            0.35,
        ],
    )

    secondary_track = make_track(
        0.90,
        [
            3.72,
            3.70,
        ],
        [
            -0.05,
            0.02,
        ],
    )

    primary_observables = make_observables(
        3.79,
        0.33,
    )

    secondary_observables = make_observables(
        3.71,
        -0.03,
    )

    solution = make_solution()

    data = HRDiagramData(

        primary_track=primary_track,

        secondary_track=secondary_track,

        primary_observables=primary_observables,

        secondary_observables=secondary_observables,

        solution=solution,
    )

    assert data.primary_track is primary_track

    assert data.secondary_track is secondary_track

    assert (
        data.primary_observables
        is primary_observables
    )

    assert (
        data.secondary_observables
        is secondary_observables
    )

    assert data.solution is solution


def test_build_binary_hrdiagram_data():

    primary_track = make_track(
        1.20,
        [
            3.80,
            3.78,
        ],
        [
            0.20,
            0.35,
        ],
    )

    secondary_track = make_track(
        0.90,
        [
            3.72,
            3.70,
        ],
        [
            -0.05,
            0.02,
        ],
    )

    primary_observables = make_observables(
        3.79,
        0.33,
    )

    secondary_observables = make_observables(
        3.71,
        -0.03,
    )

    solution = make_solution()

    data = build_binary_hrdiagram_data(

        primary_track=primary_track,

        secondary_track=secondary_track,

        primary_observables=primary_observables,

        secondary_observables=secondary_observables,

        solution=solution,
    )

    assert data.primary_track is primary_track

    assert data.secondary_track is secondary_track

    assert (
        data.primary_observables
        is primary_observables
    )

    assert (
        data.secondary_observables
        is secondary_observables
    )

    assert data.solution is solution