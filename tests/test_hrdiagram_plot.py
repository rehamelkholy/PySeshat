import matplotlib

matplotlib.use("Agg")

import numpy as np

from pyseshat.evolution.hrdiagram import (
    HRDiagramData,
)

from pyseshat.evolution.interpolate import (
    EvolutionSample,
)

from pyseshat.evolution.models import (
    EvolutionTrack,
)

from pyseshat.evolution.observables import (
    StellarObservables,
)

from pyseshat.evolution.solution import (
    BinaryEvolutionSolution,
)

from pyseshat.visualization.hrdiagram_plot import (
    plot_binary_hrdiagram,
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


def test_plot_binary_hrdiagram():

    data = HRDiagramData(

        primary_track=make_track(
            1.20,
            [
                3.80,
                3.78,
            ],
            [
                0.20,
                0.35,
            ],
        ),

        secondary_track=make_track(
            0.90,
            [
                3.72,
                3.70,
            ],
            [
                -0.05,
                0.02,
            ],
        ),

        primary_observables=make_observables(
            3.79,
            0.33,
        ),

        secondary_observables=make_observables(
            3.71,
            -0.03,
        ),

        solution=make_solution(),
    )

    ax = plot_binary_hrdiagram(
        data,
    )

    #
    # Axes.
    #
    assert ax.xaxis_inverted()

    assert (
        ax.get_xlabel()
        == r"$\log T_{\rm eff}\;(\mathrm{K})$"
    )

    assert (
        ax.get_ylabel()
        == r"$\log\left(L/L_\odot\right)$"
    )

    #
    # Legend.
    #
    legend = ax.get_legend()

    assert legend is not None

    labels = {
        text.get_text()
        for text in legend.get_texts()
    }

    assert labels == {
        "Primary Track",
        "Secondary Track",
        "Primary Observation",
        "Secondary Observation",
        "Primary Model",
        "Secondary Model",
    }