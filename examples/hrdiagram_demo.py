"""
Demonstration of the binary H-R diagram.

Run

    python examples/hrdiagram_demo.py

to generate an example H-R diagram.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from pyseshat.evolution.hrdiagram import (
    build_binary_hrdiagram_data,
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
        age=np.linspace(
            1.0,
            10.0,
            len(teff),
        ),
        log_teff=np.asarray(teff),
        log_luminosity=np.asarray(luminosity),
        log_radius=np.linspace(
            0.0,
            0.2,
            len(teff),
        ),
        log_g=np.linspace(
            4.5,
            4.1,
            len(teff),
        ),
    )


def make_sample(
    initial_mass,
    log_teff,
    log_luminosity,
):

    return EvolutionSample(
        age=5.0,
        metallicity=0.0,
        initial_mass=initial_mass,
        log_teff=log_teff,
        log_luminosity=log_luminosity,
        log_radius=0.10,
        log_g=4.30,
    )


primary_track = make_track(
    1.20,
    [
        3.82,
        3.81,
        3.80,
        3.79,
        3.78,
        3.77,
    ],
    [
        0.00,
        0.10,
        0.22,
        0.34,
        0.47,
        0.60,
    ],
)

secondary_track = make_track(
    0.90,
    [
        3.74,
        3.735,
        3.73,
        3.725,
        3.72,
        3.715,
    ],
    [
        -0.35,
        -0.25,
        -0.15,
        -0.08,
        -0.02,
        0.05,
    ],
)

primary_observables = StellarObservables(
    log_teff=3.79,
    sigma_log_teff=0.01,
    log_luminosity=0.33,
    sigma_log_luminosity=0.03,
)

secondary_observables = StellarObservables(
    log_teff=3.72,
    sigma_log_teff=0.01,
    log_luminosity=-0.03,
    sigma_log_luminosity=0.03,
)

solution = BinaryEvolutionSolution(
    primary=make_sample(
        1.20,
        3.79,
        0.34,
    ),
    secondary=make_sample(
        0.90,
        3.72,
        -0.02,
    ),
    age=5.0,
    metallicity=0.0,
    chi2=0.72,
)

data = build_binary_hrdiagram_data(
    primary_track=primary_track,
    secondary_track=secondary_track,
    primary_observables=primary_observables,
    secondary_observables=secondary_observables,
    solution=solution,
)

plot_binary_hrdiagram(data)

plt.tight_layout()

plt.show()