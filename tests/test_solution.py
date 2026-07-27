import pytest

from pyseshat.evolution.interpolate import EvolutionSample
from pyseshat.evolution.solution import BinaryEvolutionSolution


def make_sample(mass):

    return EvolutionSample(
        age=5.0,
        metallicity=0.0,
        initial_mass=mass,
        log_teff=3.8,
        log_luminosity=0.2,
        log_radius=0.1,
        log_g=4.4,
    )


def test_solution_storage():

    solution = BinaryEvolutionSolution(
        primary=make_sample(1.2),
        secondary=make_sample(0.9),
        age=5.0,
        metallicity=0.0,
        chi2=1.25,
    )

    assert solution.primary.initial_mass == pytest.approx(1.2)
    assert solution.secondary.initial_mass == pytest.approx(0.9)
    assert solution.age == pytest.approx(5.0)
    assert solution.chi2 == pytest.approx(1.25)