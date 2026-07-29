import numpy as np

from pyseshat.inference.context import (
    LikelihoodContext,
)

from pyseshat.inference.problem import (
    InferenceProblem,
)

from pyseshat.orbit.orbital_elements import (
    OrbitalElements,
)


def make_elements():

    return OrbitalElements(
        P=10.0,
        T0=0.0,
        e=0.2,
        a=1.0,
        Omega=30.0,
        omega=60.0,
        i=70.0,
    )


def test_visual_only_problem():

    context = LikelihoodContext(
        t=np.array([1.0]),
        X_obs=np.array([0.1]),
        Y_obs=np.array([0.2]),
        sx=np.array([0.01]),
        sy=np.array([0.01]),
        rv_data=None,
    )

    problem = InferenceProblem(
        elements=make_elements(),
        context=context,
    )

    assert problem.has_visual_data
    assert not problem.has_rv_data
    assert problem.is_visual_only
    assert not problem.is_combined_fit


def test_combined_problem():

    context = LikelihoodContext(
        t=np.array([1.0]),
        X_obs=np.array([0.1]),
        Y_obs=np.array([0.2]),
        sx=np.array([0.01]),
        sy=np.array([0.01]),
        rv_data=[
            {
                "time": 1.0,
                "rv": 12.0,
                "sigma": 0.5,
                "component": 1,
            }
        ],
    )

    problem = InferenceProblem(
        elements=make_elements(),
        context=context,
    )

    assert problem.has_visual_data
    assert problem.has_rv_data
    assert not problem.is_visual_only
    assert problem.is_combined_fit