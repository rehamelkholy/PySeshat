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


def test_problem_stores_elements_and_context():

    elements = OrbitalElements(
        P=10.0,
        T0=0.0,
        e=0.2,
        a=1.0,
        Omega=30.0,
        omega=60.0,
        i=70.0,
    )

    context = LikelihoodContext(
        t=np.array([1.0]),
        X_obs=np.array([0.1]),
        Y_obs=np.array([0.2]),
        sx=np.array([0.01]),
        sy=np.array([0.01]),
    )

    problem = InferenceProblem(
        elements=elements,
        context=context,
    )

    assert problem.elements is elements
    assert problem.context is context