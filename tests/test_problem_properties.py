import numpy as np

from pyseshat.inference.context import LikelihoodContext
from pyseshat.inference.problem import InferenceProblem
from pyseshat.orbit.orbital_elements import OrbitalElements


def test_problem_properties_forward_to_context():

    t = np.array([1.0])
    x = np.array([0.1])
    y = np.array([0.2])
    sx = np.array([0.01])
    sy = np.array([0.02])

    context = LikelihoodContext(
        t=t,
        X_obs=x,
        Y_obs=y,
        sx=sx,
        sy=sy,
        rv_data=None,
    )

    problem = InferenceProblem(
        elements=OrbitalElements(
            P=10.0,
            T0=0.0,
            e=0.1,
            a=1.0,
            Omega=20.0,
            omega=40.0,
            i=60.0,
        ),
        context=context,
    )

    assert problem.t is t
    assert problem.X_obs is x
    assert problem.Y_obs is y
    assert problem.sx is sx
    assert problem.sy is sy
    assert problem.rv_data is None