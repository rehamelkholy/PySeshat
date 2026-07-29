import numpy as np

from pyseshat.inference.context import LikelihoodContext
from pyseshat.inference.problem import InferenceProblem
from pyseshat.inference.solver import solve
from pyseshat.inference.solver_engine import solve_problem
from pyseshat.orbit.orbital_elements import OrbitalElements


def test_solver_engine_matches_public_solver():

    elements = OrbitalElements(
        P=20.0,
        T0=0.0,
        e=0.2,
        a=0.5,
        Omega=40.0,
        omega=80.0,
        i=60.0,
    )

    t = np.array([0.0, 5.0, 10.0])
    x = np.array([0.10, 0.15, 0.20])
    y = np.array([0.05, 0.02, -0.01])
    sx = np.full(3, 0.01)
    sy = np.full(3, 0.01)

    problem = InferenceProblem(
        elements=elements,
        context=LikelihoodContext(
            t=t,
            X_obs=x,
            Y_obs=y,
            sx=sx,
            sy=sy,
        ),
    )

    result1 = solve(problem)
    result2 = solve_problem(problem)

    np.testing.assert_allclose(
        result1.elements.to_array(),
        result2.elements.to_array(),
    )

    np.testing.assert_allclose(
        result1.residuals,
        result2.residuals,
    )