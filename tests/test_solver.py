import numpy as np

from pyseshat.inference.problem import (
    InferenceProblem,
)

from pyseshat.inference.context import (
    LikelihoodContext,
)

from pyseshat.inference.solver import (
    solve,
)

from pyseshat.inference.optimizer import (
    fit_visual_orbit,
)

from pyseshat.orbit.orbital_elements import (
    OrbitalElements,
)


def test_problem_solver_matches_optimizer():

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

    context = LikelihoodContext(
        t=t,
        X_obs=x,
        Y_obs=y,
        sx=sx,
        sy=sy,
    )

    problem = InferenceProblem(
        elements=elements,
        context=context,
    )

    direct = fit_visual_orbit(
        elements,
        t,
        x,
        y,
        sx,
        sy,
    )

    wrapped = solve(problem)

    np.testing.assert_allclose(
        direct.elements.to_array(),
        wrapped.elements.to_array(),
    )

    np.testing.assert_allclose(
        direct.residuals,
        wrapped.residuals,
    )