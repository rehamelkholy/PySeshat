import numpy as np

from pyseshat.inference.context import LikelihoodContext
from pyseshat.inference.problem import InferenceProblem
from pyseshat.orbit.orbital_elements import OrbitalElements


def test_problem_stores_solver_options():

    elements = OrbitalElements(
        P=10.0,
        T0=0.0,
        e=0.2,
        a=1.0,
        Omega=20.0,
        omega=40.0,
        i=60.0,
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
        method="dogbox",
        loss="soft_l1",
        f_scale=2.0,
        node_disambiguation=False,
        node_rtol=0.05,
    )

    assert problem.method == "dogbox"
    assert problem.loss == "soft_l1"
    assert problem.f_scale == 2.0
    assert problem.node_disambiguation is False
    assert problem.node_rtol == 0.05