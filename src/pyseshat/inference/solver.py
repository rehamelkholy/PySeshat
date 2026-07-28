from pyseshat.inference.optimizer import (
    fit_visual_orbit,
)

from pyseshat.inference.problem import (
    InferenceProblem,
)


def solve(
    problem: InferenceProblem,
    **kwargs,
):
    """
    Solve an inference problem.

    This is the preferred high-level API for orbit fitting.

    Parameters
    ----------
    problem
        Complete inference problem.
    **kwargs
        Passed directly to fit_visual_orbit().

    Returns
    -------
    FitResult
    """

    return fit_visual_orbit(
        problem.elements,
        problem.t,
        problem.X_obs,
        problem.Y_obs,
        problem.sx,
        problem.sy,
        rv_data=problem.rv_data,
        method=problem.method,
        loss=problem.loss,
        f_scale=problem.f_scale,
        bounds=problem.bounds,
        node_disambiguation=problem.node_disambiguation,
        node_rtol=problem.node_rtol,
        **kwargs,
    )