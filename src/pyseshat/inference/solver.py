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

    ctx = problem.context

    return fit_visual_orbit(
        problem.elements,
        ctx.t,
        ctx.X_obs,
        ctx.Y_obs,
        ctx.sx,
        ctx.sy,
        rv_data=ctx.rv_data,
        **kwargs,
    )