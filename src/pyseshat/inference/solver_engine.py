from pyseshat.inference.optimizer import fit_visual_orbit


def solve_problem(problem, **kwargs):
    """
    Internal implementation used by high-level solver interfaces.
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