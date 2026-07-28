from __future__ import annotations

from pyseshat.inference.optimizer import (
    fit_visual_orbit,
)

from pyseshat.observations.adapters import (
    visual_arrays,
)

from pyseshat.observations.visual import (
    VisualObservationSet,
)


def fit_visual_orbit_from_observations(
    elements,
    observations: VisualObservationSet,
    *,
    rv_data=None,
    method="trf",
    loss="linear",
    f_scale=1.0,
    bounds=None,
    node_disambiguation=True,
    node_rtol=0.01,
):
    """
    Fit a visual orbit using a VisualObservationSet.

    This is a thin wrapper around the existing public optimizer.
    """

    t, X, Y, sx, sy = visual_arrays(
        observations
    )

    return fit_visual_orbit(
        elements,
        t,
        X,
        Y,
        sx,
        sy,
        rv_data=rv_data,
        method=method,
        loss=loss,
        f_scale=f_scale,
        bounds=bounds,
        node_disambiguation=node_disambiguation,
        node_rtol=node_rtol,
    )