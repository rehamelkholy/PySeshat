import numpy as np

from pyseshat.inference.context import (
    LikelihoodContext,
)

from pyseshat.inference.likelihood import (
    combined_residuals,
)

from pyseshat.inference.terms import (
    joint_residuals_from_context,
)

from pyseshat.orbit.orbital_elements import (
    OrbitalElements,
)


def test_combined_residuals_matches_context_version():

    elements = OrbitalElements(
        P=10.0,
        T0=0.0,
        e=0.2,
        a=1.0,
        Omega=30.0,
        omega=60.0,
        i=70.0,
    )

    elements.fit_flags.update(
        P=True,
        T0=False,
        e=False,
        a=False,
        Omega=False,
        omega=False,
        i=False,
        K1=False,
        K2=False,
        V0=False,
    )

    context = LikelihoodContext(
        t=np.array([0.0, 2.0]),
        X_obs=np.array([0.1, 0.2]),
        Y_obs=np.array([0.3, 0.4]),
        sx=np.array([0.01, 0.01]),
        sy=np.array([0.01, 0.01]),
    )

    params = elements.fitted_parameters()

    np.testing.assert_allclose(
        combined_residuals(
            params,
            elements,
            context.t,
            context.X_obs,
            context.Y_obs,
            context.sx,
            context.sy,
        ),
        joint_residuals_from_context(
            params,
            elements,
            context,
        ),
    )