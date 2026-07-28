import numpy as np

from pyseshat.inference.observation_wrappers import (
    fit_visual_orbit_from_observations,
)

from pyseshat.inference.optimizer import (
    fit_visual_orbit,
)

from pyseshat.observations import (
    VisualObservationSet,
)

from pyseshat.orbit.orbital_elements import (
    OrbitalElements,
)


def test_fit_visual_orbit_wrapper():

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
        e=True,
        a=True,
        Omega=False,
        omega=False,
        i=False,
        K1=False,
        K2=False,
        V0=False,
    )

    observations = VisualObservationSet(
        epoch=np.array([0.0, 2.5, 5.0, 7.5]),
        x=np.array([0.20, 0.10, -0.15, -0.25]),
        y=np.array([0.10, 0.25, 0.20, -0.05]),
        sigma_x=np.full(4, 0.01),
        sigma_y=np.full(4, 0.01),
    )

    direct = fit_visual_orbit(
        elements,
        observations.epoch,
        observations.x,
        observations.y,
        observations.sigma_x,
        observations.sigma_y,
    )

    wrapped = fit_visual_orbit_from_observations(
        elements,
        observations,
    )

    np.testing.assert_allclose(
        direct.residuals,
        wrapped.residuals,
    )

    np.testing.assert_allclose(
        direct.elements.to_array(),
        wrapped.elements.to_array(),
    )

    np.testing.assert_allclose(
        direct.covariance,
        wrapped.covariance,
    )