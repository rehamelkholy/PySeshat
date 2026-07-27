import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pyseshat.orbit.orbital_elements import OrbitalElements
from pyseshat.visualization.orbit_plot import (
    plot_visual_orbit,
    plot_residuals,
    plot_rv_orbit,
)


def test_plot_visual_orbit():

    x = np.linspace(-1.0, 1.0, 20)
    y = np.sin(x)

    orbit = {
        "X": x,
        "Y": y,
    }

    fig, ax = plot_visual_orbit(
        x,
        y,
        orbit,
        show=False,
    )

    assert fig is not None
    assert ax is not None
    assert ax.get_xlabel() == "X (arcsec)"
    assert ax.get_ylabel() == "Y (arcsec)"

    plt.close(fig)


def test_plot_residuals():

    t = np.arange(10.0)

    residuals = np.zeros(20)

    fig, axes = plot_residuals(
        t,
        residuals,
        show=False,
    )

    assert fig is not None
    assert len(axes) == 2

    plt.close(fig)


def test_plot_rv_orbit():

    elements = OrbitalElements(
        P=10.0,
        T0=0.0,
        e=0.2,
        a=1.0,
        Omega=30.0,
        omega=60.0,
        i=70.0,
        K1=20.0,
        K2=30.0,
        V0=0.0,
    )

    rv = np.array([
        [0.0, 20.0, 1.0, 0],
        [2.5, -30.0, 1.0, 1],
        [5.0, -20.0, 1.0, 0],
        [7.5, 30.0, 1.0, 1],
    ])

    fig, ax = plot_rv_orbit(
        rv,
        elements,
        show=False,
    )

    assert fig is not None
    assert ax is not None

    plt.close(fig)


def test_plot_rv_orbit_empty():

    elements = OrbitalElements(
        P=10.0,
        T0=0.0,
        e=0.2,
        a=1.0,
        Omega=30.0,
        omega=60.0,
        i=70.0,
        K1=20.0,
        K2=30.0,
        V0=0.0,
    )

    with pytest.raises(ValueError):

        plot_rv_orbit(
            np.empty((0, 4)),
            elements,
            show=False,
        )