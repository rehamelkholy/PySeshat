import numpy as np

from pyseshat.orbit.orbital_elements import OrbitalElements


# ==========================================================
# Kepler equation solver
# ==========================================================

def solve_kepler(M, e, tol=1e-10, max_iter=100):
    """
    Solve Kepler's equation:

        E - e sin(E) = M

    Parameters
    ----------
    M : array_like
        Mean anomaly (radians)

    e : float
        Eccentricity

    Returns
    -------
    E : ndarray
        Eccentric anomaly (radians)
    """

    M = np.asarray(M)

    E = M.copy()

    for _ in range(max_iter):

        f = E - e * np.sin(E) - M
        fp = 1.0 - e * np.cos(E)

        dE = -f / fp

        E += dE

        if np.max(np.abs(dE)) < tol:
            break

    return E


# ==========================================================
# Thiele-Innes constants
# ==========================================================

def thiele_innes(elements: OrbitalElements):
    """
    Compute Thiele-Innes constants.

    Returns
    -------
    A, B, F, G
    """

    a = elements.a

    Omega = np.deg2rad(elements.Omega)
    omega = np.deg2rad(elements.omega)
    inc = np.deg2rad(elements.i)

    A = a * (
        np.cos(Omega) * np.cos(omega)
        - np.sin(Omega) * np.sin(omega) * np.cos(inc)
    )

    B = a * (
        np.sin(Omega) * np.cos(omega)
        + np.cos(Omega) * np.sin(omega) * np.cos(inc)
    )

    F = a * (
        -np.cos(Omega) * np.sin(omega)
        - np.sin(Omega) * np.cos(omega) * np.cos(inc)
    )

    G = a * (
        -np.sin(Omega) * np.sin(omega)
        + np.cos(Omega) * np.cos(omega) * np.cos(inc)
    )

    return A, B, F, G


# ==========================================================
# Visual orbit model
# ==========================================================

def compute_visual_orbit(t, elements: OrbitalElements):
    """
    Compute visual binary orbit.

    Parameters
    ----------
    t : array_like
        Observation epochs

    elements : OrbitalElements
        Orbital elements object

    Returns
    -------
    dict containing:
        X
        Y
        rho
        theta
        E
        nu

    Notes
    -----
    The Cartesian convention used throughout the visual-orbit
    fitting infrastructure is

        X = rho cos(theta)
        Y = rho sin(theta)

    where theta is the position angle as read from a parsed
    `.inp` file's PA column, i.e. the standard astronomical/WDS
    position angle: measured from North (0 deg), turning positive
    toward increasing right ascension (East) -- per the IAU's own
    definition and Washington Double Star Catalog usage. Since theta is
    measured FROM North, `X = rho*cos(theta)` is therefore the
    North (declination) offset and `Y = rho*sin(theta)` is the
    East (right-ascension*cos(declination)) offset -- NOT
    arbitrary "positive X"/"positive Y" directions with no
    physical tie. `visualization.orbit_plot.plot_visual_orbit`
    uses this fact to render the standard North-up/East-left
    orbit figure (North on the vertical axis, East on the
    horizontal axis with it inverted).

    Therefore the inverse transformation is

        theta = atan2(Y, X).
    """

    t = np.asarray(t)

    # ------------------------------------------------------
    # Mean anomaly
    # ------------------------------------------------------

    M = 2.0 * np.pi * (t - elements.T0) / elements.P

    # wrap to [0, 2pi]
    M = np.mod(M, 2.0 * np.pi)

    # ------------------------------------------------------
    # Solve Kepler equation
    # ------------------------------------------------------

    E = solve_kepler(M, elements.e)

    # ------------------------------------------------------
    # True anomaly
    # ------------------------------------------------------

    nu = 2.0 * np.arctan2(
        np.sqrt(1 + elements.e) * np.sin(E / 2.0),
        np.sqrt(1 - elements.e) * np.cos(E / 2.0)
    )

    # ------------------------------------------------------
    # Rectangular orbit coordinates
    # ------------------------------------------------------

    Xorb = np.cos(E) - elements.e

    Yorb = (
        np.sqrt(
            1 - elements.e**2
        )
        *
        np.sin(E)
    )

    # ------------------------------------------------------
    # Thiele-Innes projection
    # ------------------------------------------------------

    A, B, F, G = thiele_innes(
        elements
    )

    X = (
        A * Xorb
        +
        F * Yorb
    )

    Y = (
        B * Xorb
        +
        G * Yorb
    )

    # ------------------------------------------------------
    # Polar coordinates
    # ------------------------------------------------------

    rho = np.sqrt(
        X**2
        +
        Y**2
    )

    # The project's visual-coordinate convention is:
    #
    #     X = rho cos(theta)
    #     Y = rho sin(theta)
    #
    # Therefore:
    #
    #     theta = atan2(Y, X)
    #
    # This is also the convention used by the ORBIT3
    # visual model and by the Cartesian conversion in the
    # visual residual functions.

    theta = np.rad2deg(
        np.arctan2(
            Y,
            X
        )
    )

    theta = np.mod(
        theta,
        360.0
    )

    return {
        "X": X,
        "Y": Y,
        "rho": rho,
        "theta": theta,
        "E": E,
        "nu": nu
    }


# ==========================================================
# Smooth orbit sampling for visualization
# ==========================================================

def sample_visual_orbit(
    elements: OrbitalElements,
    n_points=1000
):
    """
    Generate a smooth visual orbit for plotting.

    Parameters
    ----------
    elements : OrbitalElements

    n_points : int

    Returns
    -------
    dict
    """

    # ------------------------------------------------------
    # Uniform mean anomaly sampling
    # ------------------------------------------------------

    M = np.linspace(
        0,
        2 * np.pi,
        n_points
    )

    # ------------------------------------------------------
    # Convert to synthetic epochs
    # ------------------------------------------------------

    t = (
        elements.T0
        +
        (
            elements.P
            *
            M
            /
            (2 * np.pi)
        )
    )

    return compute_visual_orbit(
        t,
        elements
    )