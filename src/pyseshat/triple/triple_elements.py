from dataclasses import dataclass, field

import numpy as np

from pyseshat.orbit.orbital_elements import (
    OrbitalElements
)


@dataclass
class TripleOrbitalElements:
    """
    Orbital elements for a hierarchical triple system.

    The modern representation follows the 20-parameter
    formulation used by Tokovinin's ORBIT3 program:

        Inner orbit
        -----------
        P, T0, e, a, Omega, omega, i
        K1, K2, f

        Outer orbit
        -----------
        P2, T02, e2, a2, Omega2, omega2, i2
        K3, K4, V0

    where ``f`` is the astrometric wobble factor.

    The legacy ``B`` parameter is retained temporarily for
    compatibility with the older ORBITT2-style ``.in2``
    formulation already supported by PySeshat.

    Parameters
    ----------
    outer
        Orbital elements of the outer orbit.

    inner
        Orbital elements of the inner orbit.

    K1
        RV semi-amplitude of the primary component of the
        inner spectroscopic binary.

    K2
        RV semi-amplitude of the secondary component of the
        inner spectroscopic binary.

    f
        Astrometric wobble factor of the unresolved inner
        subsystem in the outer astrometric measurement.

    K3
        RV semi-amplitude associated with the primary
        component of the outer orbit.

    K4
        RV semi-amplitude associated with the secondary
        component of the outer orbit.

    V0
        Systemic radial velocity.

    B
        Legacy scale parameter used by the older ORBITT2
        triple formulation.

    fit_flags
        Mapping specifying which parameters are fitted.
    """

    outer: OrbitalElements

    inner: OrbitalElements

    K1: float = 0.0

    K2: float = 0.0

    f: float = 0.0

    K3: float = 0.0

    K4: float = 0.0

    V0: float = 0.0

    B: float = 1.0

    fit_flags: dict = field(
        default_factory=lambda: {

            # --------------------------------------------------
            # Inner orbit
            # --------------------------------------------------

            "P": False,
            "T0": False,
            "e": False,
            "a": False,
            "Omega": False,
            "omega": False,
            "i": False,

            # --------------------------------------------------
            # Inner spectroscopic orbit / wobble
            # --------------------------------------------------

            "K1": False,
            "K2": False,
            "f": False,

            # --------------------------------------------------
            # Outer orbit
            # --------------------------------------------------

            "P2": False,
            "T02": False,
            "e2": False,
            "a2": False,
            "Omega2": False,
            "omega2": False,
            "i2": False,

            # --------------------------------------------------
            # Outer spectroscopic orbit
            # --------------------------------------------------

            "K3": False,
            "K4": False,
            "V0": False,

            # --------------------------------------------------
            # Legacy ORBITT2 parameter
            # --------------------------------------------------

            "B": False
        }
    )

    # ==========================================================
    # Fitted parameter mask
    # ==========================================================

    def fitted_mask(self):
        """
        Return the boolean mask for the complete modern
        ORBIT3 parameter vector.

        The ordering is:

            P, T0, e, a, Omega, omega, i,
            K1, K2, f,
            P2, T02, e2, a2, Omega2, omega2, i2,
            K3, K4, V0
        """

        return np.array([

            # Inner orbit

            self.fit_flags["P"],
            self.fit_flags["T0"],
            self.fit_flags["e"],
            self.fit_flags["a"],
            self.fit_flags["Omega"],
            self.fit_flags["omega"],
            self.fit_flags["i"],

            # Inner spectroscopy / wobble

            self.fit_flags["K1"],
            self.fit_flags["K2"],
            self.fit_flags["f"],

            # Outer orbit

            self.fit_flags["P2"],
            self.fit_flags["T02"],
            self.fit_flags["e2"],
            self.fit_flags["a2"],
            self.fit_flags["Omega2"],
            self.fit_flags["omega2"],
            self.fit_flags["i2"],

            # Outer spectroscopy

            self.fit_flags["K3"],
            self.fit_flags["K4"],
            self.fit_flags["V0"]

        ], dtype=bool)

    # ==========================================================
    # Fitted parameter names
    # ==========================================================

    def fitted_parameter_names(self):
        """
        Return the names of the fitted ORBIT3 parameters in
        parameter-vector order.
        """

        names = [

            # Inner orbit

            "P",
            "T0",
            "e",
            "a",
            "Omega",
            "omega",
            "i",

            # Inner spectroscopy / wobble

            "K1",
            "K2",
            "f",

            # Outer orbit

            "P2",
            "T02",
            "e2",
            "a2",
            "Omega2",
            "omega2",
            "i2",

            # Outer spectroscopy

            "K3",
            "K4",
            "V0"
        ]

        mask = self.fitted_mask()

        return [
            name
            for name, fitted
            in zip(
                names,
                mask
            )
            if fitted
        ]

    # ==========================================================
    # Modern parameter names
    # ==========================================================

    @staticmethod
    def parameter_names():
        """
        Return the complete ORBIT3 parameter ordering.
        """

        return [

            # Inner orbit

            "P",
            "T0",
            "e",
            "a",
            "Omega",
            "omega",
            "i",

            # Inner spectroscopy / wobble

            "K1",
            "K2",
            "f",

            # Outer orbit

            "P2",
            "T02",
            "e2",
            "a2",
            "Omega2",
            "omega2",
            "i2",

            # Outer spectroscopy

            "K3",
            "K4",
            "V0"
        ]

    # ==========================================================
    # Legacy compatibility
    # ==========================================================

    @property
    def scale_inner(self):
        """
        Return the legacy inner-orbit scale parameter.

        This property is retained temporarily for compatibility
        with the existing ORBITT2-style implementation.

        It must not be interpreted as the modern ORBIT3 wobble
        factor ``f``.
        """

        return self.B