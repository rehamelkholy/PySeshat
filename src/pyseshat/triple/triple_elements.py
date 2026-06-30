from dataclasses import dataclass, field

import numpy as np

from pyseshat.orbit.orbital_elements import (
    OrbitalElements
)


@dataclass
class TripleOrbitalElements:
    """
    Hierarchical triple system.
    """

    outer: OrbitalElements

    inner: OrbitalElements

    B: float

    fit_flags: dict = field(
        default_factory=lambda: {

            "P": False,
            "T0": False,
            "e": False,
            "a": False,
            "Omega": False,
            "omega": False,
            "i": False,

            "P2": False,
            "T02": False,
            "e2": False,
            "a2": False,
            "Omega2": False,
            "omega2": False,
            "i2": False,

            "B": False
        }
    )

    # ======================================================
    # Fitted parameter mask
    # ======================================================

    def fitted_mask(self):

        return np.array([

            self.fit_flags["P"],
            self.fit_flags["T0"],
            self.fit_flags["e"],
            self.fit_flags["a"],
            self.fit_flags["Omega"],
            self.fit_flags["omega"],
            self.fit_flags["i"],

            self.fit_flags["P2"],
            self.fit_flags["T02"],
            self.fit_flags["e2"],
            self.fit_flags["a2"],
            self.fit_flags["Omega2"],
            self.fit_flags["omega2"],
            self.fit_flags["i2"],

            self.fit_flags["B"]

        ])

    # ======================================================
    # Fitted parameter names
    # ======================================================

    def fitted_parameter_names(self):

        names = [

            "P",
            "T0",
            "e",
            "a",
            "Omega",
            "omega",
            "i",

            "P2",
            "T02",
            "e2",
            "a2",
            "Omega2",
            "omega2",
            "i2",

            "B"
        ]

        return [
            n
            for n, m in zip(
                names,
                self.fitted_mask()
            )
            if m
        ]