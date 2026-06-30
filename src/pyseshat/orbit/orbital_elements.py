from dataclasses import dataclass, field
import numpy as np


@dataclass
class OrbitalElements:
    """
    ORBIT-compatible orbital elements container.

    Parameters follow Tokovinin ORBIT conventions.

    Visual elements:
    ----------------
    P       : Orbital period
    T0      : Epoch of periastron passage
    e       : Eccentricity
    a       : Semi-major axis (arcsec)
    Omega   : Longitude of ascending node (deg)
    omega   : Argument of periastron (deg)
    i       : Inclination (deg)

    Spectroscopic elements:
    -----------------------
    K1      : RV semi-amplitude primary
    K2      : RV semi-amplitude secondary
    V0      : Systemic velocity

    Fit flags:
    ----------
    True  -> parameter is fitted
    False -> parameter is fixed
    """

    # --- Canonical ORBIT parameter order for array conversions ---
    PARAMETER_ORDER = [
        "P",
        "T0",
        "e",
        "a",
        "Omega",
        "omega",
        "i",
        "K1",
        "K2",
        "V0"
    ]

    # --- Visual orbit parameters ---
    P: float
    T0: float
    e: float
    a: float
    Omega: float
    omega: float
    i: float

    # --- Spectroscopic parameters ---
    K1: float = 0.0
    K2: float = 0.0
    V0: float = 0.0

    # --- ORBIT-style fit toggles ---
    fit_flags: dict = field(default_factory=lambda: {
        "P": True,
        "T0": True,
        "e": True,
        "a": True,
        "Omega": True,
        "omega": True,
        "i": True,
        "K1": True,
        "K2": True,
        "V0": True
    })

    def to_array(self):
        """
        Convert visual elements to NumPy array
        in canonical ORBIT order.
        """

        return np.array([
            getattr(self, name)
            for name in self.PARAMETER_ORDER
        ], dtype=float)

    @classmethod
    def from_array(cls, arr, fit_flags=None):
        """
        Create OrbitalElements from array.
        """

        kwargs = {
            name: value
            for name, value in zip(
                cls.PARAMETER_ORDER,
                arr
            )
        }

        obj = cls(**kwargs)

        if fit_flags is not None:
            obj.fit_flags = fit_flags

        return obj

    def fitted_mask(self):
        """
        Return Boolean mask of fitted visual parameters.
        """

        return np.array([
            self.fit_flags[name]
            for name in self.PARAMETER_ORDER
        ], dtype=bool)

    def fixed_mask(self):
        """
        Return Boolean mask of fixed visual parameters.
        """

        return ~self.fitted_mask()

    def fitted_parameters(self):
        """
        Return only fitted parameters as array.
        """

        return self.to_array()[self.fitted_mask()]
    
    def fitted_parameter_names(self):
        """
        Names of fitted parameters.
        """

        return [
            name
            for name in self.PARAMETER_ORDER
            if self.fit_flags[name]
        ]

    def update_fitted(self, fitted_values):
        """
        Update only fitted parameters.
        """

        full = self.to_array()

        full[self.fitted_mask()] = fitted_values

        for name, value in zip(
            self.PARAMETER_ORDER,
            full
        ):
            setattr(self, name, value)

    def normalize_angles(self):
        """
        Normalize angular elements to ORBIT conventions.
        """

        self.Omega = self.Omega % 360.0
        self.omega = self.omega % 360.0

        self.i = np.clip(self.i, 0.0, 180.0)
    
    def copy(self):
        """
        Return deep copy of orbital elements.
        """

        return OrbitalElements.from_array(
            self.to_array(),
            fit_flags=self.fit_flags.copy()
        )

    # ======================================================
    # Convention transforms
    # ======================================================

    def to_spectroscopic_convention(self):
        """
        Convert visual-binary convention to
        spectroscopic convention.

        In visual binaries:
            omega is measured from ascending node
            of the SECONDARY.

        In spectroscopic binaries:
            omega is measured from ascending node
            of the PRIMARY.

        Therefore:

            omega_SB = omega_VB + 180 deg
        """

        new = OrbitalElements.from_array(
            self.to_array(),
            fit_flags=self.fit_flags.copy()
        )

        new.omega = (
            new.omega + 180.0
        ) % 360.0

        return new