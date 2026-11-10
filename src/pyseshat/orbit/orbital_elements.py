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

        # --------------------------------------------------
        # Fold i into [0, 180] by REFLECTION, not clipping.
        #
        # The visual-orbit model (thiele_innes) only ever uses
        # cos(i), which is even about both i=0 and i=180:
        #   cos(-x) = cos(x)  and  cos(180+x) = cos(180-x).
        # So folding via i -> -i (for i<0) or i -> 360-i (for
        # i>180) reproduces the identical model for any input.
        #
        # np.clip(i, 0, 180) does NOT have this property: it
        # replaces the value itself, so e.g. i=-10 degrees
        # would be clipped to i=0 (cos=1) instead of correctly
        # folding to i=10 (cos~0.985) -- a real distortion of
        # the model, not just a boundary clamp. This only
        # bites when an unbounded fit (method="lm") probes i
        # outside [0, 180] -- e.g. during finite-difference
        # Jacobian steps -- since bounded methods already keep
        # i inside [0, 180] via optimizer bounds.
        # --------------------------------------------------

        i = self.i % 360.0

        if i > 180.0:
            i = 360.0 - i

        self.i = i
    
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

        WARNING: this function is not used anywhere in the
        codebase, and `orbit.rv_model.compute_rv_orbit` does
        NOT apply this shift -- it uses `elements.omega`
        directly for both components. That unshifted
        convention has been validated against real published
        joint visual+RV solutions (FIN379, GL765: see
        tests/test_rv_sign_convention_validation.py), which converge to the
        literature omega/K1/K2 without needing this +180 deg
        shift. Do not wire this function into compute_rv_orbit
        (or anywhere else) without re-deriving why those two
        benchmark systems would still need to converge
        correctly afterward -- as written today, applying it
        would very likely reintroduce a real regression.
        """

        new = OrbitalElements.from_array(
            self.to_array(),
            fit_flags=self.fit_flags.copy()
        )

        new.omega = (
            new.omega + 180.0
        ) % 360.0

        return new