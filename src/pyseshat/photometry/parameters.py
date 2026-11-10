from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class PhotometricParameters:
    """
    Immutable container for the free parameters of the
    unresolved-binary photometric model.

    Parameters
    ----------
    primary_temperature : float
        Effective temperature of the primary star [K].

    secondary_temperature : float
        Effective temperature of the secondary star [K].

    primary_radius : float
        Radius of the primary star [R_sun].

    secondary_radius : float
        Radius of the secondary star [R_sun].

    ebv : float
        Color excess E(B-V).

    fit_flags : dict
        Mirrors ``OrbitalElements.fit_flags``: True -> parameter
        is fitted, False -> parameter is held fixed at its
        current value. Defaults to all-True (every parameter
        fitted), matching the original
        always-fit-everything behavior.

    priors : dict
        Optional Gaussian priors, mapping parameter name to a
        ``(mean, sigma)`` tuple. A prior may be attached to a
        fitted parameter (a soft constraint the optimizer/sampler
        can move away from if the data disagrees strongly enough
        -- e.g. radius tied to an independently published
        literature value) or, less commonly, to a fixed one (no
        effect on the fit itself, since a fixed parameter never
        moves, but still contributes a constant term to the log-
        posterior). Empty by default: no priors.

    Notes
    -----
    This mechanism mirrors ``OrbitalElements.fit_flags`` (see
    that class) and exists specifically to replace the
    near-zero-width-bounds "pinning" workaround that
    ``validation/benchmark_runner.py::run_sed_benchmark()``
    used previously for parameters the (4- or
    9-measurement) photometric data couldn't independently
    constrain (the radius degeneracy motivated this mechanism). A HARD pin
    (fit_flags[name]=False) still fully replaces that old
    epsilon-bounds trick when no prior is warranted; a Gaussian
    prior lets a parameter move under the data's own evidence
    instead of being frozen, which is the architecturally correct
    fix when an independent literature estimate exists but
    shouldn't be treated as infinitely precise.
    """

    primary_temperature: float
    secondary_temperature: float
    primary_radius: float
    secondary_radius: float
    ebv: float

    fit_flags: dict = field(
        default_factory=lambda: {
            name: True
            for name in PhotometricParameters.PARAMETER_ORDER
        }
    )

    priors: dict = field(
        default_factory=dict
    )

    # --- Canonical parameter order for array conversions ---
    PARAMETER_ORDER = [
        "primary_temperature",
        "secondary_temperature",
        "primary_radius",
        "secondary_radius",
        "ebv",
    ]

    # ======================================================
    # Full parameter vector (fixed + fitted, canonical order)
    # ======================================================

    def to_array(self):
        """
        Convert ALL parameters (fixed and fitted) to a NumPy
        array in canonical order. Mirrors
        ``OrbitalElements.to_array()``.
        """

        return np.array(
            [
                getattr(self, name)
                for name in self.PARAMETER_ORDER
            ],
            dtype=float,
        )

    # ======================================================
    # Fit-flag masks
    # ======================================================

    def fitted_mask(self):
        """
        Return a Boolean mask of which canonical-order
        parameters are currently fitted. Mirrors
        ``OrbitalElements.fitted_mask()``.
        """

        return np.array(
            [
                self.fit_flags.get(name, True)
                for name in self.PARAMETER_ORDER
            ],
            dtype=bool,
        )

    def fixed_mask(self):
        """
        Return a Boolean mask of which canonical-order
        parameters are currently held fixed. Mirrors
        ``OrbitalElements.fixed_mask()``.
        """

        return ~self.fitted_mask()

    # ======================================================
    # Parameter vector (fitted subset only)
    # ======================================================

    def fitted_parameters(
        self
    ):
        """
        Return the parameter vector used by the optimizer/
        sampler: only the currently-fitted parameters, in
        canonical order. With the default all-True fit_flags
        this is exactly the full 5-parameter vector (the
        original, pre-fit-flags behavior).
        """

        return self.to_array()[self.fitted_mask()]

    # ======================================================
    # Parameter names
    # ======================================================

    def fitted_parameter_names(
        self
    ):
        """
        Return the names of the currently-fitted parameters, in
        canonical order. Instance method (not a staticmethod) because which parameters are
        "fitted" is now per-instance state, not fixed for the
        whole class -- mirrors
        ``OrbitalElements.fitted_parameter_names()``.
        """

        return [
            name
            for name in self.PARAMETER_ORDER
            if self.fit_flags.get(name, True)
        ]


# ==========================================================
# Parameter update
# ==========================================================

def update_parameters(
    parameters,
    values
):
    """
    Construct a new PhotometricParameters object by applying an
    optimizer/sampler trial vector to a template's currently-
    FITTED parameters, carrying its fixed parameters, fit_flags,
    and priors over unchanged.

    Parameters
    ----------
    parameters : PhotometricParameters
        Template parameter object. Its ``fit_flags`` determines
        both how many values ``values`` must contain and which
        canonical-order positions they update.

    values : array_like
        Trial parameter vector, one entry per currently-fitted
        parameter (``len(parameters.fitted_parameter_names())``),
        in canonical order. With the default all-True fit_flags
        this must be exactly 5 values -- the original, pre-fit-
        flags behavior.

    Returns
    -------
    PhotometricParameters
    """

    values = np.asarray(

        values,

        dtype=float

    )

    mask = parameters.fitted_mask()

    n_fitted = int(mask.sum())

    if values.shape != (n_fitted,):

        raise ValueError(

            f"Photometric parameter vector must contain "
            f"exactly {n_fitted} values "
            f"({n_fitted} currently fitted)."

        )

    full = parameters.to_array()

    full[mask] = values

    kwargs = {

        name: float(value)

        for name, value in zip(

            PhotometricParameters.PARAMETER_ORDER,

            full

        )

    }

    return PhotometricParameters(

        fit_flags=parameters.fit_flags.copy(),

        priors=parameters.priors.copy(),

        **kwargs

    )
