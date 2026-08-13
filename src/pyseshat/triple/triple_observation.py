from dataclasses import dataclass


@dataclass
class TripleObservation:
    """
    Single observation associated with a hierarchical
    triple-star system.

    The observation may be either:

    - a visual observation (I1 or I2), represented by
      ``theta`` and ``rho``; or

    - a radial-velocity observation (Va, Vb, or Vc),
      represented by ``velocity``.

    Parameters
    ----------
    epoch : float
        Observation epoch.

    theta : float or None
        Position angle in degrees for a visual observation.
        ``None`` for a radial-velocity observation.

    rho : float or None
        Angular separation for a visual observation.
        ``None`` for a radial-velocity observation.

    sigma : float
        Measurement uncertainty.

        For visual observations this is the positional
        uncertainty.

        For radial-velocity observations this is the
        velocity uncertainty.

    orbit_id : str
        ORBIT observation identifier:

            I1 -> inner visual observation
            I2 -> outer visual observation
            Va -> radial velocity of Aa
            Vb -> radial velocity of Ab
            Vc -> radial velocity of B

    velocity : float or None
        Radial velocity for Va, Vb, or Vc.
        ``None`` for I1 and I2.
    """

    epoch: float

    theta: float | None

    rho: float | None

    sigma: float

    orbit_id: str

    velocity: float | None = None

    # ======================================================
    # Compatibility interface
    # ======================================================

    @property
    def component(self) -> str:
        """
        Return the ORBIT radial-velocity component identifier.

        This is an alias for ``orbit_id`` retained for
        compatibility with the original parser output,
        where radial-velocity records were dictionaries
        containing a ``component`` key.
        """

        return self.orbit_id

    def __getitem__(
        self,
        key: str
    ):
        """
        Provide dictionary-style access for compatibility
        with the original radial-velocity parser output.

        Historically, ``rv_observations`` contained
        dictionaries with keys such as:

            epoch
            velocity
            sigma
            component

        The observation model is now unified under
        ``TripleObservation``, but these keys remain
        available so existing callers and tests continue
        to work.
        """

        if key == "epoch":
            return self.epoch

        if key == "velocity":
            return self.velocity

        if key == "sigma":
            return self.sigma

        if key == "component":
            return self.component

        if key == "orbit_id":
            return self.orbit_id

        if key == "theta":
            return self.theta

        if key == "rho":
            return self.rho

        raise KeyError(
            key
        )