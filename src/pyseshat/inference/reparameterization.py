"""
Sampling parameterization for near-circular orbits (added in 0.2.0).

For e -> 0 the epoch of periastron T0 and the argument of periastron
omega trade off along a curved ridge (only the mean longitude is
constrained), and an affine-invariant sampler working in (T0, e, omega)
mixes very slowly. The "sqrt_e" parameterization samples instead

    h = sqrt(e) sin(omega),   k = sqrt(e) cos(omega),
    L = omega + 360 (t_ref - T0) / P        (degrees),

the mean longitude at a reference epoch t_ref, with all other fitted
elements unchanged. The posterior is the same: the log-density is
corrected by the Jacobian of the transformation, so flat priors in
(T0, e, omega) remain flat. |d(P, T0, e, omega) / d(P, L, h, k)| =
(P / 360) * (360 / pi), so the correction is log P (P fitted) up to a
constant.

Samples are converted back to the standard elements, with T0 within
half a period of t_ref.
"""

import math

import numpy as np


_REQUIRED = ("T0", "e", "omega")


class SqrtEParameterization:
    """
    Map between standard fitted-parameter vectors (in the order of
    ``parameter_names``) and the sampled vector, in which ``T0``, ``e``
    and ``omega`` are replaced, in place, by ``L``, ``h`` and ``k``.

    Parameters
    ----------
    parameter_names : sequence of str
        Fitted parameter names; must contain T0, e and omega.
    t_ref : float
        Reference epoch (same time unit as T0 and P).
    period : float
        Orbital period, used only when ``P`` is not fitted.

    Raises
    ------
    ValueError
        If T0, e or omega is not fitted.
    """

    def __init__(self, parameter_names, t_ref, period):

        self.names = list(parameter_names)

        missing = [n for n in _REQUIRED if n not in self.names]

        if missing:

            raise ValueError(
                "parameterization='sqrt_e' needs T0, e and omega to be "
                f"fitted; not fitted: {', '.join(missing)}"
            )

        self.i_t0 = self.names.index("T0")
        self.i_e = self.names.index("e")
        self.i_w = self.names.index("omega")
        self.i_p = self.names.index("P") if "P" in self.names else None
        self.t_ref = float(t_ref)
        self.period = float(period)

    def sampled_names(self):

        out = list(self.names)
        out[self.i_t0], out[self.i_e], out[self.i_w] = "L", "h", "k"
        return out

    def _period(self, v):

        return float(v[self.i_p]) if self.i_p is not None else self.period

    def to_sampled(self, values):

        v = np.array(values, dtype=float, copy=True)
        P = self._period(v)
        T0, e, w = v[self.i_t0], v[self.i_e], v[self.i_w]
        root = math.sqrt(max(e, 0.0))
        v[self.i_t0] = w + 360.0 * (self.t_ref - T0) / P
        v[self.i_e] = root * math.sin(math.radians(w))
        v[self.i_w] = root * math.cos(math.radians(w))
        return v

    def to_standard(self, sampled):

        q = np.array(sampled, dtype=float, copy=True)
        P = self._period(q)
        L, h, k = q[self.i_t0], q[self.i_e], q[self.i_w]
        e = h * h + k * k
        w = math.degrees(math.atan2(h, k)) % 360.0
        dl = ((L - w + 180.0) % 360.0) - 180.0
        q[self.i_t0] = self.t_ref - P * dl / 360.0
        q[self.i_e] = e
        q[self.i_w] = w
        return q

    def log_jacobian(self, sampled):
        """log |d(standard) / d(sampled)| up to a constant."""

        if self.i_p is None:
            return 0.0

        P = float(sampled[self.i_p])

        return math.log(P) if P > 0.0 else -np.inf

    def to_standard_array(self, samples):

        samples = np.asarray(samples, dtype=float)

        return np.array([self.to_standard(q) for q in samples])

    def transform_covariance(self, values, covariance, step=1e-7):
        """
        Linearized covariance of the sampled vector at ``values``
        (standard), J C J^T with J = d(sampled)/d(standard), by central
        differences. Returns None if ``covariance`` is None.
        """

        if covariance is None:
            return None

        v = np.asarray(values, dtype=float)
        n = len(v)
        J = np.zeros((n, n))

        for j in range(n):

            h = step * max(abs(v[j]), 1.0)
            up, dn = v.copy(), v.copy()
            up[j] += h
            dn[j] -= h
            if j == self.i_e:          # keep e >= 0
                dn[j] = max(dn[j], 0.0)
            J[:, j] = (self.to_sampled(up) - self.to_sampled(dn)) / (up[j] - dn[j])

        C = np.asarray(covariance, dtype=float)

        out = J @ C @ J.T

        # e at (or very near) 0 makes d(h, k)/de singular; fall back to
        # the engine's default walker ball rather than an invalid matrix.
        return out if np.all(np.isfinite(out)) else None


def sqrt_e_log_probability(sampled, reparameterization, log_probability, args):
    """
    Log-density in the sampled parameters: the standard log-posterior at
    the mapped point plus the Jacobian term. ``h^2 + k^2 >= 1`` (e >= 1)
    is rejected.
    """

    i_e, i_w = reparameterization.i_e, reparameterization.i_w

    if sampled[i_e] ** 2 + sampled[i_w] ** 2 >= 1.0:
        return -np.inf

    lj = reparameterization.log_jacobian(sampled)

    if not np.isfinite(lj):
        return -np.inf

    standard = reparameterization.to_standard(sampled)

    return log_probability(standard, *args) + lj
