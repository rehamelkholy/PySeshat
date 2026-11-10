"""
Dataset-level weighting diagnostics and error rescaling.

In a combined visual + radial-velocity (RV) fit the single reduced
chi-square can hide a large imbalance between data types: a poorly
described dataset (reduced chi-square >> 1) is down-weighted only through
its own per-datum sigmas.  Following the practice of ORBIT/ORBIT3
(Tokovinin), which reports chi2/N separately for each data type and can
rescale the errors, this module provides

* per-dataset chi-square / weighted-rms diagnostics, and
* an iterative rescaling of the per-datum uncertainties of each dataset so
  that every dataset has reduced chi-square of unity.

Degrees of freedom are shared among datasets in proportion to their
number of data points: ``dof_k = N_k - J * N_k / N``.
"""

import numpy as np


def combine_error_floor(sigma, floor):
    """Add an error floor in quadrature: sqrt(sigma^2 + floor^2)."""
    sigma = np.asarray(sigma, dtype=float)
    if floor is None or floor == 0:
        return sigma.copy()
    return np.sqrt(sigma ** 2 + float(floor) ** 2)


def dataset_diagnostics(residuals, n_visual, n_rv, n_parameters):
    """
    Per-dataset fit diagnostics from the normalized residual vector.

    The residual vector is ordered as in ``joint_residuals``: visual
    (X residuals then Y residuals, ``2 * n_visual`` entries) followed by
    RV (``n_rv`` entries).

    Returns
    -------
    dict
        ``{"visual": {...}, "rv": {...}, "total": {...}}`` where each entry
        has ``n``, ``dof``, ``chi2``, ``chi2_per_dof`` and ``rms``
        (rms of the normalized residuals).  RV entry is omitted if
        ``n_rv == 0``.
    """
    r = np.asarray(residuals, dtype=float)
    n_vis_res = 2 * int(n_visual)
    n_rv = int(n_rv)
    n_tot = n_vis_res + n_rv
    if r.size != n_tot:
        raise ValueError(
            f"Residual vector has {r.size} entries, expected {n_tot}."
        )

    def _entry(x):
        n = x.size
        dof = n - n_parameters * n / n_tot
        chi2 = float(np.sum(x ** 2))
        return {
            "n": int(n),
            "dof": float(dof),
            "chi2": chi2,
            "chi2_per_dof": chi2 / dof if dof > 0 else float("nan"),
            "rms": float(np.sqrt(np.mean(x ** 2))) if n else float("nan"),
        }

    out = {"visual": _entry(r[:n_vis_res])}
    if n_rv > 0:
        out["rv"] = _entry(r[n_vis_res:])
    dof_tot = n_tot - n_parameters
    chi2_tot = float(np.sum(r ** 2))
    out["total"] = {
        "n": int(n_tot),
        "dof": float(dof_tot),
        "chi2": chi2_tot,
        "chi2_per_dof": chi2_tot / dof_tot if dof_tot > 0 else float("nan"),
        "rms": float(np.sqrt(np.mean(r ** 2))),
    }
    return out


def update_scales(diag, scales, allow_deflation=False):
    """
    One rescaling step.  Returns (new_scales, max_abs_change).
    ``scales`` maps "visual"/"rv" to the cumulative multiplicative factor
    currently applied to the sigmas.
    """
    new = dict(scales)
    change = 0.0
    for key in scales:
        c = diag[key]["chi2_per_dof"]
        if not np.isfinite(c) or c <= 0:
            continue
        step = np.sqrt(c)
        if step < 1.0 and not allow_deflation:
            step = 1.0
        new[key] = scales[key] * step
        change = max(change, abs(step - 1.0))
    return new, change
