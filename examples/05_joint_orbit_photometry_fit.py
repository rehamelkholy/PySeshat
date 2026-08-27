"""
Example 5: Joint orbit + photometry fit.

inference/joint_problem.py's JointInferenceProblem and
inference/joint_parameters.py's JointParameters combine an orbital
InferenceProblem and a PhotometricInferenceProblem into one
simultaneous fit -- a real, tested capability with no example
anywhere else in the repo (see docs/development_plan_phase2.md,
Phase 2 item 5). Unlike Examples 1-4, there is currently no
convenience ``fit_joint(...)`` wrapper: this example builds one
from the pieces the package already provides (the same small
"rebuild a trial problem at the optimizer's current vector" pattern
used internally by inference/photometric_optimizer.py -- see
CLAUDE.md's "Photometry / stellar / evolution" section), which is
itself worth seeing since it shows how the lower-level pieces
compose.

The dataset here is deliberately SYNTHETIC: this repo's real
benchmark systems each validate one side only (FIN379/GL765/
ZetaAqr/HIP111805 have real orbit data but no photometry; the four
Al-Wardat systems have real photometry but no orbit data file --
see CLAUDE.md's "Validation & reference data" section) so there is
no real combined dataset to fit yet. A known "truth" is chosen,
synthetic visual+RV astrometry and synthetic photometry are
generated from it with added noise, and the joint fit is checked
to recover that truth -- the same synthetic-recovery pattern used
throughout tests/test_orbit3_fit.py and friends.

Run

    python examples/05_joint_orbit_photometry_fit.py
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from pyseshat.orbit.orbital_elements import (
    OrbitalElements,
)

from pyseshat.orbit.orbit_model import (
    compute_visual_orbit,
)

from pyseshat.orbit.rv_model import (
    compute_rv_model,
)

from pyseshat.inference.context import (
    LikelihoodContext,
)

from pyseshat.inference.problem import (
    InferenceProblem,
)

from pyseshat.inference.terms.joint import (
    joint_residuals_from_context,
)

from pyseshat.photometry.parameters import (
    PhotometricParameters,
)

from pyseshat.photometry.observations import (
    PhotometricMeasurement,
)

from pyseshat.photometry.model import (
    synthetic_binary_magnitudes,
)

from pyseshat.inference.photometric_context import (
    PhotometricContext,
)

from pyseshat.inference.photometric_problem import (
    PhotometricInferenceProblem,
)

from pyseshat.inference.photometric_likelihood import (
    photometric_residuals_from_problem,
)

from pyseshat.inference.joint_problem import (
    JointInferenceProblem,
)

from pyseshat.inference.joint_parameters import (
    JointParameters,
)

from pyseshat.stellar.atmospheres import (
    load_ck04_grid,
)

from pyseshat.photometry.filters import (
    load_passband,
)


REPO_ROOT = Path(__file__).resolve().parent.parent

VALIDATION_DIR = REPO_ROOT / "validation"

RNG = np.random.default_rng(7)

# --------------------------------------------------
# "Truth": a synthetic Sun-like visual+RV binary, loosely
# modeled on FIN379's real orbital solution, paired with a
# synthetic photometric SED reusing HD214222's real
# atmosphere-grid/passband files. This is illustrative
# only -- it is NOT a claim about either real system's
# actual physical properties.
# --------------------------------------------------

TRUE_ELEMENTS = OrbitalElements(
    P=6.69, T0=2008.844, e=0.499, a=0.100,
    Omega=183.9, omega=191.0, i=42.0,
    K1=7.43, K2=8.05, V0=-0.51,
)

TRUE_PHOTOMETRIC = PhotometricParameters(
    primary_temperature=6075.0,
    secondary_temperature=6000.0,
    primary_radius=1.14,
    secondary_radius=1.09,
    ebv=0.01,
)


def make_synthetic_orbit_context():

    t = np.sort(
        RNG.uniform(2007.0, 2015.0, size=20)
    )

    orbit = compute_visual_orbit(t, TRUE_ELEMENTS)

    sx = np.full_like(t, 0.002)
    sy = np.full_like(t, 0.002)

    X_obs = orbit["X"] + RNG.normal(0.0, sx)
    Y_obs = orbit["Y"] + RNG.normal(0.0, sy)

    t_rv = np.sort(
        RNG.uniform(2007.0, 2015.0, size=15)
    )

    sigma_rv = 0.3

    rows = []

    for epoch in t_rv:

        for component, flag in ((1, 1), (2, 2)):

            model = compute_rv_model(
                np.array([[epoch, 0.0, sigma_rv, flag]]),
                TRUE_ELEMENTS,
            )[0]

            rows.append([
                epoch,
                model + RNG.normal(0.0, sigma_rv),
                sigma_rv,
                flag,
            ])

    rv_data = np.array(rows)

    return LikelihoodContext(
        t=t, X_obs=X_obs, Y_obs=Y_obs, sx=sx, sy=sy,
        rv_data=rv_data,
    )


def make_synthetic_photometric_context():

    atmosphere_grid = load_ck04_grid([
        VALIDATION_DIR / "atmosphere_models" / "ckp00" / f"ckp00_{teff}.fits"
        for teff in (5750, 6000, 6250)
    ])

    passbands = {
        name: load_passband(
            filename=(
                VALIDATION_DIR / "photometry" / "passbands" / filename
            ),
            name=name,
        )
        for name, filename in (
            ("Johnson B", "johnson_b.dat"),
            ("Johnson V", "johnson_v.dat"),
            ("Tycho BT", "tycho_bt.dat"),
            ("Tycho VT", "tycho_vt.dat"),
        )
    }

    zero_points = {
        "Johnson B": 6.72553e-9,
        "Johnson V": 3.5833e-9,
        "Tycho BT": 6.50910e-9,
        "Tycho VT": 3.99504e-9,
    }

    true_magnitudes = synthetic_binary_magnitudes(
        atmosphere_grid=atmosphere_grid,
        passbands=passbands,
        zero_points=zero_points,
        primary_temperature=TRUE_PHOTOMETRIC.primary_temperature,
        secondary_temperature=TRUE_PHOTOMETRIC.secondary_temperature,
        primary_logg=4.4,
        secondary_logg=4.4,
        metallicity=0.0,
        primary_radius=TRUE_PHOTOMETRIC.primary_radius,
        secondary_radius=TRUE_PHOTOMETRIC.secondary_radius,
        distance=30.0,
        ebv=TRUE_PHOTOMETRIC.ebv,
    )

    sigma_mag = 0.02

    measurements = [
        PhotometricMeasurement(
            passband=name,
            magnitude=magnitude + RNG.normal(0.0, sigma_mag),
            uncertainty=sigma_mag,
        )
        for name, magnitude in true_magnitudes.items()
    ]

    return PhotometricContext(
        atmosphere_grid=atmosphere_grid,
        passbands=passbands,
        zero_points=zero_points,
        measurements=measurements,
        primary_logg=4.4,
        secondary_logg=4.4,
        metallicity=0.0,
        distance=30.0,
    )


def joint_residuals_for_optimizer(theta, template, joint_problem):
    """
    scipy.optimize.least_squares-compatible residual function:
    splits the flat fitted-parameter vector back into orbital and
    photometric pieces, rebuilds trial orbit/photometric problems
    at that point, and concatenates their residual vectors --
    mirroring inference/photometric_optimizer.py's
    ``_photometric_residuals_for_optimizer`` wrapper, extended to
    both sides of a JointParameters vector at once.
    """

    updated = template.update_from_vector(theta)

    orbit_residuals = joint_residuals_from_context(
        fitted_values=updated.orbital.fitted_parameters(),
        elements=updated.orbital,
        context=joint_problem.orbit.context,
    )

    trial_photometric_problem = replace(
        joint_problem.photometry,
        parameters=updated.photometric,
    )

    photometric_residuals = photometric_residuals_from_problem(
        trial_photometric_problem
    )

    return np.concatenate((orbit_residuals, photometric_residuals))


def main():

    orbit_context = make_synthetic_orbit_context()

    photometric_context = make_synthetic_photometric_context()

    joint_problem = JointInferenceProblem(
        orbit_problem=InferenceProblem(
            elements=TRUE_ELEMENTS,
            context=orbit_context,
        ),
        photometric_problem=PhotometricInferenceProblem(
            parameters=TRUE_PHOTOMETRIC,
            context=photometric_context,
        ),
    )

    # --------------------------------------------------
    # Perturb away from the truth to give the fit
    # something real to do.
    # --------------------------------------------------

    start_orbital = OrbitalElements(
        P=TRUE_ELEMENTS.P * 1.01,
        T0=TRUE_ELEMENTS.T0 + 0.05,
        e=TRUE_ELEMENTS.e * 0.95,
        a=TRUE_ELEMENTS.a * 1.05,
        Omega=TRUE_ELEMENTS.Omega + 3.0,
        omega=TRUE_ELEMENTS.omega - 3.0,
        i=TRUE_ELEMENTS.i + 2.0,
        K1=TRUE_ELEMENTS.K1 * 1.05,
        K2=TRUE_ELEMENTS.K2 * 0.95,
        V0=TRUE_ELEMENTS.V0 + 0.2,
    )

    start_photometric = PhotometricParameters(
        primary_temperature=TRUE_PHOTOMETRIC.primary_temperature + 150.0,
        secondary_temperature=TRUE_PHOTOMETRIC.secondary_temperature - 150.0,
        primary_radius=TRUE_PHOTOMETRIC.primary_radius * 0.95,
        secondary_radius=TRUE_PHOTOMETRIC.secondary_radius * 1.05,
        ebv=0.0,
    )

    template = JointParameters(
        orbital=start_orbital,
        photometric=start_photometric,
    )

    print(
        f"Joint fit: {template.n_orbital} orbital + "
        f"{template.n_photometric} photometric parameters "
        f"({template.size} total)."
    )

    # --------------------------------------------------
    # Physical bounds. Needed because plain unbounded LM
    # can wander into unphysical territory mid-iteration
    # (e.g. E(B-V) < 0), which the forward photometric
    # model rejects outright -- the same reason
    # inference/photometric_optimizer.py's
    # build_parameter_bounds() exists.
    # --------------------------------------------------

    orbital_bounds = {
        "P": (0.1, 100.0), "T0": (-1e6, 1e6),
        "e": (0.0, 0.99), "a": (1e-6, 10.0),
        "Omega": (0.0, 360.0), "omega": (0.0, 360.0),
        "i": (0.0, 180.0),
        "K1": (0.0, 100.0), "K2": (0.0, 100.0),
        "V0": (-100.0, 100.0),
    }

    photometric_bounds = {
        "primary_temperature": (5750.0, 6250.0),
        "secondary_temperature": (5750.0, 6250.0),
        "primary_radius": (0.01, 100.0),
        "secondary_radius": (0.01, 100.0),
        "ebv": (0.0, 10.0),
    }

    lower = [
        orbital_bounds[name][0]
        for name in start_orbital.fitted_parameter_names()
    ] + [
        photometric_bounds[name][0]
        for name in start_photometric.fitted_parameter_names()
    ]

    upper = [
        orbital_bounds[name][1]
        for name in start_orbital.fitted_parameter_names()
    ] + [
        photometric_bounds[name][1]
        for name in start_photometric.fitted_parameter_names()
    ]

    result = least_squares(
        joint_residuals_for_optimizer,
        template.fitted_vector(),
        args=(template, joint_problem),
        method="trf",
        bounds=(lower, upper),
    )

    fitted = template.update_from_vector(result.x)

    print()

    print(
        f"{'Parameter':>12}{'Truth':>12}{'Fitted':>12}"
    )

    for name in TRUE_ELEMENTS.fitted_parameter_names():

        print(
            f"{name:>12}"
            f"{getattr(TRUE_ELEMENTS, name):>12.4f}"
            f"{getattr(fitted.orbital, name):>12.4f}"
        )

    for name in TRUE_PHOTOMETRIC.fitted_parameter_names():

        print(
            f"{name:>12}"
            f"{getattr(TRUE_PHOTOMETRIC, name):>12.4f}"
            f"{getattr(fitted.photometric, name):>12.4f}"
        )

    print()

    print(
        f"Converged: {result.success}  "
        f"cost: {result.cost:.4f}  "
        f"function evals: {result.nfev}"
    )

    print()

    print(
        "Note: the orbital parameters and both component "
        "temperatures recover well, but the two radii typically "
        "do not (this run has no Gaussian prior on either radius "
        "-- see JointParameters/PhotometricParameters.priors). "
        "This is the SAME unresolved-photometry (Teff, R) "
        "degeneracy documented in docs/project_decisions.md, and "
        "the reason Example 4 fits radius WITH a Gaussian prior "
        "rather than leaving it fully free -- add one the same "
        "way here (photometric.priors={'primary_radius': (mean, "
        "sigma), ...}) for a real fit, not just this "
        "illustrative one."
    )


if __name__ == "__main__":

    main()
