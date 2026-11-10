"""
Tests for the binary SED-fit plot: the
``PhotometricPassband.pivot_wavelength`` property, the
``photometry.model.synthetic_binary_sed()`` curve function, and
``visualization.sed_plot.plot_sed_fit()`` itself.

See docs/guide/visualization.md's "SED fit" section.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pyseshat.stellar.atmospheres import load_ck04_grid
from pyseshat.photometry.filters import PhotometricPassband, load_passband
from pyseshat.photometry.model import (
    synthetic_binary_magnitudes,
    synthetic_binary_sed,
)
from pyseshat.photometry.synthetic import (
    pogson_magnitude,
    synthetic_band_flux,
)
from pyseshat.visualization.sed_plot import plot_sed_fit


ATMOSPHERE_DIR = ROOT / "validation" / "atmosphere_models" / "ckp00"
PASSBAND_DIR = ROOT / "validation" / "photometry" / "passbands"

REQUIRED_FITS = (
    ATMOSPHERE_DIR / "ckp00_5750.fits",
    ATMOSPHERE_DIR / "ckp00_6000.fits",
)

pytestmark = pytest.mark.skipif(
    not all(path.is_file() for path in REQUIRED_FITS)
    or not PASSBAND_DIR.is_dir(),
    reason=(
        "Real CK04/passband files not present under validation/ -- "
        "these tests exercise the real, checked-in benchmark data, "
        "not synthetic fixtures."
    ),
)


# ==========================================================
# PhotometricPassband.pivot_wavelength
# ==========================================================


def test_pivot_wavelength_lies_within_passband_range():
    """
    The pivot wavelength must lie strictly inside
    [minimum_wavelength, maximum_wavelength] for any real,
    single-peaked passband -- a basic sanity bound regardless of
    the passband's exact shape.
    """

    pb = load_passband(PASSBAND_DIR / "johnson_v.dat", name="Johnson V")

    assert pb.minimum_wavelength < pb.pivot_wavelength < pb.maximum_wavelength


def test_pivot_wavelength_matches_hand_computed_formula():
    """
    Cross-check PhotometricPassband.pivot_wavelength against the
    standard pivot-wavelength formula computed independently (not
    just re-calling the same code path).
    """

    wavelength = np.array([4000.0, 4500.0, 5000.0, 5500.0, 6000.0])
    throughput = np.array([0.0, 0.3, 1.0, 0.4, 0.0])

    pb = PhotometricPassband(
        wavelength=wavelength, throughput=throughput, name="synthetic"
    )

    expected = np.sqrt(
        np.trapezoid(throughput * wavelength, wavelength)
        / np.trapezoid(throughput / wavelength, wavelength)
    )

    assert pb.pivot_wavelength == pytest.approx(expected, rel=1e-12)


# ==========================================================
# synthetic_binary_sed()
# ==========================================================


def _load_two_component_grid():

    return load_ck04_grid([
        ATMOSPHERE_DIR / "ckp00_5750.fits",
        ATMOSPHERE_DIR / "ckp00_6000.fits",
    ])


def _load_passbands(names):

    files = {
        "Johnson B": "johnson_b.dat",
        "Johnson V": "johnson_v.dat",
    }

    return {
        name: load_passband(PASSBAND_DIR / files[name], name=name)
        for name in names
    }


def test_synthetic_binary_sed_composite_equals_sum_of_components():
    """
    composite_flux must equal primary_flux + secondary_flux to
    floating-point precision (both components share the same
    E(B-V)/R_V -- extinction is a multiplicative factor applied
    identically to both terms of a sum). Uses ``np.allclose`` with a
    tight relative tolerance rather than exact equality: computing
    ``extinction * (a + b)`` (this function's composite path) and
    ``extinction * a + extinction * b`` (summing the two
    independently-extincted component arrays) are only guaranteed
    equal up to floating-point rounding, not bit-for-bit, even
    though they are mathematically identical.
    """

    grid = _load_two_component_grid()
    passbands = _load_passbands(["Johnson B", "Johnson V"])

    sed = synthetic_binary_sed(
        atmosphere_grid=grid,
        passbands=passbands,
        primary_temperature=6000.0,
        secondary_temperature=5750.0,
        primary_logg=4.5,
        secondary_logg=4.5,
        metallicity=0.0,
        primary_radius=1.0,
        secondary_radius=0.9,
        distance=20.0,
        ebv=0.05,
    )

    assert np.allclose(
        sed["composite_flux"],
        sed["primary_flux"] + sed["secondary_flux"],
        rtol=1e-12,
    )


def test_synthetic_binary_sed_matches_synthetic_binary_magnitudes():
    """
    The real regression this file exists to catch: integrating
    synthetic_binary_sed()'s composite curve through a passband
    (via the same synthetic_band_flux()/pogson_magnitude() steps
    the fitting/benchmark pipeline uses) must reproduce
    synthetic_binary_magnitudes()'s own reported magnitude for that
    passband exactly -- these are two independently-implemented
    functions (deliberately not sharing code, see
    synthetic_binary_sed()'s docstring) that must still agree
    numerically, since they claim to describe the same physical
    model.
    """

    grid = _load_two_component_grid()
    passbands = _load_passbands(["Johnson B", "Johnson V"])
    zero_points = {"Johnson B": 6.72553e-9, "Johnson V": 3.5833e-9}

    kwargs = dict(
        primary_temperature=6000.0,
        secondary_temperature=5750.0,
        primary_logg=4.5,
        secondary_logg=4.5,
        metallicity=0.0,
        primary_radius=1.0,
        secondary_radius=0.9,
        distance=20.0,
        ebv=0.05,
    )

    magnitudes = synthetic_binary_magnitudes(
        atmosphere_grid=grid,
        passbands=passbands,
        zero_points=zero_points,
        **kwargs,
    )

    sed = synthetic_binary_sed(
        atmosphere_grid=grid,
        passbands=passbands,
        **kwargs,
    )

    for name, passband in passbands.items():

        band_flux = synthetic_band_flux(
            wavelength=sed["wavelength"],
            flux=sed["composite_flux"],
            passband=passband,
        )

        recovered_magnitude = pogson_magnitude(
            flux=band_flux, zero_point=zero_points[name]
        )

        assert recovered_magnitude == pytest.approx(
            magnitudes[name], abs=1e-9
        )


def test_synthetic_binary_sed_rejects_empty_passbands():

    grid = _load_two_component_grid()

    with pytest.raises(ValueError):
        synthetic_binary_sed(
            atmosphere_grid=grid,
            passbands={},
            primary_temperature=6000.0,
            secondary_temperature=5750.0,
            primary_logg=4.5,
            secondary_logg=4.5,
            metallicity=0.0,
            primary_radius=1.0,
            secondary_radius=0.9,
            distance=20.0,
        )


# ==========================================================
# plot_sed_fit()
# ==========================================================


def test_plot_sed_fit_draws_the_data_it_is_given():
    """
    Asserts on actual plotted data, not just "runs without raising"
    -- matching this project's own established plot-testing
    convention (see tests/test_orbit_plot.py and
    tests/test_dynamics.py).
    """

    wavelength = np.linspace(4000.0, 6000.0, 50)
    composite = np.linspace(1.0, 2.0, 50)
    primary = 0.6 * composite
    secondary = 0.4 * composite

    obs_wavelength = [4200.0, 5500.0]
    obs_flux = [1.2, 1.8]
    obs_flux_uncertainty = [0.1, 0.15]

    fig, ax = plot_sed_fit(
        wavelength=wavelength,
        composite_flux=composite,
        primary_flux=primary,
        secondary_flux=secondary,
        observed_wavelength=obs_wavelength,
        observed_flux=obs_flux,
        observed_flux_uncertainty=obs_flux_uncertainty,
        observed_labels=["Band A", "Band B"],
        title="Test SED",
        show=False,
    )

    lines = {line.get_label(): line for line in ax.get_lines()}

    assert "Composite model" in lines
    assert np.allclose(lines["Composite model"].get_ydata(), composite)

    assert "Primary component" in lines
    assert np.allclose(lines["Primary component"].get_ydata(), primary)

    assert "Secondary component" in lines
    assert np.allclose(lines["Secondary component"].get_ydata(), secondary)

    # The observed points are drawn via ax.errorbar(), which returns
    # (and registers on the Axes as) an ErrorbarContainer, not a
    # plain Line2D -- its label lives on the container, and its data
    # line is container.lines[0].
    observed_containers = [
        container
        for container in ax.containers
        if container.get_label() == "Observed (fitted)"
    ]

    assert len(observed_containers) == 1

    observed_data_line = observed_containers[0].lines[0]

    assert np.allclose(observed_data_line.get_xdata(), obs_wavelength)
    assert np.allclose(observed_data_line.get_ydata(), obs_flux)

    matplotlib.pyplot.close(fig)


def test_plot_sed_fit_shows_excluded_points_distinctly():

    wavelength = np.linspace(4000.0, 6000.0, 10)
    flat = np.ones_like(wavelength)

    fig, ax = plot_sed_fit(
        wavelength=wavelength,
        composite_flux=flat,
        primary_flux=0.5 * flat,
        secondary_flux=0.5 * flat,
        observed_wavelength=[5000.0],
        observed_flux=[1.0],
        observed_flux_uncertainty=[0.1],
        excluded_wavelength=[3500.0],
        excluded_flux=[0.3],
        excluded_flux_uncertainty=[0.05],
        excluded_labels=["Stromgren u"],
        show=False,
    )

    container_labels = {
        container.get_label() for container in ax.containers
    }

    assert "Excluded (not fit)" in container_labels

    matplotlib.pyplot.close(fig)


def test_plot_sed_fit_avoids_overlapping_labels():
    """
    Regression test for label collisions: drawing each passband
    label at a fixed (0, 8)-point offset collides (as in the
    HD191854 figure) whenever two or more
    observed points were close in both wavelength and flux -- e.g.
    "Stromgren b"/"Johnson B" (~4400-4670 A) and "Johnson
    V"/"Tycho VT"/"Stromgren y" (~5340-5500 A).

    This constructs a deliberately tight, 7-point cluster (modeled on
    that collision) and asserts on the actual rendered label
    bounding boxes -- not just "it ran" -- that adjust_text() leaves
    zero pairwise overlaps, matching this project's own established
    plot-correctness-testing convention.
    """

    wavelength = np.linspace(3500.0, 9000.0, 200)
    composite = np.ones_like(wavelength)
    primary = 0.6 * composite
    secondary = 0.4 * composite

    # Mirrors the real HD191854 collision: two clusters of closely
    # spaced passbands, both in wavelength and in flux.
    obs_wavelength = [4400.0, 4470.0, 4670.0, 5340.0, 5400.0, 5470.0, 5500.0]
    obs_flux = [1.00, 1.02, 0.98, 1.05, 1.03, 1.00, 0.99]
    obs_flux_uncertainty = [0.05] * 7
    obs_labels = [
        "Stromgren b", "Johnson B", "Johnson B2",
        "Johnson V", "Tycho VT", "Stromgren y", "Extra",
    ]

    fig, ax = plot_sed_fit(
        wavelength=wavelength,
        composite_flux=composite,
        primary_flux=primary,
        secondary_flux=secondary,
        observed_wavelength=obs_wavelength,
        observed_flux=obs_flux,
        observed_flux_uncertainty=obs_flux_uncertainty,
        observed_labels=obs_labels,
        show=False,
    )

    # adjust_text() needs a real renderer (it works in display/pixel
    # coordinates) -- force one via a draw, matching how Agg-backend
    # rendering already works for the figure's actual save path.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    label_texts = [
        text for text in ax.texts if text.get_text() in obs_labels
    ]

    assert len(label_texts) == len(obs_labels)

    boxes = [text.get_window_extent(renderer=renderer) for text in label_texts]

    overlapping_pairs = [
        (label_texts[i].get_text(), label_texts[j].get_text())
        for i in range(len(boxes))
        for j in range(i + 1, len(boxes))
        if boxes[i].overlaps(boxes[j])
    ]

    assert overlapping_pairs == [], (
        f"Overlapping SED-fit labels after adjust_text(): {overlapping_pairs}"
    )

    matplotlib.pyplot.close(fig)


def test_plot_sed_fit_saves_file(tmp_path):

    wavelength = np.linspace(4000.0, 6000.0, 10)
    flat = np.ones_like(wavelength)

    save_path = tmp_path / "sed_fit.pdf"

    fig, ax = plot_sed_fit(
        wavelength=wavelength,
        composite_flux=flat,
        primary_flux=0.5 * flat,
        secondary_flux=0.5 * flat,
        observed_wavelength=[5000.0],
        observed_flux=[1.0],
        observed_flux_uncertainty=[0.1],
        show=False,
        output_format="pdf",
        save_path=save_path,
    )

    assert save_path.is_file()
    assert save_path.stat().st_size > 0

    matplotlib.pyplot.close(fig)
