"""Fourier contour accuracy, validity, and public calculation contracts."""

import numpy as np
import pytest
from astropy import units as u
from scipy.optimize import brentq
from scipy.special import j1

from ao_stats import (
    DEFAULT_FWHM_METHOD,
    DEFAULT_FWHM_REFINEMENT_FACTOR,
    STATS_FWHM_METHOD_CONTOUR,
    STATS_FWHM_METHOD_FOURIER_CONTOUR,
    STATS_FWHM_METHODS,
    PsfMetadata,
    compute_psf_fwhm,
    compute_psf_stats,
)


def _profile(kind, shape=(41, 46), offset=(0.25, 0.5), angle=30.0):
    """Sample an elliptical intensity profile with known half-maximum axes."""
    y, x = np.indices(shape, dtype=float)
    x = x - (shape[1] - 1) / 2 - offset[0]
    y = y - (shape[0] - 1) / 2 - offset[1]
    angle = np.deg2rad(angle)
    major = x * np.cos(angle) + y * np.sin(angle)
    minor = -x * np.sin(angle) + y * np.cos(angle)
    radius = np.hypot(major / 7.0, minor / 5.0)
    if kind == "gaussian":
        return np.exp(-4 * np.log(2) * radius**2)
    if kind == "moffat":
        return (1 + 4 * (2 ** (1 / 3.5) - 1) * radius**2) ** -3.5
    half_radius = brentq(lambda z: (2 * j1(z) / z) ** 2 - 0.5, 1.0, 2.0)
    z = 2 * half_radius * radius
    amplitude = np.ones_like(z)
    np.divide(2 * j1(z), z, out=amplitude, where=z != 0)
    return amplitude**2


def test_public_fwhm_selectors_and_defaults():
    assert STATS_FWHM_METHODS == ("contour", "fourier_contour")
    assert STATS_FWHM_METHODS == (
        STATS_FWHM_METHOD_CONTOUR,
        STATS_FWHM_METHOD_FOURIER_CONTOUR,
    )
    assert DEFAULT_FWHM_METHOD == "contour"
    assert DEFAULT_FWHM_REFINEMENT_FACTOR == 4


@pytest.mark.parametrize("kind", ["gaussian", "moffat", "airy"])
@pytest.mark.parametrize("shape", [(41, 46), (42, 45)])
@pytest.mark.parametrize("offset", [(0, 0), (0.25, 0.5), (-0.3, 0.1)])
def test_fourier_contour_recovers_known_widths(kind, shape, offset):
    image = _profile(kind, shape, offset)
    native = compute_psf_fwhm(image, pixel_scale=2 * u.mas, dtype=np.float64)
    refined = compute_psf_fwhm(
        image,
        pixel_scale=2 * u.mas,
        fwhm_method="fourier_contour",
        dtype=np.float64,
    )
    true_width = 2 * np.sqrt(5 * 7)
    assert abs(refined.value / true_width - 1) < 0.003
    assert abs(refined.value - true_width) < abs(native.value - true_width)


@pytest.mark.parametrize(
    ("summary", "true_width"),
    [("min", 5.0), ("max", 7.0), ("geom", np.sqrt(35)), ("mean", 6.0)],
)
def test_refined_summary_preserves_contour_definition(summary, true_width):
    result = compute_psf_fwhm(
        _profile("gaussian"),
        pixel_scale=1 * u.mas,
        fwhm_method="fourier_contour",
        fwhm_summary=summary,
        dtype=np.float64,
    )
    np.testing.assert_allclose(result.value, true_width, rtol=0.003)


@pytest.mark.parametrize("factor", [2, 3, 4, 8, np.int64(4)])
def test_requested_multiplier_refines_known_profile(factor):
    result = compute_psf_fwhm(
        _profile("gaussian", offset=(0.1, 0.2)),
        pixel_scale=1 * u.mas,
        fwhm_method="fourier_contour",
        fwhm_refinement_factor=factor,
        dtype=np.float64,
    )
    assert abs(result.value / np.sqrt(35) - 1) < 0.02 / int(factor)


def test_multiplier_changes_sampling_and_one_is_exact_native_result():
    image = _profile("gaussian", offset=(0.1, 0.2))
    native = compute_psf_fwhm(image, pixel_scale=1 * u.mas, dtype=np.float64)
    widths = []
    for factor in (1, 2, 4, 8):
        result = compute_psf_fwhm(
            image,
            pixel_scale=1 * u.mas,
            fwhm_method="fourier_contour",
            fwhm_refinement_factor=factor,
            dtype=np.float64,
        )
        widths.append(result.value)
    assert widths[0] == native.value
    assert len(set(widths)) == len(widths)
    assert abs(widths[-1] - np.sqrt(35)) < abs(widths[1] - np.sqrt(35))
    explicit_native = compute_psf_fwhm(
        image,
        pixel_scale=1 * u.mas,
        fwhm_method="contour",
        fwhm_refinement_factor=8,
        dtype=np.float64,
    )
    assert explicit_native == native


@pytest.mark.parametrize("factor", [0, -1, 2.5, 4.0, True, np.bool_(True), "4", None])
@pytest.mark.parametrize("method", ["contour", "fourier_contour"])
def test_reject_invalid_multiplier_without_silent_coercion(factor, method):
    with pytest.raises(ValueError, match="fwhm_refinement_factor.*positive integer"):
        compute_psf_fwhm(
            _profile("gaussian"),
            pixel_scale=1 * u.mas,
            fwhm_method=method,
            fwhm_refinement_factor=factor,
        )


def test_reject_unknown_fwhm_method():
    with pytest.raises(ValueError, match="fwhm_method must be one of"):
        compute_psf_fwhm(
            _profile("gaussian"), pixel_scale=1 * u.mas, fwhm_method="unknown"
        )


@pytest.mark.parametrize("factor", [1, 2, 4, 8])
def test_refinement_cannot_recover_truncated_or_constant_psfs(factor):
    y, x = np.indices((15, 17), dtype=float)
    truncated = np.exp(-4 * np.log(2) * (((x - 1) / 7) ** 2 + ((y - 7) / 5) ** 2))
    psfs = np.stack([truncated, np.ones_like(truncated), np.zeros_like(truncated)])
    values = compute_psf_fwhm(
        psfs,
        pixel_scale=1 * u.mas,
        fwhm_method="fourier_contour",
        fwhm_refinement_factor=factor,
    )
    assert np.isnan(values.value).all()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("cube", [False, True])
@pytest.mark.parametrize("per_psf", [False, True])
@pytest.mark.parametrize("factor", [4, 8])
def test_focused_combined_units_precision_and_other_metrics(
    dtype, cube, per_psf, factor
):
    image = _profile("gaussian")
    source = np.stack([image, _profile("airy")]) if cube else image
    original = source.copy()
    count = 2 if cube else 1
    scale = (np.array([2.0, 3.0])[:count] if per_psf else 2.0) * u.mas
    metadata = PsfMetadata(
        wavelength=(np.array([1.5, 1.65])[:count] if per_psf else 1.65) * u.um,
        pixel_scale=scale,
        tel_diameter=8 * u.m,
        tel_pupil=np.ones((8, 8)) * u.one,
    )
    native = compute_psf_stats(
        source, metadata, ee_apertures=[30, 60] * u.mas, dtype=dtype
    )
    refined = compute_psf_stats(
        source,
        metadata,
        ee_apertures=[30, 60] * u.mas,
        fwhm_method="fourier_contour",
        fwhm_refinement_factor=factor,
        dtype=dtype,
    )
    focused = compute_psf_fwhm(
        source,
        pixel_scale=scale,
        fwhm_method="fourier_contour",
        fwhm_refinement_factor=factor,
        dtype=dtype,
    )
    np.testing.assert_array_equal(native[0], refined[0])
    np.testing.assert_array_equal(native[1], refined[1])
    np.testing.assert_array_equal(focused, refined[2])
    assert focused.unit == u.mas
    assert focused.dtype == np.dtype(dtype)
    assert focused.shape == ((2,) if cube else ())
    np.testing.assert_array_equal(source, original)


def test_preprocess_runs_once_and_preserves_custom_flux_scale():
    source = _profile("gaussian")
    source *= 0.8 / source.sum()
    seen = []

    def prepare(values):
        seen.append(values.shape)
        return values

    result = compute_psf_fwhm(
        source, pixel_scale=1 * u.mas, fwhm_method="fourier_contour", preprocess=prepare
    )
    expected = compute_psf_fwhm(
        source, pixel_scale=1 * u.mas, fwhm_method="fourier_contour", preprocess=None
    )
    assert seen == [(1, *source.shape)]
    assert result == expected


def test_refinement_is_not_computed_for_other_metric_subsets(monkeypatch):
    def unexpected_refinement(*args, **kwargs):
        raise AssertionError("An EE-only request must not interpolate the PSF")

    monkeypatch.setattr("ao_stats.psf.resample", unexpected_refinement)
    image = _profile("gaussian")
    native = compute_psf_stats(
        image, pixel_scale=1 * u.mas, metrics=("ee",), ee_apertures=[10] * u.mas
    )
    refined = compute_psf_stats(
        image,
        pixel_scale=1 * u.mas,
        metrics=("ee",),
        ee_apertures=[10] * u.mas,
        fwhm_method="fourier_contour",
    )
    np.testing.assert_array_equal(refined[0], native[0])
    with pytest.raises(ValueError, match="fwhm_refinement_factor.*positive integer"):
        compute_psf_stats(
            image,
            pixel_scale=1 * u.mas,
            metrics=("ee",),
            ee_apertures=[10] * u.mas,
            fwhm_refinement_factor=0,
        )


def test_cube_keeps_valid_widths_when_other_members_are_truncated():
    image = _profile("gaussian")
    truncated = _profile("gaussian", offset=(-21, 0))
    values = compute_psf_fwhm(
        np.stack([image, truncated]),
        pixel_scale=[1, 2] * u.mas,
        fwhm_method="fourier_contour",
    )
    assert np.isfinite(values[0].value)
    assert np.isnan(values[1].value)
