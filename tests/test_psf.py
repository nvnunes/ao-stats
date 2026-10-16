from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy import units as u

import ao_stats.psf as psf_module
from ao_stats import (
    PEAK_METHOD_GAUSSIAN_FIT,
    PEAK_METHOD_PIXEL_MAX,
    STATS_FWHM_SUMMARY_GEOM,
    STATS_FWHM_SUMMARY_MAX,
    STATS_FWHM_SUMMARY_MEAN,
    STATS_FWHM_SUMMARY_MIN,
    PsfMetadata,
    clip_and_sum_normalize_psfs,
    compute_psf_ee,
    compute_psf_fwhm,
    compute_psf_sr,
    compute_psf_stats,
)


def _metadata(
    *,
    wavelength: float | np.ndarray = 1.65,
    pixel_scale: float | np.ndarray = 4.0,
    tel_diameter: float | np.ndarray = 8.0,
    tel_pupil: np.ndarray | None = None,
) -> PsfMetadata:
    return PsfMetadata(
        wavelength=u.Quantity(np.asarray(wavelength), u.um, copy=False),
        pixel_scale=u.Quantity(np.asarray(pixel_scale), u.mas, copy=False),
        tel_diameter=u.Quantity(np.asarray(tel_diameter), u.m, copy=False),
        tel_pupil=u.Quantity(
            np.ones((6, 6), dtype=np.float32) if tel_pupil is None else tel_pupil,
            u.dimensionless_unscaled,
            copy=False,
        ),
    )


def _gaussian_psf(
    ny: int,
    nx: int,
    center_y: float,
    center_x: float,
    sigma_y: float,
    sigma_x: float,
) -> np.ndarray:
    y, x = np.indices((ny, nx), dtype=np.float32)
    return np.exp(
        -0.5 * (((y - center_y) / sigma_y) ** 2 + ((x - center_x) / sigma_x) ** 2),
        dtype=np.float32,
    )


def test_clip_and_sum_normalize_psfs_clips_and_normalizes() -> None:
    psfs = np.array(
        [[[-1.0, 3.0], [1.0, 0.0]], [[-1.0, 0.0], [0.0, 0.0]]],
        dtype=np.float32,
    )

    actual = clip_and_sum_normalize_psfs(psfs)

    np.testing.assert_allclose(
        actual[0], np.array([[0.0, 0.75], [0.25, 0.0]], dtype=np.float32)
    )
    np.testing.assert_array_equal(actual[1], np.zeros((2, 2), dtype=np.float32))
    assert actual.dtype == np.float32


def test_compute_psf_stats_requires_ee_apertures_by_default() -> None:
    with pytest.raises(ValueError, match="ee_apertures is required when computing EE"):
        compute_psf_stats(np.zeros((2, 5, 5), dtype=np.float32), _metadata())


@pytest.mark.parametrize(
    "metrics",
    ["sr", (), ("unknown",), ("sr", "sr")],
)
def test_compute_psf_stats_rejects_invalid_metric_selections(metrics) -> None:
    with pytest.raises((TypeError, ValueError)):
        compute_psf_stats(
            np.zeros((2, 5, 5), dtype=np.float32),
            _metadata(),
            metrics=metrics,
        )


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("peak_method", "invalid", "peak_method must be one of"),
        ("fwhm_summary", "invalid", "fwhm_summary must be one of"),
        ("ee_geometry", "triangle", "ee_geometry must be one of"),
        ("preprocess", "runner", "preprocess must be None"),
    ],
)
def test_compute_psf_stats_rejects_invalid_selectors(
    keyword: str, value: str, message: str
) -> None:
    options = {keyword: value}
    with pytest.raises(ValueError, match=message):
        compute_psf_stats(
            np.zeros((2, 5, 5), dtype=np.float32),
            _metadata(),
            ee_apertures=np.array([10.0], dtype=np.float32) * u.mas,
            **options,
        )


def test_compute_psf_stats_validates_quantity_contracts() -> None:
    psfs = np.zeros((3, 5, 5), dtype=np.float32)
    apertures = np.array([10.0], dtype=np.float32) * u.mas

    with pytest.raises(TypeError, match="metadata.wavelength must be an Astropy"):
        compute_psf_stats(
            psfs,
            PsfMetadata(
                wavelength=1.65,  # type: ignore[arg-type]
                pixel_scale=4.0 * u.mas,
                tel_diameter=8.0 * u.m,
                tel_pupil=np.ones((6, 6)) * u.dimensionless_unscaled,
            ),
            ee_apertures=apertures,
        )
    with pytest.raises(ValueError, match="wavelength per-PSF vector length"):
        compute_psf_stats(
            psfs,
            _metadata(wavelength=np.array([1.6, 1.7], dtype=np.float32)),
            ee_apertures=apertures,
        )
    with pytest.raises(ValueError, match="pixel_scale per-PSF vector length"):
        compute_psf_stats(
            psfs,
            _metadata(pixel_scale=np.array([3.5, 4.5], dtype=np.float32)),
            ee_apertures=apertures,
        )
    with pytest.raises(ValueError, match="tel_diameter must be a scalar"):
        compute_psf_stats(
            psfs,
            _metadata(tel_diameter=np.array([8.0, 8.1], dtype=np.float32)),
            ee_apertures=apertures,
        )
    with pytest.raises(ValueError, match="tel_pupil must be 2D"):
        compute_psf_stats(
            psfs,
            _metadata(tel_pupil=np.ones((2, 6, 6), dtype=np.float32)),
            ee_apertures=apertures,
        )
    with pytest.raises(ValueError, match="ee_apertures per-PSF leading dimension"):
        compute_psf_stats(
            psfs,
            _metadata(),
            ee_apertures=np.ones((2, 1), dtype=np.float32) * u.mas,
        )


def test_ee_subset_uses_peak_locations_without_computing_strehl(monkeypatch) -> None:
    def _fail_strehl(*args, **kwargs):
        raise AssertionError("Strehl should not be computed")

    def _peak_locations(psfs, peak_method):
        del peak_method
        return np.full((psfs.shape[0], 2), 2.0, dtype=np.float32)

    def _enclosed_energy(
        psfs,
        ee_apertures,
        pixel_scale,
        peak_locations_yx=None,
        *,
        ee_geometry="ensquared",
    ):
        del pixel_scale, ee_geometry
        np.testing.assert_allclose(peak_locations_yx, 2.0)
        return np.full((psfs.shape[0], ee_apertures.shape[0]), 0.25, dtype=np.float32)

    monkeypatch.setattr(psf_module, "_compute_strehl", _fail_strehl)
    monkeypatch.setattr(psf_module, "_compute_peak_locations", _peak_locations)
    monkeypatch.setattr(psf_module, "_compute_enclosed_energy", _enclosed_energy)

    psfs = np.zeros((2, 5, 5), dtype=np.float32)
    psfs[:, 2, 2] = 1.0
    (ee,) = compute_psf_stats(
        psfs,
        _metadata(),
        ee_apertures=np.array([10.0], dtype=np.float32) * u.mas,
        metrics=("ee",),
    )
    np.testing.assert_allclose(ee.value, np.full((2, 1), 0.25, dtype=np.float32))


def test_strehl_peak_locations_are_reused_for_enclosed_energy(monkeypatch) -> None:
    expected_peaks = np.full((2, 2), 1.5, dtype=np.float32)

    def _strehl(psfs, *args):
        return np.full((psfs.shape[0],), 0.5, dtype=np.float32), expected_peaks

    def _enclosed_energy(
        psfs,
        ee_apertures,
        pixel_scale,
        peak_locations_yx=None,
        *,
        ee_geometry="ensquared",
    ):
        del pixel_scale, ee_geometry
        np.testing.assert_array_equal(peak_locations_yx, expected_peaks)
        return np.ones((psfs.shape[0], ee_apertures.shape[0]), dtype=np.float32)

    monkeypatch.setattr(psf_module, "_compute_strehl", _strehl)
    monkeypatch.setattr(psf_module, "_compute_enclosed_energy", _enclosed_energy)

    psfs = np.zeros((2, 5, 5), dtype=np.float32)
    psfs[:, 2, 2] = 1.0
    compute_psf_stats(
        psfs,
        _metadata(),
        ee_apertures=np.array([10.0], dtype=np.float32) * u.mas,
    )


def test_metric_subset_preserves_requested_order_and_units(monkeypatch) -> None:
    def _measure(psfs: np.ndarray, pixel_scale: float):
        del pixel_scale
        return (
            np.full((psfs.shape[0],), 4.0, dtype=np.float32),
            np.full((psfs.shape[0],), 9.0, dtype=np.float32),
        )

    def _strehl(*args, **kwargs):
        raise AssertionError("Strehl should not be computed")

    monkeypatch.setattr(psf_module, "_measure_contour_fwhms", _measure)
    monkeypatch.setattr(psf_module, "_compute_strehl", _strehl)

    (fwhm,) = compute_psf_stats(
        np.zeros((2, 5, 5), dtype=np.float32),
        _metadata(),
        metrics=("fwhm",),
    )

    assert fwhm.unit == u.mas
    np.testing.assert_allclose(fwhm.value, np.full((2,), 6.0, dtype=np.float32))


def test_pixel_scale_only_matches_full_metadata_for_scalar_ee_and_fwhm() -> None:
    psf = _gaussian_psf(31, 31, 15.0, 15.0, 2.0, 2.5)
    apertures = np.array([20.0, 40.0], dtype=np.float32) * u.mas

    expected = compute_psf_stats(
        psf,
        _metadata(),
        ee_apertures=apertures,
        metrics=("ee", "fwhm"),
    )
    actual = compute_psf_stats(
        psf,
        pixel_scale=4.0 * u.mas,
        ee_apertures=apertures,
        metrics=("ee", "fwhm"),
    )

    for measured, reference in zip(actual, expected):
        assert measured.unit == reference.unit
        assert measured.dtype == reference.dtype
        np.testing.assert_array_equal(measured.value, reference.value)


def test_pixel_scale_only_cube_supports_per_psf_scales_and_apertures() -> None:
    psf = _gaussian_psf(31, 31, 15.0, 15.0, 2.0, 2.0)
    cube = np.stack((psf, psf))
    scales = np.array([3.5, 4.5], dtype=np.float32) * u.mas
    apertures = np.array([[15.0], [25.0]], dtype=np.float32) * u.mas

    expected = compute_psf_stats(
        cube,
        _metadata(pixel_scale=scales.value),
        ee_apertures=apertures,
        metrics=("fwhm", "ee"),
    )
    actual = compute_psf_stats(
        cube,
        pixel_scale=scales,
        ee_apertures=apertures,
        metrics=("fwhm", "ee"),
    )

    assert actual[0].shape == (2,)
    assert actual[1].shape == (2, 1)
    for measured, reference in zip(actual, expected):
        np.testing.assert_array_equal(measured.value, reference.value)


def test_pixel_scale_only_wrappers_match_full_metadata() -> None:
    psf = _gaussian_psf(31, 31, 15.0, 15.0, 2.0, 2.0)
    apertures = np.array([20.0, 40.0], dtype=np.float32) * u.mas

    ee = compute_psf_ee(psf, pixel_scale=4.0 * u.mas, ee_apertures=apertures)
    ee_scalar = compute_psf_ee(
        psf,
        pixel_scale=4.0 * u.mas,
        ee_apertures=apertures[:1],
    )
    fwhm = compute_psf_fwhm(psf, pixel_scale=4.0 * u.mas)

    np.testing.assert_array_equal(
        ee.value,
        compute_psf_ee(psf, _metadata(), ee_apertures=apertures).value,
    )
    assert fwhm == compute_psf_fwhm(psf, _metadata())
    assert ee_scalar.isscalar


def test_pixel_scale_only_rejects_missing_or_conflicting_inputs() -> None:
    psf = _gaussian_psf(31, 31, 15.0, 15.0, 2.0, 2.0)

    with pytest.raises(ValueError, match="pixel_scale is required"):
        compute_psf_stats(psf, metrics=("fwhm",))
    with pytest.raises(ValueError, match="Strehl requires PsfMetadata"):
        compute_psf_stats(psf, pixel_scale=4.0 * u.mas, metrics=("sr",))
    with pytest.raises(ValueError, match="Strehl requires PsfMetadata"):
        compute_psf_stats(psf, pixel_scale=4.0 * u.mas)
    with pytest.raises(ValueError, match="Strehl requires PsfMetadata"):
        compute_psf_stats(psf)
    with pytest.raises(ValueError, match="metadata or pixel_scale"):
        compute_psf_stats(
            psf,
            _metadata(),
            pixel_scale=4.0 * u.mas,
            metrics=("fwhm",),
        )
    with pytest.raises(ValueError, match="pixel_scale per-PSF vector length"):
        compute_psf_stats(
            np.stack((psf, psf)),
            pixel_scale=np.array([4.0]) * u.mas,
            metrics=("fwhm",),
        )
    with pytest.raises(TypeError, match="pixel_scale must be an Astropy"):
        compute_psf_stats(psf, pixel_scale=4.0, metrics=("fwhm",))
    with pytest.raises(ValueError, match="pixel_scale must have units equivalent"):
        compute_psf_stats(psf, pixel_scale=4.0 * u.m, metrics=("fwhm",))
    with pytest.raises(ValueError, match="pixel_scale must be finite and > 0"):
        compute_psf_stats(psf, pixel_scale=0 * u.mas, metrics=("fwhm",))


def test_public_metric_wrappers_return_expected_shapes_and_units() -> None:
    psf = _gaussian_psf(31, 31, 15.0, 15.0, 2.0, 2.5)
    psf /= psf.sum(dtype=np.float32)
    cube = np.stack((psf, psf), axis=0)
    apertures = np.array([20.0, 40.0], dtype=np.float32) * u.mas

    sr = compute_psf_sr(cube, _metadata(), peak_method=PEAK_METHOD_PIXEL_MAX)
    ee = compute_psf_ee(cube, _metadata(), ee_apertures=apertures)
    fwhm = compute_psf_fwhm(cube, _metadata())

    assert sr.shape == (2,)
    assert ee.shape == (2, 2)
    assert fwhm.shape == (2,)
    assert sr.unit == u.dimensionless_unscaled
    assert ee.unit == u.dimensionless_unscaled
    assert fwhm.unit == u.mas
    assert sr.dtype == ee.dtype == fwhm.dtype == np.float32


def test_scalar_image_formats_scalar_and_aperture_outputs() -> None:
    psf = _gaussian_psf(31, 31, 15.0, 15.0, 2.0, 2.0)
    psf /= psf.sum(dtype=np.float32)

    sr = compute_psf_sr(psf, _metadata(), peak_method=PEAK_METHOD_PIXEL_MAX)
    ee_scalar = compute_psf_ee(
        psf,
        _metadata(),
        ee_apertures=np.array([20.0], dtype=np.float32) * u.mas,
    )
    ee_vector = compute_psf_ee(
        psf,
        _metadata(),
        ee_apertures=np.array([20.0, 40.0], dtype=np.float32) * u.mas,
    )

    assert sr.isscalar
    assert ee_scalar.isscalar
    assert ee_vector.shape == (2,)


def test_per_psf_metadata_and_apertures_are_supported() -> None:
    psf = _gaussian_psf(31, 31, 15.0, 15.0, 2.0, 2.0)
    psf /= psf.sum(dtype=np.float32)
    cube = np.stack((psf, psf), axis=0)

    (ee,) = compute_psf_stats(
        cube,
        _metadata(
            wavelength=np.array([1.5, 1.7], dtype=np.float32),
            pixel_scale=np.array([3.5, 4.5], dtype=np.float32),
        ),
        ee_apertures=np.array([[15.0], [25.0]], dtype=np.float32) * u.mas,
        metrics=("ee",),
    )

    assert ee.shape == (2, 1)
    assert np.all(np.isfinite(ee))


def test_default_and_callable_preprocessing_are_applied(monkeypatch) -> None:
    observed: list[np.ndarray] = []

    def _measure(psfs, pixel_scale):
        del pixel_scale
        observed.append(np.asarray(psfs, dtype=np.float32))
        return (
            np.ones((psfs.shape[0],), dtype=np.float32),
            np.ones((psfs.shape[0],), dtype=np.float32),
        )

    monkeypatch.setattr(psf_module, "_measure_contour_fwhms", _measure)
    source = np.array([[[-1.0, 3.0], [1.0, 0.0]]], dtype=np.float32)

    compute_psf_fwhm(source, _metadata())
    compute_psf_fwhm(source, _metadata(), preprocess=lambda value: value + 2.0)

    np.testing.assert_allclose(
        observed[0], np.array([[[0.0, 0.75], [0.25, 0.0]]], dtype=np.float32)
    )
    np.testing.assert_allclose(observed[1], source + 2.0)


def test_explicit_none_preserves_prepared_finite_field_flux() -> None:
    psf = np.zeros((9, 9), dtype=np.float32)
    psf[4, 4] = 0.5

    ee = compute_psf_ee(
        psf,
        _metadata(),
        ee_apertures=np.array([4.0], dtype=np.float32) * u.mas,
        peak_method=PEAK_METHOD_PIXEL_MAX,
        preprocess=None,
    )

    assert ee.to_value(u.dimensionless_unscaled) == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("psf", "message"),
    [
        (np.zeros((5, 5), dtype=np.float32), "positive in-frame flux"),
        (-np.ones((5, 5), dtype=np.float32), "must be nonnegative"),
        (np.full((5, 5), 0.08, dtype=np.float32), "total in-frame flux <= 1"),
    ],
)
def test_explicit_none_rejects_invalid_flux_inputs(
    psf: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        compute_psf_sr(
            psf,
            _metadata(),
            peak_method=PEAK_METHOD_PIXEL_MAX,
            preprocess=None,
        )


@pytest.mark.parametrize("shape", [(0, 5), (5, 0), (0, 5, 5), (1, 0, 5)])
def test_empty_psf_dimensions_are_rejected(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="non-empty cube and spatial dimensions"):
        compute_psf_fwhm(np.empty(shape, dtype=np.float32), _metadata())


def test_empty_and_zero_power_pupils_are_rejected() -> None:
    psf = np.zeros((5, 5), dtype=np.float32)
    psf[2, 2] = 1.0

    with pytest.raises(ValueError, match="non-empty spatial dimensions"):
        compute_psf_sr(psf, _metadata(tel_pupil=np.empty((0, 5))))
    with pytest.raises(ValueError, match="finite positive power"):
        compute_psf_sr(psf, _metadata(tel_pupil=np.zeros((5, 5))))


def test_unsupported_diffraction_sampling_is_rejected() -> None:
    psf = np.zeros((7, 7), dtype=np.float32)
    psf[3, 3] = 1.0

    with pytest.raises(ValueError, match=">= 1 pixel per wavelength/diameter"):
        compute_psf_sr(
            psf,
            _metadata(pixel_scale=100.0),
            peak_method=PEAK_METHOD_PIXEL_MAX,
        )


@pytest.mark.parametrize(
    ("pupil_size", "fft_size", "diameter", "wavelength"),
    [(128, 256, 8.0, 1.65), (396, 1584, 9.655555555555557, 2.179)],
)
@pytest.mark.parametrize("per_psf_metadata", [False, True])
def test_independent_diffraction_psf_has_unit_pixel_strehl(
    pupil_size: int,
    fft_size: int,
    diameter: float,
    wavelength: float,
    per_psf_metadata: bool,
) -> None:
    axis = np.arange(pupil_size) - pupil_size // 2
    x, y = np.meshgrid(axis, axis)
    radius = np.hypot(x, y)
    pupil = ((radius <= pupil_size / 4) & (radius >= pupil_size / 16)).astype(float)
    # The expected PSF uses an explicit FFT size, independently of AO Stats'
    # inference from rounded physical metadata. Its crop retains lost flux.
    field = np.fft.fftshift(np.fft.fft2(pupil, s=(fft_size, fft_size)))
    psf = np.abs(field) ** 2 / (fft_size**2 * np.sum(pupil**2))
    start = fft_size // 2 - 32
    psf = psf[start : start + 64, start : start + 64]
    pixel_scale = (
        wavelength * 1e-6 * pupil_size / (diameter * fft_size) * u.rad.to(u.mas)
    )
    metadata = _metadata(
        wavelength=np.array([wavelength, wavelength]) if per_psf_metadata else wavelength,
        pixel_scale=(
            np.array([pixel_scale, pixel_scale]) if per_psf_metadata else pixel_scale
        ),
        tel_diameter=diameter,
        tel_pupil=pupil,
    )
    cube = np.stack([psf, 0.4 * psf])
    actual = compute_psf_sr(
        cube, metadata, peak_method=PEAK_METHOD_PIXEL_MAX, preprocess=None
    )
    np.testing.assert_allclose(actual.value, [1.0, 0.4], rtol=1e-6, atol=0)
    if not per_psf_metadata:
        single = compute_psf_sr(
            psf, metadata, peak_method=PEAK_METHOD_PIXEL_MAX, preprocess=None
        )
        assert single.shape == ()
        assert single.value == pytest.approx(1.0, rel=1e-6)


@pytest.mark.parametrize("padding", [3.01, 3.49, 3.51, 3.99])
def test_diffraction_reference_keeps_fractional_padding_truncation(padding: float) -> None:
    pupil = np.ones((10, 10))
    wavelength = 1.65
    sampling = 1.0 + 2.0 * padding / 10
    pixel_scale = wavelength * 1e-6 / 8.0 / sampling * u.rad.to(u.mas)
    reference = psf_module._get_diffraction_limited_psf(
        8.0, pupil, wavelength, pixel_scale, center_in_one_pix=True
    )
    field = np.fft.fftshift(np.fft.fft2(np.pad(pupil, 3)))
    expected = np.abs(field) ** 2
    expected /= expected.sum()
    assert reference.shape == (16, 16)
    np.testing.assert_allclose(reference, expected, rtol=1e-6, atol=1e-8)


def test_diffraction_limited_psf_has_unit_strehl_for_both_methods() -> None:
    pupil = np.ones((6, 6), dtype=np.float32)
    for method, center_in_one_pix in (
        (PEAK_METHOD_GAUSSIAN_FIT, False),
        (PEAK_METHOD_PIXEL_MAX, True),
    ):
        diffraction_limited = psf_module._get_diffraction_limited_psf(
            8.0,
            pupil,
            1.65,
            4.0,
            center_in_one_pix=center_in_one_pix,
        )
        actual = compute_psf_sr(
            diffraction_limited,
            _metadata(tel_pupil=pupil),
            peak_method=method,
        )
        assert actual.to_value(u.dimensionless_unscaled) == pytest.approx(
            1.0, rel=1e-4, abs=1e-4
        )


@pytest.mark.parametrize("shape", [(40, 40), (41, 41), (40, 41), (41, 40)])
@pytest.mark.parametrize("center", [(7.2, 30.3), (0.2, 38.8), (38.8, 0.2)])
def test_gaussian_fit_recovers_displaced_peak(
    shape: tuple[int, int], center: tuple[float, float]
) -> None:
    psf = _gaussian_psf(*shape, *center, 1.4, 1.8)

    fit, _ = psf_module._fit_gaussian_core(psf.astype(float))

    np.testing.assert_allclose(fit[[3, 4]], center, atol=1e-3)
    assert fit[0] == pytest.approx(1.0, rel=1e-3)


@pytest.mark.parametrize("shape", [(3, 4), (4, 3), (3, 3)])
def test_gaussian_fit_supports_images_smaller_than_patch(
    shape: tuple[int, int],
) -> None:
    psf = _gaussian_psf(*shape, 1.2, 1.3, 1.4, 1.8)

    fit, _ = psf_module._fit_gaussian_core(psf.astype(float))

    np.testing.assert_allclose(fit[[3, 4]], [1.2, 1.3], atol=1e-3)


@pytest.mark.parametrize("shape", [(40, 40), (41, 41), (40, 41), (41, 40)])
def test_fixed_center_gaussian_fit_retains_optical_center(
    shape: tuple[int, int],
) -> None:
    center = 0.5 * (np.asarray(shape) - 1)
    psf = _gaussian_psf(*shape, *center, 1.4, 1.8)

    fit, _ = psf_module._fit_gaussian_core(psf.astype(float), assume_centered=True)

    np.testing.assert_allclose(fit[[3, 4]], center, atol=1e-6)
    assert fit[0] == pytest.approx(1.0, rel=1e-3)


@pytest.mark.parametrize("geometry", ["ensquared", "encircled"])
def test_gaussian_peak_measurements_are_translation_invariant(geometry: str) -> None:
    psf = _gaussian_psf(48, 50, 23.2, 24.3, 1.4, 1.8)
    psf *= 0.8 / psf.sum(dtype=np.float64)
    shifted = np.roll(psf, (-10, 12), axis=(0, 1))
    cube = np.stack((psf, shifted))
    options = {
        "ee_apertures": np.array([12.0, 24.0]) * u.mas,
        "ee_geometry": geometry,
        "preprocess": None,
        "metrics": ("sr", "ee"),
    }

    sr, ee = compute_psf_stats(cube, _metadata(), **options)
    single_sr, single_ee = compute_psf_stats(shifted, _metadata(), **options)

    assert sr.shape == (2,)
    assert ee.shape == (2, 2)
    assert single_sr.isscalar
    assert single_ee.shape == (2,)
    assert sr.dtype == ee.dtype == np.float32
    assert sr.unit == ee.unit == u.dimensionless_unscaled
    np.testing.assert_allclose(sr.value, sr.value[0], rtol=1e-4)
    np.testing.assert_allclose(ee.value[1], ee.value[0], atol=2e-6)
    np.testing.assert_allclose(single_sr.value, sr.value[1], rtol=1e-6)
    np.testing.assert_allclose(single_ee.value, ee.value[1], atol=2e-6)


def test_displaced_native_psfs_have_consistent_ensquared_energy() -> None:
    # Exact, unrenormalized float32 Specula PSFs from the 2026-10-13 native
    # component comparison: science index 28, 50 warmup + 20 retained frames.
    with np.load(
        Path(__file__).parent / "data" / "displaced-peak-psfs.npz",
        allow_pickle=False,
    ) as data:
        cube = np.stack([data[key] for key in ("baseline", "native_lgs", "native_fft")])
    options = {
        "pixel_scale": 11.637109077416333 * u.mas,
        "ee_apertures": np.array([100.0]) * u.mas,
        "preprocess": None,
    }

    ee = compute_psf_ee(cube, **options)
    single_ee = compute_psf_ee(cube[0], **options)

    assert cube.dtype == np.float32
    np.testing.assert_allclose(
        cube.sum(axis=(1, 2), dtype=np.float64), 0.891059, rtol=0, atol=1e-7
    )
    np.testing.assert_allclose(ee.value, 0.47576088, rtol=0, atol=2e-6)
    np.testing.assert_allclose(ee.value, ee.value[0, 0], rtol=0, atol=3e-7)
    assert single_ee.isscalar
    np.testing.assert_allclose(single_ee.value, ee.value[0, 0], atol=1e-7)


def test_enclosed_energy_geometries_preserve_expected_relationship() -> None:
    psf = _gaussian_psf(31, 31, 15.0, 15.0, 2.0, 2.0)
    psf /= psf.sum(dtype=np.float32)
    apertures = np.array([12.0, 20.0], dtype=np.float32) * u.mas

    encircled = compute_psf_ee(
        psf,
        _metadata(),
        ee_apertures=apertures,
        ee_geometry="encircled",
    )
    ensquared = compute_psf_ee(
        psf,
        _metadata(),
        ee_apertures=apertures,
        ee_geometry="ensquared",
    )

    assert np.all(encircled <= ensquared + 1e-6)


def test_encircled_energy_uses_exact_aperture_weights() -> None:
    psf = np.zeros((1, 5, 5), dtype=np.float32)
    psf[0, 2, 2] = 1.0

    ee = psf_module._compute_enclosed_energy(
        psf,
        np.array([2.0, 6.0], dtype=float),
        2.0,
        np.array([[2, 2]], dtype=np.int64),
        ee_geometry="encircled",
    )

    np.testing.assert_allclose(
        ee, np.array([[np.pi / 4.0, 1.0]], dtype=np.float32), atol=1e-6
    )


def test_ensquared_energy_supports_subpixel_and_integer_peaks() -> None:
    psf = np.zeros((1, 5, 5), dtype=np.float32)
    psf[0, 2:4, 2:4] = 0.25

    subpixel = psf_module._compute_enclosed_energy(
        psf,
        np.array([2.0], dtype=float),
        2.0,
        np.array([[2.5, 2.5]], dtype=np.float32),
    )
    integer = psf_module._compute_enclosed_energy(
        psf,
        np.array([2.0], dtype=float),
        2.0,
        np.array([[2, 2]], dtype=np.int64),
    )

    assert subpixel.shape == integer.shape == (1, 1)
    assert np.all(np.isfinite(subpixel))
    assert float(subpixel[0, 0]) >= float(integer[0, 0])


def test_encircled_energy_is_stable_when_more_apertures_are_requested() -> None:
    psf = _gaussian_psf(9, 9, 4.0, 4.0, 1.5, 1.5)[None, :, :]
    psf /= psf.sum(dtype=np.float32)
    peak = np.array([[4, 4]], dtype=np.int64)

    single = psf_module._compute_enclosed_energy(
        psf, np.array([4.0]), 1.0, peak, ee_geometry="encircled"
    )
    multiple = psf_module._compute_enclosed_energy(
        psf, np.array([4.0, 8.0]), 1.0, peak, ee_geometry="encircled"
    )

    np.testing.assert_allclose(single[:, 0], multiple[:, 0], atol=1e-7)


def test_peak_centered_ensquared_curves_apply_radius_factor() -> None:
    psf = np.zeros((1, 21, 21), dtype=np.float32)
    psf[0, 10, 10] = 1.0

    curves, radii = psf_module._measure_peak_centered_ensquared_energy_curves(
        psf,
        np.array([10], dtype=np.int64),
        np.array([10], dtype=np.int64),
        max_radius=4,
        pixel_scale=1.0,
    )

    assert curves.shape == (1, 5)
    np.testing.assert_allclose(radii, np.array([0.5, 1.5, 2.5, 3.5, 4.5]))


@pytest.mark.parametrize(
    ("summary", "expected"),
    [
        (STATS_FWHM_SUMMARY_GEOM, 6.0),
        (STATS_FWHM_SUMMARY_MEAN, 6.5),
        (STATS_FWHM_SUMMARY_MAX, 9.0),
        (STATS_FWHM_SUMMARY_MIN, 4.0),
    ],
)
def test_fwhm_summary_selectors(summary: str, expected: float) -> None:
    actual = psf_module._compute_fwhm_summary(
        summary,
        np.array([4.0], dtype=np.float32),
        np.array([9.0], dtype=np.float32),
    )
    np.testing.assert_allclose(actual, np.array([expected], dtype=np.float32))


def test_contour_fwhm_returns_nan_when_not_recoverable() -> None:
    fwhm = compute_psf_fwhm(
        np.zeros((5, 5), dtype=np.float32),
        _metadata(pixel_scale=2.0),
    )
    assert np.isnan(fwhm.value)


def test_contour_fwhm_recovers_square_ring_widths() -> None:
    psf = np.zeros((1, 7, 7), dtype=np.float32)
    psf[0, 1:6, 1:6] = 1.0
    psf[0, 2:5, 2:5] = 0.0

    fwhm_min, fwhm_max = psf_module._measure_contour_fwhms(psf, 2.0)

    np.testing.assert_allclose(fwhm_min, [10.0], atol=1e-6)
    np.testing.assert_allclose(fwhm_max, [12.806249], atol=1e-6)


def test_contour_fwhm_returns_nan_for_collapsed_geometry(monkeypatch) -> None:
    psf = np.zeros((1, 7, 7), dtype=np.float32)
    psf[0, 3, 3] = 1.0
    monkeypatch.setattr(
        psf_module,
        "_find_contours",
        lambda _psf, _level: [
            np.array([[3.0, 2.0], [3.0, 3.0], [3.0, 4.0]], dtype=float)
        ],
    )

    fwhm_min, fwhm_max = psf_module._measure_contour_fwhms(psf, 2.0)

    assert np.isnan(fwhm_min[0])
    assert np.isnan(fwhm_max[0])


def test_preprocess_callable_must_preserve_shape_and_finite_values() -> None:
    psfs = np.zeros((2, 5, 5), dtype=np.float32)

    with pytest.raises(ValueError, match="preprocess must return a PSF cube"):
        compute_psf_fwhm(psfs, _metadata(), preprocess=lambda value: value[:, :-1])
    with pytest.raises(ValueError, match="preprocess returned PSFs containing"):
        compute_psf_fwhm(
            psfs,
            _metadata(),
            preprocess=lambda value: np.full_like(value, np.nan),
        )
