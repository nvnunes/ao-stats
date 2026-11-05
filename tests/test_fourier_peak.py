"""Optical accuracy, flux preservation and shared Fourier calculation contracts."""

import numpy as np
import pytest
from astropy import units as u

import ao_stats.psf as module
from ao_stats import (
    DEFAULT_PEAK_METHOD,
    DEFAULT_PEAK_REFINEMENT_FACTOR,
    PEAK_METHOD_FOURIER_MAX,
    PEAK_METHODS,
    PsfMetadata,
    compute_psf_ee,
    compute_psf_fwhm,
    compute_psf_sr,
    compute_psf_stats,
)


def optical_psf(sampling=3, shape=(41, 42), offset=(0.25, 0.5), strength=0.5):
    """Direct field quadrature; symmetric aberration keeps its core at the origin."""
    size = 24
    coordinates = (np.arange(size) - (size - 1) / 2) / size
    y, x = np.meshgrid(coordinates, coordinates, indexing="ij")
    pupil = ((x*x + y*y <= 0.25) & (x*x + y*y >= 0.15**2)).astype(float)
    phase = strength * (6*(x*x + y*y) + 3*(x*x - y*y))
    field = pupil * np.exp(1j * phase)
    yy = (np.arange(shape[0]) - shape[0] // 2 - offset[0]) / sampling
    xx = (np.arange(shape[1]) - shape[1] // 2 - offset[1]) / sampling
    ey = np.exp(-2j*np.pi*np.outer(yy, coordinates))
    ex = np.exp(-2j*np.pi*np.outer(xx, coordinates))
    image = abs(ey @ field @ ex.T)**2 / (size**2 * sampling**2 * (pupil*pupil).sum())
    truth = abs(field.sum())**2 / pupil.sum()**2
    metadata = PsfMetadata(
        wavelength=2.2 * u.um,
        pixel_scale=(2.2e-6 / 8 / sampling * u.rad).to(u.mas),
        tel_diameter=8 * u.m, tel_pupil=pupil * u.one,
    )
    return image, metadata, truth


def test_public_peak_options_and_defaults():
    assert PEAK_METHOD_FOURIER_MAX == "fourier_max"
    assert PEAK_METHODS == ("gaussian_fit", "pixel_max", "fourier_max")
    assert DEFAULT_PEAK_METHOD == "gaussian_fit"
    assert DEFAULT_PEAK_REFINEMENT_FACTOR == 4


@pytest.mark.parametrize("sampling", [2, 3, 4])
@pytest.mark.parametrize("shape", [(41, 42), (42, 41)])
@pytest.mark.parametrize("offset", [(0, 0), (0.25, 0.5), (0.5, -0.25), (0.1, 0.37), (0.185, 0.37)])
@pytest.mark.parametrize("strength", [0, 0.5, 1])
def test_refined_strehl_against_independent_pupil_quadrature(sampling, shape, offset, strength):
    image, metadata, truth = optical_psf(sampling, shape, offset, strength)
    native = compute_psf_sr(image, metadata, peak_method="pixel_max", preprocess=None, dtype=np.float64)
    refined = compute_psf_sr(image, metadata, peak_method="fourier_max",
                             peak_refinement_factor=8, preprocess=None, dtype=np.float64)
    # Eightfold sampling retains a small grid error at the Nyquist boundary.
    assert abs(refined.value - truth) < 0.003
    assert abs(refined.value - truth) <= abs(native.value - truth) + 0.0003


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("factor", [1, 2, 4, 8, np.int64(8)])
def test_factor_one_and_precision(dtype, factor):
    image, metadata, _ = optical_psf()
    result = compute_psf_sr(image, metadata, peak_method="fourier_max",
                            peak_refinement_factor=factor, preprocess=None, dtype=dtype)
    assert result.dtype == np.dtype(dtype)
    if factor == 1:
        native = compute_psf_sr(image, metadata, peak_method="pixel_max", preprocess=None, dtype=dtype)
        assert result == native


@pytest.mark.parametrize("factor", [0, -1, 2.5, 4.0, True, np.bool_(True), "4", None])
@pytest.mark.parametrize("method", ["gaussian_fit", "pixel_max", "fourier_max"])
def test_reject_invalid_refinement_factor(factor, method):
    image, _, _ = optical_psf()
    with pytest.raises(ValueError, match="peak_refinement_factor.*positive integer"):
        compute_psf_ee(image, pixel_scale=1*u.mas, ee_apertures=[10]*u.mas,
                       peak_method=method, peak_refinement_factor=factor)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("per_psf", [False, True])
def test_cube_focused_methods_preserve_flux_and_preprocessing(dtype, per_psf):
    image, metadata, _ = optical_psf()
    source = np.stack([image, image * 0.6])
    original = source.copy()
    if per_psf:
        metadata = PsfMetadata(metadata.wavelength * np.ones(2), metadata.pixel_scale * np.ones(2),
                               metadata.tel_diameter, metadata.tel_pupil)
    calls = []
    def prepare(images):
        calls.append(images.shape)
        return images
    result = compute_psf_stats(source, metadata, ee_apertures=[15,30]*u.mas,
                               peak_method="fourier_max", fwhm_method="fourier_contour",
                               fwhm_refinement_factor=8, preprocess=prepare, dtype=dtype)
    assert calls == [(2,*image.shape)]
    np.testing.assert_allclose(result[0][1].value, 0.6*result[0][0].value, rtol=2e-6)
    np.testing.assert_allclose(result[1][1].value, 0.6*result[1][0].value, rtol=2e-6)
    assert all(x.dtype == np.dtype(dtype) for x in result)
    assert result[0].shape == (2,)
    assert result[1].shape == (2,2)
    sr = compute_psf_sr(source, metadata, peak_method="fourier_max", preprocess=None, dtype=dtype)
    ee = compute_psf_ee(source, metadata, ee_apertures=[15,30]*u.mas,
                        peak_method="fourier_max", preprocess=None, dtype=dtype)
    width = compute_psf_fwhm(source, metadata, fwhm_method="fourier_contour",
                             fwhm_refinement_factor=8, preprocess=None, dtype=dtype)
    for actual, expected in zip(result, (sr,ee,width)):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(source, original)


@pytest.mark.parametrize("peak", [None, (20.25, 21.5), (20, 21)])
def test_combined_call_reuses_each_refined_image(monkeypatch, peak):
    image, metadata, _ = optical_psf()
    locations = [peak] * 2 if peak is not None else None
    original = module._refine_psf
    seen = []
    def counted(image, factor):
        seen.append((image.shape, factor))
        return original(image, factor)
    monkeypatch.setattr(module, "_refine_psf", counted)
    compute_psf_stats(np.stack([image,image]), metadata, ee_apertures=[15]*u.mas,
                      peak_method="fourier_max", fwhm_method="fourier_contour",
                      peak_locations=locations, fwhm_refinement_factor=4,
                      preprocess=None, dtype=np.float64)
    # Two science images plus one reference; no extra interpolation for FWHM.
    assert len(seen) == 3


@pytest.mark.parametrize("peak", [None, (20, 21)])
def test_fwhm_only_does_not_compute_unused_fourier_peaks(monkeypatch, peak):
    image, _, _ = optical_psf()
    def forbidden(*args, **kwargs):
        raise AssertionError("Unused peak interpolation")
    monkeypatch.setattr(module, "_refine_psf", forbidden)
    result = compute_psf_stats(
        image, pixel_scale=1*u.mas, metrics=("fwhm",), peak_method="fourier_max",
        peak_locations=peak,
    )
    assert np.isfinite(result[0].value)


def test_shared_refinement_retains_native_contour_rejection():
    y,x = np.indices((15,17))
    image = np.exp(-4*np.log(2)*(((x-1)/7)**2+((y-7)/5)**2))
    ee,width = compute_psf_stats(image, pixel_scale=1*u.mas, metrics=("ee","fwhm"),
                                 ee_apertures=[10]*u.mas, peak_method="fourier_max",
                                 fwhm_method="fourier_contour", fwhm_refinement_factor=8)
    assert np.isfinite(ee.value)
    assert np.isnan(width.value)


def test_peak_search_does_not_extend_into_padding():
    image = np.zeros((5,7))
    image[-1,-1] = 1
    result = compute_psf_ee(image, pixel_scale=1*u.mas, ee_apertures=[2]*u.mas,
                            peak_method="fourier_max", preprocess=None)
    np.testing.assert_allclose(result.value, 1)


def test_fourier_peak_supports_distinct_metadata_and_per_psf_apertures():
    image, metadata, _ = optical_psf()
    images = np.stack([image, 0.6 * image])
    factors = np.array([1.0, 1.25])
    vector_metadata = PsfMetadata(
        metadata.wavelength * factors, metadata.pixel_scale * factors,
        metadata.tel_diameter, metadata.tel_pupil,
    )
    apertures = factors[:, None] * np.array([15.0, 30.0]) * u.mas
    combined = compute_psf_stats(
        images, vector_metadata, ee_apertures=apertures,
        peak_method="fourier_max", peak_refinement_factor=8,
        fwhm_method="fourier_contour", fwhm_refinement_factor=4,
        preprocess=None, dtype=np.float64,
    )
    for index in range(2):
        selected_metadata = PsfMetadata(
            vector_metadata.wavelength[index], vector_metadata.pixel_scale[index],
            metadata.tel_diameter, metadata.tel_pupil,
        )
        individual = compute_psf_stats(
            images[index], selected_metadata, ee_apertures=apertures[index],
            peak_method="fourier_max", peak_refinement_factor=8,
            fwhm_method="fourier_contour", fwhm_refinement_factor=4,
            preprocess=None, dtype=np.float64,
        )
        for values, expected in zip(combined, individual):
            np.testing.assert_array_equal(values[index], expected)


def test_default_fourier_peak_factor_is_explicit_four():
    image, metadata, _ = optical_psf()
    options = dict(peak_method="fourier_max", ee_apertures=[15]*u.mas,
                   fwhm_method="fourier_contour", preprocess=None, dtype=np.float64)
    default = compute_psf_stats(image, metadata, **options)
    explicit = compute_psf_stats(image, metadata, peak_refinement_factor=4, **options)
    for actual, expected in zip(default, explicit):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("factor", [1, 4])
def test_supplied_detected_peak_matches_single_core_measurements(factor):
    image, metadata, _ = optical_psf()
    refined = module._refine_psf(image, factor)
    peak_yx = np.asarray(np.unravel_index(refined.argmax(), refined.shape)) / factor
    options = dict(
        ee_apertures=[100] * u.mas, peak_method="fourier_max",
        peak_refinement_factor=factor, fwhm_method="fourier_contour",
        fwhm_refinement_factor=factor, preprocess=None, dtype=np.float64,
    )
    automatic = compute_psf_stats(image, metadata, **options)
    supplied = compute_psf_stats(image, metadata, peak_locations=peak_yx, **options)
    for actual, expected in zip(supplied, automatic, strict=True):
        assert np.isfinite(actual.value)
        np.testing.assert_allclose(actual.value, expected.value, rtol=1e-12)


def test_supplied_fractional_peak_under_default_peak_method():
    image, metadata, _ = optical_psf(sampling=2, offset=(0.5, 0.5), strength=0)
    fitted_peak = module._compute_peak_locations(image[None], DEFAULT_PEAK_METHOD)[0]
    options = dict(
        ee_apertures=[100] * u.mas, fwhm_method="fourier_contour",
        preprocess=None, dtype=np.float64,
    )
    automatic = compute_psf_stats(image, metadata, **options)
    supplied = compute_psf_stats(image, metadata, peak_locations=fitted_peak, **options)
    for actual, expected in zip(supplied, automatic, strict=True):
        np.testing.assert_allclose(actual.value, expected.value, rtol=0.005)


def test_supplied_peak_accepts_pixel_quantities():
    image, _, _ = optical_psf()
    locations = np.array([20.25, 21.5])
    options = dict(pixel_scale=1 * u.mas, metrics=("fwhm",))
    pixels = compute_psf_stats(image, peak_locations=locations * u.pix, **options)
    plain = compute_psf_stats(image, peak_locations=locations, **options)
    np.testing.assert_array_equal(pixels[0], plain[0])


def test_supplied_aperture_edge_limit_applies_to_each_psf():
    y, x = np.indices((41, 41))
    locations = [(20, 20), (1, 1)]
    psfs = np.array([
        np.exp(-((y - cy)**2 + (x - cx)**2) / 18)
        for cy, cx in locations
    ])
    options = dict(pixel_scale=10 * u.mas, ee_apertures=[100] * u.mas, metrics=("ee",))
    (cube_ee,) = compute_psf_stats(psfs, peak_locations=locations, **options)
    for i, (psf, location) in enumerate(zip(psfs, locations, strict=True)):
        (single_ee,) = compute_psf_stats(psf, peak_locations=location, **options)
        np.testing.assert_array_equal(cube_ee[i, 0], single_ee)


def test_supplied_peak_rejects_negative_strehl_sample():
    _, metadata, _ = optical_psf()
    psf = np.zeros((25, 27))
    psf[12, 13] = 1
    options = dict(metrics=("sr",), peak_method="fourier_max", preprocess=None, dtype=np.float64)
    (zero,) = compute_psf_stats(psf, metadata, peak_locations=(13, 13), **options)
    assert zero.value == 0
    with pytest.raises(ValueError, match="negative Strehl intensity"):
        compute_psf_stats(psf, metadata, peak_locations=(13.5, 13), **options)


@pytest.mark.parametrize("per_psf", [False, True])
def test_supplied_peak_measures_core_with_brighter_separate_speckle(per_psf):
    _, metadata, _ = optical_psf()
    y, x = np.indices((65, 71))
    core_yx, speckle_yx = (28, 22), (36, 53)
    core = 0.004 * np.exp(-0.5 * (((y - 28) / 2.0)**2 + ((x - 22) / 2.5)**2))
    speckle = 0.008 * np.exp(-0.5 * (((y - 36) / 3.0)**2 + ((x - 53) / 4.0)**2))
    image = core + speckle
    options = dict(
        ee_apertures=[100] * u.mas, peak_method="pixel_max",
        preprocess=None, dtype=np.float64,
    )
    vector_metadata = metadata
    if per_psf:
        vector_metadata = PsfMetadata(
            metadata.wavelength * np.ones(2), metadata.pixel_scale * np.ones(2),
            metadata.tel_diameter, metadata.tel_pupil,
        )
    anchored = compute_psf_stats(
        np.stack([image, image]), vector_metadata,
        peak_locations=np.array([core_yx, speckle_yx]), **options,
    )
    expected_core = compute_psf_stats(core, metadata, **options)
    expected_speckle = compute_psf_stats(image, metadata, **options)
    for actual, core_metric, speckle_metric in zip(
        anchored, expected_core, expected_speckle, strict=True
    ):
        np.testing.assert_allclose(actual[0].value, core_metric.value, rtol=1e-8)
        np.testing.assert_allclose(actual[1].value, speckle_metric.value, rtol=1e-12)
        assert actual[0] < actual[1]


@pytest.mark.parametrize(
    ("cube", "locations", "message"),
    [
        (False, [[4, 4]], "shape"),
        (True, [4, 4], "shape"),
        (True, [[4, 4]], "shape"),
        (False, [np.nan, 4], "finite real"),
        (False, [4, np.inf], "finite real"),
        (False, [4, 4 + 1j], "finite real"),
        (False, [4, 4] * u.mas, "units equivalent to 'pix'"),
        (False, [-0.01, 4], "bounds"),
        (False, [8, 10.01], "bounds"),
    ],
)
def test_supplied_peak_validation(cube, locations, message):
    image = np.ones((9, 11))
    psfs = np.stack([image, image]) if cube else image
    with pytest.raises(ValueError, match=f"peak_locations.*{message}"):
        compute_psf_stats(
            psfs, pixel_scale=1 * u.mas, metrics=("fwhm",),
            peak_locations=locations,
        )
