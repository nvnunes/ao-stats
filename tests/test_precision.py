"""Precision at public preparation, callbacks, and physical metric boundaries."""

import numpy as np
import pytest
from astropy import units as u

from ao_stats import (
    PsfMetadata,
    clip_and_sum_normalize_psfs,
    compute_psf_stats,
    compute_psf_sr,
    compute_psf_ee,
    compute_psf_fwhm,
)


@pytest.mark.parametrize(
    "dtype", [np.float32, np.float64, "float64", np.dtype("float32")]
)
@pytest.mark.parametrize("cube", [False, True])
def test_selected_precision_reaches_metrics_and_callback(dtype, cube):
    dtype = np.dtype(dtype)
    y, x = np.mgrid[-16:17, -16:17]
    image = np.exp(-(x * x + y * y) / 8) + 1e-10
    psfs = np.stack([image, image * 1.25]) if cube else image
    metadata = PsfMetadata(
        wavelength=1.65 * u.um,
        pixel_scale=20 * u.mas,
        tel_diameter=8 * u.m,
        tel_pupil=np.ones((8, 8)) * u.one,
    )
    seen = []

    def prepare(values):
        seen.append(values.dtype)
        return clip_and_sum_normalize_psfs(values, dtype=dtype).astype(np.float64)

    combined = compute_psf_stats(
        psfs,
        metadata,
        ee_apertures=[60] * u.mas,
        peak_method="pixel_max",
        preprocess=prepare,
        dtype=dtype,
    )
    focused = [
        compute_psf_sr(psfs, metadata, peak_method="pixel_max", dtype=dtype),
        compute_psf_ee(
            psfs,
            metadata,
            ee_apertures=[60] * u.mas,
            peak_method="pixel_max",
            dtype=dtype,
        ),
        compute_psf_fwhm(psfs, metadata, dtype=dtype),
    ]
    assert seen == [dtype]
    for actual, expected in zip(combined, focused):
        assert actual.dtype == dtype
        np.testing.assert_array_equal(actual, expected)
    normalized = clip_and_sum_normalize_psfs(psfs, dtype=dtype)
    reference = np.maximum(psfs.astype(dtype), 0)
    reference /= reference.sum(axis=(-2, -1), keepdims=True, dtype=dtype)
    np.testing.assert_array_equal(normalized, reference)
    if dtype == np.dtype("float64"):
        assert np.any(normalized != normalized.astype(np.float32).astype(np.float64))


@pytest.mark.parametrize("dtype", [np.int64, np.float16, np.complex64])
def test_reject_unsupported_precision(dtype):
    with pytest.raises(ValueError, match="float32 or float64"):
        clip_and_sum_normalize_psfs(np.ones((3, 3)), dtype=dtype)
    with pytest.raises(ValueError, match="float32 or float64"):
        compute_psf_fwhm(np.ones((3, 3)), pixel_scale=1 * u.mas, dtype=dtype)


def test_gaussian_callbacks_keep_selected_precision(monkeypatch):
    import ao_stats.psf as implementation

    original = implementation.curve_fit
    observed = []

    def recording(model, coords, values, **kwargs):
        # Reproduce SciPy's callback input promotion explicitly and probe the
        # actual wrappers passed by the shared fit, including their Jacobian.
        promoted = np.asarray(coords, dtype=np.float64)
        observed.append(model(promoted, *kwargs["p0"]).dtype)
        if kwargs.get("jac") is not None:
            observed.append(kwargs["jac"](promoted, *kwargs["p0"]).dtype)
        return original(model, coords, values, **kwargs)

    monkeypatch.setattr(implementation, "curve_fit", recording)
    y, x = np.mgrid[-4:5, -4:5].astype(np.float32)
    image = np.exp(-(x * x + y * y) / 4.0)
    implementation._fit_gaussian_core(
        image, assume_centered=False, assume_circular=False
    )
    assert observed and all(dtype == np.float32 for dtype in observed)


def test_unit_conversion_uses_requested_arithmetic_and_dtype_rejection():
    from ao_stats._units import quantity_value

    value = np.float32(0.02) * u.arcsec
    converted = quantity_value(value, u.mas, label="scale", dtype=np.float64)
    assert converted == value.astype(np.float64).to_value(u.mas)
    assert converted != value.to_value(u.mas)
    from ao_stats.psf import _real_dtype

    for bad in (None, "not_a_dtype", object(), np.float16):
        with pytest.raises(ValueError, match="float32 or float64"):
            _real_dtype(bad)
