from __future__ import annotations

import inspect

import ao_stats


def test_root_public_surface() -> None:
    expected = {
        "PsfMetadata",
        "clip_and_sum_normalize_psfs",
        "compute_psf_ee",
        "compute_psf_fwhm",
        "compute_psf_sr",
        "compute_psf_stats",
    }

    assert expected <= set(ao_stats.__all__)
    for name in expected:
        assert getattr(ao_stats, name).__module__.startswith("ao_stats")


def test_public_calculation_signatures_are_available() -> None:
    assert tuple(inspect.signature(ao_stats.compute_psf_stats).parameters) == (
        "psfs",
        "metadata",
        "pixel_scale",
        "ee_apertures",
        "peak_method",
        "peak_locations",
        "peak_refinement_factor",
        "fwhm_method",
        "fwhm_refinement_factor",
        "fwhm_summary",
        "ee_geometry",
        "ee_center",
        "preprocess",
        "metrics",
        "dtype",
    )
    assert tuple(inspect.signature(ao_stats.compute_psf_sr).parameters) == (
        "psfs",
        "metadata",
        "peak_method",
        "peak_refinement_factor",        "preprocess",
        "dtype",
    )
    assert tuple(inspect.signature(ao_stats.compute_psf_ee).parameters) == (
        "psfs",
        "metadata",
        "ee_apertures",
        "pixel_scale",
        "peak_method",
        "peak_refinement_factor",
        "ee_geometry",
        "ee_center",
        "preprocess",
        "dtype",
    )
    assert tuple(inspect.signature(ao_stats.compute_psf_fwhm).parameters) == (
        "psfs",
        "metadata",
        "pixel_scale",
        "fwhm_method",
        "fwhm_refinement_factor",
        "fwhm_summary",
        "preprocess",
        "dtype",
    )


def test_ee_center_constants_are_available() -> None:
    assert {"EE_CENTER_PEAK", "EE_CENTER_IMAGE", "EE_CENTERS", "DEFAULT_EE_CENTER"} <= set(
        ao_stats.__all__
    )
    assert ao_stats.EE_CENTERS == (ao_stats.EE_CENTER_PEAK, ao_stats.EE_CENTER_IMAGE) == ("peak", "image")
    assert ao_stats.DEFAULT_EE_CENTER == ao_stats.EE_CENTER_PEAK


def test_version_present() -> None:
    assert ao_stats.__version__ == "0.1.1"
