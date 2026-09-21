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
        "ee_apertures",
        "peak_method",
        "fwhm_summary",
        "ee_geometry",
        "preprocess",
        "metrics",
    )
    assert tuple(inspect.signature(ao_stats.compute_psf_sr).parameters) == (
        "psfs",
        "metadata",
        "peak_method",
        "preprocess",
    )
    assert tuple(inspect.signature(ao_stats.compute_psf_ee).parameters) == (
        "psfs",
        "metadata",
        "ee_apertures",
        "peak_method",
        "ee_geometry",
        "preprocess",
    )
    assert tuple(inspect.signature(ao_stats.compute_psf_fwhm).parameters) == (
        "psfs",
        "metadata",
        "fwhm_summary",
        "preprocess",
    )


def test_version_present() -> None:
    assert ao_stats.__version__ == "0.1.0"
