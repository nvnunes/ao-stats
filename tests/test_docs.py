from __future__ import annotations

import numpy as np
from astropy import units as u

from ao_stats import PsfMetadata, compute_psf_stats


def test_readme_quick_start() -> None:
    size = 65
    y, x = np.indices((size, size), dtype=np.float32)
    center = 0.5 * (size - 1)
    psf = np.exp(
        -0.5 * (((y - center) / 4.5) ** 2 + ((x - center) / 5.0) ** 2)
    )
    pupil = np.ones((32, 32), dtype=np.float32)

    metadata = PsfMetadata(
        wavelength=1.65 * u.um,
        pixel_scale=4.0 * u.mas,
        tel_diameter=8.0 * u.m,
        tel_pupil=pupil * u.dimensionless_unscaled,
    )

    strehl, energy, fwhm = compute_psf_stats(
        psf,
        metadata,
        ee_apertures=np.array([50.0, 100.0]) * u.mas,
    )

    assert strehl.isscalar
    assert strehl.unit == u.dimensionless_unscaled
    assert energy.shape == (2,)
    assert energy.unit == u.dimensionless_unscaled
    assert fwhm.isscalar
    assert fwhm.unit == u.mas
    assert np.all(np.isfinite([strehl.value, *energy.value, fwhm.value]))
