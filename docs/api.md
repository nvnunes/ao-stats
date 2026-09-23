# Python API

AO Stats exposes its supported user-facing API from the `ao_stats` package
root. The API accepts a single PSF image or a cube, validates the associated
physical metadata, applies its default preprocessing unless explicitly
overridden, and returns the requested measurements as Astropy quantities.

## Core Inputs

PSF data is a NumPy-compatible array with one of these shapes:

- `[Ny, Nx]` for one PSF;
- `[M, Ny, Nx]` for a cube of `M` PSFs.

Strehl calculations require full `PsfMetadata`:

```python
import numpy as np
from astropy import units as u

from ao_stats import PsfMetadata

metadata = PsfMetadata(
    wavelength=1.65 * u.um,
    pixel_scale=4.0 * u.mas,
    tel_diameter=8.0 * u.m,
    tel_pupil=np.ones((32, 32)) * u.dimensionless_unscaled,
)
```

`wavelength` and `pixel_scale` may be scalar quantities shared by the input
cube or one-dimensional quantities with one value per PSF. `tel_diameter` is a
shared scalar and `tel_pupil` is a shared two-dimensional image.

## Computing Measurements

Use `compute_psf_stats` to calculate several measurements in one call:

```python
import numpy as np
from astropy import units as u

from ao_stats import compute_psf_stats

size = 65
y, x = np.indices((size, size), dtype=np.float32)
center = 0.5 * (size - 1)
psf = np.exp(
    -0.5 * (((y - center) / 4.5) ** 2 + ((x - center) / 5.0) ** 2)
)

sr, ee, fwhm = compute_psf_stats(
    psf,
    metadata,
    ee_apertures=np.array([50.0, 100.0]) * u.mas,
)
```

When `metrics` is omitted, the result is `(sr, ee, fwhm)`. Pass a sequence of
`"sr"`, `"ee"`, and `"fwhm"` to compute a subset; the returned tuple follows
the requested order. The focused `compute_psf_sr`, `compute_psf_ee`, and
`compute_psf_fwhm` functions return one measurement directly.

Enclosed energy and FWHM need only the angular `pixel_scale`. Pass it instead
of `metadata` when wavelength or pupil information is unavailable:

```python
ee, fwhm = compute_psf_stats(
    psf,
    pixel_scale=4.0 * u.mas,
    ee_apertures=np.array([50.0, 100.0]) * u.mas,
    metrics=("ee", "fwhm"),
)
```

The focused `compute_psf_ee` and `compute_psf_fwhm` functions accept the same
`pixel_scale` alternative. A scalar scale applies to every PSF; a vector must
have one value per PSF. Supplying both `metadata` and `pixel_scale` is an error.
Strehl cannot be requested through the pixel-scale-only path.

## Measurement Options

| Measurement | Options | Default | Result unit |
| --- | --- | --- | --- |
| Shared peak estimator | `gaussian_fit`, `pixel_max` | `gaussian_fit` | — |
| Strehl ratio | Uses the shared peak estimator | — | dimensionless |
| Enclosed energy geometry | `ensquared`, `encircled` | `ensquared` | dimensionless |
| FWHM summary | `geom`, `mean`, `min`, `max` | `geom` | milliarcseconds |

`ee_apertures` contains aperture diameters in angular units. Supply a
one-dimensional array to use the same apertures for every PSF, or an `[M, A]`
array to use `A` aperture sizes for each PSF in a cube.

## Preprocessing

Calculations clip negative pixels to zero and normalize every positive-flux PSF
by its pixel sum by default. Pass `preprocess=None` only when the supplied PSFs
are already prepared on their intended physical-flux scale. Prepared inputs
used for Strehl or enclosed energy must be nonnegative and have positive total
in-frame flux no greater than one; finite-field PSFs may retain flux below one.

A callable can implement a different preprocessing rule. It must accept and
return a PSF cube with the same shape and finite values, and its result must
satisfy the same prepared-input contract.

The built-in operation is also available directly as
`clip_and_sum_normalize_psfs`.

## Result Shapes

For a cube, Strehl and FWHM results have shape `[M]`, while enclosed-energy
results have shape `[M, A]`. For a single PSF, Strehl and FWHM are scalar. Its
enclosed-energy result is scalar for one aperture and has shape `[A]` for
multiple apertures.

See the [Python reference](reference/api.md) for complete signatures, public
types and selector constants, validation behavior, and function docstrings.
