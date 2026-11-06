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

### Supplied Peak Positions

Pass `peak_locations=(y, x)` to `compute_psf_stats` to bypass automatic
peak selection. For a cube, supply one pair per PSF as an `[M, 2]` array.
Use plain numbers or a quantity in pixels. Coordinates are zero-based and
may be fractional within the first and last native pixel centers on each axis.

- **Strehl and FWHM** use the native intensity at an integer position.
  For a fractional position, they sample the Fourier-refined image by
  bilinear interpolation using `peak_refinement_factor`. Strehl uses the
  Fourier diffraction reference at that refinement and rejects negative
  samples. FWHM measures the enclosing contour at half the sampled intensity
  using the selected [contour method](#fwhm-sampling).
- **Enclosed energy** uses the supplied position with `ee_center="peak"`.
  With `ee_center="image"`, the aperture stays at the cutout midpoint.

## Measurement Options

| Measurement | Options | Default | Result unit |
| --- | --- | --- | --- |
| Automatic peak estimator | `gaussian_fit`, `pixel_max`, `fourier_max` | `gaussian_fit` | — |
| Peak refinement factor | Positive integer | `4` | — |
| Strehl ratio | Automatic or supplied peak | — | dimensionless |
| Enclosed energy geometry | `ensquared`, `encircled` | `ensquared` | dimensionless |
| Enclosed energy center | `peak`, `image` | `peak` | — |
| FWHM method | `contour`, `fourier_contour` | `contour` | milliarcseconds |
| FWHM refinement factor | Positive integer | `4` | — |
| FWHM summary | `geom`, `mean`, `min`, `max` | `geom` | milliarcseconds |

For automatic peak selection, `gaussian_fit` locates the brightest pixel
in each PSF and fits a Gaussian within a 5×5 patch around it, regardless of
image parity. At an image edge, the patch shifts inward to keep its full
size. Axes shorter than five pixels use all available pixels. The fitted
amplitude feeds Strehl and the fitted subpixel location centers apertures
in peak-centered EE. Strehl uses the same
estimator for the diffraction-limited reference. `pixel_max` uses the brightest
pixel's value and integer location directly.

Diffraction-reference pupil padding is snapped to an integer when the
difference is within metadata roundoff at the selected precision. Genuinely fractional padding
is truncated toward zero. This preserves the intended Fourier grid when
physical metadata describe an integer padding size.

### Enclosed Energy

In `compute_psf_ee` and `compute_psf_stats`, the default `ee_center="peak"`
uses the estimated peak or the position supplied to `compute_psf_stats`.
Use `ee_center="image"` when the source is nominally at the center of the
cutout, as in a source-centered simulation. The aperture stays at the geometric
image center, halfway between the middle pixels on even-sized axes. Motion
of the PSF core relative to the source contributes to the measurement.
SR and FWHM use their own peak and contour settings.

`ee_apertures` contains aperture diameters in angular units. Supply a
one-dimensional array to use the same apertures for every PSF, or an `[M, A]`
array to use `A` aperture sizes for each PSF in a cube.

Square apertures sample a cumulative curve up to the largest sampled aperture
that fits in the image. A cube measured with automatic peaks, shared apertures,
and shared metadata uses the smallest such limit across its PSFs. With peak
centering, supplied positions give each PSF its own limit. Larger requested
apertures use extrapolation clipped to `[0, 1]`. A curve with one sample returns
that value at every requested diameter.

Circular apertures integrate the native pixels directly at either selected
center. Peak-centered circular EE at fractional locations can differ from
older AO Stats results, which resampled the image before integration.

### Fourier Peak Estimation

For automatic peak selection, set `peak_method="fourier_max"` to measure the
maximum after padded Fourier interpolation of the intensity image.
`compute_psf_sr`, `compute_psf_ee` and `compute_psf_stats` accept
`peak_refinement_factor`:

```python
from ao_stats import compute_psf_sr

sr = compute_psf_sr(
    normalized_psfs,
    metadata,
    peak_method="fourier_max",
    peak_refinement_factor=8,
    preprocess=None,
)
```

The factor defaults to 4 and sets the Fourier grid for peak searches and
intensity sampling. Use an explicit factor of 8 for the more coarsely sampled
ELT PSFs, as in the example above. The factor must be a positive integer.
AO Stats validates it for every metric subset. A factor of one uses the native
grid. The default automatic peak estimator is `gaussian_fit`.

Peak estimation and Fourier FWHM share the padding and resampling operation
described below. AO Stats searches for the maximum within the original sample
extent and converts its location back to native pixels. With `ee_center="peak"`,
EE centers the aperture at that location. With `ee_center="image"`, EE uses
the cutout midpoint. See [Enclosed Energy](#enclosed-energy) for aperture
integration. Fourier refinement preserves interpolation undershoots and skips
repeated preprocessing.

Strehl applies the same peak estimator to a diffraction-limited reference
normalized over its complete Fourier plane. Before interpolation the reference
is centrally cropped to at most the science-image dimensions, preserving its
full-plane flux scale. Neither reference nor science crop is renormalized by
the interpolation. Callers must still supply the actual pupil, its physical
support diameter, wavelength and angular pixel scale.

The peak and FWHM factors are independent. A combined call reuses each refined
science image when peak sampling and Fourier FWHM use the same factor.
It processes one science image at a time. Refinement reduces grid-position
error for adequately sampled PSFs. It cannot recover an undersampled or missing
peak, certify that a crop contains the true maximum, or suppress noise.
Small crops and discontinuous edges can cause interpolation artifacts.
A higher factor alone does not establish accuracy.

### FWHM Sampling

`fwhm_method="contour"` measures the half-maximum contour at native sampling.
Select `"fourier_contour"` to interpolate the intensity image before measuring
the same contour geometry. Both `compute_psf_fwhm` and `compute_psf_stats`
accept `fwhm_method` and `fwhm_refinement_factor`:

```python
from ao_stats import compute_psf_fwhm

fwhm = compute_psf_fwhm(
    psf,
    pixel_scale=4.0 * u.mas,
    fwhm_method="fourier_contour",
    fwhm_refinement_factor=4,
)
```

The multiplier defaults to 4; one gives the exact native measurement. It must
be a positive integer, including when unused by `"contour"`; booleans and
floating-point values are rejected. FWHM options are validated even when
another metric subset is requested.

Fourier interpolation zero-pads each axis by half its native length on each
side, rounded up, then resamples each axis by the selected multiplier. Widths
use the original pixel scale divided by that multiplier. Interpolation does
not clip negative undershoots or repeat preprocessing. Contours must be valid
on the native image and remain within its original extent after refinement;
otherwise FWHM is `NaN`. Padding cannot recover a truncated PSF.

Refinement changes only FWHM. It leaves Strehl, enclosed energy and
`fwhm_summary` semantics unchanged. Cube inputs are refined one image at a
time. Interpolation reduces sampling error for sufficiently sampled, smooth
PSFs; it does not recover information lost to undersampling or suppress noise.

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

## Numerical Precision

Preprocessing and all measurement functions accept keyword-only `dtype`, either
`np.float32` (default) or `np.float64`. NumPy dtype and string spellings are
also accepted; other types raise `ValueError`. The choice applies to images,
physical metadata preparation, reductions and returned quantities, including
scalars. Custom preprocessing receives images in that dtype, and its validated
return is converted back to it. SciPy solvers and interpolation, Photutils
aperture integration and ContourPy contour extraction may use their own
internal precision; results are converted at those boundaries.
