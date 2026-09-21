# AO Stats

AO Stats is a Python library for measuring adaptive-optics point-spread
functions (PSFs). It calculates Strehl ratio, encircled or ensquared energy,
and full width at half maximum (FWHM) from a single PSF image or a cube of
PSFs.

The package combines NumPy image arrays with explicit Astropy quantities for
wavelength, pixel scale, telescope diameter, pupil geometry, aperture size,
and returned physical values. It can measure all three statistics together or
compute only the requested subset.

AO Stats measures PSFs; it does not generate them or prescribe a simulation,
instrument, dataset, persistence, or plotting workflow. This makes the same
measurement layer usable from standalone analysis code and larger AO systems.

## Measurements

- **Strehl ratio** using a fitted core peak or the brightest pixel, compared
  with the diffraction-limited PSF implied by the telescope pupil
- **Enclosed energy** using square (ensquared) or circular (encircled)
  apertures
- **FWHM** from the half-maximum contour, summarized by its geometric mean,
  arithmetic mean, minimum, or maximum width
- **Default preprocessing** through the built-in clip-and-normalize operation,
  with explicit opt-out or caller-provided alternatives

## Installation

To install AO Stats from a local checkout:

```bash
python -m pip install .
```

For contributor setup, use the repository-managed environment described in
[`docs/development.md`](docs/development.md).

## Quick Start

Import the supported API from the package root. This self-contained example
measures a synthetic PSF:

```python
import numpy as np
from astropy import units as u

from ao_stats import PsfMetadata, compute_psf_stats

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
```

The results are Astropy quantities. For a single PSF, `strehl` and `fwhm` are
scalars and `energy` follows the requested aperture shape. Cube inputs retain
a leading PSF dimension.

## Documentation

- [`docs/api.md`](docs/api.md): inputs, measurements, options, and result shapes
- [`docs/architecture.md`](docs/architecture.md): package structure and ownership boundaries
- [`docs/development.md`](docs/development.md): local setup and daily workflow
- [`docs/testing.md`](docs/testing.md): canonical verification commands
