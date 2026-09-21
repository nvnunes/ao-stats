# AO Stats

AO Stats measures adaptive-optics point-spread functions (PSFs). Given a PSF
image or cube and its physical metadata, it calculates:

- Strehl ratio from a fitted core peak or the brightest pixel;
- enclosed energy in square or circular apertures; and
- contour-derived full width at half maximum (FWHM).

Inputs use NumPy arrays and Astropy quantities. Results are returned as
Astropy quantities with scalar or array shapes that follow the input PSFs and
requested apertures.

AO Stats is independent of any particular simulator or instrument. It can be
used directly in analysis code or as the common measurement layer for other AO
software.

## Start Here

- [Python API](api.md) explains how to supply PSFs and metadata, select
  measurements, and interpret results.
- [Python Reference](reference/api.md) provides generated signatures and
  docstrings for the public package API.
- [Architecture](architecture.md) defines calculation flow, package scope, and
  ownership boundaries.
- [Development](development.md) and [Testing](testing.md) cover contributor
  setup and verification.
