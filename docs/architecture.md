# Architecture

This document defines how AO Stats turns PSF arrays and physical metadata into
unit-aware measurements, and where the package boundary ends.

## Calculation Model

Every public calculation follows the same flow:

1. Validate and shape a two-dimensional PSF or three-dimensional PSF cube.
2. Validate full `PsfMetadata`, or a pixel scale for EE/FWHM-only requests,
   along with any enclosed-energy apertures.
3. Apply the built-in clip-and-normalize preprocessing by default, or the
   caller's explicit preprocessing choice.
4. Validate the prepared physical-flux contract required by selected metrics.
5. Calculate only the requested Strehl, enclosed-energy, and FWHM values.
6. Return `float32` Astropy quantities with shapes appropriate to a single PSF
   or cube.

The combined `compute_psf_stats` function shares validation, preprocessing,
and one selected peak estimate across requested measurements. Its amplitude
feeds Strehl and its location anchors enclosed energy. Focused public functions
provide direct access to one measurement.

## Ownership

AO Stats owns:

- the neutral full `PsfMetadata` contract and pixel-scale-only measurement path;
- PSF input and physical-metadata validation required by its calculations;
- the named clip-and-normalize preprocessing operation;
- Strehl, enclosed-energy, and FWHM algorithms and their selectors; and
- calculation result units and shapes.

AO Stats does not own simulation lifecycle, dataset schemas, persistence,
plotting, interpolation, or instrument-specific policy. Callers translate
their own contracts into NumPy arrays, full `PsfMetadata` when measuring
Strehl, and aperture quantities before invoking the measurement API.

## Public API

The supported user-facing API is exported from the `ao_stats` package root.
Implementation modules may change without providing compatibility aliases.
Physical values use Astropy quantities and image arrays use NumPy.

The public surface includes:

- `PsfMetadata`;
- `compute_psf_stats` and the focused measurement functions;
- `clip_and_sum_normalize_psfs`;
- public selector constants and type aliases; and
- `__version__`.

## Numerical Contracts

Shapes, units, selector meanings, preprocessing behavior, result ordering,
dtypes, warnings, exceptions, and numerical results are part of the supported
calculation contract. Changes to these behaviors require explicit tests and
corresponding API documentation updates.
