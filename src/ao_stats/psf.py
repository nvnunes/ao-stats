"""PSF-statistics algorithms and public contracts."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import contourpy
import numpy as np
from astropy import units as u
from photutils.profiles import CurveOfGrowth
from scipy.interpolate import interp1d
from scipy.ndimage import shift
from scipy.optimize import OptimizeWarning, curve_fit

from ._units import quantity_value

StatValue: TypeAlias = u.Quantity
PsfMetricName: TypeAlias = Literal["sr", "ee", "fwhm"]
PsfPreprocessor: TypeAlias = Callable[[np.ndarray], np.ndarray]
PsfPreprocessOption: TypeAlias = PsfPreprocessor | Literal["default"] | None

PEAK_METHOD_GAUSSIAN_FIT = "gaussian_fit"
PEAK_METHOD_PIXEL_MAX = "pixel_max"
STATS_FWHM_SUMMARY_GEOM = "geom"
STATS_FWHM_SUMMARY_MEAN = "mean"
STATS_FWHM_SUMMARY_MAX = "max"
STATS_FWHM_SUMMARY_MIN = "min"

PEAK_METHODS = (
    PEAK_METHOD_GAUSSIAN_FIT,
    PEAK_METHOD_PIXEL_MAX,
)
STATS_FWHM_SUMMARIES = (
    STATS_FWHM_SUMMARY_GEOM,
    STATS_FWHM_SUMMARY_MEAN,
    STATS_FWHM_SUMMARY_MAX,
    STATS_FWHM_SUMMARY_MIN,
)
DEFAULT_PEAK_METHOD = PEAK_METHOD_GAUSSIAN_FIT
DEFAULT_FWHM_SUMMARY = STATS_FWHM_SUMMARY_GEOM

EE_GEOMETRY_ENSQUARED = "ensquared"
EE_GEOMETRY_ENCIRCLED = "encircled"
EE_GEOMETRIES = (EE_GEOMETRY_ENSQUARED, EE_GEOMETRY_ENCIRCLED)
DEFAULT_EE_GEOMETRY = EE_GEOMETRY_ENSQUARED

STATS_METRIC_SR = "sr"
STATS_METRIC_EE = "ee"
STATS_METRIC_FWHM = "fwhm"
STATS_METRICS = (STATS_METRIC_SR, STATS_METRIC_EE, STATS_METRIC_FWHM)


# Data structures


@dataclass(frozen=True)
class PsfMetadata:
    """Physical metadata needed to measure PSF-derived statistics.

    ``wavelength`` and ``pixel_scale`` may be scalar values shared by
    every PSF in a cube, or one-dimensional arrays with length matching the PSF
    cube length. ``tel_diameter`` is scalar-only, and ``tel_pupil`` is a
    shared two-dimensional pupil image.

    Attributes:
        wavelength: PSF wavelength quantity convertible to microns, scalar or
            per-PSF vector.
        pixel_scale: PSF pixel-scale quantity convertible to
            milliarcseconds, scalar or per-PSF vector.
        tel_diameter: Telescope-diameter quantity convertible to meters and
            shared by all PSFs.
        tel_pupil: Dimensionless two-dimensional telescope pupil image shared
            by all PSFs.
    """

    wavelength: u.Quantity
    pixel_scale: u.Quantity
    tel_diameter: u.Quantity
    tel_pupil: u.Quantity


# Input validation and shaping


@dataclass(frozen=True)
class _PreparedPsfMetadata:
    wavelength: float | np.ndarray | None
    pixel_scale: float | np.ndarray
    tel_diameter: float | None
    tel_pupil: np.ndarray | None
    has_per_psf_values: bool


def _prepare_psf_cube(psfs: np.ndarray) -> tuple[np.ndarray, bool]:
    """Validate a PSF image or cube and return a 3D float32 cube."""
    psf_cube = np.asarray(psfs, dtype=np.float32)
    scalar_output = False
    if psf_cube.ndim == 2:
        psf_cube = psf_cube[None, :, :]
        scalar_output = True
    elif psf_cube.ndim != 3:
        raise ValueError(
            f"PSFs must be [Ny, Nx] or [M, Ny, Nx], got shape {psf_cube.shape}"
        )
    if any(size == 0 for size in psf_cube.shape):
        raise ValueError(
            "PSFs must have non-empty cube and spatial dimensions, "
            f"got shape {psf_cube.shape}."
        )
    if not np.all(np.isfinite(psf_cube)):
        raise ValueError("PSFs contain non-finite values.")
    return psf_cube, scalar_output


def _prepare_shared_or_per_psf_positive_vector(
    value: Any,
    *,
    unit: u.UnitBase,
    label: str,
    num_psfs: int,
) -> tuple[float | np.ndarray, bool]:
    values = quantity_value(value, unit, label=label, dtype=np.float32)
    if values.ndim == 0:
        scalar = float(values)
        if not np.isfinite(scalar) or scalar <= 0.0:
            raise ValueError(f"{label} must be finite and > 0.")
        return scalar, False
    if values.ndim != 1:
        raise ValueError(f"{label} must be a scalar or 1D per-PSF vector.")
    if values.shape[0] != num_psfs:
        raise ValueError(
            f"{label} per-PSF vector length must match the PSF cube length {num_psfs}."
        )
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError(f"{label} must contain only finite values > 0.")
    return values.astype(np.float32, copy=False), True


def _prepare_psf_metadata(metadata: PsfMetadata, num_psfs: int) -> _PreparedPsfMetadata:
    if not isinstance(metadata, PsfMetadata):
        raise TypeError("metadata must be a PsfMetadata instance.")

    wavelength, wavelength_is_per_psf = _prepare_shared_or_per_psf_positive_vector(
        metadata.wavelength,
        unit=u.um,
        label="metadata.wavelength",
        num_psfs=num_psfs,
    )
    pixel_scale, pixel_scale_is_per_psf = _prepare_shared_or_per_psf_positive_vector(
        metadata.pixel_scale,
        unit=u.mas,
        label="metadata.pixel_scale",
        num_psfs=num_psfs,
    )

    tel_diameter = quantity_value(
        metadata.tel_diameter, u.m, label="metadata.tel_diameter", dtype=np.float32
    )
    if tel_diameter.ndim != 0:
        raise ValueError("metadata.tel_diameter must be a scalar.")
    tel_diameter = float(tel_diameter)
    if not np.isfinite(tel_diameter) or tel_diameter <= 0.0:
        raise ValueError("metadata.tel_diameter must be finite and > 0.")

    tel_pupil = quantity_value(
        metadata.tel_pupil,
        u.dimensionless_unscaled,
        label="metadata.tel_pupil",
        dtype=np.float32,
    )
    if tel_pupil.ndim != 2:
        raise ValueError("metadata.tel_pupil must be 2D [Ny, Nx].")
    if any(size == 0 for size in tel_pupil.shape):
        raise ValueError("metadata.tel_pupil must have non-empty spatial dimensions.")
    if not np.all(np.isfinite(tel_pupil)):
        raise ValueError("metadata.tel_pupil must contain only finite values.")
    pupil_power = np.sum(
        np.abs(np.asarray(tel_pupil, dtype=np.float64)) ** 2,
        dtype=np.float64,
    )
    if not np.isfinite(pupil_power) or pupil_power <= 0.0:
        raise ValueError("metadata.tel_pupil must have finite positive power.")

    return _PreparedPsfMetadata(
        wavelength=wavelength,
        pixel_scale=pixel_scale,
        tel_diameter=tel_diameter,
        tel_pupil=tel_pupil,
        has_per_psf_values=wavelength_is_per_psf or pixel_scale_is_per_psf,
    )


def _prepare_pixel_scale_metadata(
    pixel_scale: u.Quantity, num_psfs: int
) -> _PreparedPsfMetadata:
    scale, per_psf = _prepare_shared_or_per_psf_positive_vector(
        pixel_scale,
        unit=u.mas,
        label="pixel_scale",
        num_psfs=num_psfs,
    )
    return _PreparedPsfMetadata(
        wavelength=None,
        pixel_scale=scale,
        tel_diameter=None,
        tel_pupil=None,
        has_per_psf_values=per_psf,
    )


def _prepare_ee_apertures(ee_apertures: Any, num_psfs: int) -> tuple[np.ndarray, bool]:
    apertures = quantity_value(
        ee_apertures, u.mas, label="ee_apertures", dtype=np.float32
    )
    if apertures.ndim == 1:
        if apertures.size == 0:
            raise ValueError("ee_apertures must be non-empty.")
        if not np.all(np.isfinite(apertures)) or np.any(apertures <= 0.0):
            raise ValueError("ee_apertures must contain only finite values > 0.")
        return apertures, False
    if apertures.ndim == 2:
        if apertures.shape[0] != num_psfs:
            raise ValueError(
                f"ee_apertures per-PSF leading dimension must match the PSF cube length {num_psfs}."
            )
        if apertures.shape[1] == 0:
            raise ValueError("ee_apertures must include at least one aperture per PSF.")
        if not np.all(np.isfinite(apertures)) or np.any(apertures <= 0.0):
            raise ValueError("ee_apertures must contain only finite values > 0.")
        return apertures, True
    raise ValueError(
        "ee_apertures must be a 1D aperture vector or 2D per-PSF aperture array."
    )


def _require_peak_method(peak_method: str) -> str:
    value = str(peak_method).strip()
    if value not in PEAK_METHODS:
        raise ValueError(f"peak_method must be one of: {', '.join(PEAK_METHODS)}.")
    return value


def _require_fwhm_summary(fwhm_summary: str) -> str:
    value = str(fwhm_summary).strip()
    if value not in STATS_FWHM_SUMMARIES:
        raise ValueError(
            f"fwhm_summary must be one of: {', '.join(STATS_FWHM_SUMMARIES)}."
        )
    return value


def _prepare_metrics(metrics: Sequence[str] | None) -> tuple[str, ...]:
    if metrics is None:
        return STATS_METRICS
    if isinstance(metrics, str):
        raise TypeError("metrics must be a sequence of metric names, not a string.")
    values = tuple(str(metric).strip() for metric in metrics)
    if not values:
        raise ValueError("metrics must include at least one metric name.")
    invalid = sorted(set(values) - set(STATS_METRICS))
    if invalid:
        raise ValueError(f"metrics contains unsupported names: {', '.join(invalid)}.")
    if len(set(values)) != len(values):
        raise ValueError("metrics must not contain duplicate names.")
    return values


# Strehl helpers


def _axis_indices_and_coords(length: int, nbox: int) -> tuple[np.ndarray, np.ndarray]:
    """Return centered patch indices and coordinates for one PSF axis."""
    nbox = min(int(nbox), int(length))
    if (length % 2) == 0:
        size = nbox + 1 if (nbox % 2) == 1 else nbox
    else:
        size = nbox + 1 if (nbox % 2) == 0 else nbox
    size = min(size, int(length))

    center = int(length // 2)
    half = int(size // 2)
    if (length % 2) == 1:
        start = center - half
        stop = center + half + 1
        idx = np.arange(start, stop, dtype=np.int64)
        coord = idx - center
    else:
        start = center - half
        stop = start + size
        idx = np.arange(start, stop, dtype=np.int64)
        coord = (idx - center) + 0.5
    return idx, np.asarray(coord, dtype=float)


def _extract_core_patch(
    psf: np.ndarray, nbox: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract a parity-aware core patch centered on the optical center."""
    y_idx, y_coord = _axis_indices_and_coords(int(psf.shape[0]), nbox)
    x_idx, x_coord = _axis_indices_and_coords(int(psf.shape[1]), nbox)
    patch = psf[np.ix_(y_idx, x_idx)]
    grid_x, grid_y = np.meshgrid(x_coord, y_coord)
    return patch, grid_x, grid_y


def _gaussian2d(
    coords: tuple[np.ndarray, np.ndarray],
    amplitude: float,
    sigma_x: float,
    sigma_y: float,
    x0: float,
    y0: float,
    theta: float,
) -> np.ndarray:
    """Evaluate a rotated 2D Gaussian."""
    x, y = coords
    xr = x - x0
    yr = y - y0
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    x_rot = cos_theta * xr + sin_theta * yr
    y_rot = -sin_theta * xr + cos_theta * yr
    return amplitude * np.exp(-0.5 * ((x_rot / sigma_x) ** 2 + (y_rot / sigma_y) ** 2))


def _fit_gaussian_core(
    psf: np.ndarray,
    *,
    assume_centered: bool = False,
    assume_circular: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a core Gaussian and return parameters in full-image coordinates.

    Free-center fits use a 5x5 patch about the brightest pixel, shifted inward
    at image edges and reduced only when an image axis is shorter than five.
    Fixed-center fits retain the parity-aware optical-center patch.
    """
    if assume_centered:
        patch, grid_x, grid_y = _extract_core_patch(psf, 4)
        y_center, x_center = 0.5 * (np.asarray(psf.shape) - 1)
    else:
        y_center, x_center = np.unravel_index(np.argmax(psf), psf.shape)
        height, width = min(5, psf.shape[0]), min(5, psf.shape[1])
        y_start = min(max(y_center - 2, 0), psf.shape[0] - height)
        x_start = min(max(x_center - 2, 0), psf.shape[1] - width)
        patch = psf[y_start : y_start + height, x_start : x_start + width]
        grid_x, grid_y = np.meshgrid(
            np.arange(x_start, x_start + width) - x_center,
            np.arange(y_start, y_start + height) - y_center,
        )
    coords = np.vstack([grid_x.ravel(), grid_y.ravel()])
    values = patch.ravel()

    amplitude0 = float(patch.max())
    sigma_x0 = (
        max(1.0, float(np.std(grid_x[patch == patch.max()])))
        if np.any(patch == patch.max())
        else 1.0
    )
    sigma_y0 = (
        max(1.0, float(np.std(grid_y[patch == patch.max()])))
        if np.any(patch == patch.max())
        else 1.0
    )
    p0 = [amplitude0, sigma_x0, sigma_y0, 0.0, 0.0, 0.0]

    x_min, x_max = float(grid_x.min()), float(grid_x.max())
    y_min, y_max = float(grid_y.min()), float(grid_y.max())
    if assume_centered:
        x_min = y_min = -1e-6
        x_max = y_max = 1e-6
    theta_min = -np.pi / 2
    theta_max = np.pi / 2
    if assume_circular:
        theta_min = -1e-6
        theta_max = 1e-6

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        popt, pcov = curve_fit(
            _gaussian2d,
            coords,
            values,
            p0=p0,
            bounds=(
                [0.0, 0.1, 0.1, x_min, y_min, theta_min],
                [np.inf, 100.0, 100.0, x_max, y_max, theta_max],
            ),
        )

    amplitude_fit, sigma_x_fit, sigma_y_fit, x0_fit, y0_fit, theta_fit = popt
    return (
        np.asarray(
            [
                amplitude_fit,
                sigma_x_fit,
                sigma_y_fit,
                y_center + y0_fit,
                x_center + x0_fit,
                theta_fit,
            ],
            dtype=float,
        ),
        np.asarray(pcov, dtype=float),
    )


def _diffraction_padding(length: int, sampling: float) -> int:
    """Truncate padding after correcting near-integer metadata roundoff."""
    padding = (sampling - 1.0) * length / 2.0
    nearest = round(padding)
    # Sampling combines three float32 metadata values. Allow their roundoff
    # with a margin, scaled by the half-size of the Fourier grid.
    tolerance = 4 * np.finfo(np.float32).eps * max(1.0, sampling * length / 2.0)
    if abs(padding - nearest) <= tolerance:
        padding = nearest
    return int(padding)


def _get_diffraction_limited_psf(
    tel_diameter: float,
    tel_pupil: np.ndarray,
    wavelength: float,
    pixel_scale: float,
    *,
    center_in_one_pix: bool,
) -> np.ndarray:
    """Compute the diffraction-limited PSF from the pupil."""
    wavelength_m = float(wavelength) * 1e-6
    pixel_scale_rad = pixel_scale / 1000.0 / 3600.0 / 180.0 * np.pi
    sampling = wavelength_m / float(tel_diameter) / pixel_scale_rad
    if not np.isfinite(sampling) or sampling < 1.0:
        raise ValueError(
            "Diffraction-limited PSF sampling must be finite and >= 1 pixel "
            f"per wavelength/diameter, got {sampling:.6g}."
        )

    pupil = np.abs(np.asarray(tel_pupil, dtype=float))
    nx, ny = pupil.shape
    pad_x = _diffraction_padding(nx, sampling)
    pad_y = _diffraction_padding(ny, sampling)
    pupil_pad = np.pad(pupil, [(pad_x, pad_x), (pad_y, pad_y)])

    field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil_pad)))
    psf = np.abs(field) ** 2

    n_x, n_y = psf.shape
    if (n_x % 2) == 0 and not center_in_one_pix:
        u_1d = np.fft.fftshift(np.fft.fftfreq(n_x))
        v_1d = np.fft.fftshift(np.fft.fftfreq(n_y))
        u_2d, v_2d = np.meshgrid(u_1d, v_1d, indexing="ij")
        otf = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(psf)))
        otf *= np.exp(1j * np.pi * (u_2d + v_2d))
        psf = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(otf))))

    psf = np.clip(psf, 0.0, None)
    total = float(psf.sum(dtype=np.float64))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(
            "The telescope pupil did not produce a positive reference PSF."
        )
    psf /= total
    return psf.astype(np.float32, copy=False)


def _compute_strehl_gaussian_fit(
    psfs: np.ndarray,
    pixel_scale: float,
    wavelength: float,
    tel_diameter: float,
    tel_pupil: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Strehl via fitted core amplitude."""
    peak_fits = np.asarray(
        [
            _fit_gaussian_core(np.asarray(psf, dtype=float), assume_centered=False)[0]
            for psf in np.asarray(psfs, dtype=float)
        ],
        dtype=float,
    )
    peak_amplitude = peak_fits[:, 0]
    peak_locations_yx = peak_fits[:, [3, 4]].astype(np.float32)

    psf_dl = _get_diffraction_limited_psf(
        tel_diameter,
        tel_pupil,
        wavelength,
        pixel_scale,
        center_in_one_pix=False,
    )
    peak_amplitude_dl = float(_fit_gaussian_core(psf_dl, assume_centered=False)[0][0])
    sr = peak_amplitude / peak_amplitude_dl
    return np.asarray(sr, dtype=np.float32), peak_locations_yx


def _compute_strehl_pixel_max(
    psfs: np.ndarray,
    pixel_scale: float,
    wavelength: float,
    tel_diameter: float,
    tel_pupil: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Strehl via native peak pixel."""
    psfs = np.asarray(psfs, dtype=np.float32)
    num_sci = int(psfs.shape[0])
    peak_idx = np.argmax(psfs.reshape(num_sci, -1), axis=1)
    peak_amplitude = psfs.reshape(num_sci, -1)[np.arange(num_sci), peak_idx]
    peak_y, peak_x = np.unravel_index(peak_idx, psfs.shape[1:])
    peak_locations_yx = np.stack((peak_y, peak_x), axis=1).astype(np.float32)

    psf_dl = _get_diffraction_limited_psf(
        tel_diameter,
        tel_pupil,
        wavelength,
        pixel_scale,
        center_in_one_pix=True,
    )
    peak_amplitude_dl = float(psf_dl.max())
    sr = peak_amplitude / peak_amplitude_dl
    return np.asarray(sr, dtype=np.float32), peak_locations_yx


def _compute_strehl(
    psfs: np.ndarray,
    peak_method: str,
    pixel_scale: float,
    wavelength: float,
    tel_diameter: float,
    tel_pupil: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch to the selected Strehl implementation."""
    if peak_method == PEAK_METHOD_GAUSSIAN_FIT:
        return _compute_strehl_gaussian_fit(
            psfs, pixel_scale, wavelength, tel_diameter, tel_pupil
        )
    if peak_method == PEAK_METHOD_PIXEL_MAX:
        return _compute_strehl_pixel_max(
            psfs, pixel_scale, wavelength, tel_diameter, tel_pupil
        )
    raise ValueError(f"Unsupported peak method: {peak_method}")


def _compute_peak_locations(psfs: np.ndarray, peak_method: str) -> np.ndarray:
    """Return the peak-anchor locations used by the selected SR method."""
    psfs = np.asarray(psfs, dtype=np.float32)
    num_sci = int(psfs.shape[0])
    if peak_method == PEAK_METHOD_GAUSSIAN_FIT:
        peak_fits = np.asarray(
            [
                _fit_gaussian_core(np.asarray(psf, dtype=float), assume_centered=False)[
                    0
                ]
                for psf in np.asarray(psfs, dtype=float)
            ],
            dtype=float,
        )
        return peak_fits[:, [3, 4]].astype(np.float32)
    if peak_method == PEAK_METHOD_PIXEL_MAX:
        peak_idx = np.argmax(psfs.reshape(num_sci, -1), axis=1)
        peak_y, peak_x = np.unravel_index(peak_idx, psfs.shape[1:])
        return np.stack((peak_y, peak_x), axis=1).astype(np.float32)
    raise ValueError(f"Unsupported peak method: {peak_method}")


# Enclosed Energy helpers


def _anchor_psfs_for_ee(
    psfs: np.ndarray,
    peak_locations_yx: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return PSFs plus integer peak anchors for enclosed energy."""
    psfs = np.asarray(psfs, dtype=np.float32)
    num_sci, _ny, nx = psfs.shape

    if peak_locations_yx is not None:
        peak_locations_yx = np.asarray(peak_locations_yx)
        if np.issubdtype(peak_locations_yx.dtype, np.integer):
            peak_y = peak_locations_yx[:, 0].astype(np.int64, copy=False)
            peak_x = peak_locations_yx[:, 1].astype(np.int64, copy=False)
        elif np.issubdtype(peak_locations_yx.dtype, np.floating):
            eps = 1e-12
            anchor_yx = np.floor(peak_locations_yx + 0.5 - eps).astype(np.int64)
            shift_yx = anchor_yx - peak_locations_yx
            shifted_psfs = np.empty_like(psfs)
            for i in range(num_sci):
                shifted_psfs[i] = shift(
                    psfs[i],
                    shift=tuple(shift_yx[i]),
                    order=3,
                    mode="constant",
                    cval=0.0,
                    prefilter=True,
                )
            psfs = np.clip(shifted_psfs, 0.0, None)
            peak_y = anchor_yx[:, 0]
            peak_x = anchor_yx[:, 1]
        else:
            raise TypeError("peak_locations_yx must have an integer or floating dtype")
    else:
        peak_idx = np.argmax(psfs.reshape(num_sci, -1), axis=1)
        peak_y, peak_x = np.divmod(peak_idx, nx)

    return psfs, peak_y, peak_x


def _measure_peak_centered_ensquared_energy_curves(
    psfs: np.ndarray,
    peak_y: np.ndarray,
    peak_x: np.ndarray,
    max_radius: int,
    pixel_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure cumulative odd-sized square-box EE curves about integer anchors."""
    num_sci = psfs.shape[0]
    radii = np.arange(max_radius + 1, dtype=np.int64)
    ee_curves = np.zeros((num_sci, radii.size), dtype=psfs.dtype)
    idx = np.arange(num_sci, dtype=np.int64)
    unique_radii = (
        np.arange(1, ee_curves.shape[1] * 2, 2, dtype=float) * pixel_scale * 0.5
    )

    integral = np.pad(
        psfs.cumsum(axis=1).cumsum(axis=2),
        ((0, 0), (1, 0), (1, 0)),
        mode="constant",
    )

    def _rect_sum(
        idx: np.ndarray, y0: np.ndarray, y1: np.ndarray, x0: np.ndarray, x1: np.ndarray
    ) -> np.ndarray:
        return (
            integral[idx, y1, x1]
            - integral[idx, y0, x1]
            - integral[idx, y1, x0]
            + integral[idx, y0, x0]
        )

    for j, radius in enumerate(radii):
        y0 = peak_y - radius
        y1 = peak_y + radius + 1
        x0 = peak_x - radius
        x1 = peak_x + radius + 1
        ee_curves[:, j] = _rect_sum(idx, y0, y1, x0, x1)

    return ee_curves, unique_radii


def _measure_peak_centered_encircled_energy_cog(
    psfs: np.ndarray,
    peak_y: np.ndarray,
    peak_x: np.ndarray,
    ee_radii: np.ndarray,
    pixel_scale: float,
) -> np.ndarray:
    """Measure circular-aperture EE using Photutils curve-of-growth apertures."""
    radii_pix = np.asarray(ee_radii, dtype=float) / float(pixel_scale)
    if np.any(radii_pix <= 0.0):
        raise ValueError("Encircled EE apertures must have positive radii.")
    cog_radii_pix = radii_pix
    if radii_pix.size == 1:
        cog_radii_pix = np.array([radii_pix[0], radii_pix[0] + 1.0], dtype=float)
    ee = np.empty((psfs.shape[0], radii_pix.size), dtype=np.float32)
    for i, psf in enumerate(psfs):
        curve = CurveOfGrowth(
            psf, (float(peak_x[i]), float(peak_y[i])), cog_radii_pix, method="exact"
        )
        ee[i, :] = np.asarray(curve.profile[: radii_pix.size], dtype=np.float32)
    return ee


def _interpolate_ee_at_radii(
    ee_curves: np.ndarray,
    curve_radii_mas: np.ndarray,
    ee_radii: np.ndarray,
) -> np.ndarray:
    """Interpolate cumulative EE curves onto requested physical radii."""
    curve_radii_mas = np.asarray(curve_radii_mas, dtype=float)
    ee_at_radius = np.empty((ee_curves.shape[0], ee_radii.size), dtype=float)
    for i, values in enumerate(ee_curves):
        if curve_radii_mas.size == 1:
            interpolated = np.full(ee_radii.shape, float(values[0]), dtype=float)
        else:
            interp_kind = "cubic" if curve_radii_mas.size >= 4 else "linear"
            interpolated = interp1d(
                curve_radii_mas,
                values,
                kind=interp_kind,
                bounds_error=False,
                fill_value="extrapolate",
            )(ee_radii)
        ee_at_radius[i, :] = np.clip(interpolated, 0.0, 1.0)
    return ee_at_radius


def _normalize_ee_geometry(ee_geometry: str) -> str:
    """Validate and return the requested EE aperture geometry."""
    value = str(ee_geometry).strip()
    if value not in EE_GEOMETRIES:
        raise ValueError(f"ee_geometry must be one of: {', '.join(EE_GEOMETRIES)}.")
    return value


def _compute_enclosed_energy(
    psfs: np.ndarray,
    ee_apertures: np.ndarray,
    pixel_scale: float,
    peak_locations_yx: np.ndarray | None,
    *,
    ee_geometry: str = EE_GEOMETRY_ENSQUARED,
) -> np.ndarray:
    """Compute EE for the requested aperture geometry."""
    ee_geometry = _normalize_ee_geometry(ee_geometry)
    ee_apertures = np.asarray(ee_apertures, dtype=float)
    ee_radii = ee_apertures / 2.0

    psfs, peak_y, peak_x = _anchor_psfs_for_ee(psfs, peak_locations_yx)

    if ee_geometry == EE_GEOMETRY_ENSQUARED:
        _, ny, nx = psfs.shape

        # Use only aperture radii that fit around every peak anchor in the cube.
        # This keeps all PSFs on the same valid peak-centered grid.
        edge_limited_radius = int(
            np.min(np.stack([peak_y, ny - 1 - peak_y, peak_x, nx - 1 - peak_x]))
        )

        # Square aperture index r has physical half-width (r + 0.5) pixels.
        # So, r=0 corresponds to the peak pixel alone, r=1 includes the 8 surrounding pixels, etc.
        required_radius = int(np.ceil(ee_radii.max() / pixel_scale - 0.5))

        # Compute beyond the requested radius so interpolation is not pinned to
        # the final sampled point of the cumulative EE curve.
        max_radius_factor = 2.0
        max_radius = min(
            edge_limited_radius,
            int(np.ceil(max(0, required_radius) * max_radius_factor)),
        )
        ee_curves, curve_radii_mas = _measure_peak_centered_ensquared_energy_curves(
            psfs,
            peak_y,
            peak_x,
            max_radius,
            pixel_scale,
        )
        ee_at_radius = _interpolate_ee_at_radii(ee_curves, curve_radii_mas, ee_radii)
    else:
        ee_at_radius = _measure_peak_centered_encircled_energy_cog(
            psfs,
            peak_y,
            peak_x,
            ee_radii,
            pixel_scale,
        )

    return np.asarray(ee_at_radius, dtype=np.float32)


# FWHM helpers


def _find_contours(psf: np.ndarray, level: float) -> list[np.ndarray]:
    """Return contour vertices in `[N, 2]` arrays with `(x, y)` columns."""
    generator = contourpy.contour_generator(z=np.asarray(psf, dtype=float))
    return [np.asarray(line, dtype=float) for line in generator.lines(float(level))]


def _measure_contour_fwhms(
    psfs: np.ndarray, pixel_scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """Measure contour-derived minimum and maximum FWHM vectors."""
    psfs = np.asarray(psfs, dtype=np.float32)
    num_sci, ny, nx = psfs.shape
    fwhm_min = np.full((num_sci,), np.nan, dtype=np.float32)
    fwhm_max = np.full((num_sci,), np.nan, dtype=np.float32)
    edge_eps = 1e-6

    for i, psf in enumerate(psfs):
        # Non-finite PSFs do not define a usable half-max contour.
        if not np.all(np.isfinite(psf)):
            continue

        max_val = float(np.max(psf))
        min_val = float(np.min(psf))
        half_max = 0.5 * max_val
        # Non-positive peaks have no scientifically meaningful FWHM threshold.
        if max_val <= 0.0:
            continue
        # If the image never drops below half max, no contour crossing exists.
        if min_val >= half_max:
            continue

        contours = _find_contours(psf, half_max)
        # No extracted contour means the half-max geometry is unrecoverable.
        if not contours:
            continue
        contour_points = max(contours, key=len)
        # At least three vertices are required to define a 2D contour shape.
        if len(contour_points) < 3:
            continue

        x_coords = contour_points[:, 0]
        y_coords = contour_points[:, 1]
        # Edge-touching contours are treated as truncated and therefore invalid.
        if x_coords.min() <= edge_eps or x_coords.max() >= (nx - 1) - edge_eps:
            continue
        if y_coords.min() <= edge_eps or y_coords.max() >= (ny - 1) - edge_eps:
            continue

        x_span = x_coords.max() - x_coords.min()
        y_span = y_coords.max() - y_coords.min()
        # Collapsed spans indicate degenerate contour geometry on the native grid.
        if x_span <= edge_eps or y_span <= edge_eps:
            continue
        rounded_points = np.round(contour_points, decimals=6)
        # Too few distinct vertices after rounding means the contour is numerically degenerate.
        if np.unique(rounded_points, axis=0).shape[0] < 3:
            continue

        contour_center = 0.5 * np.array(
            [
                x_coords.max() + x_coords.min(),
                y_coords.max() + y_coords.min(),
            ],
            dtype=float,
        )
        radial_distances = (
            np.hypot(
                x_coords - contour_center[0],
                y_coords - contour_center[1],
            )
            * pixel_scale
        )
        radial_max = float(radial_distances.max())
        radial_min = float(radial_distances.min())
        # Non-positive radial widths indicate collapsed contour-derived FWHM.
        if radial_max <= 0.0 or radial_min <= 0.0:
            continue

        fwhm_max[i] = np.float32(2.0 * radial_max)
        fwhm_min[i] = np.float32(2.0 * radial_min)

    return fwhm_min, fwhm_max


def _compute_fwhm_summary(
    fwhm_summary: str, fwhm_min: np.ndarray, fwhm_max: np.ndarray
) -> np.ndarray:
    """Select one FWHM summary vector from contour-derived minimum/maximum widths."""
    if fwhm_summary == STATS_FWHM_SUMMARY_GEOM:
        return np.sqrt(fwhm_max * fwhm_min, dtype=np.float32)
    if fwhm_summary == STATS_FWHM_SUMMARY_MEAN:
        return 0.5 * (fwhm_max + fwhm_min)
    if fwhm_summary == STATS_FWHM_SUMMARY_MAX:
        return fwhm_max
    if fwhm_summary == STATS_FWHM_SUMMARY_MIN:
        return fwhm_min
    raise ValueError(f"Unsupported FWHM summary: {fwhm_summary}")


# Public entrypoints


def clip_and_sum_normalize_psfs(psfs: np.ndarray) -> np.ndarray:
    """Clip negative PSF pixels and normalize each positive-flux PSF.

    Negative pixels are clipped to zero. Each PSF with positive total pixel
    flux is then divided by its pixel sum.

    Args:
        psfs: PSF image ``[Ny, Nx]`` or PSF cube ``[M, Ny, Nx]``.

    Returns:
        A `float32` array with the same shape as ``psfs``. A PSF with no
        positive flux remains zero.
    """
    psf_cube = np.maximum(np.asarray(psfs, dtype=np.float32), 0.0)
    denom = np.sum(psf_cube, axis=(-2, -1), dtype=np.float32)
    positive = denom > 0.0
    if np.any(positive):
        psf_cube = psf_cube.copy()
        psf_cube[positive] /= denom[positive, np.newaxis, np.newaxis]
    return psf_cube


def _resolve_preprocess(preprocess: PsfPreprocessOption) -> PsfPreprocessor | None:
    if preprocess == "default":
        return clip_and_sum_normalize_psfs
    if isinstance(preprocess, str) or not (preprocess is None or callable(preprocess)):
        raise ValueError("preprocess must be None, 'default', or a callable.")
    return preprocess


def _apply_preprocess(
    psf_cube: np.ndarray, preprocess: PsfPreprocessOption
) -> np.ndarray:
    preprocess = _resolve_preprocess(preprocess)
    if preprocess is None:
        return psf_cube
    prepared = np.asarray(preprocess(psf_cube), dtype=np.float32)
    if prepared.shape != psf_cube.shape:
        raise ValueError(
            f"preprocess must return a PSF cube with shape {psf_cube.shape}, got {prepared.shape}"
        )
    if not np.all(np.isfinite(prepared)):
        raise ValueError("preprocess returned PSFs containing non-finite values.")
    return prepared


def _validate_flux_inputs(psf_cube: np.ndarray, metrics: tuple[str, ...]) -> None:
    """Validate prepared PSFs used by flux-dependent measurements."""
    if not ({STATS_METRIC_SR, STATS_METRIC_EE} & set(metrics)):
        return
    if np.any(psf_cube < 0.0):
        raise ValueError(
            "Prepared PSFs must be nonnegative when computing Strehl or "
            "enclosed energy."
        )
    total_flux = np.sum(psf_cube, axis=(-2, -1), dtype=np.float64)
    if np.any(total_flux <= 0.0):
        raise ValueError(
            "Prepared PSFs must have positive in-frame flux when computing "
            "Strehl or enclosed energy."
        )
    if np.any(total_flux > 1.0 + 1.0e-5):
        raise ValueError(
            "Prepared PSFs must have total in-frame flux <= 1 when computing "
            "Strehl or enclosed energy. Use the default preprocessing or pass "
            "PSFs already expressed on their intended physical-flux scale."
        )


def _select_metadata_value(value: float | np.ndarray, index: int) -> float:
    if isinstance(value, np.ndarray):
        return float(value[index])
    return float(value)


def _compute_psf_stats_core(
    psf_cube: np.ndarray,
    *,
    wavelength: float | None,
    pixel_scale: float,
    tel_diameter: float | None,
    tel_pupil: np.ndarray | None,
    ee_apertures: np.ndarray,
    peak_method: str,
    fwhm_summary: str,
    ee_geometry: str,
    metrics: tuple[str, ...],
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    peak_locations_yx = None

    if STATS_METRIC_SR in metrics:
        if wavelength is None or tel_diameter is None or tel_pupil is None:
            raise ValueError("Strehl requires PsfMetadata with full physical metadata.")
        sr, peak_locations_yx = _compute_strehl(
            psf_cube,
            peak_method,
            pixel_scale,
            wavelength,
            tel_diameter,
            tel_pupil,
        )
        output[STATS_METRIC_SR] = np.asarray(sr, dtype=np.float32)
    elif STATS_METRIC_EE in metrics:
        peak_locations_yx = _compute_peak_locations(psf_cube, peak_method)

    if STATS_METRIC_EE in metrics:
        output[STATS_METRIC_EE] = np.asarray(
            _compute_enclosed_energy(
                psf_cube,
                ee_apertures,
                pixel_scale,
                peak_locations_yx,
                ee_geometry=ee_geometry,
            ),
            dtype=np.float32,
        )

    if STATS_METRIC_FWHM in metrics:
        fwhm_min, fwhm_max = _measure_contour_fwhms(psf_cube, pixel_scale)
        output[STATS_METRIC_FWHM] = np.asarray(
            _compute_fwhm_summary(fwhm_summary, fwhm_min, fwhm_max),
            dtype=np.float32,
        )

    return output


def _format_metric_value(
    name: str, value: np.ndarray, *, scalar_output: bool
) -> StatValue:
    unit = u.mas if name == STATS_METRIC_FWHM else u.dimensionless_unscaled
    if not scalar_output:
        return value * unit
    if name == STATS_METRIC_EE:
        ee_value = value[0].copy()
        if ee_value.shape[0] == 1:
            return float(ee_value[0]) * unit
        return ee_value * unit
    return float(value[0]) * unit


def _format_metric_results(
    results: dict[str, np.ndarray], metrics: tuple[str, ...], *, scalar_output: bool
) -> tuple[StatValue, ...]:
    return tuple(
        _format_metric_value(name, results[name], scalar_output=scalar_output)
        for name in metrics
    )


def compute_psf_stats(
    psfs: np.ndarray,
    metadata: PsfMetadata | None = None,
    *,
    pixel_scale: u.Quantity | None = None,
    ee_apertures: u.Quantity | None = None,
    peak_method: str = DEFAULT_PEAK_METHOD,
    fwhm_summary: str = DEFAULT_FWHM_SUMMARY,
    ee_geometry: str = DEFAULT_EE_GEOMETRY,
    preprocess: PsfPreprocessOption = "default",
    metrics: Sequence[str] | None = None,
) -> tuple[StatValue, ...]:
    """Compute core PSF statistics for one PSF image or cube.

    Args:
        psfs: PSF image ``[Ny, Nx]`` or PSF cube ``[M, Ny, Nx]``.
        metadata: Full physical PSF metadata, required for Strehl. Omit only
            when requesting EE and/or FWHM with ``pixel_scale`` instead.
            ``wavelength`` and ``pixel_scale`` may be scalar or one-dimensional
            per-PSF vectors with length ``M``. ``tel_diameter`` and ``tel_pupil``
            are shared across the cube.
        pixel_scale: Angular pixel scale when ``metadata`` is omitted. May be
            scalar or a per-PSF vector. Cannot be combined with ``metadata``.
        ee_apertures: EE aperture diameters in milliarcseconds. Required
            when ``"ee"`` is selected. Pass a 1D vector ``[A]`` for shared
            apertures, or a 2D array ``[M, A]`` for per-PSF apertures.
        peak_method: Shared peak estimator used for Strehl amplitude and
            enclosed-energy anchoring. Select ``"gaussian_fit"`` or
            ``"pixel_max"``. Gaussian fits use a 5x5 patch about the brightest
            pixel, shifted inward at edges and reduced for smaller images.
        fwhm_summary: FWHM contour summary selector: ``"geom"``, ``"mean"``,
            ``"min"``, or ``"max"``.
        ee_geometry: EE aperture geometry. Defaults to ``"ensquared"`` to
            preserve the existing square-aperture statistic.
        preprocess: Preprocessing control. ``"default"`` clips negative
            pixels and sum-normalizes every positive-flux PSF. ``None`` leaves
            already prepared PSFs unchanged. A callable with signature
            ``preprocess(psfs)`` can provide custom preprocessing. Prepared
            inputs used for Strehl or enclosed energy must be nonnegative and
            have positive total in-frame flux no greater than one.
        metrics: Optional metric names to compute. Omit it to return the
            standard tuple ``(sr, ee, fwhm)``. Pass a sequence drawn from
            ``"sr"``, ``"ee"``, and ``"fwhm"`` to compute only those
            metrics and return a tuple in the requested order.

    Returns:
        Tuple of requested Astropy quantities. Strehl ratio and enclosed energy
        are dimensionless; FWHM is in milliarcseconds. When ``metrics`` is
        omitted, the tuple is ``(sr, ee, fwhm)``. Successful results may return
        ``NaN`` in ``fwhm`` when the contour-based FWHM is not scientifically
        recoverable.

    Raises:
        TypeError: If metadata, physical quantities, or metric selection types
            do not satisfy the public contract.
        ValueError: If PSF shapes or values, metadata values, apertures,
            selectors, or preprocessing results are invalid.
    """
    psf_cube, scalar_output = _prepare_psf_cube(psfs)
    num_psfs = int(psf_cube.shape[0])
    metrics = _prepare_metrics(metrics)
    if metadata is None:
        if STATS_METRIC_SR in metrics:
            raise ValueError("Strehl requires PsfMetadata with full physical metadata.")
        if pixel_scale is None:
            raise ValueError("pixel_scale is required when metadata is omitted.")
        prepared_metadata = _prepare_pixel_scale_metadata(pixel_scale, num_psfs)
    else:
        if pixel_scale is not None:
            raise ValueError("Provide metadata or pixel_scale, not both.")
        prepared_metadata = _prepare_psf_metadata(metadata, num_psfs)
    if STATS_METRIC_EE in metrics:
        if ee_apertures is None:
            raise ValueError("ee_apertures is required when computing EE.")
        ee_apertures, ee_apertures_are_per_psf = _prepare_ee_apertures(
            ee_apertures, num_psfs
        )
    else:
        ee_apertures = np.empty((0,), dtype=np.float32)
        ee_apertures_are_per_psf = False
    peak_method = _require_peak_method(peak_method)
    fwhm_summary = _require_fwhm_summary(fwhm_summary)
    ee_geometry = _normalize_ee_geometry(ee_geometry)
    psf_cube = _apply_preprocess(psf_cube, preprocess)
    _validate_flux_inputs(psf_cube, metrics)

    if not prepared_metadata.has_per_psf_values and not ee_apertures_are_per_psf:
        results = _compute_psf_stats_core(
            psf_cube,
            wavelength=(
                float(prepared_metadata.wavelength)
                if prepared_metadata.wavelength is not None
                else None
            ),
            pixel_scale=float(prepared_metadata.pixel_scale),
            tel_diameter=prepared_metadata.tel_diameter,
            tel_pupil=prepared_metadata.tel_pupil,
            ee_apertures=ee_apertures,
            peak_method=peak_method,
            fwhm_summary=fwhm_summary,
            ee_geometry=ee_geometry,
            metrics=metrics,
        )
    else:
        result_rows: dict[str, list[np.ndarray]] = {name: [] for name in metrics}
        for index in range(num_psfs):
            row_results = _compute_psf_stats_core(
                psf_cube[index : index + 1],
                wavelength=(
                    _select_metadata_value(prepared_metadata.wavelength, index)
                    if prepared_metadata.wavelength is not None
                    else None
                ),
                pixel_scale=_select_metadata_value(
                    prepared_metadata.pixel_scale, index
                ),
                tel_diameter=prepared_metadata.tel_diameter,
                tel_pupil=prepared_metadata.tel_pupil,
                ee_apertures=ee_apertures[index]
                if ee_apertures_are_per_psf
                else ee_apertures,
                peak_method=peak_method,
                fwhm_summary=fwhm_summary,
                ee_geometry=ee_geometry,
                metrics=metrics,
            )
            for name in metrics:
                result_rows[name].append(row_results[name])
        results = {
            name: np.concatenate(rows, axis=0).astype(np.float32, copy=False)
            for name, rows in result_rows.items()
        }

    return _format_metric_results(results, metrics, scalar_output=scalar_output)


def compute_psf_sr(
    psfs: np.ndarray,
    metadata: PsfMetadata,
    *,
    peak_method: str = DEFAULT_PEAK_METHOD,
    preprocess: PsfPreprocessOption = "default",
) -> StatValue:
    """Measure PSF Strehl ratio.

    Args:
        psfs: PSF image ``[Ny, Nx]`` or PSF cube ``[M, Ny, Nx]``.
        metadata: Physical metadata for the PSFs.
        peak_method: Peak estimator, ``"gaussian_fit"`` or ``"pixel_max"``.
        preprocess: Preprocessing control accepted by ``compute_psf_stats``.

    Returns:
        A dimensionless Astropy quantity, scalar for one PSF or shape ``[M]``
        for a cube.

    Raises:
        TypeError: If metadata or physical quantities do not satisfy the
            public contract.
        ValueError: If PSFs, metadata, the selector, or preprocessing results
            are invalid.
    """
    result = compute_psf_stats(
        psfs,
        metadata,
        peak_method=peak_method,
        preprocess=preprocess,
        metrics=(STATS_METRIC_SR,),
    )
    return result[0]


def compute_psf_ee(
    psfs: np.ndarray,
    metadata: PsfMetadata | None = None,
    *,
    ee_apertures: u.Quantity,
    pixel_scale: u.Quantity | None = None,
    peak_method: str = DEFAULT_PEAK_METHOD,
    ee_geometry: str = DEFAULT_EE_GEOMETRY,
    preprocess: PsfPreprocessOption = "default",
) -> StatValue:
    """Measure PSF enclosed energy.

    Args:
        psfs: PSF image ``[Ny, Nx]`` or PSF cube ``[M, Ny, Nx]``.
        metadata: Full physical metadata for the PSFs. Omit when supplying
            ``pixel_scale`` without Strehl metadata.
        ee_apertures: Aperture-diameter quantity. Pass shape ``[A]`` for
            shared apertures or ``[M, A]`` for per-PSF apertures.
        pixel_scale: Angular pixel scale used when ``metadata`` is omitted.
        peak_method: Peak estimator, ``"gaussian_fit"`` or ``"pixel_max"``.
        ee_geometry: Aperture geometry, ``"ensquared"`` or ``"encircled"``.
        preprocess: Preprocessing control accepted by ``compute_psf_stats``.

    Returns:
        A dimensionless Astropy quantity. Cube results have shape ``[M, A]``;
        a single-PSF result is scalar for one aperture or shape ``[A]``.

    Raises:
        TypeError: If metadata or physical quantities do not satisfy the
            public contract.
        ValueError: If PSFs, metadata, apertures, selectors, or preprocessing
            results are invalid.
    """
    result = compute_psf_stats(
        psfs,
        metadata,
        pixel_scale=pixel_scale,
        ee_apertures=ee_apertures,
        peak_method=peak_method,
        ee_geometry=ee_geometry,
        preprocess=preprocess,
        metrics=(STATS_METRIC_EE,),
    )
    return result[0]


def compute_psf_fwhm(
    psfs: np.ndarray,
    metadata: PsfMetadata | None = None,
    *,
    pixel_scale: u.Quantity | None = None,
    fwhm_summary: str = DEFAULT_FWHM_SUMMARY,
    preprocess: PsfPreprocessOption = "default",
) -> StatValue:
    """Measure contour-derived PSF FWHM.

    Args:
        psfs: PSF image ``[Ny, Nx]`` or PSF cube ``[M, Ny, Nx]``.
        metadata: Full physical metadata for the PSFs. Omit when supplying
            ``pixel_scale`` without Strehl metadata.
        pixel_scale: Angular pixel scale used when ``metadata`` is omitted.
        fwhm_summary: Contour summary, ``"geom"``, ``"mean"``, ``"min"``, or
            ``"max"``.
        preprocess: Optional preprocessing control accepted by
            ``compute_psf_stats``.

    Returns:
        An Astropy quantity in milliarcseconds, scalar for one PSF or shape
        ``[M]`` for a cube. A value may be ``NaN`` when a half-maximum contour
        is not scientifically recoverable.

    Raises:
        TypeError: If metadata or physical quantities do not satisfy the
            public contract.
        ValueError: If PSFs, metadata, the selector, or preprocessing results
            are invalid.
    """
    result = compute_psf_stats(
        psfs,
        metadata,
        pixel_scale=pixel_scale,
        fwhm_summary=fwhm_summary,
        preprocess=preprocess,
        metrics=(STATS_METRIC_FWHM,),
    )
    return result[0]
