# Python Reference

The supported API is imported from the `ao_stats` package root.

## Types

| Name | Contract |
| --- | --- |
| `PsfMetadata` | Full physical metadata required for Strehl; EE and FWHM also accept a pixel scale alone. |
| `StatValue` | Astropy quantity returned by a measurement function. |
| `PsfMetricName` | Metric name: `"sr"`, `"ee"`, or `"fwhm"`. |
| `PsfPreprocessor` | Callable that accepts and returns a PSF cube. |
| `PsfPreprocessOption` | A preprocessor, `"default"`, or `None`. |

## Metric Constants

| Name | Value |
| --- | --- |
| `STATS_METRIC_SR` | `"sr"` |
| `STATS_METRIC_EE` | `"ee"` |
| `STATS_METRIC_FWHM` | `"fwhm"` |
| `STATS_METRICS` | `("sr", "ee", "fwhm")` |

## Peak-estimation Constants

| Name | Value |
| --- | --- |
| `PEAK_METHOD_GAUSSIAN_FIT` | `"gaussian_fit"` |
| `PEAK_METHOD_PIXEL_MAX` | `"pixel_max"` |
| `PEAK_METHODS` | `("gaussian_fit", "pixel_max")` |
| `DEFAULT_PEAK_METHOD` | `"gaussian_fit"` |

## Enclosed-energy Constants

| Name | Value |
| --- | --- |
| `EE_GEOMETRY_ENSQUARED` | `"ensquared"` |
| `EE_GEOMETRY_ENCIRCLED` | `"encircled"` |
| `EE_GEOMETRIES` | `("ensquared", "encircled")` |
| `DEFAULT_EE_GEOMETRY` | `"ensquared"` |

## FWHM Constants

| Name | Value |
| --- | --- |
| `STATS_FWHM_SUMMARY_GEOM` | `"geom"` |
| `STATS_FWHM_SUMMARY_MEAN` | `"mean"` |
| `STATS_FWHM_SUMMARY_MAX` | `"max"` |
| `STATS_FWHM_SUMMARY_MIN` | `"min"` |
| `STATS_FWHM_SUMMARIES` | `("geom", "mean", "max", "min")` |
| `DEFAULT_FWHM_SUMMARY` | `"geom"` |

## Package Version

`__version__` contains the installed AO Stats version.

## Classes and Functions

::: ao_stats
