"""Private Astropy quantity conversion helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from astropy import units as u


def _unit_string(unit: u.UnitBase) -> str:
    """Return the canonical generic string for one Astropy unit."""
    unit = u.Unit(unit)
    if unit == u.dimensionless_unscaled:
        return "1"
    return unit.to_string("generic")


def _require_quantity(
    value: Any,
    unit: u.UnitBase,
    *,
    label: str,
) -> u.Quantity:
    """Require a quantity and convert it to one canonical equivalent unit."""
    if not isinstance(value, u.Quantity):
        raise TypeError(f"{label} must be an Astropy Quantity.")
    try:
        return value.to(unit, copy=False)
    except u.UnitConversionError as exc:
        raise ValueError(
            f"{label} must have units equivalent to {_unit_string(unit)!r}."
        ) from exc


def quantity_value(
    value: Any,
    unit: u.UnitBase,
    *,
    label: str,
    dtype: Any | None = None,
) -> np.ndarray:
    """Return the unit-normalized numerical view of one required quantity."""
    quantity = _require_quantity(value, unit, label=label)
    return np.asarray(quantity.value, dtype=dtype)
