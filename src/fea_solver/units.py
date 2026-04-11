"""Unit system definitions, conversion tables, and converter for the FEA solver.

Supports two canonical unit systems. All incoming YAML values are converted to
the chosen canonical basis before any assembly or solve step. The math in
elements.py, assembler.py, and solver.py is unit-agnostic; it relies solely on
numerical consistency within whichever canonical system is selected.

Conversion pipeline: input unit -> canonical SI -> canonical Empirical (if needed).
Routing through SI as an intermediate avoids an N*M conversion table.

UnitSystem:       Enum identifying the two supported canonical systems.
CANONICAL_UNITS:  Default input units assumed per system when `units:` block is absent.
UNIT_LABELS:      Human-readable unit strings for reporter column headers.
_TO_SI:           Conversion factors from any supported input unit to canonical SI.
_SI_TO_EMP:       Conversion factors from canonical SI to canonical Empirical.
UnitConverter:    Frozen dataclass that converts a scalar from its declared input unit
                  to the canonical unit of the chosen system.
validate_unit:    Raises ValueError if a unit string is not recognised for a quantity type.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


# ---------------------------------------------------------------------------
# Enum
# ---------------------------------------------------------------------------


class UnitSystem(str, Enum):
    """Identifies the canonical unit system used by an FEAModel.

    Fields:
        SI: SI units -- metres, Newtons, Pascals, m^2, m^4, N/m, N-m.
        EMPIRICAL: US Customary units -- inches, pounds-force, psi, in^2, in^4, lb/in, in-lb.

    Notes:
        Used as a field on FEAModel and as a key in CANONICAL_UNITS and UNIT_LABELS.
        Inherits from str so YAML strings ("SI", "empirical") compare equal to members.
    """

    SI = "SI"
    EMPIRICAL = "empirical"


# ---------------------------------------------------------------------------
# Conversion tables
# ---------------------------------------------------------------------------

# Multiply any supported input value by the factor to obtain canonical SI.
_TO_SI: dict[str, dict[str, float]] = {
    "length": {
        "m":  1.0,
        "cm": 0.01,
        "mm": 1.0e-3,
        "ft": 0.3048,
        "in": 0.0254,
    },
    "force": {
        "N":   1.0,
        "kN":  1.0e3,
        "lb":  4.44822162,
        "kip": 4448.22162,
    },
    "modulus": {
        "Pa":  1.0,
        "kPa": 1.0e3,
        "MPa": 1.0e6,
        "GPa": 1.0e9,
        "psi": 6894.757293,
        "ksi": 6894757.293,
    },
    "area": {
        "m^2":  1.0,
        "cm^2": 1.0e-4,
        "mm^2": 1.0e-6,
        "in^2": 6.4516e-4,
        "ft^2": 0.09290304,
    },
    "second_moment": {
        "m^4":  1.0,
        "cm^4": 1.0e-8,
        "mm^4": 1.0e-12,
        "in^4": 4.16231426e-7,
        "ft^4": 8.63097481e-3,
    },
    "distributed": {
        "N/m":   1.0,
        "kN/m":  1.0e3,
        "lb/ft": 14.5939029,
        "lb/in": 175.126835,
    },
    "moment": {
        "N-m":  1.0,
        "N*m":  1.0,
        "kN-m": 1.0e3,
        "ft-lb": 1.35581795,
        "lb-ft": 1.35581795,
        "in-lb": 0.112984829,
        "lb-in": 0.112984829,
    },
}

# Multiply canonical SI value by factor to obtain canonical Empirical.
_SI_TO_EMP: dict[str, float] = {
    "length":        1.0 / 0.0254,
    "force":         1.0 / 4.44822162,
    "modulus":       1.0 / 6894.757293,
    "area":          1.0 / 6.4516e-4,
    "second_moment": 1.0 / 4.16231426e-7,
    "distributed":   1.0 / 175.126835,
    "moment":        1.0 / 0.112984829,
}


# ---------------------------------------------------------------------------
# Public maps
# ---------------------------------------------------------------------------

# Canonical input units for each system -- used when units: block is omitted.
CANONICAL_UNITS: dict[UnitSystem, dict[str, str]] = {
    UnitSystem.SI: {
        "length":        "m",
        "force":         "N",
        "modulus":       "Pa",
        "area":          "m^2",
        "second_moment": "m^4",
        "distributed":   "N/m",
        "moment":        "N-m",
    },
    UnitSystem.EMPIRICAL: {
        "length":        "in",
        "force":         "lb",
        "modulus":       "psi",
        "area":          "in^2",
        "second_moment": "in^4",
        "distributed":   "lb/in",
        "moment":        "in-lb",
    },
}

# Reporter column header labels keyed by unit system.
UNIT_LABELS: dict[UnitSystem, dict[str, str]] = {
    UnitSystem.SI: {
        "length":       "m",
        "displacement": "m",
        "rotation":     "rad",
        "force":        "N",
        "moment":       "N-m",
        "distributed":  "N/m",
    },
    UnitSystem.EMPIRICAL: {
        "length":       "in",
        "displacement": "in",
        "rotation":     "rad",
        "force":        "lb",
        "moment":       "in-lb",
        "distributed":  "lb/in",
    },
}


# ---------------------------------------------------------------------------
# Converter
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class UnitConverter:
    """Converts scalar values from declared input units to a canonical system.

    Fields:
        unit_system (UnitSystem): Target canonical system (SI or EMPIRICAL).
        units (dict[str, str]): Maps quantity type name to input unit string.
            Example: {"length": "ft", "force": "lb", "distributed": "lb/ft", ...}

    Notes:
        Conversion pipeline: input -> SI (via _TO_SI) -> Empirical if needed (via _SI_TO_EMP).
        Routing through SI as an intermediate avoids a combinatorial input*output table.
        Quantity type keys must be present in _TO_SI (length, force, modulus, area,
        second_moment, distributed, moment). Unknown keys raise KeyError.
        Unknown unit strings for a given quantity type raise KeyError from the inner dict.
    """

    unit_system: UnitSystem
    units: dict[str, str]

    def convert(self, value: float, qty: str) -> float:
        """Convert a scalar value from its declared input unit to the canonical system.

        Args:
            value (float): Numeric value in the input unit for quantity type qty.
            qty (str): Quantity type key -- one of "length", "force", "modulus",
                "area", "second_moment", "distributed", "moment".

        Returns:
            float: Value converted to the canonical unit of self.unit_system.

        Notes:
            Raises KeyError if qty is not in _TO_SI or if the input unit string
            is not recognised for that quantity type.
        """
        si_val = value * _TO_SI[qty][self.units[qty]]
        if self.unit_system == UnitSystem.EMPIRICAL:
            return si_val * _SI_TO_EMP[qty]
        return si_val


# ---------------------------------------------------------------------------
# Public validation helper
# ---------------------------------------------------------------------------


def validate_unit(qty: str, unit_str: str) -> None:
    """Raise ValueError if unit_str is not a recognised unit for the given quantity type.

    Args:
        qty (str): Quantity type key -- one of "length", "force", "modulus",
            "area", "second_moment", "distributed", "moment".
        unit_str (str): Unit label string to validate (e.g. "ft", "psi", "lb/ft").

    Returns:
        None

    Raises:
        ValueError: If qty is not a known quantity type, or if unit_str is not
            a supported unit for that quantity type.

    Notes:
        Exposed as a public API so callers do not need to import _TO_SI directly.
        Use this in YAML parsing to report clean errors with supported-unit lists.
    """
    if qty not in _TO_SI:
        raise ValueError(
            f"Unknown quantity type '{qty}'. Supported: {sorted(_TO_SI)}"
        )
    if unit_str not in _TO_SI[qty]:
        raise ValueError(
            f"Unknown unit '{unit_str}' for quantity '{qty}'. "
            f"Supported: {sorted(_TO_SI[qty])}"
        )
