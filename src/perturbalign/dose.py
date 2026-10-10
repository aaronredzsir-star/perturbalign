"""Module for Multi-Dimensional Dose Classification, Standardization, and Conversion.

Strictly adheres to physical dimensional laws, cleanly separating molar concentration,
mass concentration, in vivo animal body weight dose, biological activity potency,
multiplicity of infection (MOI), and percentage concentration without conflating non-molar dimensions.
Cross-dimensional conversion (mass concentration <-> molar concentration) requires callers
to explicitly provide molecular weight (MW).
"""
from __future__ import annotations

import math
from typing import Optional

# Dimensional constants
DIMENSION_MOLAR = "molar"
DIMENSION_MASS_CONCENTRATION = "mass_concentration"
DIMENSION_BODY_WEIGHT = "body_weight"
DIMENSION_ACTIVITY = "activity_conc"
DIMENSION_PERCENTAGE = "percentage"
DIMENSION_INFECTION_RATIO = "infection_ratio"
DIMENSION_AMOUNT = "amount"
DIMENSION_VOLUME = "volume"
DIMENSION_RADIATION = "radiation"
DIMENSION_SHEAR_STRESS = "shear_stress"
DIMENSION_OSMOLARITY = "osmolarity"
DIMENSION_PRESSURE = "pressure"
DIMENSION_TEMPERATURE = "temperature"
DIMENSION_MULTIPLIER = "multiplier"
DIMENSION_VIRAL_TITER = "viral_titer"

# 1. Molar Concentration (base: M = mol/L, default normalized: µM)
_MOLAR_FACTORS = {
    "m": (1.0, "M"),
    "mol/l": (1.0, "M"),
    "mm": (1e-3, "mM"),
    "mmol/l": (1e-3, "mM"),
    "um": (1e-6, "µM"),
    "μm": (1e-6, "µM"),
    "µm": (1e-6, "µM"),
    "umol/l": (1e-6, "µM"),
    "μmol/l": (1e-6, "µM"),
    "µmol/l": (1e-6, "µM"),
    "nm": (1e-9, "nM"),
    "nmol/l": (1e-9, "nM"),
    "pm": (1e-12, "pM"),
    "pmol/l": (1e-12, "pM"),
    "fm": (1e-15, "fM"),
    "fmol/l": (1e-15, "fM"),
}

# 2. Mass Concentration (base: g/L = mg/mL, default normalized: µg/ml)
_MASS_CONC_FACTORS = {
    "g/l": (1.0, "g/L"),
    "mg/ml": (1.0, "mg/ml"),
    "mg/l": (1e-3, "mg/L"),
    "ug/ml": (1e-3, "µg/ml"),
    "μg/ml": (1e-3, "µg/ml"),
    "µg/ml": (1e-3, "µg/ml"),
    "ug/l": (1e-6, "µg/L"),
    "μg/l": (1e-6, "µg/L"),
    "µg/l": (1e-6, "µg/L"),
    "ng/ml": (1e-6, "ng/ml"),
    "ng/l": (1e-9, "ng/L"),
    "pg/ml": (1e-9, "pg/ml"),
    "ug/ul": (1.0, "µg/µl"),
    "μg/ul": (1.0, "µg/µl"),
    "µg/ul": (1.0, "µg/µl"),
    "ng/ul": (1e-3, "ng/µl"),
    "pg/ul": (1e-6, "pg/µl"),
}

# 3. Body Weight dose (base: mg/kg, default normalized: mg/kg)
_BODY_WEIGHT_FACTORS = {
    "g/kg": (1000.0, "g/kg"),
    "mg/kg": (1.0, "mg/kg"),
    "ug/kg": (1e-3, "µg/kg"),
    "μg/kg": (1e-3, "µg/kg"),
    "µg/kg": (1e-3, "µg/kg"),
}

# 4. Biological Activity Concentration (base: U/ml or IU/ml)
_ACTIVITY_CONC_FACTORS = {
    "ku/ml": (1000.0, "kU/ml"),
    "u/ml": (1.0, "U/ml"),
    "units/ml": (1.0, "units/ml"),
    "mu/ml": (1e-3, "mU/ml"),
    "iu/ml": (1.0, "IU/ml"),
    "miu/ml": (1e-3, "mIU/ml"),
}

# 5. Percentage Concentration
_PERCENT_UNITS = {
    "%", "% v/v", "% w/v", "% w/w", "wt%", "vol%"
}

# 6. Infection Ratio & Multiplicity of Infection (MOI)
_INFECTION_UNITS = {
    "moi", "copies/cell"
}

# 7. Amount of Substance (base: mol)
_AMOUNT_FACTORS = {
    "mol": (1.0, "mol"),
    "mmol": (1e-3, "mmol"),
    "umol": (1e-6, "µmol"),
    "μmol": (1e-6, "µmol"),
    "µmol": (1e-6, "µmol"),
    "nmol": (1e-9, "nmol"),
    "pmol": (1e-12, "pmol"),
}

# 8. Volume (base: L)
_VOLUME_FACTORS = {
    "l": (1.0, "L"),
    "ml": (1e-3, "mL"),
    "ul": (1e-6, "µL"),
    "μl": (1e-6, "µL"),
    "µl": (1e-6, "µL"),
    "nl": (1e-9, "nL"),
    "pl": (1e-12, "pL"),
}

# 9. Radiation Dose (base: Gy)
_RADIATION_FACTORS = {
    "gy": (1.0, "Gy"),
    "cgy": (0.01, "cGy"),
    "mgy": (1e-3, "mGy"),
    "rad": (0.01, "rad"),
}

# 10. Fluid Shear Stress (base: dyn/cm2)
_SHEAR_STRESS_FACTORS = {
    "dyn/cm2": (1.0, "dyn/cm2"),
    "dyn/cm²": (1.0, "dyn/cm2"),
    "dyne/cm2": (1.0, "dyn/cm2"),
    "pa": (10.0, "Pa"),
    "n/m2": (10.0, "N/m2"),
    "mpa": (0.01, "mPa"),
}

# 11. Osmolarity (base: mOsm)
_OSMOLARITY_FACTORS = {
    "mosm": (1.0, "mOsm"),
    "mosm/l": (1.0, "mOsm/L"),
    "mosm/kg": (1.0, "mOsm/kg"),
    "osm": (1000.0, "Osm"),
    "osm/l": (1000.0, "Osm/L"),
    "osm/kg": (1000.0, "Osm/kg"),
}

# 12. Pressure (base: atm)
_PRESSURE_FACTORS = {
    "atm": (1.0, "atm"),
    "bar": (0.986923, "bar"),
    "kpa": (0.00986923, "kPa"),
    "pa": (9.86923e-6, "Pa"),
    "mmhg": (0.00131579, "mmHg"),
    "torr": (0.00131579, "torr"),
    "psi": (0.068046, "psi"),
}

# 13. Temperature
_TEMPERATURE_FACTORS = {
    "°c": (1.0, "°C"),
    "c": (1.0, "°C"),
    "degc": (1.0, "°C"),
}

# 14. Multiplier
_MULTIPLIER_FACTORS = {
    "x": (1.0, "x"),
    "-fold": (1.0, "x"),
    "fold": (1.0, "x"),
}

# 15. Viral Titer & Genome Copies (base: vg/ml)
_VIRAL_TITER_FACTORS = {
    "vg/ml": (1.0, "vg/ml"),
    "vg/ul": (1000.0, "vg/µl"),
    "vg/µl": (1000.0, "vg/µl"),
    "tu/ml": (1.0, "TU/ml"),
    "ifu/ml": (1.0, "IFU/ml"),
    "pfu/ml": (1.0, "PFU/ml"),
    "gc/ml": (1.0, "gc/ml"),
}

_BUILTIN_FACTORS: dict[str, dict[str, tuple[float, str]]] = {
    DIMENSION_MOLAR: _MOLAR_FACTORS,
    DIMENSION_MASS_CONCENTRATION: _MASS_CONC_FACTORS,
    DIMENSION_BODY_WEIGHT: _BODY_WEIGHT_FACTORS,
    DIMENSION_ACTIVITY: _ACTIVITY_CONC_FACTORS,
    DIMENSION_AMOUNT: _AMOUNT_FACTORS,
    DIMENSION_VOLUME: _VOLUME_FACTORS,
    DIMENSION_RADIATION: _RADIATION_FACTORS,
    DIMENSION_SHEAR_STRESS: _SHEAR_STRESS_FACTORS,
    DIMENSION_OSMOLARITY: _OSMOLARITY_FACTORS,
    DIMENSION_PRESSURE: _PRESSURE_FACTORS,
    DIMENSION_TEMPERATURE: _TEMPERATURE_FACTORS,
    DIMENSION_MULTIPLIER: _MULTIPLIER_FACTORS,
    DIMENSION_VIRAL_TITER: _VIRAL_TITER_FACTORS,
}

# Dynamic custom dimension registry
_CUSTOM_DIMENSION_FACTORS: dict[str, dict[str, tuple[float, str]]] = {}
_CUSTOM_DIMENSION_DEFAULTS: dict[str, str] = {}


def register_dose_dimension(
    dimension: str,
    factors: dict[str, tuple[float, str]],
    default_unit: str,
) -> None:
    """Register custom dose or environmental physical dimension with conversion factors and base unit.

    Parameters
    ----------
    dimension : str
        Unique dimension identifier (e.g., 'optics', 'sound_pressure').
    factors : dict
        Mapping {unit_string: (factor_to_base, canonical_display_name)}.
    default_unit : str
        Recommended normalized unit (e.g., 'µW/cm2').
    """
    dim_norm = dimension.strip().lower()
    normed = {_norm_unit_str(k): v for k, v in factors.items()}
    _CUSTOM_DIMENSION_FACTORS[dim_norm] = normed
    _CUSTOM_DIMENSION_DEFAULTS[dim_norm] = default_unit


def _norm_unit_str(unit: str) -> str:
    """Clean and normalize unit string."""
    return unit.strip().replace(" ", "").lower()


def classify_dose_dimension(unit: str | None) -> str | None:
    """Determine the physical dimension of a dosing or concentration unit.

    Returns
    -------
    str | None
        One of 'molar', 'mass_concentration', 'body_weight', 'activity_conc',
        'percentage', 'infection_ratio', 'amount', 'volume', 'radiation',
        'shear_stress', 'osmolarity', 'pressure', 'temperature', 'multiplier',
        'viral_titer', custom dimension, or None if unrecognized.
    """
    if not unit:
        return None
    raw = unit.strip()
    u = _norm_unit_str(raw)

    for dim_name, factors in _BUILTIN_FACTORS.items():
        if u in factors:
            return dim_name

    if raw.lower() in _PERCENT_UNITS or u == "%":
        return DIMENSION_PERCENTAGE
    if u in _INFECTION_UNITS:
        return DIMENSION_INFECTION_RATIO

    for c_dim, c_factors in _CUSTOM_DIMENSION_FACTORS.items():
        if u in c_factors:
            return c_dim

    return None


def get_default_canonical_unit(dimension: str) -> str:
    """Get the recommended standard unit for a given dimension."""
    defaults = {
        DIMENSION_MOLAR: "µM",
        DIMENSION_MASS_CONCENTRATION: "µg/ml",
        DIMENSION_BODY_WEIGHT: "mg/kg",
        DIMENSION_ACTIVITY: "U/ml",
        DIMENSION_PERCENTAGE: "%",
        DIMENSION_INFECTION_RATIO: "MOI",
        DIMENSION_AMOUNT: "µmol",
        DIMENSION_VOLUME: "µL",
        DIMENSION_RADIATION: "Gy",
        DIMENSION_SHEAR_STRESS: "dyn/cm2",
        DIMENSION_OSMOLARITY: "mOsm",
        DIMENSION_PRESSURE: "atm",
        DIMENSION_TEMPERATURE: "°C",
        DIMENSION_MULTIPLIER: "x",
        DIMENSION_VIRAL_TITER: "vg/ml",
    }
    if dimension in defaults:
        return defaults[dimension]
    return _CUSTOM_DIMENSION_DEFAULTS.get(dimension, "")


def standardize_dose(
    dose: float | None,
    unit: str | None,
    target_unit: str | None = None,
    mw: float | None = None,
) -> tuple[float | None, str | None]:
    """Standardize and convert multi-dimensional concentration and dose values.

    Parameters
    ----------
    dose : float | None
        Numerical dose value.
    unit : str | None
        Raw unit string (e.g., '100 nM', '500 ng/ml', '2 mg/kg', '1000 IU/ml', '4 Gy').
    target_unit : str | None, optional
        Target unit. If None, automatically adopts the recommended unit for the dimension.
    mw : float | None, optional
        Molecular weight in g/mol. Strictly required when converting between mass and molar dimensions.

    Returns
    -------
    tuple[float | None, str | None]
        (converted_value, standardized_unit)

    Raises
    ------
    ValueError
        If attempting an incompatible cross-dimensional conversion or missing mw for mass-molar conversion.
    """
    if dose is None or unit is None:
        return dose, unit

    dim = classify_dose_dimension(unit)
    if dim is None:
        # Unknown dimension, passthrough as-is
        return dose, unit

    u_norm = _norm_unit_str(unit)
    target_dim = classify_dose_dimension(target_unit) if target_unit else dim

    if target_unit is None:
        target_unit = get_default_canonical_unit(dim)
        target_dim = dim

    t_norm = _norm_unit_str(target_unit)

    # 1) Same-dimension conversion
    if dim == target_dim:
        factors = _BUILTIN_FACTORS.get(dim) or _CUSTOM_DIMENSION_FACTORS.get(dim)
        if factors and u_norm in factors and t_norm in factors:
            factor_in, _ = factors[u_norm]
            factor_out, canon_out = factors[t_norm]
            val_base = dose * factor_in
            return round(val_base / factor_out, 10), canon_out

        # Percentage and MOI passthrough directly
        return dose, target_unit

    # 2) Cross-dimension conversion: mass_concentration <-> molar (requires MW)
    if {dim, target_dim} == {DIMENSION_MASS_CONCENTRATION, DIMENSION_MOLAR}:
        if mw is None or mw <= 0:
            raise ValueError(
                f"Cross-dimensional conversion [{dim} ({unit})] -> [{target_dim} ({target_unit})] "
                "must provide positive molecular weight parameter mw (g/mol)."
            )
        if dim == DIMENSION_MASS_CONCENTRATION and target_dim == DIMENSION_MOLAR:
            # Mass concentration (g/L) / mw = Molar concentration (M)
            factor_in, _ = _MASS_CONC_FACTORS[u_norm]
            val_g_per_l = dose * factor_in
            val_molar = val_g_per_l / mw
            factor_out, canon_out = _MOLAR_FACTORS[t_norm]
            return round(val_molar / factor_out, 10), canon_out
        else:
            # Molar concentration (M) * mw = Mass concentration (g/L)
            factor_in, _ = _MOLAR_FACTORS[u_norm]
            val_molar = dose * factor_in
            val_g_per_l = val_molar * mw
            factor_out, canon_out = _MASS_CONC_FACTORS[t_norm]
            return round(val_g_per_l / factor_out, 10), canon_out

    # 3) Incompatible cross-dimension conversion
    raise ValueError(
        f"Cannot convert dose between incompatible physical dimensions: {dim} ({unit}) -> {target_dim} ({target_unit})"
    )


def to_pconc(dose: float | None, unit: str | None) -> float | None:
    """Calculate pharmacological logarithmic molar concentration pConc = -log10[M].

    Guardrails: Returns a computed float ONLY when the unit belongs to the molar dimension
    (M, mM, µM, nM, pM, fM). For mass concentrations (ng/ml), in vivo doses (mg/kg),
    biological activities (U/ml), percentages, or non-positive numbers, safely returns None.

    Examples
    --------
    >>> to_pconc(10.0, "µM")
    5.0
    >>> to_pconc(100.0, "nM")
    7.0
    >>> to_pconc(10.0, "ng/ml")
    None
    """
    if dose is None or unit is None or dose <= 0:
        return None
    dim = classify_dose_dimension(unit)
    if dim != DIMENSION_MOLAR:
        return None
    u_norm = _norm_unit_str(unit)
    factor, _ = _MOLAR_FACTORS[u_norm]
    molar_conc = dose * factor
    if molar_conc <= 0:
        return None
    return round(-math.log10(molar_conc), 4)
