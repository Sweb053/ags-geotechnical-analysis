from __future__ import annotations

import math

import pandas as pd

from ags_app.common import (
    REQUIRED_GEOL_COLUMNS,
    add_unmatched_geology,
    attach_geology_by_depth,
    prepare_geology_table,
    required_table,
    to_number,
)
from ags_app.geolmodel import add_geological_model_fields


REQUIRED_GCHM_COLUMNS = {"LOCA_ID", "SAMP_TOP", "GCHM_CODE", "GCHM_RESL"}
CHEMISTRY_COLUMNS = {
    "WS": "WS_MG_L",
    "SO4": "WS_MG_L",
    "PH": "PH_VALUE",
    "AS": "AS_PERCENT",
    "TS": "TS_PERCENT",
    "CL": "CL_MG_L",
    "NO3": "NO3_MG_L",
    "MG": "MG_MG_L",
}
SOIL_SULFATE_LIMITS = [(500, "DS-1"), (1500, "DS-2"), (3000, "DS-3"), (6000, "DS-4")]
GROUNDWATER_SULFATE_LIMITS = [(400, "DS-1"), (1400, "DS-2"), (3000, "DS-3"), (6000, "DS-4")]


def build_bre_sulphate_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    gchm = required_table(tables, "GCHM", REQUIRED_GCHM_COLUMNS).copy()
    geol = tables.get("GEOL", pd.DataFrame()).copy()

    gchm["LOCA_ID"] = gchm["LOCA_ID"].astype(str).str.strip()
    spec_depth = gchm["SPEC_DPTH"] if "SPEC_DPTH" in gchm.columns else pd.Series(pd.NA, index=gchm.index)
    gchm["BRE_DEPTH"] = spec_depth.fillna(gchm["SAMP_TOP"])
    gchm["BRE_DEPTH_NUM"] = to_number(gchm["BRE_DEPTH"])
    gchm["GCHM_CODE_NORM"] = gchm["GCHM_CODE"].fillna("").astype(str).str.upper().str.strip()
    gchm["GCHM_RESL_NUM"] = to_number(gchm["GCHM_RESL"])
    gchm["BRE_SAMPLE_TYPE"] = gchm.apply(detect_bre_sample_type, axis=1)

    key_columns = [
        column
        for column in ["LOCA_ID", "SAMP_TOP", "SAMP_REF", "SAMP_TYPE", "SAMP_ID", "SPEC_REF", "SPEC_DPTH"]
        if column in gchm.columns
    ]
    if not key_columns:
        key_columns = ["LOCA_ID", "BRE_DEPTH"]

    base = gchm.groupby(key_columns, dropna=False).first().reset_index()
    keep_columns = [
        column
        for column in [
            *key_columns,
            "BRE_DEPTH",
            "BRE_DEPTH_NUM",
            "BRE_SAMPLE_TYPE",
            "GCHM_METH",
            "GCHM_TTYP",
        ]
        if column in base.columns
    ]
    result = base[keep_columns].copy()

    for source_code, target_column in CHEMISTRY_COLUMNS.items():
        values = (
            gchm[gchm["GCHM_CODE_NORM"] == source_code]
            .groupby(key_columns, dropna=False)["GCHM_RESL_NUM"]
            .first()
            .rename(target_column)
            .reset_index()
        )
        if target_column in result.columns:
            continue
        result = result.merge(values, on=key_columns, how="left")

    result = result.dropna(subset=["LOCA_ID", "BRE_DEPTH_NUM"], how="any").copy()
    result = result[
        result[["WS_MG_L", "PH_VALUE", "AS_PERCENT", "TS_PERCENT", "CL_MG_L", "NO3_MG_L", "MG_MG_L"]]
        .notna()
        .any(axis=1)
    ].copy()
    result["TPS_PERCENT"] = result["TS_PERCENT"] * 3
    result["OS_PERCENT"] = result["TPS_PERCENT"] - result["AS_PERCENT"]
    result["SOIL_DS_CLASS"] = result["WS_MG_L"].map(lambda value: sulphate_class(value, "soil"))
    result["GROUNDWATER_DS_CLASS"] = result["WS_MG_L"].map(lambda value: sulphate_class(value, "groundwater"))

    if REQUIRED_GEOL_COLUMNS.issubset(set(geol.columns)):
        geol = prepare_geology_table(geol)
        result = attach_geology_by_depth(result, geol, "BRE_DEPTH_NUM")
    else:
        result = add_unmatched_geology(result)

    return add_geological_model_fields(result.sort_values(["LOCA_ID", "BRE_DEPTH_NUM"]).reset_index(drop=True))


def detect_bre_sample_type(row: pd.Series) -> str:
    text = " ".join(str(row.get(column, "")) for column in ["GCHM_TTYP", "GCHM_METH", "GCHM_NAME"]).upper()
    if "GROUNDWATER" in text or "WATER SAMPLE" in text or "AQUEOUS" in text:
        return "Groundwater"
    return "Soil"


def sulphate_class(value: object, sample_type: str = "soil") -> str | None:
    if pd.isna(value):
        return None
    numeric = float(value)
    limits = GROUNDWATER_SULFATE_LIMITS if sample_type == "groundwater" else SOIL_SULFATE_LIMITS
    for limit, label in limits:
        if (label == "DS-1" and numeric < limit) or (label != "DS-1" and numeric <= limit):
            return label
    return "DS-5"


def characteristic_high(values: pd.Series, round_to: float | None = None) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float).sort_values(ascending=False)
    if clean.empty:
        return None
    if len(clean) < 5:
        value = float(clean.iloc[0])
    elif len(clean) < 10:
        value = float(clean.iloc[:2].mean())
    else:
        count = max(1, math.ceil(len(clean) * 0.2))
        value = float(clean.iloc[:count].mean())
    if round_to:
        return math.ceil(value / round_to) * round_to
    return value


def characteristic_low(values: pd.Series) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float).sort_values()
    if clean.empty:
        return None
    if len(clean) < 5:
        return float(clean.iloc[0])
    count = max(1, math.ceil(len(clean) * 0.2))
    return float(clean.iloc[:count].mean())


def ds_rank(ds_class: str | None) -> int:
    if not ds_class:
        return 0
    for rank, label in enumerate(["DS-1", "DS-2", "DS-3", "DS-4", "DS-4m", "DS-5", "DS-5m"], start=1):
        if ds_class == label:
            return rank
    return 0


def highest_ds(classes: list[str | None]) -> str | None:
    clean = [value for value in classes if value]
    if not clean:
        return None
    return max(clean, key=ds_rank)


def apply_brownfield_magnesium_suffix(ds_class: str | None, magnesium: float | None, sample_type: str = "soil") -> str | None:
    if ds_class not in {"DS-4", "DS-5"} or magnesium is None:
        return ds_class
    threshold = 1000 if sample_type == "groundwater" else 1200
    if magnesium > threshold:
        return f"{ds_class}m"
    return ds_class


def classify_acec(ds_class: str | None, ph: float | None, site_type: str, water_mobility: str) -> str | None:
    if not ds_class or ph is None or ph < 2.5:
        return None
    site = site_type.lower()
    mobility = water_mobility.lower()
    if site == "brownfield":
        return classify_brownfield_acec(ds_class, ph, mobility)
    return classify_natural_acec(ds_class, ph, mobility)


def classify_natural_acec(ds_class: str, ph: float, mobility: str) -> str:
    if mobility == "static":
        if ds_class == "DS-1":
            return "AC-1s"
        if ds_class == "DS-2":
            return "AC-1s" if ph > 3.5 else "AC-2s"
        if ds_class == "DS-3":
            return "AC-2s" if ph > 3.5 else "AC-3s"
        if ds_class == "DS-4":
            return "AC-3s" if ph > 3.5 else "AC-4s"
        return "AC-4s" if ph > 3.5 else "AC-5"

    if ds_class == "DS-1":
        return "AC-1" if ph > 5.5 else "AC-2z"
    if ds_class == "DS-2":
        return "AC-2" if ph > 5.5 else "AC-3z"
    if ds_class == "DS-3":
        return "AC-3" if ph > 5.5 else "AC-4"
    if ds_class == "DS-4":
        return "AC-4" if ph > 5.5 else "AC-5"
    return "AC-5"


def classify_brownfield_acec(ds_class: str, ph: float, mobility: str) -> str:
    magnesium_suffix = "m" if ds_class.endswith("m") else ""
    base_ds = ds_class.removesuffix("m")
    if mobility == "static":
        if base_ds == "DS-1":
            return "AC-1s"
        if base_ds == "DS-2":
            return "AC-1s" if ph > 5.5 else "AC-2s"
        if base_ds == "DS-3":
            return "AC-2s" if ph > 5.5 else "AC-3s"
        if base_ds == "DS-4":
            return f"AC-3{magnesium_suffix}s" if ph > 5.5 else f"AC-4{magnesium_suffix}s"
        return f"AC-4{magnesium_suffix}s" if ph > 5.5 else f"AC-5{magnesium_suffix}"

    if base_ds == "DS-1":
        if ph > 6.5:
            return "AC-1"
        if ph > 5.5:
            return "AC-2z"
        if ph > 4.5:
            return "AC-3z"
        return "AC-4z"
    if base_ds == "DS-2":
        if ph > 6.5:
            return "AC-2"
        if ph > 5.5:
            return "AC-3z"
        if ph > 4.5:
            return "AC-4z"
        return "AC-5z"
    if base_ds == "DS-3":
        if ph > 6.5:
            return "AC-3"
        if ph > 5.5:
            return "AC-4"
        return "AC-5"
    if base_ds == "DS-4":
        return f"AC-4{magnesium_suffix}" if ph > 6.5 else f"AC-5{magnesium_suffix}"
    return f"AC-5{magnesium_suffix}"


def calculate_bre_summary(data: pd.DataFrame, site_type: str, water_mobility: str) -> dict[str, object]:
    soil = data[data["BRE_SAMPLE_TYPE"] == "Soil"] if "BRE_SAMPLE_TYPE" in data.columns else data
    groundwater = data[data["BRE_SAMPLE_TYPE"] == "Groundwater"] if "BRE_SAMPLE_TYPE" in data.columns else data.iloc[0:0]

    soil_sulfate = characteristic_high(soil["WS_MG_L"], round_to=100 if len(soil["WS_MG_L"].dropna()) >= 10 else None)
    groundwater_sulfate = characteristic_high(groundwater["WS_MG_L"], round_to=100)
    soil_ds = sulphate_class(soil_sulfate, "soil") if soil_sulfate is not None else None
    groundwater_ds = sulphate_class(groundwater_sulfate, "groundwater") if groundwater_sulfate is not None else None

    ph = characteristic_low(data["PH_VALUE"])
    magnesium = characteristic_high(data["MG_MG_L"]) if "MG_MG_L" in data.columns else None
    if site_type.lower() == "brownfield":
        soil_ds = apply_brownfield_magnesium_suffix(soil_ds, magnesium, "soil")
        groundwater_ds = apply_brownfield_magnesium_suffix(groundwater_ds, magnesium, "groundwater")

    ds_class = highest_ds([soil_ds, groundwater_ds])
    acec_class = classify_acec(ds_class, ph, site_type, water_mobility)
    return {
        "Site Type": site_type,
        "Groundwater Mobility": water_mobility,
        "Samples": int(len(data)),
        "Soil WS Characteristic (mg/l SO4)": soil_sulfate,
        "Groundwater SO4 Characteristic (mg/l SO4)": groundwater_sulfate,
        "Characteristic pH": ph,
        "Characteristic Mg (mg/l)": magnesium,
        "Soil DS": soil_ds,
        "Groundwater DS": groundwater_ds,
        "Design Sulfate Class": ds_class,
        "ACEC Class": acec_class,
    }
