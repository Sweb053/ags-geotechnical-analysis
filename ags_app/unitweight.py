from __future__ import annotations

import pandas as pd

from ags_app.geolmodel import add_geological_model_fields
from ags_app.common import (
    REQUIRED_GEOL_COLUMNS,
    add_unmatched_geology,
    attach_geology_by_depth,
    prepare_geology_table,
    required_table,
    to_number,
)


REQUIRED_LDEN_COLUMNS = {"LOCA_ID", "SAMP_TOP"}
WATER_UNIT_WEIGHT = 9.81


def build_unit_weight_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    lden = required_table(tables, "LDEN", REQUIRED_LDEN_COLUMNS).copy()
    geol = tables.get("GEOL", pd.DataFrame()).copy()

    if "LDEN_BDEN" not in lden.columns and "LDEN_DDEN" not in lden.columns:
        raise ValueError("LDEN is missing required density columns: LDEN_BDEN or LDEN_DDEN")

    lden["LOCA_ID"] = lden["LOCA_ID"].astype(str).str.strip()
    lden["SAMP_TOP_NUM"] = to_number(lden["SAMP_TOP"])
    lden["LDEN_BDEN_NUM"] = to_number(lden["LDEN_BDEN"]) if "LDEN_BDEN" in lden.columns else pd.NA
    lden["LDEN_DDEN_NUM"] = to_number(lden["LDEN_DDEN"]) if "LDEN_DDEN" in lden.columns else pd.NA
    lden["LDEN_MC_NUM"] = to_number(lden["LDEN_MC"]) if "LDEN_MC" in lden.columns else pd.NA
    lden["BULK_UNIT_WEIGHT_NUM"] = lden["LDEN_BDEN_NUM"] * WATER_UNIT_WEIGHT
    lden["DRY_UNIT_WEIGHT_NUM"] = lden["LDEN_DDEN_NUM"] * WATER_UNIT_WEIGHT

    unit_weight = lden.dropna(subset=["LOCA_ID", "SAMP_TOP_NUM"]).copy()
    unit_weight = unit_weight.dropna(subset=["BULK_UNIT_WEIGHT_NUM", "DRY_UNIT_WEIGHT_NUM"], how="all").copy()
    unit_weight = unit_weight.sort_values(["LOCA_ID", "SAMP_TOP_NUM"]).reset_index(drop=True)

    if REQUIRED_GEOL_COLUMNS.issubset(set(geol.columns)):
        geol = prepare_geology_table(geol)
        unit_weight = attach_geology_by_depth(unit_weight, geol, "SAMP_TOP_NUM")
    else:
        unit_weight = add_unmatched_geology(unit_weight)

    return add_geological_model_fields(unit_weight)
