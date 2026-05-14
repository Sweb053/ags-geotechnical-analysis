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

REQUIRED_ISPT_COLUMNS = {"LOCA_ID", "ISPT_TOP", "ISPT_MAIN"}


def build_spt_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    ispt = required_table(tables, "ISPT", REQUIRED_ISPT_COLUMNS).copy()
    geol = tables.get("GEOL", pd.DataFrame()).copy()

    ispt["LOCA_ID"] = ispt["LOCA_ID"].astype(str).str.strip()
    ispt["ISPT_TOP_NUM"] = to_number(ispt["ISPT_TOP"])
    ispt["ISPT_MAIN_NUM"] = to_number(ispt["ISPT_MAIN"])
    ispt["ISPT_ERAT_NUM"] = to_number(ispt["ISPT_ERAT"]) if "ISPT_ERAT" in ispt.columns else pd.NA
    ispt["ISPT_N60_NUM"] = ispt["ISPT_MAIN_NUM"] * ispt["ISPT_ERAT_NUM"] / 60

    spt = ispt.dropna(subset=["LOCA_ID", "ISPT_TOP_NUM", "ISPT_MAIN_NUM"]).copy()
    spt = spt.sort_values(["LOCA_ID", "ISPT_TOP_NUM"]).reset_index(drop=True)

    if REQUIRED_GEOL_COLUMNS.issubset(set(geol.columns)):
        geol = prepare_geology_table(geol)
        spt = attach_geology_by_depth(spt, geol, "ISPT_TOP_NUM")
    else:
        spt = add_unmatched_geology(spt)

    return add_geological_model_fields(spt)
