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


REQUIRED_LNMC_COLUMNS = {"LOCA_ID", "SAMP_TOP", "LNMC_MC"}


def build_moisture_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    lnmc = required_table(tables, "LNMC", REQUIRED_LNMC_COLUMNS).copy()
    geol = tables.get("GEOL", pd.DataFrame()).copy()

    lnmc["LOCA_ID"] = lnmc["LOCA_ID"].astype(str).str.strip()
    lnmc["SAMP_TOP_NUM"] = to_number(lnmc["SAMP_TOP"])
    lnmc["LNMC_MC_NUM"] = to_number(lnmc["LNMC_MC"])

    moisture = lnmc.dropna(subset=["LOCA_ID", "SAMP_TOP_NUM", "LNMC_MC_NUM"]).copy()
    moisture = moisture.sort_values(["LOCA_ID", "SAMP_TOP_NUM"]).reset_index(drop=True)

    if REQUIRED_GEOL_COLUMNS.issubset(set(geol.columns)):
        geol = prepare_geology_table(geol)
        moisture = attach_geology_by_depth(moisture, geol, "SAMP_TOP_NUM")
    else:
        moisture = add_unmatched_geology(moisture)

    return add_geological_model_fields(moisture)
