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


REQUIRED_CORE_COLUMNS = {"LOCA_ID", "CORE_TOP", "CORE_RQD"}


def build_rqd_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    core = required_table(tables, "CORE", REQUIRED_CORE_COLUMNS).copy()
    geol = tables.get("GEOL", pd.DataFrame()).copy()

    core["LOCA_ID"] = core["LOCA_ID"].astype(str).str.strip()
    core["CORE_TOP_NUM"] = to_number(core["CORE_TOP"])
    core["CORE_RQD_NUM"] = to_number(core["CORE_RQD"])

    rqd = core.dropna(subset=["LOCA_ID", "CORE_TOP_NUM", "CORE_RQD_NUM"]).copy()
    rqd = rqd.sort_values(["LOCA_ID", "CORE_TOP_NUM"]).reset_index(drop=True)

    if REQUIRED_GEOL_COLUMNS.issubset(set(geol.columns)):
        geol = prepare_geology_table(geol)
        rqd = attach_geology_by_depth(rqd, geol, "CORE_TOP_NUM")
    else:
        rqd = add_unmatched_geology(rqd)

    return add_geological_model_fields(rqd)
