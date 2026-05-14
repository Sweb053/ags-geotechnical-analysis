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


REQUIRED_RUCS_COLUMNS = {"LOCA_ID", "SAMP_TOP", "RUCS_UCS"}


def build_ucs_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rucs = required_table(tables, "RUCS", REQUIRED_RUCS_COLUMNS).copy()
    geol = tables.get("GEOL", pd.DataFrame()).copy()

    rucs["LOCA_ID"] = rucs["LOCA_ID"].astype(str).str.strip()
    rucs["SAMP_TOP_NUM"] = to_number(rucs["SAMP_TOP"])
    rucs["RUCS_UCS_NUM"] = to_number(rucs["RUCS_UCS"])

    ucs = rucs.dropna(subset=["LOCA_ID", "SAMP_TOP_NUM", "RUCS_UCS_NUM"]).copy()
    ucs = ucs.sort_values(["LOCA_ID", "SAMP_TOP_NUM"]).reset_index(drop=True)

    if REQUIRED_GEOL_COLUMNS.issubset(set(geol.columns)):
        geol = prepare_geology_table(geol)
        ucs = attach_geology_by_depth(ucs, geol, "SAMP_TOP_NUM")
    else:
        ucs = add_unmatched_geology(ucs)

    return add_geological_model_fields(ucs)
