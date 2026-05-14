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


REQUIRED_RPLT_COLUMNS = {"LOCA_ID", "SAMP_TOP", "RPLT_PLSI"}


def build_pointload_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rplt = required_table(tables, "RPLT", REQUIRED_RPLT_COLUMNS).copy()
    geol = tables.get("GEOL", pd.DataFrame()).copy()

    rplt["LOCA_ID"] = rplt["LOCA_ID"].astype(str).str.strip()
    rplt["SAMP_TOP_NUM"] = to_number(rplt["SAMP_TOP"])
    rplt["RPLT_PLSI_NUM"] = to_number(rplt["RPLT_PLSI"])

    if "SPEC_DPTH" in rplt.columns:
        rplt["SPEC_DPTH_NUM"] = to_number(rplt["SPEC_DPTH"])
    else:
        rplt["SPEC_DPTH"] = pd.NA
        rplt["SPEC_DPTH_NUM"] = pd.Series(index=rplt.index, dtype="float64")

    rplt["POINTLOAD_DEPTH_NUM"] = rplt["SPEC_DPTH_NUM"].combine_first(rplt["SAMP_TOP_NUM"])

    pointload = rplt.dropna(subset=["LOCA_ID", "POINTLOAD_DEPTH_NUM", "RPLT_PLSI_NUM"]).copy()
    pointload = pointload.sort_values(["LOCA_ID", "POINTLOAD_DEPTH_NUM"]).reset_index(drop=True)

    if REQUIRED_GEOL_COLUMNS.issubset(set(geol.columns)):
        geol = prepare_geology_table(geol)
        pointload = attach_geology_by_depth(pointload, geol, "POINTLOAD_DEPTH_NUM")
    else:
        pointload = add_unmatched_geology(pointload)

    return add_geological_model_fields(pointload)
