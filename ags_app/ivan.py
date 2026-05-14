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


REQUIRED_IVAN_COLUMNS = {"LOCA_ID", "IVAN_DPTH", "IVAN_IVAN"}


def build_ivan_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    ivan = required_table(tables, "IVAN", REQUIRED_IVAN_COLUMNS).copy()
    geol = tables.get("GEOL", pd.DataFrame()).copy()

    ivan["LOCA_ID"] = ivan["LOCA_ID"].astype(str).str.strip()
    ivan["IVAN_DPTH_NUM"] = to_number(ivan["IVAN_DPTH"])
    ivan["IVAN_IVAN_NUM"] = to_number(ivan["IVAN_IVAN"])

    vane = ivan.dropna(subset=["LOCA_ID", "IVAN_DPTH_NUM", "IVAN_IVAN_NUM"]).copy()
    vane = vane.sort_values(["LOCA_ID", "IVAN_DPTH_NUM"]).reset_index(drop=True)

    if REQUIRED_GEOL_COLUMNS.issubset(set(geol.columns)):
        geol = prepare_geology_table(geol)
        vane = attach_geology_by_depth(vane, geol, "IVAN_DPTH_NUM")
    else:
        vane = add_unmatched_geology(vane)

    return add_geological_model_fields(vane)
