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


REQUIRED_LLPL_COLUMNS = {"LOCA_ID", "SAMP_TOP", "LLPL_LL", "LLPL_PL"}


def build_atterberg_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    llpl = required_table(tables, "LLPL", REQUIRED_LLPL_COLUMNS).copy()
    geol = tables.get("GEOL", pd.DataFrame()).copy()

    llpl["LOCA_ID"] = llpl["LOCA_ID"].astype(str).str.strip()
    llpl["SAMP_TOP_NUM"] = to_number(llpl["SAMP_TOP"])
    llpl["LLPL_LL_NUM"] = to_number(llpl["LLPL_LL"])
    llpl["LLPL_PL_NUM"] = to_number(llpl["LLPL_PL"])

    if "LLPL_PI" in llpl.columns:
        pi_from_field = to_number(llpl["LLPL_PI"])
    else:
        llpl["LLPL_PI"] = pd.NA
        pi_from_field = pd.Series(index=llpl.index, dtype="float64")

    calculated_pi = llpl["LLPL_LL_NUM"] - llpl["LLPL_PL_NUM"]
    llpl["LLPL_PI_NUM"] = pi_from_field.combine_first(calculated_pi)

    value_columns = ["LLPL_LL_NUM", "LLPL_PL_NUM", "LLPL_PI_NUM"]
    atterberg = llpl.dropna(subset=["LOCA_ID", "SAMP_TOP_NUM"]).copy()
    atterberg = atterberg.dropna(subset=value_columns, how="all").copy()
    atterberg = atterberg.sort_values(["LOCA_ID", "SAMP_TOP_NUM"]).reset_index(drop=True)

    if REQUIRED_GEOL_COLUMNS.issubset(set(geol.columns)):
        geol = prepare_geology_table(geol)
        atterberg = attach_geology_by_depth(atterberg, geol, "SAMP_TOP_NUM")
    else:
        atterberg = add_unmatched_geology(atterberg)

    return add_geological_model_fields(atterberg)
