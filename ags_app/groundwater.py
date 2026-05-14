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


REQUIRED_WSTG_COLUMNS = {"LOCA_ID", "WSTG_DPTH"}


def build_groundwater_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    wstg = required_table(tables, "WSTG", REQUIRED_WSTG_COLUMNS).copy()
    geol = tables.get("GEOL", pd.DataFrame()).copy()

    wstg["LOCA_ID"] = wstg["LOCA_ID"].astype(str).str.strip()
    wstg["WSTG_DPTH_NUM"] = to_number(wstg["WSTG_DPTH"])

    groundwater = wstg.dropna(subset=["LOCA_ID", "WSTG_DPTH_NUM"]).copy()
    groundwater = attach_post_strike_readings(groundwater, tables.get("WSTD", pd.DataFrame()))
    groundwater = groundwater.sort_values(["LOCA_ID", "WSTG_DPTH_NUM"]).reset_index(drop=True)
    groundwater["GROUNDWATER_PLOT_NUM"] = pd.factorize(groundwater["LOCA_ID"])[0] + 1

    if REQUIRED_GEOL_COLUMNS.issubset(set(geol.columns)):
        geol = prepare_geology_table(geol)
        groundwater = attach_geology_by_depth(groundwater, geol, "WSTG_DPTH_NUM")
    else:
        groundwater = add_unmatched_geology(groundwater)

    return add_geological_model_fields(groundwater)


def attach_post_strike_readings(wstg: pd.DataFrame, wstd: pd.DataFrame) -> pd.DataFrame:
    if wstd.empty or not {"LOCA_ID", "WSTG_DPTH", "WSTD_POST"}.issubset(set(wstd.columns)):
        result = wstg.copy()
        result["WSTD_POST"] = pd.NA
        result["WSTD_POST_NUM"] = pd.NA
        result["WSTD_NMIN"] = pd.NA
        return result

    readings = wstd.copy()
    readings["LOCA_ID"] = readings["LOCA_ID"].astype(str).str.strip()
    readings["WSTG_DPTH_NUM"] = to_number(readings["WSTG_DPTH"])
    readings["WSTD_POST_NUM"] = to_number(readings["WSTD_POST"])

    aggregation = {
        "WSTD_POST": "first",
        "WSTD_POST_NUM": "first",
    }
    if "WSTD_NMIN" in readings.columns:
        aggregation["WSTD_NMIN"] = "first"

    grouped = (
        readings.dropna(subset=["LOCA_ID", "WSTG_DPTH_NUM"])
        .sort_values(["LOCA_ID", "WSTG_DPTH_NUM"])
        .groupby(["LOCA_ID", "WSTG_DPTH_NUM"], as_index=False)
        .agg(aggregation)
    )
    if "WSTD_NMIN" not in grouped.columns:
        grouped["WSTD_NMIN"] = pd.NA
    return wstg.merge(grouped, on=["LOCA_ID", "WSTG_DPTH_NUM"], how="left")
