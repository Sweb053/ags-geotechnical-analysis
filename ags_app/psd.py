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


REQUIRED_GRAT_COLUMNS = {"LOCA_ID", "SAMP_TOP", "GRAT_SIZE", "GRAT_PERP"}


def build_psd_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    grat = required_table(tables, "GRAT", REQUIRED_GRAT_COLUMNS).copy()
    geol = tables.get("GEOL", pd.DataFrame()).copy()

    grat["LOCA_ID"] = grat["LOCA_ID"].astype(str).str.strip()
    grat["SAMP_TOP_NUM"] = to_number(grat["SAMP_TOP"])
    grat["GRAT_SIZE_NUM"] = to_number(grat["GRAT_SIZE"])
    grat["GRAT_PERP_NUM"] = to_number(grat["GRAT_PERP"])

    if "SPEC_DPTH" in grat.columns:
        grat["SPEC_DPTH_NUM"] = to_number(grat["SPEC_DPTH"])
    else:
        grat["SPEC_DPTH"] = pd.NA
        grat["SPEC_DPTH_NUM"] = pd.Series(index=grat.index, dtype="float64")

    grat["PSD_DEPTH_NUM"] = grat["SPEC_DPTH_NUM"].combine_first(grat["SAMP_TOP_NUM"])
    grat["PSD_SAMPLE_ID"] = build_sample_ids(grat)

    psd = grat.dropna(subset=["LOCA_ID", "PSD_DEPTH_NUM", "GRAT_SIZE_NUM", "GRAT_PERP_NUM"]).copy()
    psd = psd[(psd["GRAT_SIZE_NUM"] > 0) & (psd["GRAT_PERP_NUM"].between(0, 100))].copy()
    psd = psd.sort_values(["LOCA_ID", "PSD_DEPTH_NUM", "PSD_SAMPLE_ID", "GRAT_SIZE_NUM"]).reset_index(drop=True)

    if REQUIRED_GEOL_COLUMNS.issubset(set(geol.columns)):
        geol = prepare_geology_table(geol)
        psd = attach_geology_by_depth(psd, geol, "PSD_DEPTH_NUM")
    else:
        psd = add_unmatched_geology(psd)

    return add_geological_model_fields(psd)


def build_sample_ids(data: pd.DataFrame) -> pd.Series:
    parts = [data["LOCA_ID"].astype(str).str.strip(), data["SAMP_TOP"].astype(str).str.strip()]
    for column in ("SAMP_REF", "SAMP_TYPE", "SAMP_ID", "SPEC_REF"):
        if column in data.columns:
            parts.append(data[column].fillna("").astype(str).str.strip())

    labels = parts[0] + " @ " + parts[1] + "m"
    for part in parts[2:]:
        labels = labels + " " + part
    return labels.str.replace(r"\s+", " ", regex=True).str.strip()
