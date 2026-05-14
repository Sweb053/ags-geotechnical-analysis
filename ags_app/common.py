from __future__ import annotations

import re

import pandas as pd


REQUIRED_GEOL_COLUMNS = {"LOCA_ID", "GEOL_TOP", "GEOL_BASE", "GEOL_GEOL"}


def to_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(series.map(_first_number))


def required_table(tables: dict[str, pd.DataFrame], group: str, columns: set[str]) -> pd.DataFrame:
    table = tables.get(group, pd.DataFrame())
    missing = sorted(columns - set(table.columns))
    if table.empty:
        raise ValueError(f"Missing required AGS group: {group}")
    if missing:
        raise ValueError(f"{group} is missing required columns: {', '.join(missing)}")
    return table


def prepare_geology_table(geol: pd.DataFrame) -> pd.DataFrame:
    geol = geol.copy()
    geol["LOCA_ID"] = geol["LOCA_ID"].astype(str).str.strip()
    geol["GEOL_TOP_NUM"] = to_number(geol["GEOL_TOP"])
    geol["GEOL_BASE_NUM"] = to_number(geol["GEOL_BASE"])
    geol["GEOL_GEOL"] = geol["GEOL_GEOL"].fillna("Unspecified").astype(str).str.strip()
    return geol.dropna(subset=["LOCA_ID", "GEOL_TOP_NUM", "GEOL_BASE_NUM"])


def add_unmatched_geology(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["GEOL_GEOL"] = "Unmatched"
    data["GEOL_TOP"] = pd.NA
    data["GEOL_BASE"] = pd.NA
    data["GEOL_DESC"] = pd.NA
    data["GEOLOGY_MATCHED"] = False
    return data


def attach_geology_by_depth(data: pd.DataFrame, geol: pd.DataFrame, depth_column: str) -> pd.DataFrame:
    geol_columns = [
        "LOCA_ID",
        "GEOL_TOP",
        "GEOL_BASE",
        "GEOL_DESC",
        "GEOL_GEOL",
        "GEOL_TOP_NUM",
        "GEOL_BASE_NUM",
    ]
    geol = geol[[column for column in geol_columns if column in geol.columns]].copy()

    matched = []
    for loca_id, data_group in data.groupby("LOCA_ID", dropna=False, sort=False):
        geol_group = geol[geol["LOCA_ID"] == loca_id].sort_values("GEOL_TOP_NUM")
        data_group = data_group.sort_values(depth_column)

        if geol_group.empty:
            matched.append(add_unmatched_geology(data_group))
            continue

        merged = pd.merge_asof(
            data_group,
            geol_group.drop(columns=["LOCA_ID"]),
            left_on=depth_column,
            right_on="GEOL_TOP_NUM",
            direction="backward",
        )
        deepest_base = geol_group["GEOL_BASE_NUM"].max()
        in_interval = (
            (merged[depth_column] >= merged["GEOL_TOP_NUM"])
            & (
                (merged[depth_column] < merged["GEOL_BASE_NUM"])
                | (
                    (merged["GEOL_BASE_NUM"] == deepest_base)
                    & (merged[depth_column] == merged["GEOL_BASE_NUM"])
                )
            )
        )
        merged["GEOLOGY_MATCHED"] = in_interval.fillna(False)
        merged.loc[~merged["GEOLOGY_MATCHED"], ["GEOL_TOP", "GEOL_BASE", "GEOL_DESC"]] = pd.NA
        merged.loc[~merged["GEOLOGY_MATCHED"], "GEOL_GEOL"] = "Unmatched"
        merged["GEOL_GEOL"] = merged["GEOL_GEOL"].fillna("Unspecified")
        matched.append(merged)

    if not matched:
        return add_unmatched_geology(data)

    return pd.concat(matched, ignore_index=True).sort_values(["LOCA_ID", depth_column])


def _first_number(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    match = re.search(r"[-+]?\d*\.?\d+", str(value))
    return float(match.group(0)) if match else None
