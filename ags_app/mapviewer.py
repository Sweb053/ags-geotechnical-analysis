from __future__ import annotations

import pandas as pd

from ags_app.common import required_table, to_number


REQUIRED_LOCA_COLUMNS = {"LOCA_ID"}
REQUIRED_GEOL_COLUMNS = {"LOCA_ID", "GEOL_TOP", "GEOL_BASE", "GEOL_GEOL"}
COORDINATE_PAIRS = [
    ("LOCA_LON", "LOCA_LAT", "Longitude", "Latitude"),
    ("LOCA_ELON", "LOCA_ELAT", "Longitude", "Latitude"),
    ("LOCA_NATE", "LOCA_NATN", "Easting", "Northing"),
    ("LOCA_LOCX", "LOCA_LOCY", "X", "Y"),
]


def build_map_locations(tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, str, str, str, str]:
    loca = required_table(tables, "LOCA", REQUIRED_LOCA_COLUMNS).copy()
    loca["LOCA_ID"] = loca["LOCA_ID"].astype(str).str.strip()

    x_column, y_column, x_label, y_label = choose_coordinate_pair(loca)
    loca["MAP_X"] = to_number(loca[x_column])
    loca["MAP_Y"] = to_number(loca[y_column])

    columns = ["LOCA_ID", "MAP_X", "MAP_Y", x_column, y_column]
    for optional in ("LOCA_TYPE", "LOCA_FDEP", "LOCA_GL", "LOCA_REM"):
        if optional in loca.columns:
            columns.append(optional)

    locations = loca.dropna(subset=["LOCA_ID", "MAP_X", "MAP_Y"])[columns].copy()
    locations = locations.sort_values("LOCA_ID").reset_index(drop=True)
    if locations.empty:
        raise ValueError("LOCA does not contain plottable coordinate values.")
    return locations, "MAP_X", "MAP_Y", x_label, y_label


def choose_coordinate_pair(loca: pd.DataFrame) -> tuple[str, str, str, str]:
    for x_column, y_column, x_label, y_label in COORDINATE_PAIRS:
        if x_column not in loca.columns or y_column not in loca.columns:
            continue
        x_values = to_number(loca[x_column])
        y_values = to_number(loca[y_column])
        if int(x_values.notna().sum()) > 0 and int(y_values.notna().sum()) > 0:
            return x_column, y_column, x_label, y_label
    raise ValueError("LOCA is missing supported coordinate pairs: latitude/longitude, easting/northing, or X/Y.")


def build_geology_intervals(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    geol = required_table(tables, "GEOL", REQUIRED_GEOL_COLUMNS).copy()
    geol["LOCA_ID"] = geol["LOCA_ID"].astype(str).str.strip()
    geol["GEOL_GEOL"] = geol["GEOL_GEOL"].fillna("Unspecified").astype(str).str.strip()
    geol["GEOL_TOP_NUM"] = to_number(geol["GEOL_TOP"])
    geol["GEOL_BASE_NUM"] = to_number(geol["GEOL_BASE"])
    geol["THICKNESS_NUM"] = geol["GEOL_BASE_NUM"] - geol["GEOL_TOP_NUM"]
    geol = geol.dropna(subset=["LOCA_ID", "GEOL_GEOL", "GEOL_TOP_NUM", "GEOL_BASE_NUM"]).copy()
    return geol.sort_values(["LOCA_ID", "GEOL_TOP_NUM"]).reset_index(drop=True)
