import pandas as pd

from app import build_aerial_map_data, build_geology_summary, build_map_geology_labels, merge_geology_intervals
from ags_app.mapviewer import build_geology_intervals, build_map_locations


def test_build_map_locations_uses_easting_northing_when_available() -> None:
    tables = {
        "LOCA": pd.DataFrame(
            [
                {"LOCA_ID": "BH02", "LOCA_NATE": "101.0", "LOCA_NATN": "202.0", "LOCA_TYPE": "RC"},
                {"LOCA_ID": "BH01", "LOCA_NATE": "100.0m", "LOCA_NATN": "200.0m", "LOCA_TYPE": "CP"},
            ]
        )
    }

    locations, x_column, y_column, x_label, y_label = build_map_locations(tables)

    assert x_column == "MAP_X"
    assert y_column == "MAP_Y"
    assert x_label == "Easting"
    assert y_label == "Northing"
    assert locations["LOCA_ID"].tolist() == ["BH01", "BH02"]
    assert locations.loc[0, "MAP_X"] == 100.0
    assert locations.loc[0, "MAP_Y"] == 200.0


def test_build_geology_intervals_calculates_thickness() -> None:
    tables = {
        "GEOL": pd.DataFrame(
            [
                {"LOCA_ID": "BH01", "GEOL_TOP": "0", "GEOL_BASE": "1.2", "GEOL_GEOL": "PEAT"},
                {"LOCA_ID": "BH01", "GEOL_TOP": "1.2", "GEOL_BASE": "2.5", "GEOL_GEOL": "GT"},
            ]
        )
    }

    intervals = build_geology_intervals(tables)

    assert intervals.loc[0, "GEOL_GEOL"] == "PEAT"
    assert intervals.loc[0, "THICKNESS_NUM"] == 1.2
    assert intervals.loc[1, "THICKNESS_NUM"] == 1.3


def test_map_geology_summary_merges_touching_intervals() -> None:
    geology = pd.DataFrame(
        [
            {"LOCA_ID": "BH01", "GEOL_GEOL": "PEAT", "GEOL_TOP_NUM": 0.0, "GEOL_BASE_NUM": 0.2, "GEOL_DESC": "Peat 1"},
            {"LOCA_ID": "BH01", "GEOL_GEOL": "PEAT", "GEOL_TOP_NUM": 0.2, "GEOL_BASE_NUM": 1.0, "GEOL_DESC": "Peat 2"},
            {"LOCA_ID": "BH01", "GEOL_GEOL": "PEAT", "GEOL_TOP_NUM": 1.4, "GEOL_BASE_NUM": 1.8, "GEOL_DESC": "Peat 3"},
            {"LOCA_ID": "BH04", "GEOL_GEOL": "PEAT", "GEOL_TOP_NUM": 0.0, "GEOL_BASE_NUM": 0.5, "GEOL_DESC": "Peat 4"},
        ]
    )

    merged = merge_geology_intervals(geology)
    summary = build_geology_summary(geology)
    labels = build_map_geology_labels(merged)

    assert merged[merged["LOCA_ID"] == "BH01"]["GEOL_TOP_NUM"].tolist() == [0.0, 1.4]
    assert merged[merged["LOCA_ID"] == "BH01"]["GEOL_BASE_NUM"].tolist() == [1.0, 1.8]
    assert summary.loc[0, "Depth range"] == "0-1 m"
    assert "PEAT 0-1m" in labels["BH01"]
    assert "PEAT 0-0.5m" in labels["BH04"]


def test_build_aerial_map_data_converts_british_grid_to_lat_lon_and_labels_geology() -> None:
    locations = pd.DataFrame(
        [
            {"LOCA_ID": "BH01", "MAP_X": 189092.88, "MAP_Y": 811600.90},
            {"LOCA_ID": "BH02", "MAP_X": 192141.50, "MAP_Y": 809293.55},
        ]
    )
    geology = pd.DataFrame(
        [
            {"LOCA_ID": "BH01", "GEOL_GEOL": "PEAT", "GEOL_TOP_NUM": 0.0, "GEOL_BASE_NUM": 1.0},
        ]
    )

    result = build_aerial_map_data(locations, geology, "MAP_X", "MAP_Y", "Easting", "Northing")

    assert len(result) == 2
    assert result["LATITUDE"].between(57, 58).all()
    assert result["LONGITUDE"].between(-7, -5).all()
    assert result.loc[result["LOCA_ID"] == "BH01", "HAS_SELECTED_GEOLOGY"].iloc[0]
    assert "PEAT 0-1m" in result.loc[result["LOCA_ID"] == "BH01", "MAP_LABEL"].iloc[0]
