import pandas as pd

from ags_app.groundwater import build_groundwater_table


def test_build_groundwater_table_matches_geology_by_strike_depth() -> None:
    tables = {
        "WSTG": pd.DataFrame(
            [
                {"LOCA_ID": "BH01", "WSTG_DPTH": "1.20", "WSTG_DTIM": "2026-01-01"},
                {"LOCA_ID": "BH01", "WSTG_DPTH": "3.00", "WSTG_DTIM": "2026-01-01"},
                {"LOCA_ID": "BH02", "WSTG_DPTH": "2.00", "WSTG_DTIM": "2026-01-01"},
            ]
        ),
        "WSTD": pd.DataFrame(
            [
                {"LOCA_ID": "BH01", "WSTG_DPTH": "1.20", "WSTD_NMIN": "20", "WSTD_POST": "0.80"},
                {"LOCA_ID": "BH01", "WSTG_DPTH": "3.00", "WSTD_NMIN": "20", "WSTD_POST": "2.40"},
            ]
        ),
        "GEOL": pd.DataFrame(
            [
                {
                    "LOCA_ID": "BH01",
                    "GEOL_TOP": "0.00",
                    "GEOL_BASE": "2.00",
                    "GEOL_GEOL": "SAND",
                    "GEOL_DESC": "Sand",
                },
                {
                    "LOCA_ID": "BH01",
                    "GEOL_TOP": "2.00",
                    "GEOL_BASE": "4.00",
                    "GEOL_GEOL": "ROCK",
                    "GEOL_DESC": "Rock",
                },
            ]
        ),
    }

    result = build_groundwater_table(tables)

    assert result.loc[0, "GEOL_GEOL"] == "SAND"
    assert result.loc[0, "WSTD_POST_NUM"] == 0.8
    assert result.loc[1, "GEOL_GEOL"] == "ROCK"
    assert result.loc[1, "WSTD_POST_NUM"] == 2.4
    assert result.loc[2, "GEOL_GEOL"] == "Unmatched"


def test_build_groundwater_table_uses_first_number_in_depth() -> None:
    tables = {
        "WSTG": pd.DataFrame(
            [{"LOCA_ID": "BH01", "WSTG_DPTH": "4.10m"}]
        )
    }

    result = build_groundwater_table(tables)

    assert result.loc[0, "WSTG_DPTH_NUM"] == 4.1
    assert pd.isna(result.loc[0, "WSTD_POST_NUM"])
