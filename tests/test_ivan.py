import pandas as pd

from ags_app.ivan import build_ivan_table


def test_build_ivan_table_matches_geology_by_loca_and_depth() -> None:
    tables = {
        "IVAN": pd.DataFrame(
            [
                {"LOCA_ID": "BH01", "IVAN_DPTH": "0.50", "IVAN_IVAN": "24"},
                {"LOCA_ID": "BH01", "IVAN_DPTH": "1.50", "IVAN_IVAN": "31"},
                {"LOCA_ID": "BH02", "IVAN_DPTH": "0.20", "IVAN_IVAN": "18"},
            ]
        ),
        "GEOL": pd.DataFrame(
            [
                {
                    "LOCA_ID": "BH01",
                    "GEOL_TOP": "0.00",
                    "GEOL_BASE": "1.00",
                    "GEOL_GEOL": "PEAT",
                    "GEOL_DESC": "Peat",
                },
                {
                    "LOCA_ID": "BH01",
                    "GEOL_TOP": "1.00",
                    "GEOL_BASE": "2.00",
                    "GEOL_GEOL": "CLAY",
                    "GEOL_DESC": "Clay",
                },
            ]
        ),
    }

    result = build_ivan_table(tables)

    assert result.loc[0, "GEOL_GEOL"] == "PEAT"
    assert result.loc[1, "GEOL_GEOL"] == "CLAY"
    assert result.loc[2, "GEOL_GEOL"] == "Unmatched"
    assert result["GEOLOGY_MATCHED"].tolist() == [True, True, False]


def test_build_ivan_table_uses_first_number_in_reading() -> None:
    tables = {
        "IVAN": pd.DataFrame(
            [{"LOCA_ID": "BH01", "IVAN_DPTH": "0.30m", "IVAN_IVAN": "22 kPa"}]
        )
    }

    result = build_ivan_table(tables)

    assert result.loc[0, "IVAN_DPTH_NUM"] == 0.3
    assert result.loc[0, "IVAN_IVAN_NUM"] == 22
