import pandas as pd

from ags_app.rqd import build_rqd_table


def test_build_rqd_table_matches_geology_by_loca_and_depth() -> None:
    tables = {
        "CORE": pd.DataFrame(
            [
                {"LOCA_ID": "BH01", "CORE_TOP": "1.20", "CORE_BASE": "2.20", "CORE_RQD": "45"},
                {"LOCA_ID": "BH01", "CORE_TOP": "3.00", "CORE_BASE": "4.00", "CORE_RQD": "82"},
                {"LOCA_ID": "BH02", "CORE_TOP": "2.00", "CORE_BASE": "3.00", "CORE_RQD": "10"},
            ]
        ),
        "GEOL": pd.DataFrame(
            [
                {
                    "LOCA_ID": "BH01",
                    "GEOL_TOP": "0.00",
                    "GEOL_BASE": "2.00",
                    "GEOL_GEOL": "WEATHERED",
                    "GEOL_DESC": "Weathered rock",
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

    result = build_rqd_table(tables)

    assert result.loc[0, "GEOL_GEOL"] == "WEATHERED"
    assert result.loc[1, "GEOL_GEOL"] == "ROCK"
    assert result.loc[2, "GEOL_GEOL"] == "Unmatched"
    assert result["GEOLOGY_MATCHED"].tolist() == [True, True, False]


def test_build_rqd_table_uses_first_number_in_rqd() -> None:
    tables = {
        "CORE": pd.DataFrame(
            [{"LOCA_ID": "BH01", "CORE_TOP": "4.10m", "CORE_RQD": "91 %"}]
        )
    }

    result = build_rqd_table(tables)

    assert result.loc[0, "CORE_TOP_NUM"] == 4.1
    assert result.loc[0, "CORE_RQD_NUM"] == 91
