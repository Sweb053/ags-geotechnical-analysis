import pandas as pd

from ags_app.ucs import build_ucs_table


def test_build_ucs_table_matches_geology_by_loca_and_depth() -> None:
    tables = {
        "RUCS": pd.DataFrame(
            [
                {"LOCA_ID": "BH01", "SAMP_TOP": "1.20", "RUCS_UCS": "12.5"},
                {"LOCA_ID": "BH01", "SAMP_TOP": "3.00", "RUCS_UCS": "42.1"},
                {"LOCA_ID": "BH02", "SAMP_TOP": "2.00", "RUCS_UCS": "8.4"},
            ]
        ),
        "GEOL": pd.DataFrame(
            [
                {
                    "LOCA_ID": "BH01",
                    "GEOL_TOP": "0.00",
                    "GEOL_BASE": "2.00",
                    "GEOL_GEOL": "CLAY",
                    "GEOL_DESC": "Clay",
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

    result = build_ucs_table(tables)

    assert result.loc[0, "GEOL_GEOL"] == "CLAY"
    assert result.loc[1, "GEOL_GEOL"] == "ROCK"
    assert result.loc[2, "GEOL_GEOL"] == "Unmatched"
    assert result["GEOLOGY_MATCHED"].tolist() == [True, True, False]


def test_build_ucs_table_uses_first_number_in_ucs() -> None:
    tables = {
        "RUCS": pd.DataFrame(
            [{"LOCA_ID": "BH01", "SAMP_TOP": "4.10m", "RUCS_UCS": "91.5 MPa"}]
        )
    }

    result = build_ucs_table(tables)

    assert result.loc[0, "SAMP_TOP_NUM"] == 4.1
    assert result.loc[0, "RUCS_UCS_NUM"] == 91.5
