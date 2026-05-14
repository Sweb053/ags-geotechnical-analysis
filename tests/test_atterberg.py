import pandas as pd

from ags_app.atterberg import build_atterberg_table


def test_build_atterberg_table_matches_geology_and_calculates_pi() -> None:
    tables = {
        "LLPL": pd.DataFrame(
            [
                {"LOCA_ID": "BH01", "SAMP_TOP": "0.50", "LLPL_LL": "60", "LLPL_PL": "25", "LLPL_PI": None},
                {"LOCA_ID": "BH01", "SAMP_TOP": "1.50", "LLPL_LL": "72", "LLPL_PL": "30", "LLPL_PI": "41"},
                {"LOCA_ID": "BH02", "SAMP_TOP": "0.20", "LLPL_LL": "50", "LLPL_PL": "NP", "LLPL_PI": None},
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

    result = build_atterberg_table(tables)

    assert result.loc[0, "GEOL_GEOL"] == "PEAT"
    assert result.loc[1, "GEOL_GEOL"] == "CLAY"
    assert result.loc[2, "GEOL_GEOL"] == "Unmatched"
    assert result.loc[0, "LLPL_PI_NUM"] == 35
    assert result.loc[1, "LLPL_PI_NUM"] == 41
    assert pd.isna(result.loc[2, "LLPL_PL_NUM"])


def test_build_atterberg_table_uses_first_number_in_limits() -> None:
    tables = {
        "LLPL": pd.DataFrame(
            [{"LOCA_ID": "BH01", "SAMP_TOP": "1.20m", "LLPL_LL": "55 %", "LLPL_PL": "22 %"}]
        )
    }

    result = build_atterberg_table(tables)

    assert result.loc[0, "SAMP_TOP_NUM"] == 1.2
    assert result.loc[0, "LLPL_LL_NUM"] == 55
    assert result.loc[0, "LLPL_PL_NUM"] == 22
    assert result.loc[0, "LLPL_PI_NUM"] == 33
