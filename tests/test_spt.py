import pandas as pd

from ags_app.spt import build_spt_table


def test_build_spt_table_matches_geology_by_loca_and_depth() -> None:
    tables = {
        "ISPT": pd.DataFrame(
            [
                {"LOCA_ID": "BH01", "ISPT_TOP": "1.20", "ISPT_MAIN": "12"},
                {"LOCA_ID": "BH01", "ISPT_TOP": "2.20", "ISPT_MAIN": "18"},
                {"LOCA_ID": "BH02", "ISPT_TOP": "0.50", "ISPT_MAIN": "7"},
            ]
        ),
        "GEOL": pd.DataFrame(
            [
                {
                    "LOCA_ID": "BH01",
                    "GEOL_TOP": "0.00",
                    "GEOL_BASE": "2.20",
                    "GEOL_GEOL": "CLAY",
                    "GEOL_DESC": "Clay",
                },
                {
                    "LOCA_ID": "BH01",
                    "GEOL_TOP": "2.20",
                    "GEOL_BASE": "4.00",
                    "GEOL_GEOL": "GRAVEL",
                    "GEOL_DESC": "Gravel",
                },
            ]
        ),
    }

    result = build_spt_table(tables)

    assert result.loc[0, "GEOL_GEOL"] == "CLAY"
    assert result.loc[1, "GEOL_GEOL"] == "GRAVEL"
    assert result.loc[2, "GEOL_GEOL"] == "Unmatched"
    assert result["GEOLOGY_MATCHED"].tolist() == [True, True, False]


def test_build_spt_table_uses_first_number_in_blow_count() -> None:
    tables = {
        "ISPT": pd.DataFrame(
            [{"LOCA_ID": "BH01", "ISPT_TOP": "1.20m", "ISPT_MAIN": "50 blows"}]
        )
    }

    result = build_spt_table(tables)

    assert result.loc[0, "ISPT_TOP_NUM"] == 1.2
    assert result.loc[0, "ISPT_MAIN_NUM"] == 50


def test_build_spt_table_calculates_n60_from_energy_ratio() -> None:
    tables = {
        "ISPT": pd.DataFrame(
            [
                {"LOCA_ID": "BH01", "ISPT_TOP": "1.20", "ISPT_MAIN": "30", "ISPT_ERAT": "70"},
                {"LOCA_ID": "BH01", "ISPT_TOP": "2.20", "ISPT_MAIN": "18", "ISPT_ERAT": "60"},
            ]
        )
    }

    result = build_spt_table(tables)

    assert result.loc[0, "ISPT_ERAT_NUM"] == 70
    assert result.loc[0, "ISPT_N60_NUM"] == 35
    assert result.loc[1, "ISPT_N60_NUM"] == 18


def test_build_spt_table_keeps_n60_blank_without_energy_ratio() -> None:
    tables = {
        "ISPT": pd.DataFrame(
            [{"LOCA_ID": "BH01", "ISPT_TOP": "1.20", "ISPT_MAIN": "30"}]
        )
    }

    result = build_spt_table(tables)

    assert pd.isna(result.loc[0, "ISPT_ERAT_NUM"])
    assert pd.isna(result.loc[0, "ISPT_N60_NUM"])
