import pandas as pd

from ags_app.psd import build_psd_table


def test_build_psd_table_matches_geology_and_groups_curve_points() -> None:
    tables = {
        "GRAT": pd.DataFrame(
            [
                {"LOCA_ID": "BH01", "SAMP_TOP": "1.20", "SAMP_REF": "1", "GRAT_SIZE": "0.063", "GRAT_PERP": "22"},
                {"LOCA_ID": "BH01", "SAMP_TOP": "1.20", "SAMP_REF": "1", "GRAT_SIZE": "2.0", "GRAT_PERP": "71"},
                {"LOCA_ID": "BH02", "SAMP_TOP": "3.00", "SAMP_REF": "2", "GRAT_SIZE": "0.063", "GRAT_PERP": "12"},
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
                }
            ]
        ),
    }

    result = build_psd_table(tables)

    assert result.loc[0, "GEOL_GEOL"] == "CLAY"
    assert result.loc[1, "GEOL_GEOL"] == "CLAY"
    assert result.loc[2, "GEOL_GEOL"] == "Unmatched"
    assert result.loc[0, "PSD_SAMPLE_ID"].startswith("BH01 @ 1.20m")
    assert result["GRAT_SIZE_NUM"].tolist() == [0.063, 2.0, 0.063]


def test_build_psd_table_filters_invalid_sizes_and_percentages() -> None:
    tables = {
        "GRAT": pd.DataFrame(
            [
                {"LOCA_ID": "BH01", "SAMP_TOP": "1.20", "GRAT_SIZE": "0", "GRAT_PERP": "50"},
                {"LOCA_ID": "BH01", "SAMP_TOP": "1.20", "GRAT_SIZE": "0.063 mm", "GRAT_PERP": "101"},
                {"LOCA_ID": "BH01", "SAMP_TOP": "1.20", "GRAT_SIZE": "2 mm", "GRAT_PERP": "88 %"},
            ]
        )
    }

    result = build_psd_table(tables)

    assert len(result) == 1
    assert result.loc[0, "GRAT_SIZE_NUM"] == 2
    assert result.loc[0, "GRAT_PERP_NUM"] == 88
