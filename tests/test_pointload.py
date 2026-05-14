import pandas as pd

from ags_app.pointload import build_pointload_table


def test_build_pointload_table_matches_geology_using_specimen_depth_first() -> None:
    tables = {
        "RPLT": pd.DataFrame(
            [
                {"LOCA_ID": "BH01", "SAMP_TOP": "1.20", "SPEC_DPTH": "1.50", "RPLT_PLSI": "2.1"},
                {"LOCA_ID": "BH01", "SAMP_TOP": "3.00", "SPEC_DPTH": None, "RPLT_PLSI": "4.8"},
                {"LOCA_ID": "BH02", "SAMP_TOP": "2.00", "SPEC_DPTH": None, "RPLT_PLSI": "1.4"},
            ]
        ),
        "GEOL": pd.DataFrame(
            [
                {
                    "LOCA_ID": "BH01",
                    "GEOL_TOP": "0.00",
                    "GEOL_BASE": "2.00",
                    "GEOL_GEOL": "ROCK-A",
                    "GEOL_DESC": "Rock A",
                },
                {
                    "LOCA_ID": "BH01",
                    "GEOL_TOP": "2.00",
                    "GEOL_BASE": "4.00",
                    "GEOL_GEOL": "ROCK-B",
                    "GEOL_DESC": "Rock B",
                },
            ]
        ),
    }

    result = build_pointload_table(tables)

    assert result.loc[0, "POINTLOAD_DEPTH_NUM"] == 1.5
    assert result.loc[0, "GEOL_GEOL"] == "ROCK-A"
    assert result.loc[1, "POINTLOAD_DEPTH_NUM"] == 3.0
    assert result.loc[1, "GEOL_GEOL"] == "ROCK-B"
    assert result.loc[2, "GEOL_GEOL"] == "Unmatched"


def test_build_pointload_table_uses_first_number_in_strength_index() -> None:
    tables = {
        "RPLT": pd.DataFrame(
            [{"LOCA_ID": "BH01", "SAMP_TOP": "4.10m", "RPLT_PLSI": "6.5 MPa"}]
        )
    }

    result = build_pointload_table(tables)

    assert result.loc[0, "POINTLOAD_DEPTH_NUM"] == 4.1
    assert result.loc[0, "RPLT_PLSI_NUM"] == 6.5
