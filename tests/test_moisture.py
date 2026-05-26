import pandas as pd

from ags_app.moisture import build_moisture_table


def test_build_moisture_table_matches_geology_by_depth() -> None:
    tables = {
        "LNMC": pd.DataFrame(
            [{"LOCA_ID": "BH01", "SAMP_TOP": "1.20", "LNMC_MC": "24.5"}]
        ),
        "GEOL": pd.DataFrame(
            [
                {
                    "LOCA_ID": "BH01",
                    "GEOL_TOP": "0.00",
                    "GEOL_BASE": "2.00",
                    "GEOL_GEOL": "GT",
                    "GEOL_DESC": "Firm sandy gravelly CLAY.",
                }
            ]
        ),
    }

    result = build_moisture_table(tables)

    assert result.loc[0, "LNMC_MC_NUM"] == 24.5
    assert result.loc[0, "GEOL_GEOL"] == "GT"
    assert result.loc[0, "MATERIAL_CLASS"] == "Cohesive"
