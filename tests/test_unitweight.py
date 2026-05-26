import pandas as pd

from ags_app.unitweight import build_unit_weight_table


def test_build_unit_weight_table_calculates_bulk_and_dry_unit_weight() -> None:
    tables = {
        "LDEN": pd.DataFrame(
            [
                {
                    "LOCA_ID": "BH01",
                    "SAMP_TOP": "1.20",
                    "LDEN_BDEN": "2.05",
                    "LDEN_DDEN": "1.82",
                }
            ]
        ),
        "GEOL": pd.DataFrame(
            [
                {
                    "LOCA_ID": "BH01",
                    "GEOL_TOP": "0.00",
                    "GEOL_BASE": "2.00",
                    "GEOL_GEOL": "GT",
                    "GEOL_DESC": "Dense clayey gravelly SAND.",
                }
            ]
        ),
    }

    result = build_unit_weight_table(tables)

    assert round(result.loc[0, "BULK_UNIT_WEIGHT_NUM"], 3) == 20.11
    assert round(result.loc[0, "DRY_UNIT_WEIGHT_NUM"], 3) == 17.854
    assert result.loc[0, "GEOL_GEOL"] == "GT"
    assert result.loc[0, "MATERIAL_CLASS"] == "Granular"
