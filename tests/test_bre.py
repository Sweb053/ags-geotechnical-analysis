import pandas as pd

from ags_app.bre import (
    build_bre_sulphate_table,
    calculate_bre_summary,
    characteristic_high,
    characteristic_low,
    classify_acec,
    sulphate_class,
)


def test_build_bre_sulphate_table_pivots_gchm_results_and_matches_geology() -> None:
    tables = {
        "GCHM": pd.DataFrame(
            [
                {"LOCA_ID": "BH01", "SAMP_TOP": "1.00", "SAMP_REF": "1", "GCHM_CODE": "WS", "GCHM_RESL": "1200"},
                {"LOCA_ID": "BH01", "SAMP_TOP": "1.00", "SAMP_REF": "1", "GCHM_CODE": "PH", "GCHM_RESL": "5.8"},
                {"LOCA_ID": "BH01", "SAMP_TOP": "1.00", "SAMP_REF": "1", "GCHM_CODE": "TS", "GCHM_RESL": "0.2"},
                {"LOCA_ID": "BH01", "SAMP_TOP": "1.00", "SAMP_REF": "1", "GCHM_CODE": "AS", "GCHM_RESL": "0.1"},
            ]
        ),
        "GEOL": pd.DataFrame(
            [
                {
                    "LOCA_ID": "BH01",
                    "GEOL_TOP": "0.00",
                    "GEOL_BASE": "2.00",
                    "GEOL_GEOL": "GT",
                    "GEOL_DESC": "Firm gravelly CLAY.",
                }
            ]
        ),
    }

    result = build_bre_sulphate_table(tables)

    assert len(result) == 1
    assert result.loc[0, "WS_MG_L"] == 1200
    assert result.loc[0, "PH_VALUE"] == 5.8
    assert round(result.loc[0, "TPS_PERCENT"], 3) == 0.6
    assert round(result.loc[0, "OS_PERCENT"], 3) == 0.5
    assert result.loc[0, "GEOL_GEOL"] == "GT"
    assert result.loc[0, "MODEL_UNIT"] == "GT - Cohesive"


def test_sulphate_class_uses_soil_and_groundwater_limits() -> None:
    assert sulphate_class(499, "soil") == "DS-1"
    assert sulphate_class(500, "soil") == "DS-2"
    assert sulphate_class(399, "groundwater") == "DS-1"
    assert sulphate_class(400, "groundwater") == "DS-2"
    assert sulphate_class(7000, "soil") == "DS-5"


def test_classify_acec_natural_and_brownfield() -> None:
    assert classify_acec("DS-2", 6.0, "Natural", "Mobile") == "AC-2"
    assert classify_acec("DS-2", 5.0, "Natural", "Mobile") == "AC-3z"
    assert classify_acec("DS-4m", 7.0, "Brownfield", "Mobile") == "AC-4m"
    assert classify_acec("DS-4m", 6.0, "Brownfield", "Mobile") == "AC-5m"


def test_bre_summary_uses_characteristic_values_for_filtered_data() -> None:
    data = pd.DataFrame(
        {
            "LOCA_ID": ["BH01"] * 5,
            "BRE_SAMPLE_TYPE": ["Soil"] * 5,
            "WS_MG_L": [100, 200, 300, 400, 500],
            "PH_VALUE": [7.0, 6.5, 6.0, 5.5, 5.0],
            "MG_MG_L": [pd.NA] * 5,
        }
    )

    summary = calculate_bre_summary(data, "Natural", "Mobile")

    assert characteristic_high(data["WS_MG_L"]) == 450
    assert characteristic_low(data["PH_VALUE"]) == 5.0
    assert summary["Design Sulfate Class"] == "DS-1"
    assert summary["ACEC Class"] == "AC-2z"
