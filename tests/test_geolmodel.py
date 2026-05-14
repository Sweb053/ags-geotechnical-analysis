import pandas as pd

from app import build_filtered_investigation_summary
from ags_app.geolmodel import build_geological_model, classify_material, extract_bedrock_type


def test_classify_material_distinguishes_glacial_till_cohesive_and_granular() -> None:
    assert classify_material("GT", "Firm brown sandy gravelly CLAY.") == "Cohesive"
    assert classify_material("GT", "Dense brown clayey gravelly SAND.") == "Granular"


def test_extract_bedrock_type_uses_capitalised_rock_word() -> None:
    assert extract_bedrock_type("Medium strong grey fine grained PSAMMITE.") == "PSAMMITE"
    assert extract_bedrock_type("Strong pink GRANITE with bands of SCHIST.") == "GRANITE / SCHIST"


def test_build_geological_model_adds_model_unit_and_thickness() -> None:
    tables = {
        "GEOL": pd.DataFrame(
            [
                {
                    "LOCA_ID": "BH01",
                    "GEOL_TOP": "0",
                    "GEOL_BASE": "1.2",
                    "GEOL_GEOL": "PEAT",
                    "GEOL_DESC": "Dark brown PEAT.",
                },
                {
                    "LOCA_ID": "BH01",
                    "GEOL_TOP": "1.2",
                    "GEOL_BASE": "3.5",
                    "GEOL_GEOL": "GT",
                    "GEOL_DESC": "Firm sandy gravelly CLAY.",
                },
                {
                    "LOCA_ID": "BH01",
                    "GEOL_TOP": "3.5",
                    "GEOL_BASE": "6.0",
                    "GEOL_GEOL": "MORR",
                    "GEOL_DESC": "Strong grey PSAMMITE.",
                },
            ]
        )
    }

    result = build_geological_model(tables)

    assert result["MATERIAL_CLASS"].tolist() == ["Peat", "Cohesive", "Rock / Bedrock"]
    assert result.loc[1, "MODEL_UNIT"] == "GT - Cohesive"
    assert result.loc[2, "BEDROCK_TYPE"] == "PSAMMITE"
    assert result.loc[2, "MODEL_UNIT"] == "MORR - PSAMMITE"
    assert result.loc[2, "THICKNESS_NUM"] == 2.5


def test_build_filtered_investigation_summary_lists_matching_investigations() -> None:
    filtered = pd.DataFrame(
        [
            {
                "LOCA_ID": "BH01",
                "GEOL_GEOL": "GT",
                "MATERIAL_CLASS": "Cohesive",
                "MODEL_UNIT": "GT - Cohesive",
                "BEDROCK_TYPE": pd.NA,
                "GEOL_TOP_NUM": 0.0,
                "GEOL_BASE_NUM": 1.0,
                "THICKNESS_NUM": 1.0,
            },
            {
                "LOCA_ID": "BH04",
                "GEOL_GEOL": "GT",
                "MATERIAL_CLASS": "Cohesive",
                "MODEL_UNIT": "GT - Cohesive",
                "BEDROCK_TYPE": pd.NA,
                "GEOL_TOP_NUM": 0.5,
                "GEOL_BASE_NUM": 2.0,
                "THICKNESS_NUM": 1.5,
            },
        ]
    )

    summary = build_filtered_investigation_summary(filtered)

    assert summary["Investigation"].tolist() == ["BH01", "BH04"]
    assert summary.loc[0, "Model units"] == "GT - Cohesive"
    assert summary.loc[0, "Matching depth ranges"] == "0-1 m"
    assert summary.loc[1, "Total matching thickness (m)"] == 1.5
