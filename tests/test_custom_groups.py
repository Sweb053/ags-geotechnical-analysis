import pandas as pd

from app import (
    build_custom_bre_group_rows,
    build_custom_group_summary_rows,
    build_model_unit_summary_pdf,
    build_model_unit_summary_report_rows,
    parse_custom_groups,
)


def test_parse_custom_groups_reads_named_combinations() -> None:
    groups, errors = parse_custom_groups(
        """
        Alluvium = ALV, ALV(G)
        GT = GT
        broken line
        """
    )

    assert groups == [
        {"name": "Alluvium", "members": ["ALV", "ALV(G)"]},
        {"name": "GT", "members": ["GT"]},
    ]
    assert len(errors) == 1


def test_build_custom_group_summary_rows_creates_one_set_of_stats_per_group() -> None:
    data = pd.DataFrame(
        {
            "LOCA_ID": ["BH01", "BH02", "BH03", "BH04"],
            "GEOL_GEOL": ["ALV", "ALV(G)", "GT", "GT"],
            "ISPT_MAIN_NUM": [10, 20, 30, 40],
            "ISPT_N60_NUM": [11, 22, 33, 44],
        }
    )
    groups = [
        {"name": "Alluvium", "members": ["ALV", "ALV(G)"]},
        {"name": "Glacial Till", "members": ["GT"]},
    ]

    rows = build_custom_group_summary_rows("SPT", data, "GEOL_GEOL", groups)

    assert {row["Group"] for row in rows} == {"Alluvium", "Glacial Till"}
    assert {row["Parameter"] for row in rows} == {"Raw SPT N", "Corrected SPT N60"}
    alluvium_raw = next(row for row in rows if row["Group"] == "Alluvium" and row["Parameter"] == "Raw SPT N")
    assert alluvium_raw["Records"] == 2
    assert alluvium_raw["Minimum"] == 10
    assert alluvium_raw["Mean"] == 15
    assert alluvium_raw["Maximum"] == 20


def test_build_custom_bre_group_rows_creates_classification_per_group() -> None:
    data = pd.DataFrame(
        {
            "LOCA_ID": ["BH01", "BH02", "BH03"],
            "GEOL_GEOL": ["ALV", "ALV(G)", "GT"],
            "BRE_SAMPLE_TYPE": ["Soil", "Soil", "Soil"],
            "WS_MG_L": [100, 300, 2000],
            "PH_VALUE": [7.0, 6.5, 5.0],
            "MG_MG_L": [pd.NA, pd.NA, pd.NA],
        }
    )
    groups = [
        {"name": "Alluvium", "members": ["ALV", "ALV(G)"]},
        {"name": "Glacial Till", "members": ["GT"]},
    ]

    rows = build_custom_bre_group_rows(data, "GEOL_GEOL", groups, "Natural", "Mobile")

    assert [row["Group"] for row in rows] == ["Alluvium", "Glacial Till"]
    assert rows[0]["Design Sulfate Class"] == "DS-1"
    assert rows[1]["Design Sulfate Class"] == "DS-3"


def test_build_model_unit_summary_report_rows_adds_dashes_for_missing_module_values() -> None:
    module_tables = {
        "SPT": pd.DataFrame(
            {
                "LOCA_ID": ["BH01", "BH02"],
                "MODEL_UNIT": ["GT - Cohesive", "GT - Cohesive"],
                "ISPT_MAIN_NUM": [10, 20],
                "ISPT_N60_NUM": [11, 22],
            }
        ),
        "Hand Shear Vane": pd.DataFrame(),
    }

    rows = build_model_unit_summary_report_rows(module_tables)

    spt_raw = next(row for row in rows if row["Module"] == "SPT" and row["Parameter"] == "Raw SPT N")
    vane = next(row for row in rows if row["Module"] == "Hand Shear Vane")
    assert spt_raw["Model Unit"] == "GT - Cohesive"
    assert spt_raw["Records"] == 2
    assert spt_raw["Mean"] == 15
    assert vane["Model Unit"] == "GT - Cohesive"
    assert vane["Records"] == "-"
    assert vane["Cautious Estimate"] == "-"


def test_build_model_unit_summary_pdf_returns_pdf_bytes() -> None:
    rows = [
        {
            "Model Unit": "GT - Cohesive",
            "Module": "SPT",
            "Parameter": "Raw SPT N",
            "Unit": "blows",
            "Records": 2,
            "Minimum": 10,
            "Mean": 15,
            "Maximum": 20,
            "Lower 95% Estimate": 0,
            "Upper 95% Estimate": 30,
            "Cautious Estimate": 0,
            "Cautious Side": "Lower",
        }
    ]

    pdf_bytes = build_model_unit_summary_pdf(rows)

    assert pdf_bytes.startswith(b"%PDF")
