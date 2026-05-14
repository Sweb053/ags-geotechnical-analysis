import pandas as pd

from app import build_custom_bre_group_rows, build_custom_group_summary_rows, parse_custom_groups


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
    assert alluvium_raw["Mean"] == 15


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
