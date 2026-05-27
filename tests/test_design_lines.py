import pandas as pd

from app import (
    apply_table_edits,
    calculate_design_line,
    calculate_horizontal_depth_design_line,
    calculate_psd_design_line,
    calculate_scalar_summary,
    dataframe_from_editor_state,
    interpolate_psd_d_value,
    parse_custom_design_lines,
    t_critical_one_sided_95,
)


def test_calculate_design_line_returns_lower_bound_below_mean_trend() -> None:
    data = pd.DataFrame(
        {
            "VALUE": [10.0, 13.0, 15.0, 18.0, 21.0],
            "DEPTH": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    mean_line = calculate_design_line(data, "VALUE", "DEPTH", "Mean trend")
    lower_line = calculate_design_line(data, "VALUE", "DEPTH", "Lower cautious estimate")

    assert mean_line is not None
    assert lower_line is not None
    mean_x, mean_y, mean_label = mean_line
    lower_x, lower_y, lower_label = lower_line
    assert mean_y == lower_y
    assert mean_label == "Mean trend"
    assert lower_label == "Lower 95% cautious line"
    assert all(lower <= mean for lower, mean in zip(lower_x, mean_x))


def test_calculate_design_line_returns_none_for_small_dataset() -> None:
    data = pd.DataFrame({"VALUE": [10.0, 12.0], "DEPTH": [1.0, 2.0]})

    assert calculate_design_line(data, "VALUE", "DEPTH", "Lower cautious estimate") is None


def test_calculate_horizontal_depth_design_line_uses_depth_statistic() -> None:
    data = pd.DataFrame(
        {
            "INVESTIGATION": [1, 2, 3, 4, 5],
            "DEPTH": [0.8, 1.0, 1.2, 1.6, 2.0],
        }
    )

    lower_line = calculate_horizontal_depth_design_line(data, "INVESTIGATION", "DEPTH", "Lower cautious estimate")
    mean_line = calculate_horizontal_depth_design_line(data, "INVESTIGATION", "DEPTH", "Mean trend")
    upper_line = calculate_horizontal_depth_design_line(data, "INVESTIGATION", "DEPTH", "Upper cautious estimate")

    assert lower_line is not None
    assert mean_line is not None
    assert upper_line is not None
    lower_x, lower_y, _ = lower_line
    mean_x, mean_y, _ = mean_line
    upper_x, upper_y, _ = upper_line
    assert lower_x == mean_x == upper_x == [1.0, 5.0]
    assert lower_y[0] == lower_y[1]
    assert mean_y[0] == mean_y[1]
    assert upper_y[0] == upper_y[1]
    assert lower_y[0] < mean_y[0] < upper_y[0]


def test_parse_custom_design_lines_groups_multiline_coordinates() -> None:
    rows = pd.DataFrame(
        {
            "Line": ["Line A", "Line A", "Line B", "Line B", "Line C"],
            "X": [10, 15, 20, 25, 30],
            "Y": [1, 3, 2, 4, 5],
        }
    )

    parsed = parse_custom_design_lines(rows)

    assert parsed == (
        ("Line A", ((10.0, 1.0), (15.0, 3.0))),
        ("Line B", ((20.0, 2.0), (25.0, 4.0))),
    )


def test_parse_custom_design_lines_can_require_positive_x_values() -> None:
    rows = pd.DataFrame(
        {
            "Line": ["PSD line", "PSD line", "PSD line"],
            "X": [0.0, 0.063, 2.0],
            "Y": [10.0, 25.0, 70.0],
        }
    )

    parsed = parse_custom_design_lines(rows, positive_x=True)

    assert parsed == (("PSD line", ((0.063, 25.0), (2.0, 70.0))),)


def test_dataframe_from_editor_state_applies_streamlit_edit_delta() -> None:
    base = pd.DataFrame(
        {
            "GEOL_GEOL": ["ALV(G)", "GT"],
            "__GEOL_SOURCE_ROW_INDEX": [4, 9],
        }
    )
    state = {"edited_rows": {"0": {"GEOL_GEOL": "ALV - Granular"}}}

    edited = dataframe_from_editor_state(base, state)

    assert edited.loc[0, "GEOL_GEOL"] == "ALV - Granular"
    assert edited.loc[1, "GEOL_GEOL"] == "GT"


def test_apply_table_edits_adds_geology_override_columns() -> None:
    tables = {
        "GEOL": pd.DataFrame(
            [
                {
                    "LOCA_ID": "BH01",
                    "GEOL_TOP": "0",
                    "GEOL_BASE": "1",
                    "GEOL_GEOL": "GDU",
                    "GEOL_DESC": "Grey medium grained psammite boulder.",
                }
            ]
        )
    }
    edited = pd.DataFrame(
        [
            {
                "__GEOL_SOURCE_ROW_INDEX": 0,
                "MATERIAL_CLASS": "Granular",
                "MODEL_UNIT": "GDU - Granular",
                "BEDROCK_TYPE": "Manual rock type",
            }
        ]
    )

    updates = apply_table_edits(tables, "GEOL", edited, "__GEOL_SOURCE_ROW_INDEX", geol_only=True)

    assert updates == 1
    assert tables["GEOL"].loc[0, "MATERIAL_CLASS"] == "Granular"
    assert tables["GEOL"].loc[0, "MODEL_UNIT"] == "GDU - Granular"
    assert tables["GEOL"].loc[0, "BEDROCK_TYPE"] == "Manual rock type"


def test_calculate_psd_design_line_uses_selected_curves() -> None:
    data = pd.DataFrame(
        {
            "PSD_SAMPLE_ID": ["A", "B", "C", "A", "B", "C"],
            "GRAT_SIZE_NUM": [0.063, 0.063, 0.063, 2.0, 2.0, 2.0],
            "GRAT_PERP_NUM": [20.0, 30.0, 40.0, 70.0, 80.0, 90.0],
        }
    )

    mean_curve = calculate_psd_design_line(data, "Mean trend")
    lower_curve = calculate_psd_design_line(data, "Lower cautious estimate")

    assert mean_curve is not None
    assert lower_curve is not None
    mean_x, mean_y, _ = mean_curve
    lower_x, lower_y, _ = lower_curve
    assert mean_x == [0.063, 2.0]
    assert lower_x == mean_x
    assert lower_y[0] <= mean_y[0]
    assert lower_y[1] <= mean_y[1]


def test_t_critical_one_sided_95_uses_large_sample_normal_limit() -> None:
    assert t_critical_one_sided_95(200) == 1.645


def test_calculate_scalar_summary_uses_lower_cautious_estimate() -> None:
    values = pd.Series([10, 12, 14, 16, 18])

    stats = calculate_scalar_summary(values, "lower")

    assert stats is not None
    assert stats["count"] == 5
    assert stats["minimum"] == 10
    assert stats["mean"] == 14
    assert stats["maximum"] == 18
    assert stats["cautious"] < stats["mean"]
    assert stats["cautious"] == stats["lower_95"]


def test_calculate_scalar_summary_can_use_upper_cautious_estimate() -> None:
    values = pd.Series([1.0, 2.0, 3.0, 4.0])

    stats = calculate_scalar_summary(values, "upper")

    assert stats is not None
    assert stats["cautious"] == stats["upper_95"]
    assert stats["cautious"] > stats["mean"]


def test_interpolate_psd_d_value_interpolates_on_log_size() -> None:
    curve = pd.DataFrame(
        {
            "GRAT_SIZE_NUM": [0.01, 0.1, 1.0],
            "GRAT_PERP_NUM": [0, 50, 100],
        }
    )

    d50 = interpolate_psd_d_value(curve, 50)
    d25 = interpolate_psd_d_value(curve, 25)

    assert d50 == 0.1
    assert round(d25, 3) == 0.032
