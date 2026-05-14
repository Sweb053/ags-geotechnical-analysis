import pandas as pd

from app import (
    calculate_design_line,
    calculate_psd_design_line,
    calculate_scalar_summary,
    interpolate_psd_d_value,
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
    assert stats["mean"] == 14
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
