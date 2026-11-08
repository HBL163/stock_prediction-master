import pandas as pd
import pytest

from scripts.online_analyze import add_indicators, normalize_code


def test_normalize_code_accepts_common_a_share_formats():
    assert normalize_code("000001") == "000001"
    assert normalize_code("600519.SH") == "600519"
    assert normalize_code("300750.sz") == "300750"


@pytest.mark.parametrize("value", ["ABC", "12345", "000001.US", "600519.SZ", ""])
def test_normalize_code_rejects_invalid_codes(value):
    with pytest.raises(ValueError):
        normalize_code(value)


def test_add_indicators_creates_expected_columns():
    close = list(range(1, 121))
    frame = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=120),
        "open": close, "close": close, "high": [x + 1 for x in close],
        "low": [max(0.1, x - 1) for x in close], "volume": [1000] * 120,
    })
    result = add_indicators(frame)
    assert {
        "close_ma_ratio_20", "close_ma_ratio_60", "rsi_14",
        "macd_ratio", "rolling_high_distance_60", "rolling_low_distance_60",
    } <= set(result.columns)
    assert result["close_ma_ratio_60"].iloc[-1] > 0
