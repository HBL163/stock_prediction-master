import pandas as pd
import pytest

from scripts.online_analyze import add_indicators, normalize_code


def test_normalize_code_accepts_common_a_share_formats():
    assert normalize_code("000001") == "000001"
    assert normalize_code("600519.SH") == "600519"
    assert normalize_code(" 300750.sz ") == "300750"


@pytest.mark.parametrize("value", ["ABC", "12345", "000001.US", ""])
def test_normalize_code_rejects_invalid_codes(value):
    with pytest.raises(ValueError):
        normalize_code(value)


def test_add_indicators_creates_expected_columns():
    close = list(range(1, 81))
    frame = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=80),
        "open": close, "close": close, "high": [x + 1 for x in close],
        "low": [x - 1 for x in close], "volume": [1000] * 80,
    })
    result = add_indicators(frame)
    assert {"ma20", "ma60", "rsi14", "macd", "high_60d", "low_60d"} <= set(result.columns)
    assert result["ma60"].iloc[-1] == pytest.approx(50.5)
