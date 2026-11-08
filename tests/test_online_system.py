import numpy as np
import pandas as pd
import pytest

from stock_prediction.online_system import (
    FEATURE_COLUMNS, HORIZONS, add_indicators, fit_forecasts,
    make_supervised_data, normalize_code,
)


def sample_history(rows=620, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=rows)
    daily = rng.normal(0.0004, 0.015, rows)
    close = 12 * np.exp(np.cumsum(daily))
    volume = rng.lognormal(12, 0.25, rows)
    return pd.DataFrame({
        "date": dates,
        "open": close * (1 + rng.normal(0, 0.003, rows)),
        "close": close,
        "high": close * (1 + rng.uniform(0.001, 0.02, rows)),
        "low": close * (1 - rng.uniform(0.001, 0.02, rows)),
        "volume": volume,
    })


def test_normalize_code_and_exchange_suffix():
    assert normalize_code(" 000001.sz ") == "000001"
    assert normalize_code("600519.SH") == "600519"
    assert normalize_code("830799.BJ") == "830799"


@pytest.mark.parametrize("value", ["ABC", "12345", "000001.US", "600519.SZ", "300750.SH", ""])
def test_reject_invalid_codes(value):
    with pytest.raises(ValueError):
        normalize_code(value)


def test_indicators_and_supervised_target_are_finite_and_forward_only():
    data = add_indicators(sample_history())
    assert set(FEATURE_COLUMNS).issubset(data.columns)
    assert not any("future" in col.lower() or "target" in col.lower() for col in FEATURE_COLUMNS)
    x, y = make_supervised_data(data, 3)
    assert len(x) > 300
    assert np.isfinite(x.to_numpy()).all()
    assert np.isfinite(y.to_numpy()).all()
    assert "close" not in x.columns
    first = data[FEATURE_COLUMNS].notna().all(axis=1).to_numpy().nonzero()[0][0]
    assert y.iloc[0] == pytest.approx(data.loc[first + 3, "close"] / data.loc[first, "close"] - 1)
    assert x.columns.tolist() == FEATURE_COLUMNS


def test_all_horizons_train_validate_and_return_sensible_intervals():
    data = add_indicators(sample_history())
    forecasts, diagnostics = fit_forecasts(data)
    assert tuple(item.horizon for item in forecasts) == HORIZONS
    assert all(item.lower_price <= item.prediction_price <= item.upper_price for item in forecasts)
    assert all(item.validation_samples >= 20 for item in forecasts)
    assert set(diagnostics) == {str(h) for h in HORIZONS}
    assert all(item.selected_model for item in forecasts)
