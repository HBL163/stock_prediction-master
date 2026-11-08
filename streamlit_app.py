# -*- coding: utf-8 -*-
"""Interactive single-stock research dashboard for the stock_prediction project.

This dashboard is deliberately independent from the legacy PKL/checkpoint inference
path: a user can enter a ticker, fetch recent OHLCV data, train a small per-ticker
baseline model, and inspect technical/risk diagnostics without uploading local files.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

FEATURE_COLUMNS = [
    "return_1d", "return_2d", "return_5d", "return_10d",
    "ma5_gap", "ma20_gap", "ma60_gap", "ema12_gap", "ema26_gap",
    "macd_gap", "macd_signal_gap", "rsi14", "atr_pct",
    "volatility20", "bb_position", "volume_ratio", "intraday_range",
]


def normalize_ticker(raw: str) -> tuple[str, str]:
    """Return (provider ticker, market label) for common A-share and Yahoo symbols."""
    ticker = str(raw).strip().upper().replace(" ", "")
    if not ticker:
        raise ValueError("请输入股票代码。")
    ticker = ticker.replace(".XSHG", ".SS").replace(".XSHE", ".SZ")
    base = ticker.split(".", 1)[0]
    if base.isdigit() and len(base) <= 6:
        base = base.zfill(6)
        if base.startswith(("5", "6", "9")):
            return base, "中国 A 股（上交所）"
        if base.startswith(("0", "1", "2", "3")):
            return base, "中国 A 股（深交所）"
        if base.startswith(("4", "8")):
            return base, "中国 A 股（北交所）"
    return ticker, "国际市场 / Yahoo Finance"


def _standardize_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert AkShare or yfinance columns into a common OHLCV schema."""
    if frame is None or frame.empty:
        raise ValueError("数据源没有返回行情，请检查代码或稍后重试。")
    data = frame.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    aliases = {
        "日期": "date", "交易日期": "date", "Date": "date",
        "开盘": "open", "Open": "open",
        "收盘": "close", "Close": "close",
        "最高": "high", "High": "high",
        "最低": "low", "Low": "low",
        "成交量": "volume", "成交额": "amount", "Volume": "volume",
        "Adj Close": "adj_close",
    }
    data = data.rename(columns={c: aliases.get(str(c), str(c).lower()) for c in data.columns})
    if "date" not in data.columns:
        data = data.rename_axis("date").reset_index()
    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"行情字段不完整，缺少：{', '.join(missing)}")
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    for col in required + ["volume"]:
        if col not in data.columns:
            data[col] = 0.0
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["date", "open", "high", "low", "close"])
    data = data.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    data = data[(data["close"] > 0) & (data["high"] > 0) & (data["low"] > 0)]
    if len(data) < 100:
        raise ValueError(f"有效历史行情只有 {len(data)} 条，至少需要 100 个交易日。")
    return data


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_history(raw_ticker: str, years: int = 3) -> tuple[pd.DataFrame, str]:
    ticker, market = normalize_ticker(raw_ticker)
    start = (date.today() - timedelta(days=365 * years)).strftime("%Y%m%d")
    end = date.today().strftime("%Y%m%d")

    # Prefer AkShare for mainland A-shares; keep Yahoo as a fallback.
    if ticker.isdigit() and len(ticker) == 6:
        try:
            import akshare as ak
            if ticker.startswith(("4", "8")) and hasattr(ak, "stock_bj_a_hist"):
                raw = ak.stock_bj_a_hist(symbol=ticker, period="daily",
                                         start_date=start, end_date=end, adjust="qfq")
            else:
                raw = ak.stock_zh_a_hist(symbol=ticker, period="daily",
                                         start_date=start, end_date=end, adjust="qfq")
            return _standardize_ohlcv(raw), "AkShare（前复权日线）"
        except Exception as ak_error:
            suffix = ".SS" if ticker.startswith(("5", "6", "9")) else ".SZ"
            if ticker.startswith(("4", "8")):
                suffix = ".BJ"
            yahoo_ticker = ticker + suffix
            try:
                import yfinance as yf
                raw = yf.download(yahoo_ticker, period=f"{years}y", interval="1d",
                                  auto_adjust=True, progress=False, threads=False)
                return _standardize_ohlcv(raw), f"Yahoo Finance（AkShare 回退：{type(ak_error).__name__}）"
            except Exception as yahoo_error:
                raise RuntimeError(
                    f"AkShare 与 Yahoo Finance 均无法获取 {ticker}：{ak_error}；{yahoo_error}"
                ) from yahoo_error

    import yfinance as yf
    raw = yf.download(ticker, period=f"{years}y", interval="1d",
                      auto_adjust=True, progress=False, threads=False)
    return _standardize_ohlcv(raw), "Yahoo Finance（日线）"


def add_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    close = df["close"]
    high, low = df["high"], df["low"]
    df["return_1d"] = close.pct_change()
    df["return_2d"] = close.pct_change(2)
    df["return_5d"] = close.pct_change(5)
    df["return_10d"] = close.pct_change(10)
    for n in (5, 20, 60):
        ma = close.rolling(n).mean()
        df[f"ma{n}_gap"] = close / ma - 1
    for n in (12, 26):
        ema = close.ewm(span=n, adjust=False).mean()
        df[f"ema{n}_gap"] = close / ema - 1
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["macd_gap"] = macd / close
    df["macd_signal_gap"] = (macd - signal) / close

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi14"] = (100 - 100 / (1 + rs)).fillna(50)

    previous_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(14).mean()
    df["atr_pct"] = atr / close
    df["volatility20"] = df["return_1d"].rolling(20).std() * np.sqrt(252)

    middle = close.rolling(20).mean()
    band_width = close.rolling(20).std() * 2
    upper, lower = middle + band_width, middle - band_width
    df["bb_position"] = (close - lower) / (upper - lower).replace(0, np.nan)
    volume_mean = df["volume"].rolling(20).mean().replace(0, np.nan)
    df["volume_ratio"] = df["volume"] / volume_mean
    df["intraday_range"] = (high - low) / close
    return df.replace([np.inf, -np.inf], np.nan)


def fit_forecast(indicators: pd.DataFrame, horizon: int) -> dict:
    """Time-ordered holdout evaluation, then refit on all available labeled rows."""
    data = indicators.copy()
    data["target_return"] = data["close"].shift(-horizon) / data["close"] - 1
    supervised = data.dropna(subset=FEATURE_COLUMNS + ["target_return"]).copy()
    if len(supervised) < 100:
        raise ValueError("可用于训练的样本不足，请选择更长的历史区间或缩短预测天数。")
    split = int(len(supervised) * 0.8)
    split = min(max(split, 60), len(supervised) - 20)
    train, test = supervised.iloc[:split], supervised.iloc[split:]
    params = dict(n_estimators=180, max_depth=5, min_samples_leaf=4,
                  max_features=0.85, random_state=42, n_jobs=-1)
    evaluation_model = RandomForestRegressor(**params)
    evaluation_model.fit(train[FEATURE_COLUMNS], train["target_return"])
    test_pred = evaluation_model.predict(test[FEATURE_COLUMNS])
    actual = test["target_return"].to_numpy()
    errors = actual - test_pred
    direction_accuracy = float(np.mean((test_pred >= 0) == (actual >= 0)))
    mae = float(mean_absolute_error(actual, test_pred))
    rmse = float(np.sqrt(mean_squared_error(actual, test_pred)))

    final_model = RandomForestRegressor(**params)
    final_model.fit(supervised[FEATURE_COLUMNS], supervised["target_return"])
    latest = indicators.dropna(subset=FEATURE_COLUMNS).iloc[[-1]]
    prediction = float(final_model.predict(latest[FEATURE_COLUMNS])[0])
    tree_predictions = np.array([
        tree.predict(latest[FEATURE_COLUMNS])[0] for tree in final_model.estimators_
    ], dtype=float)
    direction_probability = float(np.mean(tree_predictions > 0))
    residual_low, residual_high = np.quantile(errors, [0.10, 0.90])
    lower_return = prediction + float(residual_low)
    upper_return = prediction + float(residual_high)

    feature_importance = pd.Series(
        final_model.feature_importances_, index=FEATURE_COLUMNS
    ).sort_values(ascending=False).head(8)
    backtest = pd.DataFrame({
        "date": test["date"].to_numpy(),
        "actual_return": actual,
        "predicted_return": test_pred,
    })
    return {
        "prediction_return": prediction,
        "lower_return": lower_return,
        "upper_return": upper_return,
        "direction_probability": direction_probability,
        "mae": mae,
        "rmse": rmse,
        "direction_accuracy": direction_accuracy,
        "feature_importance": feature_importance,
        "backtest": backtest,
        "train_samples": len(train),
        "test_samples": len(test),
    }


def technical_summary(df: pd.DataFrame) -> tuple[str, str, int]:
    latest = df.iloc[-1]
    score = 0
    score += 1 if latest["close"] > latest["close_ma20"] else -1
    score += 1 if latest["close"] > latest["close_ma60"] else -1
    score += 1 if latest["macd_value"] > latest["macd_signal"] else -1
    if latest["rsi14"] < 30:
        score += 1
    elif latest["rsi14"] > 70:
        score -= 1
    if score >= 2:
        return "偏强", "均线与动量指标整体偏多，但仍需结合成交量和风险区间判断。", score
    if score <= -2:
        return "偏弱", "均线与动量指标整体偏空，注意回撤与趋势反转风险。", score
    return "震荡 / 中性", "多空信号暂不一致，等待趋势与成交量进一步确认。", score


def prepare_display_frame(indicators: pd.DataFrame) -> pd.DataFrame:
    df = indicators.copy()
    df["close_ma20"] = df["close"].rolling(20).mean()
    df["close_ma60"] = df["close"].rolling(60).mean()
    e12 = df["close"].ewm(span=12, adjust=False).mean()
    e26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd_value"] = e12 - e26
    df["macd_signal"] = df["macd_value"].ewm(span=9, adjust=False).mean()
    return df


def main() -> None:
    st.set_page_config(page_title="股票智能分析", page_icon="📈", layout="wide")
    st.title("📈 股票智能分析系统")
    st.caption("基于项目原有行情 / 技术指标 / 时序建模方向扩展的交互式分析入口。研究用途，不构成投资建议。")

    with st.sidebar:
        st.header("分析设置")
        ticker = st.text_input("股票代码", value="AAPL", help="示例：AAPL、MSFT、600519、000001")
        horizon = st.select_slider("预测周期（交易日）", options=[1, 3, 5, 10, 20], value=5)
        years = st.select_slider("历史数据范围", options=[1, 2, 3, 5, 8], value=3)
        run = st.button("开始分析", type="primary", use_container_width=True)
        st.caption("中国 A 股优先使用 AkShare；其他市场使用 Yahoo Finance。")

    if run or "stock_analysis_result" not in st.session_state:
        if run:
            st.session_state.pop("stock_analysis_result", None)
        if not ticker.strip():
            st.warning("请先输入股票代码。")
            return
        try:
            with st.spinner("正在获取行情、计算指标并进行时间序列回测…"):
                prices, provider = fetch_history(ticker, years)
                indicators = add_indicators(prices)
                result = fit_forecast(indicators, int(horizon))
                display = prepare_display_frame(indicators)
                st.session_state["stock_analysis_result"] = {
                    "ticker": ticker, "provider": provider, "horizon": int(horizon),
                    "prices": prices, "indicators": indicators, "display": display,
                    "result": result,
                }
        except Exception as exc:
            st.error(f"分析失败：{exc}")
            st.info("可检查代码是否正确、数据源是否可访问，或稍后重试。")
            return

    state = st.session_state.get("stock_analysis_result")
    if not state:
        return
    if state["ticker"].strip().upper() != ticker.strip().upper() or state["horizon"] != int(horizon):
        st.info("修改代码或预测周期后，请点击「开始分析」重新计算。")
        return

    prices, display, result = state["prices"], state["display"], state["result"]
    latest = prices.iloc[-1]
    label, explanation, score = technical_summary(display)
    forecast_price = float(latest["close"] * (1 + result["prediction_return"]))
    lower_price = float(latest["close"] * (1 + result["lower_return"]))
    upper_price = float(latest["close"] * (1 + result["upper_return"]))
    change_pct = float(latest["close"] / prices.iloc[-2]["close"] - 1) if len(prices) > 1 else 0.0

    st.caption(f"数据源：{state['provider']}　|　最近交易日：{latest['date'].date()}　|　有效样本：{len(prices)}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("最新收盘价", f"{latest['close']:,.2f}", f"{change_pct:+.2%}")
    c2.metric(f"{state['horizon']} 日模型目标价", f"{forecast_price:,.2f}", f"{result['prediction_return']:+.2%}")
    c3.metric("模型方向概率", f"{result['direction_probability']:.1%}", "模型树投票比例")
    c4.metric("技术面判断", label, f"信号分数 {score:+d}")

    st.info(explanation)
    st.caption(
        f"模型区间（经验残差区间，非统计保证）：{lower_price:,.2f} – {upper_price:,.2f}。"
        "预测目标为未来所选交易日的累计收益率。"
    )

    chart = make_subplots(rows=2, cols=1, shared_xaxes=True,
                          vertical_spacing=0.04, row_heights=[0.72, 0.28])
    chart.add_trace(go.Candlestick(
        x=display["date"], open=display["open"], high=display["high"],
        low=display["low"], close=display["close"], name="K线"
    ), row=1, col=1)
    chart.add_trace(go.Scatter(x=display["date"], y=display["close_ma20"],
                               name="MA20", line=dict(width=1.3)), row=1, col=1)
    chart.add_trace(go.Scatter(x=display["date"], y=display["close_ma60"],
                               name="MA60", line=dict(width=1.3)), row=1, col=1)
    chart.add_trace(go.Bar(x=display["date"], y=display["volume"], name="成交量"),
                    row=2, col=1)
    chart.update_layout(height=620, xaxis_rangeslider_visible=False,
                        legend_orientation="h", margin=dict(l=10, r=10, t=35, b=10))
    st.subheader("行情与趋势")
    st.plotly_chart(chart, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("技术指标")
        tech = pd.DataFrame({
            "指标": ["RSI(14)", "MACD", "MA20 偏离", "MA60 偏离", "年化波动率", "ATR / 价格"],
            "当前值": [
                f"{display.iloc[-1]['rsi14']:.1f}",
                f"{display.iloc[-1]['macd_value']:.4f}",
                f"{display.iloc[-1]['ma20_gap']:+.2%}",
                f"{display.iloc[-1]['ma60_gap']:+.2%}",
                f"{display.iloc[-1]['volatility20']:.2%}",
                f"{display.iloc[-1]['atr_pct']:.2%}",
            ],
        })
        st.dataframe(tech, hide_index=True, use_container_width=True)
    with right:
        st.subheader("模型评估（时间顺序留出集）")
        st.metric("方向准确率", f"{result['direction_accuracy']:.1%}")
        st.metric("收益率 MAE", f"{result['mae']:.2%}")
        st.metric("收益率 RMSE", f"{result['rmse']:.2%}")
        st.caption(f"训练样本 {result['train_samples']} 条；测试样本 {result['test_samples']} 条。历史表现不代表未来效果。")

    st.subheader("历史留出集：预测收益率 vs 实际收益率")
    bt = result["backtest"]
    backtest_fig = go.Figure()
    backtest_fig.add_trace(go.Scatter(x=bt["date"], y=bt["actual_return"],
                                      name="实际收益率", mode="lines"))
    backtest_fig.add_trace(go.Scatter(x=bt["date"], y=bt["predicted_return"],
                                      name="模型预测", mode="lines"))
    backtest_fig.add_hline(y=0, line_dash="dot")
    backtest_fig.update_layout(height=340, yaxis_tickformat=".1%",
                               margin=dict(l=10, r=10, t=25, b=10))
    st.plotly_chart(backtest_fig, use_container_width=True)

    st.subheader("模型关注的特征")
    importance = result["feature_importance"].sort_values()
    imp_fig = go.Figure(go.Bar(x=importance.values, y=importance.index, orientation="h"))
    imp_fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10),
                          xaxis_title="随机森林特征重要性")
    st.plotly_chart(imp_fig, use_container_width=True)

    csv = prices.to_csv(index=False).encode("utf-8-sig")
    st.download_button("下载历史行情 CSV", data=csv,
                       file_name=f"{ticker.strip().replace('.', '_')}_history.csv",
                       mime="text/csv")
    st.warning("风险提示：这是基于历史数据的研究型基线模型，未纳入实时新闻、财报、交易成本、停牌及涨跌停等全部因素；模型输出不应作为单独买卖依据。")


if __name__ == "__main__":
    main()
