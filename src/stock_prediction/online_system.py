#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""可在 GitHub Actions 运行的 A 股机器学习研究流水线。

流程：行情/可选资讯 -> 无未来函数特征 -> 时间切分训练与验证 -> 多周期预测
-> 滚动验证回测 -> Markdown/JSON/CSV/PNG 报告。
本模块不依赖仓库中尚未提交的旧版深度学习权重。
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


HORIZONS = (1, 3, 5, 10)
POSITIVE_WORDS = ("增长", "盈利", "中标", "回购", "增持", "突破", "创新高", "上调", "改善", "利好", "上涨", "获批", "订单", "分红")
NEGATIVE_WORDS = ("下滑", "亏损", "减持", "处罚", "立案", "退市", "暴跌", "下调", "风险", "违规", "诉讼", "债务", "利空", "跌停", "预亏")


def normalize_code(value: str) -> str:
    """校验 A 股代码并返回六位数字代码。"""
    code = str(value).strip().upper()
    if not re.fullmatch(r"\d{6}(?:\.(?:SH|SZ|BJ))?", code):
        raise ValueError("请输入六位 A 股代码，例如 000001、600519.SH 或 300750.SZ")
    digits = code.split(".", 1)[0]
    suffix = code.split(".", 1)[1] if "." in code else None
    if suffix == "SH" and not digits.startswith(("5", "6", "9")):
        raise ValueError(f"{digits} 与 .SH 后缀不匹配")
    if suffix == "SZ" and not digits.startswith(("0", "1", "2", "3")):
        raise ValueError(f"{digits} 与 .SZ 后缀不匹配")
    if suffix == "BJ" and not digits.startswith(("4", "8")):
        raise ValueError(f"{digits} 与 .BJ 后缀不匹配")
    return digits


def _first_existing(frame: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    return None


def fetch_history(code: str, years: int = 3) -> pd.DataFrame:
    """通过 AkShare 获取前复权日线，并转换为统一英文列名。"""
    import akshare as ak

    end = date.today()
    start = end - timedelta(days=365 * years + 35)
    raw = ak.stock_zh_a_hist(
        symbol=code,
        period="daily",
        start_date=start.strftime("%Y%m%d"),
        end_date=end.strftime("%Y%m%d"),
        adjust="qfq",
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"行情接口没有返回 {code} 的数据，请稍后重试")
    mapping = {
        "日期": "date", "开盘": "open", "收盘": "close", "最高": "high",
        "最低": "low", "成交量": "volume", "成交额": "amount",
        "涨跌幅": "pct_change", "振幅": "amplitude", "换手率": "turnover",
    }
    frame = raw.rename(columns=mapping).copy()
    date_col = _first_existing(frame, ("date", "trade_date", "日期"))
    if date_col is None:
        raise RuntimeError(f"行情数据缺少日期列，实际字段：{list(frame.columns)}")
    if date_col != "date":
        frame = frame.rename(columns={date_col: "date"})
    required = ("open", "close", "high", "low", "volume")
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise RuntimeError(f"行情数据缺少必要字段：{', '.join(missing)}")
    for col in required + ("amount", "pct_change", "amplitude", "turnover"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date", "open", "close", "high", "low", "volume"])
    frame = frame.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    if len(frame) < 180:
        raise RuntimeError(f"只有 {len(frame)} 个有效交易日；机器学习研究至少需要 180 个交易日")
    return frame


def add_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """生成仅使用当日及历史数据的技术特征，避免未来函数。"""
    data = frame.copy().sort_values("date").reset_index(drop=True)
    close = data["close"].astype(float)
    high = data["high"].astype(float)
    low = data["low"].astype(float)
    volume = data["volume"].astype(float).clip(lower=0)
    returns = close.pct_change()
    log_returns = np.log(close.clip(lower=1e-8)).diff()

    for lag in (1, 2, 3, 5, 10, 20):
        data[f"return_{lag}"] = close.pct_change(lag)
        data[f"log_return_{lag}"] = np.log(close.clip(lower=1e-8)).diff(lag)
    for window in (5, 10, 20, 60):
        ma = close.rolling(window).mean()
        data[f"close_ma_ratio_{window}"] = close / ma - 1
        data[f"volatility_{window}"] = returns.rolling(window).std()
        data[f"momentum_{window}"] = close / close.shift(window) - 1
        data[f"volume_ratio_{window}"] = volume / volume.rolling(window).mean().replace(0, np.nan)
        data[f"rolling_high_distance_{window}"] = close / high.rolling(window).max() - 1
        data[f"rolling_low_distance_{window}"] = close / low.rolling(window).min() - 1

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    data["rsi_14"] = (100 - 100 / (1 + rs)).fillna(50)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    data["macd_ratio"] = (ema12 - ema26) / close.replace(0, np.nan)
    macd_signal = (ema12 - ema26).ewm(span=9, adjust=False).mean()
    data["macd_hist_ratio"] = ((ema12 - ema26) - macd_signal) / close.replace(0, np.nan)
    mid = close.rolling(20).mean()
    band_std = close.rolling(20).std()
    data["bollinger_position"] = (close - mid) / (2 * band_std).replace(0, np.nan)
    prev_close = close.shift(1)
    true_range = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    data["atr_14_ratio"] = true_range.rolling(14).mean() / close.replace(0, np.nan)
    data["return_skew_20"] = returns.rolling(20).skew()
    data["return_kurtosis_20"] = returns.rolling(20).kurt()
    data["volume_change_5"] = volume.pct_change(5)
    data["intraday_range"] = (high - low) / close.replace(0, np.nan)
    data["open_gap"] = data["open"] / prev_close - 1
    data["weekday"] = pd.to_datetime(data["date"]).dt.weekday
    data["month"] = pd.to_datetime(data["date"]).dt.month
    data["log_return_1d"] = log_returns
    return data.replace([np.inf, -np.inf], np.nan)


FEATURE_COLUMNS = [
    *(f"{kind}_{lag}" for kind in ("return", "log_return") for lag in (1, 2, 3, 5, 10, 20)),
    *(f"{prefix}_{window}" for window in (5, 10, 20, 60) for prefix in (
        "close_ma_ratio", "volatility", "momentum", "volume_ratio",
        "rolling_high_distance", "rolling_low_distance",
    )),
    "rsi_14", "macd_ratio", "macd_hist_ratio", "bollinger_position",
    "atr_14_ratio", "return_skew_20", "return_kurtosis_20",
    "volume_change_5", "intraday_range", "open_gap", "weekday", "month",
]


def make_supervised_data(indicators: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.Series]:
    """用未来 horizon 日收益作为标签；特征严格截取自预测时点。"""
    if horizon not in HORIZONS:
        raise ValueError(f"不支持的预测周期：{horizon}")
    x = indicators[FEATURE_COLUMNS].copy()
    y = indicators["close"].shift(-horizon) / indicators["close"] - 1
    valid = x.notna().all(axis=1) & y.notna() & np.isfinite(y)
    return x.loc[valid].reset_index(drop=True), y.loc[valid].astype(float).reset_index(drop=True)


@dataclass
class HorizonResult:
    horizon: int
    prediction_return: float
    lower_return: float
    upper_return: float
    prediction_price: float
    lower_price: float
    upper_price: float
    validation_mae: float
    baseline_mae: float
    validation_rmse: float
    directional_accuracy: float
    validation_samples: int
    selected_model: str
    beats_zero_baseline: bool
    backtest_trades: int
    backtest_hit_rate: float
    backtest_total_return: float


def _make_candidates(seed: int) -> dict[str, Any]:
    return {
        "extra_trees": ExtraTreesRegressor(
            n_estimators=260, min_samples_leaf=3, max_features=0.85,
            random_state=seed, n_jobs=-1,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=220, min_samples_leaf=4, max_features=0.85,
            random_state=seed, n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=180, learning_rate=0.045, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=seed,
        ),
    }


def _score_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "directional_accuracy": float(np.mean(np.sign(y_true) == np.sign(y_pred))),
    }


def _walk_forward_backtest(y_true: np.ndarray, y_pred: np.ndarray, horizon: int) -> tuple[int, float, float]:
    """验证段按 horizon 间隔采样，避免重叠持仓收益被重复计算。"""
    if len(y_true) == 0:
        return 0, 0.0, 0.0
    indices = np.arange(0, len(y_true), horizon)
    actual = y_true[indices]
    forecast = y_pred[indices]
    # 0.10% 往返成本假设；只有模型给出正收益才开多仓。
    active = forecast > 0.001
    trades = int(active.sum())
    if trades == 0:
        return 0, 0.0, 0.0
    net = actual[active] - 0.001
    hit_rate = float(np.mean(net > 0))
    total_return = float(np.prod(1 + net) - 1)
    return trades, hit_rate, total_return


def fit_forecasts(
    indicators: pd.DataFrame, horizons: tuple[int, ...] = HORIZONS, seed: int = 42
) -> tuple[list[HorizonResult], dict[str, Any]]:
    """时间顺序验证并留出 horizon 天 embargo，最后用全量可用样本重训。"""
    latest_price = float(indicators["close"].iloc[-1])
    results: list[HorizonResult] = []
    diagnostics: dict[str, Any] = {}
    for horizon in horizons:
        x, y = make_supervised_data(indicators, horizon)
        n = len(x)
        if n < 120:
            raise RuntimeError(f"{horizon} 日预测只有 {n} 条有效样本，至少需要 120 条")
        split = max(int(n * 0.8), 80)
        train_end = max(split - horizon, 60)
        if n - split < 20 or train_end < 60:
            raise RuntimeError(f"{horizon} 日预测的时间验证样本不足")
        x_train, y_train = x.iloc[:train_end], y.iloc[:train_end]
        x_valid, y_valid = x.iloc[split:], y.iloc[split:]
        candidates: list[tuple[float, str, Any, np.ndarray]] = []
        candidate_metrics: dict[str, Any] = {}
        for name, estimator in _make_candidates(seed + horizon).items():
            estimator.fit(x_train, y_train)
            pred = np.asarray(estimator.predict(x_valid), dtype=float)
            scores = _score_predictions(y_valid.to_numpy(), pred)
            candidate_metrics[name] = scores
            candidates.append((scores["mae"], name, estimator, pred))
        zero_pred = np.zeros(len(y_valid), dtype=float)
        baseline_mae = float(mean_absolute_error(y_valid, zero_pred))
        candidates.sort(key=lambda item: item[0])
        best_mae, best_name, _, best_valid_pred = candidates[0]
        beats_baseline = best_mae < baseline_mae
        # 模型未优于零收益基线时，不把噪声包装成预测优势。
        chosen_valid_pred = best_valid_pred if beats_baseline else zero_pred
        chosen_name = best_name if beats_baseline else "zero_return_baseline"
        valid_scores = _score_predictions(y_valid.to_numpy(), chosen_valid_pred)
        residuals = y_valid.to_numpy() - chosen_valid_pred
        q_low, q_high = np.quantile(residuals, [0.05, 0.95])
        final_model = _make_candidates(seed + horizon)[best_name]
        final_model.fit(x, y)
        latest_features = indicators[FEATURE_COLUMNS].iloc[[-1]]
        if latest_features.isna().any(axis=None):
            raise RuntimeError(f"{horizon} 日预测的最新特征仍含缺失值，请增加历史数据")
        raw_prediction = float(final_model.predict(latest_features)[0])
        prediction = raw_prediction if beats_baseline else 0.0
        lower = prediction + float(q_low)
        upper = prediction + float(q_high)
        lower_price = max(0.01, latest_price * (1 + lower))
        upper_price = max(lower_price, latest_price * (1 + upper))
        pred_price = max(0.01, latest_price * (1 + prediction))
        trades, hit_rate, total_return = _walk_forward_backtest(
            y_valid.to_numpy(), chosen_valid_pred, horizon
        )
        result = HorizonResult(
            horizon=horizon,
            prediction_return=prediction,
            lower_return=lower,
            upper_return=upper,
            prediction_price=pred_price,
            lower_price=lower_price,
            upper_price=upper_price,
            validation_mae=valid_scores["mae"],
            baseline_mae=baseline_mae,
            validation_rmse=valid_scores["rmse"],
            directional_accuracy=valid_scores["directional_accuracy"],
            validation_samples=len(y_valid),
            selected_model=chosen_name,
            beats_zero_baseline=beats_baseline,
            backtest_trades=trades,
            backtest_hit_rate=hit_rate,
            backtest_total_return=total_return,
        )
        results.append(result)
        diagnostics[str(horizon)] = {
            "candidate_validation": candidate_metrics,
            "chosen_model": chosen_name,
            "validation_start": str(indicators.loc[split, "date"].date()) if split < len(indicators) else None,
            "train_samples": int(len(x_train)),
            "validation_samples": int(len(x_valid)),
            "embargo_rows": int(horizon),
        }
    return results, diagnostics


def fetch_company_context(code: str) -> dict[str, Any]:
    """尽力获取公司资料和新闻；外部接口失败时不影响量化分析主流程。"""
    import akshare as ak

    context: dict[str, Any] = {"fundamentals": {}, "news": {"count": 0, "sentiment_score": None, "headlines": []}}
    try:
        info = ak.stock_individual_info_em(symbol=code)
        if isinstance(info, pd.DataFrame) and not info.empty and {"item", "value"} <= set(info.columns):
            raw = dict(zip(info["item"].astype(str), info["value"].astype(str)))
            keep = ("股票简称", "行业", "总市值", "流通市值", "市盈率-动态", "市净率", "上市时间")
            context["fundamentals"] = {k: raw[k] for k in keep if k in raw}
    except Exception as exc:
        context["fundamentals_error"] = str(exc)[:240]
    try:
        news = ak.stock_news_em(symbol=code)
        if isinstance(news, pd.DataFrame) and not news.empty:
            title_col = _first_existing(news, ("新闻标题", "标题", "title"))
            content_col = _first_existing(news, ("新闻内容", "内容", "content"))
            time_col = _first_existing(news, ("发布时间", "时间", "publish_time"))
            if title_col:
                recent = news.head(40).copy()
                texts = (recent[title_col].fillna("").astype(str) + " " +
                         (recent[content_col].fillna("").astype(str) if content_col else ""))
                positive = sum(text.count(word) for text in texts for word in POSITIVE_WORDS)
                negative = sum(text.count(word) for text in texts for word in NEGATIVE_WORDS)
                total = positive + negative
                headlines = []
                for _, row in recent.head(8).iterrows():
                    headlines.append({
                        "time": str(row[time_col]) if time_col else "",
                        "title": str(row[title_col])[:180],
                    })
                context["news"] = {
                    "count": int(len(recent)),
                    "sentiment_score": float((positive - negative) / total) if total else 0.0,
                    "positive_keyword_hits": int(positive),
                    "negative_keyword_hits": int(negative),
                    "headlines": headlines,
                    "method": "中文关键词词典计数（非大语言模型情绪识别）",
                }
    except Exception as exc:
        context["news_error"] = str(exc)[:240]
    return context


def build_report(code: str, indicators: pd.DataFrame, forecasts: list[HorizonResult],
                 diagnostics: dict[str, Any], context: dict[str, Any]) -> str:
    last = indicators.iloc[-1]
    prev = indicators.iloc[-2]
    price = float(last["close"])
    daily_change = (price / float(prev["close"]) - 1) * 100
    rsi = float(last["rsi_14"])
    ma20 = float(last["close_ma_ratio_20"])
    ma60 = float(last["close_ma_ratio_60"])
    trend = "偏多" if ma20 > 0 and ma60 > 0 else (
        "偏空" if ma20 < 0 and ma60 < 0 else "震荡/分化"
    )
    sentiment = context.get("news", {}).get("sentiment_score")
    lines = [
        f"# {code} 全维度股票研究报告", "",
        f"- 行情截止日期：{pd.Timestamp(last['date']).date()}",
        f"- 最新收盘价（前复权）：{price:.2f}",
        f"- 最近交易日涨跌幅：{daily_change:+.2f}%",
        f"- 近 20 日收益：{float(last['momentum_20']) * 100:+.2f}%",
        f"- 近 20 日年化波动率估算：{float(last['volatility_20']) * math.sqrt(252) * 100:.2f}%",
        f"- RSI(14)：{rsi:.1f}",
        f"- 价格相对 MA20 / MA60：{ma20 * 100:+.2f}% / {ma60 * 100:+.2f}%",
        f"- 均线趋势参考：**{trend}**", "",
        "## 多周期机器学习预测", "",
        "| 周期 | 预测收益 | 预测价格 | 90%经验区间 | 验证 MAE | 零收益基线 MAE | 模型 |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in forecasts:
        lines.append(
            f"| {item.horizon} 日 | {item.prediction_return * 100:+.2f}% | {item.prediction_price:.2f} | "
            f"{item.lower_price:.2f}–{item.upper_price:.2f} | {item.validation_mae * 100:.3f}% | "
            f"{item.baseline_mae * 100:.3f}% | {item.selected_model} |"
        )
    lines += ["", "预测模型包含 ExtraTrees、RandomForest、HistGradientBoosting 候选模型；"
              "按时间顺序验证选择，若验证 MAE 不优于零收益基线，则该周期预测回退为零收益基线。"]
    lines += ["", "## 验证集与策略回测", "",
              "| 周期 | 验证样本 | 方向准确率 | 开仓次数 | 命中率 | 扣费后累计收益* | 是否优于基线 |",
              "|---:|---:|---:|---:|---:|---:|---|"]
    for item in forecasts:
        lines.append(
            f"| {item.horizon} 日 | {item.validation_samples} | {item.directional_accuracy * 100:.1f}% | "
            f"{item.backtest_trades} | {item.backtest_hit_rate * 100:.1f}% | "
            f"{item.backtest_total_return * 100:+.2f}% | {'是' if item.beats_zero_baseline else '否'} |"
        )
    lines += ["", "*回测为验证区间的简化多头信号演示：预测收益超过 0.1% 才开仓，假设单次往返成本 0.1%，"
              "按预测周期间隔抽样以减少重叠；不代表可成交的实盘策略。"]
    lines += ["", "## 公司资料"]
    fundamentals = context.get("fundamentals") or {}
    if fundamentals:
        lines.extend([f"- {key}：{value}" for key, value in fundamentals.items()])
    else:
        lines.append("- 公司资料接口未返回有效数据；请勿将缺失字段理解为零。")
    lines += ["", "## 新闻情绪（辅助参考）"]
    news = context.get("news", {})
    if news.get("count"):
        score = news.get("sentiment_score")
        label = "偏正面" if score > 0.15 else ("偏负面" if score < -0.15 else "中性/信号较弱")
        lines += [f"- 最近抓取新闻：{news['count']} 条",
                  f"- 词典情绪分数：{score:+.3f}（{label}）",
                  f"- 正面/负面词命中：{news.get('positive_keyword_hits', 0)} / {news.get('negative_keyword_hits', 0)}",
                  f"- 计算方式：{news.get('method', '关键词计数')}"]
        lines += ["", "近期标题："]
        lines.extend([f"- {item['time']} {item['title']}" for item in news.get("headlines", [])])
    else:
        lines.append("- 新闻接口暂不可用或没有返回可解析标题，本次未纳入新闻情绪。")
    lines += ["", "## 风险与方法边界",
              "- 所有模型特征均由预测时点及此前数据计算；标签为未来 1/3/5/10 个交易日收益。",
              "- 验证按时间顺序切分，并在训练集末端留出预测周期长度的隔离区，降低标签重叠泄漏。",
              "- 90%区间由验证残差经验分位数构造，不是严格统计保证，也不是模型置信度。",
              "- 新闻情绪使用关键词词典，容易误判反讽、否定、上下文和公告重要性，不等于人工或大模型研判。",
              "- 基于历史数据的回测可能受幸存者偏差、复权方式、滑点、涨跌停、停牌和数据源修订影响。",
              "- 本报告仅用于研究与学习，不构成投资建议；模型结果不保证未来收益。", ""]
    return "\n".join(lines)


def save_chart(code: str, indicators: pd.DataFrame, forecasts: list[HorizonResult], path: Path) -> None:
    fig, (ax_price, ax_rsi) = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax_price.plot(indicators["date"], indicators["close"], label="Adjusted close", linewidth=1.3)
    for window in (20, 60):
        ax_price.plot(indicators["date"], indicators["close"].rolling(window).mean(),
                      label=f"MA{window}", linewidth=1)
    ax_price.set_title(f"{code} | Price trend and ML forecast")
    ax_price.set_ylabel("Price")
    ax_price.grid(alpha=0.2)
    ax_price.legend()
    ax_rsi.plot(indicators["date"], indicators["rsi_14"], label="RSI(14)", linewidth=1)
    ax_rsi.axhline(70, linestyle="--", linewidth=0.8)
    ax_rsi.axhline(30, linestyle="--", linewidth=0.8)
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_ylabel("RSI")
    ax_rsi.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_analysis(code: str, years: int = 3, output: str | Path = "output",
                 include_context: bool = True) -> dict[str, Any]:
    code = normalize_code(code)
    history = fetch_history(code, years=years)
    indicators = add_indicators(history)
    forecasts, diagnostics = fit_forecasts(indicators)
    context: dict[str, Any] = {"fundamentals": {}, "news": {"count": 0, "sentiment_score": None, "headlines": []}}
    if include_context:
        try:
            context = fetch_company_context(code)
        except Exception as exc:
            context["context_error"] = str(exc)[:240]
    report = build_report(code, indicators, forecasts, diagnostics, context)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{code}_analysis.md").write_text(report, encoding="utf-8")
    indicators.to_csv(out / f"{code}_features.csv", index=False, encoding="utf-8-sig")
    forecast_payload = {
        "code": code,
        "as_of": str(pd.Timestamp(indicators.iloc[-1]["date"]).date()),
        "latest_close": float(indicators.iloc[-1]["close"]),
        "forecasts": [asdict(item) for item in forecasts],
        "validation_diagnostics": diagnostics,
        "company_context": context,
        "methodology": {
            "models": ["ExtraTreesRegressor", "RandomForestRegressor", "HistGradientBoostingRegressor"],
            "horizons": list(HORIZONS),
            "validation": "chronological 80/20 split with horizon-sized embargo",
            "prediction_interval": "empirical 5th/95th percentile of validation residuals",
        },
    }
    (out / f"{code}_forecast.json").write_text(
        json.dumps(forecast_payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    (out / f"{code}_analysis.md").write_text(report, encoding="utf-8")
    save_chart(code, indicators, forecasts, out / f"{code}_chart.png")
    print(report)
    print(f"\n报告文件已生成：{out.resolve()}")
    return forecast_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="A 股全维度机器学习研究流水线")
    parser.add_argument("--code", required=True, help="A 股代码，例如 000001 或 600519.SH")
    parser.add_argument("--years", type=int, choices=(2, 3, 5), default=3)
    parser.add_argument("--output", default="output")
    parser.add_argument("--skip-context", action="store_true", help="跳过公司资料和新闻接口")
    args = parser.parse_args()
    run_analysis(args.code, years=args.years, output=args.output, include_context=not args.skip_context)


if __name__ == "__main__":
    main()
