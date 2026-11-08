#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""基于项目行情数据规范的 A 股在线技术分析入口。

该入口不依赖本地训练权重，适合 GitHub Actions 手动输入股票代码后运行。
输出为 Markdown 分析报告、行情 CSV 和趋势图；不构成投资建议。
"""
from __future__ import annotations

import argparse
import re
from datetime import date, timedelta
from pathlib import Path

import akshare as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def normalize_code(value: str) -> str:
    """校验并规范化 A 股代码，允许 000001、000001.SZ 等格式。"""
    code = value.strip().upper()
    if not re.fullmatch(r"\d{6}(\.(SH|SZ|BJ))?", code):
        raise ValueError("目前在线分析入口仅支持 A 股代码，例如 000001、600519.SH")
    return code.split(".", 1)[0]


def fetch_history(code: str, years: int = 3) -> pd.DataFrame:
    """通过 AkShare 获取前复权日线，并统一成项目使用的英文列名。"""
    end = date.today()
    start = end - timedelta(days=365 * years + 30)
    raw = ak.stock_zh_a_hist(
        symbol=code,
        period="daily",
        start_date=start.strftime("%Y%m%d"),
        end_date=end.strftime("%Y%m%d"),
        adjust="qfq",
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"没有获取到 {code} 的行情数据，请检查代码或稍后重试")
    mapping = {
        "日期": "date", "开盘": "open", "收盘": "close",
        "最高": "high", "最低": "low", "成交量": "volume",
        "成交额": "amount", "涨跌幅": "pct_change",
    }
    frame = raw.rename(columns=mapping)
    required = ["date", "open", "close", "high", "low", "volume"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise RuntimeError(f"行情字段不完整，缺少：{', '.join(missing)}")
    frame = frame[required + [c for c in ["amount", "pct_change"] if c in frame.columns]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for col in required[1:]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date")
    if len(frame) < 60:
        raise RuntimeError(f"有效交易日只有 {len(frame)} 天，至少需要 60 天才能分析")
    return frame.reset_index(drop=True)


def add_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """计算均线、RSI、MACD、波动率和区间位置。"""
    data = frame.copy()
    close = data["close"]
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    data["rsi14"] = (100 - 100 / (1 + rs)).fillna(50)
    data["ma5"] = close.rolling(5).mean()
    data["ma10"] = close.rolling(10).mean()
    data["ma20"] = close.rolling(20).mean()
    data["ma60"] = close.rolling(60).mean()
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    data["macd"] = ema12 - ema26
    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()
    data["return_20d"] = close.pct_change(20) * 100
    data["volatility_20d"] = close.pct_change().rolling(20).std() * np.sqrt(252) * 100
    data["high_60d"] = data["high"].rolling(60).max()
    data["low_60d"] = data["low"].rolling(60).min()
    return data


def build_report(code: str, data: pd.DataFrame) -> str:
    last = data.iloc[-1]
    prev = data.iloc[-2]
    price = float(last["close"])
    daily_change = (price / float(prev["close"]) - 1) * 100
    ma20, ma60 = float(last["ma20"]), float(last["ma60"])
    rsi = float(last["rsi14"])
    macd_bull = float(last["macd"]) > float(last["macd_signal"])
    trend = "偏多" if price > ma20 > ma60 else ("偏空" if price < ma20 < ma60 else "震荡/分化")
    momentum = "偏强" if rsi >= 55 and macd_bull else ("偏弱" if rsi <= 45 and not macd_bull else "中性")
    score = 0
    score += 1 if price > ma20 else -1
    score += 1 if ma20 > ma60 else -1
    score += 1 if macd_bull else -1
    score += 1 if rsi >= 50 else -1
    stance = "技术面偏强" if score >= 2 else ("技术面偏弱" if score <= -2 else "技术面中性")
    high, low = float(last["high_60d"]), float(last["low_60d"])
    vol = float(last["volatility_20d"]) if pd.notna(last["volatility_20d"]) else float("nan")
    return f"""# {code} 股票技术分析

- 数据日期：{last['date'].date()}
- 最新收盘价：{price:.2f}
- 当日涨跌幅：{daily_change:+.2f}%
- 近 20 个交易日涨跌幅：{float(last['return_20d']):+.2f}%
- 近 20 日年化波动率估算：{vol:.2f}%

## 趋势与指标
- 趋势判断：**{trend}**（收盘价 {price:.2f}；MA20 {ma20:.2f}；MA60 {ma60:.2f}）
- 动能判断：**{momentum}**（RSI(14) {rsi:.1f}；MACD {'多头' if macd_bull else '空头'}）
- 60 日价格区间：{low:.2f} ～ {high:.2f}
- 综合技术信号：**{stance}**（简单规则评分 {score:+d}/4）

## 解读
- {'价格位于 MA20 与 MA60 上方，短中期结构相对积极。' if price > ma20 > ma60 else '价格位于 MA20 与 MA60 下方，短中期结构偏弱。' if price < ma20 < ma60 else '均线结构尚未形成一致方向，建议结合后续交易日确认。'}
- {'RSI 偏高，短线需留意回撤风险。' if rsi >= 70 else 'RSI 偏低，可能存在超跌反弹，也可能反映下行动能仍强。' if rsi <= 30 else 'RSI 未处于常见极端区间。'}
- {'MACD 位于信号线上方，动能信号偏正面。' if macd_bull else 'MACD 位于信号线下方，动能信号偏负面。'}
- 60 日区间高低点仅作历史参考，不代表确定的支撑位或阻力位。

## 风险提示
本报告是基于历史行情的规则化技术分析，不是对未来价格的可靠保证，也不是个性化投资建议。技术指标存在滞后性；请结合基本面、公告、流动性和自身风险承受能力独立判断。
"""


def save_chart(code: str, data: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data["date"], data["close"], label="Close", linewidth=1.2)
    ax.plot(data["date"], data["ma20"], label="MA20", linewidth=1)
    ax.plot(data["date"], data["ma60"], label="MA60", linewidth=1)
    ax.set_title(f"{code} - Adjusted Daily Close")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="A股在线行情分析")
    parser.add_argument("--code", required=True, help="股票代码，例如 000001 或 600519.SH")
    parser.add_argument("--output", default="output", help="报告输出目录")
    args = parser.parse_args()
    code = normalize_code(args.code)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    history = add_indicators(fetch_history(code))
    report = build_report(code, history)
    (out / f"{code}_analysis.md").write_text(report, encoding="utf-8")
    history.to_csv(out / f"{code}_history.csv", index=False, encoding="utf-8-sig")
    save_chart(code, history, out / f"{code}_chart.png")
    print(report)
    print(f"\n分析文件已生成：{out.resolve()}")


if __name__ == "__main__":
    main()
