#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GitHub Actions / 本地 CLI：股票多周期机器学习分析。"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stock_prediction.online_system import (
    add_indicators, build_report, fetch_company_context, fetch_history,
    fit_forecasts, make_supervised_data, normalize_code, run_analysis,
)

__all__ = [
    "add_indicators", "build_report", "fetch_company_context", "fetch_history",
    "fit_forecasts", "make_supervised_data", "normalize_code", "run_analysis",
]

if __name__ == "__main__":
    from stock_prediction.online_system import main
    main()
