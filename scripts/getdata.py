#!/usr/bin/env python
# coding: utf-8
"""
数据获取脚本
用于从各种数据源获取股票数据
"""
import sys
from pathlib import Path

# 添加 src 路径
# __file__：当前脚本路径，如 .../stock_prediction-master/scripts/getdata.py
# Path(__file__)：转为 Path 对象
# .resolve()：解析为绝对路径
# .parents[1]：上两级目录，即项目根目录（stock_prediction-master）
# / "src"：在项目根下拼接 src 目录
src_path = Path(__file__).resolve().parents[1] / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from stock_prediction.getdata import main

if __name__ == "__main__":
    main()