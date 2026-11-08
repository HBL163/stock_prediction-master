# 在线全维度股票研究系统

## 当前能力
- AkShare 获取 A 股前复权日线；统一日期、价格、成交量字段。
- 生成收益率、均线偏离、动量、波动率、成交量比、RSI、MACD、布林位置、ATR 等时序特征。
- 对未来 1/3/5/10 个交易日收益分别训练 ExtraTrees、RandomForest、HistGradientBoosting 候选模型。
- 按时间顺序做训练/验证，并在训练尾部隔离预测周期长度，降低未来标签重叠造成的数据泄漏。
- 若模型验证 MAE 不优于零收益基线，该周期预测自动回退为零收益基线。
- 使用验证残差 5%/95% 分位数构造经验预测区间。
- 提供验证集方向准确率、MAE、RMSE 和扣除假设交易成本后的简化多头回测。
- 自动聚合 Google News RSS 多语言新闻：公司/股票代码、行业链条，以及利率、贸易、政策等宏观主题；覆盖英语、简体中文和日语检索。\n- 新闻按标题去重，展示来源、时间、原文链接、规则情绪标签和主题计数；抓取失败/部分成功会明确记录，不把空结果伪装成成功。\n- 新闻情绪目前是透明的标题关键词规则，不是 FinBERT 或大语言模型；行业/宏观关联仅作背景提示，不代表对个股的确定性影响。
- 输出 Markdown 报告、JSON 预测数据、特征 CSV 和价格/RSI 图。

## GitHub Actions 使用方法
1. 打开仓库 Actions。
2. 选择“股票代码全维度分析”。
3. 点击 Run workflow，输入 A 股代码（例如 000001 或 600519.SH）。
4. 选择 2、3 或 5 年行情；公司资料和全球新闻自动纳入分析。
5. 运行成功后，在本次 workflow run 的 Artifacts 下载 ZIP。

## 本地运行
```bash
python -m pip install -r requirements-online.txt
python -m pytest -q tests/test_online_system.py tests/test_online_analyze.py tests/test_news_research.py
python scripts/online_analyze.py --code 000001 --years 3 --output output
```

## 方法边界
- 当前在线系统使用 scikit-learn 树模型集成，**不是**原仓库 README 中提到的 LSTM/Transformer/PTFT/VSSM 权重推理。
- 当前仓库快照未包含 README 所描述的 `src/stock_prediction/models/` 源码目录和训练权重，因此不能声称已完成旧深度学习模型加载。该缺口需在模型目录/权重恢复后再接入。
- 新闻情绪为词典规则；公司资料和新闻 API 为可选数据源，接口失败不会阻断行情建模。
- 验证和回测仅是历史研究，不等于实盘收益；未完整建模涨跌停、滑点、停牌、税费、资金约束和幸存者偏差。
- 90%经验区间来自验证残差分位数，不是统计覆盖保证或投资置信度。
