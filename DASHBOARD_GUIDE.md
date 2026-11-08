# 交互式股票分析看板

本项目保留原有 PyTorch 训练 / 推理命令，同时新增一个不依赖本地 PKL 和预训练权重的单股票交互式分析入口。

## 本地运行

```bash
python -m pip install -r requirements-dashboard.txt
streamlit run streamlit_app.py
```

打开终端输出的本地地址，在侧边栏输入代码，例如：

- 美股：`AAPL`、`MSFT`、`TSLA`
- A 股：`600519`、`000001`、`300750`

A 股优先使用 AkShare；其他市场使用 Yahoo Finance。数据源可用性受网络、地区和上游服务限制。

## 部署到 Streamlit Community Cloud

1. 将本仓库推送到 GitHub（默认分支或本功能分支均可）。
2. 登录 Streamlit Community Cloud，选择 **New app**。
3. 选择仓库 `HBL163/stock_prediction-master`、分支 `feature/interactive-stock-dashboard`。
4. Main file path 填写 `streamlit_app.py`。
5. 点击 Deploy。平台会根据 `requirements-dashboard.txt` 安装依赖。

注意：GitHub 仓库本身用于存放源码和 GitHub Actions 自动检查；交互式网页需要 Streamlit Community Cloud 等应用托管服务，不能仅靠普通 GitHub 仓库页面提供动态 Python 服务。

## 设计说明与限制

- 复用原项目“行情采集、技术指标、时序建模、回测评估”的产品方向。
- 新增看板采用按股票代码即时拟合的轻量随机森林基线，避免原 CLI 推理入口对 `pkl_handle/train.pkl` 和模型权重的硬依赖。
- 指标包含均线、MACD、RSI、ATR、波动率、布林带位置、成交量比率。
- 使用时间顺序留出集评估 MAE、RMSE 和方向准确率，避免随机打乱时间序列。
- 预测区间来自留出集残差分位数，只是经验区间，不是保证覆盖率的置信区间。
- 预测结果仅供研究和学习，不构成投资建议；历史表现不代表未来收益。
