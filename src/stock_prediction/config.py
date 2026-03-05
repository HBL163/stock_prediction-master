"""
Centralised configuration utilities for managing project paths.
"""
from __future__ import annotations

from pathlib import Path


class Config:
    """Project configuration helper providing canonical directory layout."""

    def __init__(self, root_path: str | Path | None = None) -> None:
        # Determine repository root
        if root_path is None:
            self.root_path = Path(__file__).resolve().parents[2]
        else:
            self.root_path = Path(root_path)

        # Ensure root exists before defining sub-paths
        self.root_path.mkdir(parents=True, exist_ok=True)

        # Data locations
        self.data_path = self.root_path / "stock_data"
        self.daily_path = self.root_path / "stock_daily"
        self.handle_path = self.root_path / "stock_handle"
        self.pkl_path = self.root_path / "pkl_handle"
        self.bert_data_path = self.root_path / "bert_data"

        # Output locations
        self.png_path = self.root_path / "png"
        self.output_path = self.root_path / "output"

        # Canonical files
        self.train_path = self.handle_path / "stock_train.csv"
        self.test_path = self.handle_path / "stock_test.csv"
        self.train_pkl_path = self.pkl_path / "train.pkl"

        # Model paths
        self.models_path = self.root_path / "models"
        self.lstm_path = self.models_path / "LSTM"
        self.transformer_path = self.models_path / "TRANSFORMER"
        self.cnnlstm_path = self.models_path / "CNNLSTM"

        # Pre-create frequently used directories
        self._create_directories()

    def _create_directories(self) -> None:
        """Create directories eagerly so downstream code can assume they exist."""

        directories = [
            self.data_path,
            self.daily_path,
            self.handle_path,
            self.pkl_path,
            self.bert_data_path,
            self.png_path,
            self.output_path,
            self.models_path,
            self.png_path / "train_loss",
            self.png_path / "predict",
            self.png_path / "test",
            self.bert_data_path / "model",
            self.bert_data_path / "data",
            self.data_path / "log",
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        # model_type：模型名，如 "LSTM"、"TRANSFORMER"、"ITRANSFORMER"。
        # symbol：股票代码，如 "000001.SZ" 或默认 "Generic.Data"。
        # symbol_clean = symbol.replace(".", "")：把代码里的点去掉，避免路径里出现 .（如 000001.SZ → 000001SZ）。
        # model_dir = self.models_path / symbol_clean / model_type.upper()：在 models/ 下按「股票 → 模型类型」建子目录，例如 models/000001SZ/LSTM/。
        # model_dir.mkdir(parents=True, exist_ok=True)：若目录不存在则创建（包括父目录）。
        # return model_dir / model_type.upper()：返回的是该目录下的「模型名前缀」，例如 models/000001SZ/LSTM/LSTM，实际存文件时会用这个前缀加后缀（如 LSTM.pth、LSTM_Model_args.json）。
    def get_model_path(self, model_type: str, symbol: str = "Generic.Data") -> Path:
        """Return the canonical checkpoint path for the given model type and symbol."""
        # 去掉点号
        symbol_clean = symbol.replace(".", "")
        model_dir = self.models_path / symbol_clean / model_type.upper()
        # 创建目录，包括上层父目录
        model_dir.mkdir(parents=True, exist_ok=True)
        # model_type.upper()----把model的值，全部大写
        return model_dir / model_type.upper()

    def __str__(self) -> str:
        return f"Config(root_path={self.root_path})"


# Global configuration instance for backwards compatibility
config = Config()

# Provide string aliases for legacy imports
root_path = str(config.root_path)
train_path = str(config.train_path)
test_path = str(config.test_path)
train_pkl_path = str(config.train_pkl_path)
png_path = str(config.png_path)
daily_path = str(config.daily_path)
handle_path = str(config.handle_path)
pkl_path = str(config.pkl_path)
bert_data_path = str(config.bert_data_path)
data_path = str(config.data_path)
lstm_path = str(config.lstm_path)
transformer_path = str(config.transformer_path)
cnnlstm_path = str(config.cnnlstm_path)
