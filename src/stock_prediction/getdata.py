"""Data acquisition helpers for the stock_prediction package."""
from __future__ import annotations

import argparse
import datetime
import random
from pathlib import Path
from typing import Iterable, Sequence

try:
    from .init import (
        TQDM_NCOLS,
        NoneDataFrame,
        daily_path,
        pd,
        stock_data_queue,
        stock_list_queue,
        threading,
        time,
        tqdm,
    )
except ImportError:  # pragma: no cover
    import sys

    current_dir = Path(__file__).resolve().parent
    root_dir = current_dir.parent.parent
    src_dir = root_dir / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from stock_prediction.init import (
        TQDM_NCOLS,
        NoneDataFrame,
        daily_path,
        pd,
        stock_data_queue,
        stock_list_queue,
        threading,
        time,
        tqdm,
    )

try:
    import akshare as ak
except ImportError:
    ak = None


class DataConfig:
    """Runtime switches for akshare-based data fetching."""

    def __init__(self) -> None:
        self.adjust = "qfq"
        self.code = "0000001"


config = DataConfig()

def _rename_first_column(frame: pd.DataFrame, target_name: str) -> pd.DataFrame:
    """Rename the first column without referencing locale specific headers."""

    if frame.columns.empty:
        return frame
    first = frame.columns[0]
    if first != target_name:
        frame = frame.rename(columns={first: target_name})
    return frame


def get_stock_list() -> Sequence[str]:

    if ak is None:
        raise ImportError("akshare not installed")

    df = ak.stock_info_a_code_name()
    if df is None or df.empty:
        raise RuntimeError("get stk list fail")
    df.columns = [c.strip() for c in df.columns]
    if "代码" in df.columns:
        df = df.rename(columns={"代码": "code", "名称": "name"})
    #转成字符串，并补齐6位，如1转成000001
    df["code"] = df["code"].astype(str).str.zfill(6)
    df = df[~df["code"].str.startswith("8") & ~df["code"].str.startswith("4")]
    df = df[["code"]].drop_duplicates("code").reset_index(drop=True)
    stock_list = df["code"].astype(str).tolist()
    stock_list_queue.put(stock_list)
    return stock_list


    # iterable_from_code 的作用：把可能是单个代码或一堆代码的输入，统一成可遍历的字符串列表，方便后面用 for 循环处理。
    # ts_code 是字符串且非空：包装成 [ts_code]，如 "000001.SZ" → ["000001.SZ"]
    # ts_code 是空字符串：返回空列表 []
    # ts_code 是 Sequence（如列表、元组）：原样返回，如 ["000001.SZ", "600519.SH"] → 不变
    # 这样，无论是单个代码还是多个代码，都可以用同一个循环逻辑处理。
    # isinstance(ts_code, str) 用来判断 ts_code 是不是字符串类型。
    # 为 True：说明传入的是单个股票代码（一个 str），后面会包成列表或按空串处理。
    # 为 False：说明传入的是列表、元组等序列，后面直接 return ts_code 当可迭代对象用。
    # 这样就能区分「传了一个字符串」和「传了一串代码」两种调用方式。

def _iterable_from_code(ts_code: str | Sequence[str]) -> Iterable[str]:
    #Sequence[str] 是 Python 的类型提示，表示「字符串序列」——可索引、可遍历的一串字符串，不限定是列表还是元组。
    # Sequence：来自 typing，泛指 list、tuple、range 等有序容器。
    # [str]：表示元素都是 str。
    # 因此 Sequence[str] ≈ 元素类型为字符串的列表或元组，例如 ["000001.SZ", "600519.SH"] 或 ("000001.SZ",)。
    # 在这个函数里，ts_code 可以是一个 str，也可以是 Sequence[str]，所以需要用 isinstance(ts_code, str) 来区分。
    if isinstance(ts_code, str):
        if ts_code:
            return [ts_code]
        return []
    return ts_code


def get_stock_data(ts_code: Sequence[str] | str = "", save: bool = True, start_code: str = "", save_path: Path | str = "", datediff: int = -1):
    """Download historical bar data for the provided symbols."""

    if isinstance(save_path, str):
        save_path = Path(save_path)

    
    if ak is None:
        raise ImportError("akshare not installed")
        #如果用户给定stock list就用用户给定的list，否则用akshare获取股票list
    stock_list = list(_iterable_from_code(ts_code)) or get_stock_list()
    if start_code:
        stock_list = stock_list[stock_list.index(start_code):]  #注意看冒号切片
    # 进度条
    pbar = tqdm(total=len(stock_list), leave=False, ncols=TQDM_NCOLS) if save else None
    lock = threading.Lock()

    with lock:
        end_date = (datetime.datetime.now() + datetime.timedelta(days=datediff)).strftime("%Y%m%d")
        for code in stock_list:
            try:
                df = ak.stock_zh_a_hist(symbol=code, period="daily", end_date=end_date, adjust=config.adjust)
                df.columns = [
                    "trade_date",
                    "ts_code",
                    "open",
                    "close",
                    "high",
                    "low",
                    "vol",
                    "amount",
                    "amplitude",
                    "pct_change",
                    "change",
                    "exchange_rate",
                ]
                columns = list(df.columns)
                columns[0], columns[1] = columns[1], columns[0]
                df = df[columns]
                #把日期列转成YYYYMMDD格式
                df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y%m%d")
                df.sort_values(by=["trade_date"], ascending=False, inplace=True)
                df = df.reindex(columns=[
                    "ts_code",
                    "trade_date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "change",
                    "pct_change",
                    "vol",
                    "amount",
                    "amplitude",
                    "exchange_rate",
                ])
            except Exception as exc:  # pragma: no cover
                message = f"{code} {exc}"
                if save and pbar is not None:
                    tqdm.write(message)
                    pbar.update(1)
                else:
                    print(message)
                if getattr(exc, "args", []) and isinstance(exc.args[0], Exception):
                    inner = exc.args[0]
                    text = str(inner)
                    if "Connection aborted" in text or "Remote end closed connection" in text:
                        break
                continue

            time.sleep(random.uniform(0.1, 0.9))
            if save:
                save_path.mkdir(parents=True, exist_ok=True)
                df.to_csv(save_path / f"{code}.csv", index=False)
                if pbar is not None:
                    pbar.update(1)
            else:
                stock_data_queue.put(df if not df.empty else NoneDataFrame)
                return df if not df.empty else None

    if pbar is not None:
        pbar.close()
    return None


def main() -> None:
    """Command-line entry point for fetching quote data."""
    # 命令行参数解析
    #type=str ， 对应的那个参数会被解析成字符串
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", default="", type=str, help="single stock code or ticker")
    parser.add_argument("--api", default="akshare", type=str, help="api-provider: tushare, akshare or yfinance")
    parser.add_argument("--adjust", default="qfq", type=str, help="adjustment: none, qfq, or hfq")
    args = parser.parse_args()

    config.api = args.api
    config.adjust = args.adjust
    config.code = args.code
    if args.code:
        get_stock_data(args.code, save=True, save_path=daily_path)
    else:
        get_stock_data("000001", save=True, save_path=daily_path, datediff=-1)


if __name__ == "__main__":
    main()
