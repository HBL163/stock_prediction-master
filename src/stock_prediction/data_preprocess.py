"""Utilities for converting daily CSV data into serialized queues."""
from __future__ import annotations

import argparse
import os
import queue
import random
from pathlib import Path

try:
    from .common import load_data
    from .init import TQDM_NCOLS, data_queue, daily_path, dill, glob, pd, pkl_path, tqdm
except ImportError:  # pragma: no cover
    import sys

    current_dir = Path(__file__).resolve().parent
    root_dir = current_dir.parent.parent
    src_dir = root_dir / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from stock_prediction.common import load_data
    from stock_prediction.init import TQDM_NCOLS, data_queue, daily_path, dill, glob, pd, pkl_path, tqdm


def preprocess_data(pkl_name: str = "train.pkl") -> int:
    """Convert the prepared daily CSV files into a serialized queue for training."""

    # daily_dir = Path(daily_path)
    # 把传进来的 daily_path（通常是字符串，比如 "stock_daily" 或绝对路径）转换成一个 Path 对象，方便后面用 glob 等方法操作目录。
    # pkl_dir = Path(pkl_path)
    # 同理，把 pkl_path 转成 Path，用于后面保存 train.pkl 等文件。
    # csv_files = [str(path) for path in daily_dir.glob("*.csv")]
    # daily_dir.glob("*.csv")：在 daily_dir 目录下找到所有扩展名为 .csv 的文件，返回一个 Path 的迭代器。
    # 外面的列表推导式把每个 Path 转换成字符串路径，最后得到一个列表，比如：
    # ["stock_daily/000001.csv", "stock_daily/000002.csv", ...]。
    # ts_codes: list[str] = []
    # 显式声明 ts_codes 是一个“字符串列表”，用来存股票代码（ts_code）。
    # 初始为空，后面会从文件名中抽取代码放进去。
    # dump_queue: queue.Queue[pd.DataFrame] = queue.Queue()
    # 创建一个队列对象 dump_queue，类型提示为 queue.Queue[pd.DataFrame]，表示这个队列里会放很多 DataFrame。
    # 预处理时通常会把每个股票的处理结果 DataFrame 放进队列，然后再统一 dump 成 pkl。
    # for csv_file in csv_files:
    # 遍历前面收集到的每一个 CSV 文件路径字符串。
    # os.path.basename(csv_file).rsplit(".", 1)[0]
    # os.path.basename(csv_file)：取出文件名部分，去掉前面的目录。例如：
    # "stock_daily/000001.csv" → "000001.csv"
    # .rsplit(".", 1)：从右边开始最多切一刀，用 . 分割：
    # "000001.csv".rsplit(".", 1) → ["000001", "csv"]
    # [0]：取第一个元素 "000001"，也就是不带扩展名的股票代码。
    # 然后 ts_codes.append(...) 把每个股票代码依次加入列表 ts_codes。
    # random.shuffle(ts_codes)
    # 原地打乱 ts_codes 的顺序，让后面的处理/训练数据的股票顺序是随机的，这样能减少“按代码排序带来的顺序偏差”，对训练通常更有利。
    daily_dir = Path(daily_path)
    pkl_dir = Path(pkl_path)

    csv_files = [str(path) for path in daily_dir.glob("*.csv")]
    ts_codes: list[str] = []
    dump_queue: queue.Queue[pd.DataFrame] = queue.Queue()

    for csv_file in csv_files:
        ts_codes.append(os.path.basename(csv_file).rsplit(".", 1)[0])
    random.shuffle(ts_codes)

    load_data(ts_codes, True)
    pbar = tqdm(total=len(ts_codes), leave=False, ncols=TQDM_NCOLS)

    while not data_queue.empty():
        try:
            data = data_queue.get(timeout=1)
            if data.empty or data["ts_code"].iloc[0] == "None":
                tqdm.write("data is empty or has invalid columns")
                pbar.update(1)
                continue
            ts_code = data["ts_code"].iloc[0]
            dump_queue.put(data)
            pbar.update(1)
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"Error processing {ts_code}: {exc}")
            pbar.update(1)
            continue

    with open(pkl_dir / pkl_name, "wb") as fh:
        dill.dump(dump_queue, fh)
    pbar.close()
    print("dump_queue size:", dump_queue.qsize())
    return dump_queue.qsize()


def main() -> None:
    """Command-line entrypoint."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--pklname", default="train.pkl", type=str, help="output pkl filename")
    args = parser.parse_args()

    preprocess_data(args.pklname)


if __name__ == "__main__":
    main()
