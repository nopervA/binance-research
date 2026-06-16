"""Production-quality data loader for Binance Futures research datasets."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

DATA_ROOT = Path("/var/lib/binance-futures-collector/data")


class DataQualityError(Exception):
    pass


def _list_parquet_files(
    data_root: Path, stream: str, symbol: str, date: str
) -> list[Path]:
    date_dir = data_root / stream / f"symbol={symbol}" / f"date={date}"
    if not date_dir.is_dir():
        return []

    compacted = date_dir / f"{stream}.parquet"
    if compacted.is_file():
        return [compacted]

    return sorted(date_dir.glob("*.segment.*.parquet"))


def _read_and_concat(files: list[Path]) -> pd.DataFrame:
    frames = [pd.read_parquet(path) for path in files]
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, ignore_index=True)


def _convert_timestamps(df: pd.DataFrame, ts_cols: list[str]) -> pd.DataFrame:
    result = df.copy()
    for col in ts_cols:
        if col in result.columns:
            result[col] = pd.to_datetime(result[col], unit="ms", utc=True)
    return result


def _iter_dates(start: str, end: str) -> list[str]:
    return [
        day.strftime("%Y-%m-%d")
        for day in pd.date_range(start=start, end=end, freq="D")
    ]


def _collect_files(
    data_root: Path, stream: str, symbol: str, start: str, end: str
) -> list[Path]:
    files: list[Path] = []
    for date in _iter_dates(start, end):
        files.extend(_list_parquet_files(data_root, stream, symbol, date))
    return files


def _set_timestamp_index(df: pd.DataFrame, index_col: str) -> pd.DataFrame:
    indexed = df.set_index(index_col)
    indexed.index.name = "timestamp"
    return indexed


def _filter_timestamp_range(
    df: pd.DataFrame, start: str, end: str
) -> pd.DataFrame:
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = (
        pd.Timestamp(end, tz="UTC")
        + pd.Timedelta(days=1)
        - pd.Timedelta(milliseconds=1)
    )
    return df[(df.index >= start_ts) & (df.index <= end_ts)]


def _validate_no_nan(df: pd.DataFrame, columns: list[str]) -> None:
    for col in columns:
        if col in df.columns and df[col].isna().any():
            raise DataQualityError(f"{col} contains NaN values")


def _dedupe_with_warning(
    df: pd.DataFrame, dedupe_fn: Callable[[pd.DataFrame], pd.DataFrame]
) -> pd.DataFrame:
    before = len(df)
    result = dedupe_fn(df)
    removed = before - len(result)
    if removed > 0:
        warnings.warn(
            f"Removed {removed} duplicate row(s)",
            stacklevel=3,
        )
    return result


def _dedupe_trades(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset=["symbol", "trade_id"], keep="first")


def _dedupe_full_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(keep="first")


def _validate_trades(df: pd.DataFrame) -> pd.DataFrame:
    _validate_no_nan(df, ["price", "quantity"])
    return df


def _validate_liquidations(df: pd.DataFrame) -> pd.DataFrame:
    _validate_no_nan(df, ["price", "quantity"])
    return df


def _validate_top_of_book(df: pd.DataFrame) -> pd.DataFrame:
    _validate_no_nan(df, ["best_bid_price", "best_ask_price"])

    non_positive = (df["best_bid_price"] <= 0) | (df["best_ask_price"] <= 0)
    if non_positive.any():
        raise DataQualityError(
            "best_bid_price and best_ask_price must be positive"
        )

    crossed = df["best_bid_price"] > df["best_ask_price"]
    crossed_count = int(crossed.sum())
    if crossed_count:
        warnings.warn(
            f"Dropped {crossed_count} crossed-book row(s) "
            f"where best_bid_price > best_ask_price",
            stacklevel=3,
        )
        df = df[~crossed]
    return df


@dataclass(frozen=True)
class _StreamConfig:
    stream: str
    index_col: str
    datetime_cols: tuple[str, ...]
    deduplicate: Callable[[pd.DataFrame], pd.DataFrame]
    validate: Callable[[pd.DataFrame], pd.DataFrame]


_STREAM_CONFIGS: dict[str, _StreamConfig] = {
    "trades": _StreamConfig(
        stream="trades",
        index_col="timestamp",
        datetime_cols=("timestamp", "received_at"),
        deduplicate=_dedupe_trades,
        validate=_validate_trades,
    ),
    "top_of_book": _StreamConfig(
        stream="top_of_book",
        index_col="event_time",
        datetime_cols=("event_time", "received_at"),
        deduplicate=_dedupe_full_rows,
        validate=_validate_top_of_book,
    ),
    "liquidations": _StreamConfig(
        stream="liquidations",
        index_col="timestamp",
        datetime_cols=("timestamp", "received_at"),
        deduplicate=_dedupe_full_rows,
        validate=_validate_liquidations,
    ),
}


def _load_stream(
    config: _StreamConfig,
    symbol: str,
    start: str,
    end: str,
    *,
    data_root: Path | None = None,
) -> pd.DataFrame:
    root = DATA_ROOT if data_root is None else data_root
    files = _collect_files(root, config.stream, symbol, start, end)
    if not files:
        raise FileNotFoundError(
            f"No {config.stream} data found for {symbol} from {start} to {end}"
        )

    df = _read_and_concat(files)
    df = _convert_timestamps(df, list(config.datetime_cols))
    df = _set_timestamp_index(df, config.index_col)
    df = _filter_timestamp_range(df, start, end)
    df = _dedupe_with_warning(df, config.deduplicate)
    df = config.validate(df)
    return df.sort_index()


def load_trades(symbol: str, start: str, end: str) -> pd.DataFrame:
    return _load_stream(_STREAM_CONFIGS["trades"], symbol, start, end)


def load_top_of_book(symbol: str, start: str, end: str) -> pd.DataFrame:
    return _load_stream(_STREAM_CONFIGS["top_of_book"], symbol, start, end)


def load_liquidations(symbol: str, start: str, end: str) -> pd.DataFrame:
    return _load_stream(_STREAM_CONFIGS["liquidations"], symbol, start, end)
