"""Tests for the Binance Futures research data loader."""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

import pandas as pd
import pytest

_LOADER_PATH = Path(__file__).resolve().parent.parent / "loader.py"
_SPEC = importlib.util.spec_from_file_location("loader", _LOADER_PATH)
loader = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules["loader"] = loader
_SPEC.loader.exec_module(loader)


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "DATA_ROOT", tmp_path)
    return tmp_path


def _write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _date_dir(data_root: Path, stream: str, symbol: str, date: str) -> Path:
    return data_root / stream / f"symbol={symbol}" / f"date={date}"


def _date_ms(date: str, *, hours: int = 0, minutes: int = 0, seconds: int = 0) -> int:
    ts = pd.Timestamp(date, tz="UTC") + pd.Timedelta(
        hours=hours, minutes=minutes, seconds=seconds
    )
    return int(ts.value // 1_000_000)


def _range_bounds(start: str, end: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = (
        pd.Timestamp(end, tz="UTC")
        + pd.Timedelta(days=1)
        - pd.Timedelta(milliseconds=1)
    )
    return start_ts, end_ts


def _trades_row(
    timestamp_ms: int,
    trade_id: int,
    *,
    symbol: str = "BTCUSDT",
    price: float = 100.0,
    quantity: float = 1.0,
    received_at_ms: int | None = None,
) -> dict:
    return {
        "timestamp": timestamp_ms,
        "symbol": symbol,
        "price": price,
        "quantity": quantity,
        "is_buyer_maker": False,
        "trade_id": trade_id,
        "is_recovered": False,
        "received_at": received_at_ms if received_at_ms is not None else timestamp_ms + 1,
    }


def _tob_row(
    event_time_ms: int,
    *,
    symbol: str = "BTCUSDT",
    best_bid_price: float = 100.0,
    best_ask_price: float = 100.1,
) -> dict:
    return {
        "event_time": event_time_ms,
        "symbol": symbol,
        "received_at": event_time_ms + 1,
        "best_bid_price": best_bid_price,
        "best_bid_qty": 1.0,
        "best_ask_price": best_ask_price,
        "best_ask_qty": 1.0,
        "spread": best_ask_price - best_bid_price,
        "spread_bps": 10.0,
        "mid_price": (best_bid_price + best_ask_price) / 2,
    }


def _liq_row(
    timestamp_ms: int,
    *,
    symbol: str = "BTCUSDT",
    price: float = 100.0,
    quantity: float = 1.0,
) -> dict:
    return {
        "timestamp": timestamp_ms,
        "symbol": symbol,
        "side": "SELL",
        "price": price,
        "quantity": quantity,
        "notional": price * quantity,
        "order_timestamp": timestamp_ms - 10,
        "received_at": timestamp_ms + 1,
    }


class TestTradesLoader:
    def test_duplicate_handling(self, data_root):
        date = "2026-06-13"
        rows = [
            _trades_row(_date_ms(date, hours=1), 1),
            _trades_row(_date_ms(date, hours=2), 1),
            _trades_row(_date_ms(date, hours=3), 2),
        ]
        _write_parquet(
            _date_dir(data_root, "trades", "BTCUSDT", date) / "trades.parquet",
            pd.DataFrame(rows),
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = loader.load_trades("BTCUSDT", date, date)

        assert len(result) == 2
        assert result["trade_id"].tolist() == [1, 2]
        assert any("duplicate" in str(w.message).lower() for w in caught)

    def test_nan_raises_data_quality_error(self, data_root):
        date = "2026-06-13"
        rows = [_trades_row(_date_ms(date, hours=1), 1, price=float("nan"))]
        _write_parquet(
            _date_dir(data_root, "trades", "BTCUSDT", date) / "trades.parquet",
            pd.DataFrame(rows),
        )

        with pytest.raises(loader.DataQualityError):
            loader.load_trades("BTCUSDT", date, date)

    def test_timestamp_conversion_and_sorting(self, data_root):
        date = "2026-06-13"
        rows = [
            _trades_row(_date_ms(date, hours=3), 3),
            _trades_row(_date_ms(date, hours=1), 1),
            _trades_row(_date_ms(date, hours=2), 2),
        ]
        _write_parquet(
            _date_dir(data_root, "trades", "BTCUSDT", date) / "trades.parquet",
            pd.DataFrame(rows),
        )

        result = loader.load_trades("BTCUSDT", date, date)

        assert result.index.name == "timestamp"
        assert str(result.index.tz) == "UTC"
        assert result.index.is_monotonic_increasing
        assert result["trade_id"].tolist() == [1, 2, 3]
        assert pd.api.types.is_datetime64_any_dtype(result["received_at"])

    def test_compacted_file_ignores_segments(self, data_root):
        date = "2026-06-13"
        date_dir = _date_dir(data_root, "trades", "BTCUSDT", date)
        _write_parquet(
            date_dir / "trades.parquet",
            pd.DataFrame([_trades_row(_date_ms(date, hours=1), 1)]),
        )
        _write_parquet(
            date_dir / "trades.segment.001.parquet",
            pd.DataFrame([_trades_row(_date_ms(date, hours=2), 99)]),
        )

        result = loader.load_trades("BTCUSDT", date, date)

        assert len(result) == 1
        assert result["trade_id"].iloc[0] == 1

    def test_segment_files_when_no_compacted(self, data_root):
        date = "2026-06-13"
        date_dir = _date_dir(data_root, "trades", "BTCUSDT", date)
        _write_parquet(
            date_dir / "trades.segment.001.parquet",
            pd.DataFrame([_trades_row(_date_ms(date, hours=1), 1)]),
        )
        _write_parquet(
            date_dir / "trades.segment.002.parquet",
            pd.DataFrame([_trades_row(_date_ms(date, hours=2), 2)]),
        )

        result = loader.load_trades("BTCUSDT", date, date)

        assert len(result) == 2
        assert sorted(result["trade_id"].tolist()) == [1, 2]

    def test_missing_data_raises_file_not_found(self, data_root):
        with pytest.raises(FileNotFoundError):
            loader.load_trades("BTCUSDT", "2026-06-13", "2026-06-15")

    def test_inclusive_date_range(self, data_root):
        dates = ["2026-06-13", "2026-06-14", "2026-06-15"]
        for offset, date in enumerate(dates):
            _write_parquet(
                _date_dir(data_root, "trades", "BTCUSDT", date) / "trades.parquet",
                pd.DataFrame(
                    [_trades_row(_date_ms(date, hours=12), offset + 1)]
                ),
            )

        start = "2026-06-13"
        end = "2026-06-15"
        result = loader.load_trades("BTCUSDT", start, end)

        assert len(result) == 3
        start_ts, end_ts = _range_bounds(start, end)
        assert result.index.min() >= start_ts
        assert result.index.max() <= end_ts

    def test_multi_day_timestamps_within_range(self, data_root):
        start = "2026-06-13"
        end = "2026-06-15"
        dates = ["2026-06-13", "2026-06-14", "2026-06-15"]
        for day_index, date in enumerate(dates):
            _write_parquet(
                _date_dir(data_root, "trades", "BTCUSDT", date) / "trades.parquet",
                pd.DataFrame(
                    [
                        _trades_row(_date_ms(date, hours=8), day_index * 10 + 1),
                        _trades_row(_date_ms(date, hours=16), day_index * 10 + 2),
                    ]
                ),
            )

        result = loader.load_trades("BTCUSDT", start, end)
        start_ts, end_ts = _range_bounds(start, end)

        assert len(result) == 6
        assert (result.index >= start_ts).all()
        assert (result.index <= end_ts).all()

    def test_received_at_correctness(self, data_root):
        date = "1970-01-01"
        timestamp_ms = 4_000
        received_at_ms = 5_000
        rows = [
            _trades_row(
                timestamp_ms,
                1,
                received_at_ms=received_at_ms,
            )
        ]
        _write_parquet(
            _date_dir(data_root, "trades", "BTCUSDT", date) / "trades.parquet",
            pd.DataFrame(rows),
        )

        result = loader.load_trades("BTCUSDT", date, date)

        assert result["received_at"].iloc[0] == pd.Timestamp(
            "1970-01-01 00:00:05", tz="UTC"
        )

    def test_duplicate_trade_id_different_symbols_kept(self, data_root):
        date = "2026-06-13"
        rows = [
            _trades_row(_date_ms(date, hours=1), 1, symbol="BTCUSDT"),
            _trades_row(_date_ms(date, hours=2), 1, symbol="ETHUSDT"),
        ]
        _write_parquet(
            _date_dir(data_root, "trades", "BTCUSDT", date) / "trades.parquet",
            pd.DataFrame(rows),
        )

        result = loader.load_trades("BTCUSDT", date, date)

        assert len(result) == 2
        assert set(result["symbol"]) == {"BTCUSDT", "ETHUSDT"}
        assert result["trade_id"].tolist() == [1, 1]


class TestTopOfBookLoader:
    def test_crossed_book_dropped_with_warning(self, data_root):
        date = "2026-06-13"
        rows = [
            _tob_row(
                _date_ms(date, hours=1),
                best_bid_price=100.5,
                best_ask_price=100.0,
            ),
            _tob_row(_date_ms(date, hours=2), best_bid_price=100.0, best_ask_price=100.1),
        ]
        _write_parquet(
            _date_dir(data_root, "top_of_book", "BTCUSDT", date)
            / "top_of_book.parquet",
            pd.DataFrame(rows),
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = loader.load_top_of_book("BTCUSDT", date, date)

        assert len(result) == 1
        assert result.index[0] == pd.Timestamp(
            f"{date} 02:00:00", tz="UTC"
        )
        assert any("crossed-book" in str(w.message).lower() for w in caught)

    def test_nan_raises_data_quality_error(self, data_root):
        date = "2026-06-13"
        rows = [_tob_row(_date_ms(date, hours=1), best_bid_price=float("nan"))]
        _write_parquet(
            _date_dir(data_root, "top_of_book", "BTCUSDT", date)
            / "top_of_book.parquet",
            pd.DataFrame(rows),
        )

        with pytest.raises(loader.DataQualityError):
            loader.load_top_of_book("BTCUSDT", date, date)

    def test_non_positive_price_raises_data_quality_error(self, data_root):
        date = "2026-06-13"
        rows = [_tob_row(_date_ms(date, hours=1), best_bid_price=0.0)]
        _write_parquet(
            _date_dir(data_root, "top_of_book", "BTCUSDT", date)
            / "top_of_book.parquet",
            pd.DataFrame(rows),
        )

        with pytest.raises(loader.DataQualityError):
            loader.load_top_of_book("BTCUSDT", date, date)

    def test_duplicate_rows_dropped(self, data_root):
        date = "2026-06-13"
        row = _tob_row(_date_ms(date, hours=1))
        _write_parquet(
            _date_dir(data_root, "top_of_book", "BTCUSDT", date)
            / "top_of_book.parquet",
            pd.DataFrame([row, row]),
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = loader.load_top_of_book("BTCUSDT", date, date)

        assert len(result) == 1
        assert any("duplicate" in str(w.message).lower() for w in caught)


class TestLiquidationsLoader:
    def test_nan_raises_data_quality_error(self, data_root):
        date = "2026-06-13"
        rows = [_liq_row(_date_ms(date, hours=1), quantity=float("nan"))]
        _write_parquet(
            _date_dir(data_root, "liquidations", "BTCUSDT", date)
            / "liquidations.parquet",
            pd.DataFrame(rows),
        )

        with pytest.raises(loader.DataQualityError):
            loader.load_liquidations("BTCUSDT", date, date)

    def test_timestamp_index_and_received_at(self, data_root):
        date = "2026-06-13"
        rows = [_liq_row(_date_ms(date, hours=5))]
        _write_parquet(
            _date_dir(data_root, "liquidations", "BTCUSDT", date)
            / "liquidations.parquet",
            pd.DataFrame(rows),
        )

        result = loader.load_liquidations("BTCUSDT", date, date)

        assert result.index.name == "timestamp"
        assert str(result.index.tz) == "UTC"
        assert pd.api.types.is_datetime64_any_dtype(result["received_at"])
        assert result["order_timestamp"].dtype == "int64"
