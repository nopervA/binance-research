"""Tests for gap detection in qa/check_window.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

_CHECK_WINDOW_PATH = Path(__file__).resolve().parent.parent / "check_window.py"
_SPEC = importlib.util.spec_from_file_location("check_window", _CHECK_WINDOW_PATH)
check_window = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules["check_window"] = check_window
_SPEC.loader.exec_module(check_window)


def _df_with_timestamps(timestamps: list[pd.Timestamp]) -> pd.DataFrame:
    index = pd.DatetimeIndex(timestamps, name="timestamp")
    return pd.DataFrame({"received_at": index}, index=index)


def _ts(date: str, hour: int, minute: int = 0, second: int = 0) -> pd.Timestamp:
    return pd.Timestamp(f"{date} {hour:02d}:{minute:02d}:{second:02d}", tz="UTC")


@pytest.fixture
def mock_loader(monkeypatch):
    datasets: dict[tuple[str, str], pd.DataFrame] = {}

    def make_loader(name: str):
        def loader(symbol: str, start: str, end: str) -> pd.DataFrame:
            del start, end
            return datasets[(symbol, name)]

        return loader

    monkeypatch.setitem(
        check_window.STREAM_LOADERS,
        "trades",
        make_loader("trades"),
    )
    monkeypatch.setitem(
        check_window.STREAM_LOADERS,
        "top_of_book",
        make_loader("top_of_book"),
    )

    class LoaderRegistry:
        def register(self, symbol: str, stream: str, df: pd.DataFrame) -> None:
            datasets[(symbol, stream)] = df

    return LoaderRegistry()


class TestFindGaps:
    def test_known_gap_detection(self, mock_loader):
        mock_loader.register(
            "BTCUSDT",
            "trades",
            _df_with_timestamps(
                [
                    _ts("2026-06-13", 0, 0, 0),
                    _ts("2026-06-13", 0, 0, 45),
                ]
            ),
        )

        result = check_window.find_gaps(
            "BTCUSDT", "2026-06-13", "2026-06-13", "trades", min_gap_seconds=10.0
        )

        assert len(result) == 1
        assert result.loc[0, "gap_seconds"] == pytest.approx(45.0)
        assert result.loc[0, "gap_start"] == _ts("2026-06-13", 0, 0, 0)
        assert result.loc[0, "gap_end"] == _ts("2026-06-13", 0, 0, 45)

    def test_threshold_filtering(self, mock_loader):
        mock_loader.register(
            "BTCUSDT",
            "trades",
            _df_with_timestamps(
                [
                    _ts("2026-06-13", 0, 0, 0),
                    _ts("2026-06-13", 0, 0, 5),
                ]
            ),
        )

        result = check_window.find_gaps(
            "BTCUSDT", "2026-06-13", "2026-06-13", "trades", min_gap_seconds=10.0
        )

        assert result.empty


class TestOverlapDetection:
    def test_overlap_detection(self):
        gaps = pd.DataFrame(
            [
                {
                    "gap_start": _ts("2026-06-13", 12, 0, 0),
                    "gap_end": _ts("2026-06-13", 12, 0, 45),
                    "gap_seconds": 45.0,
                    "symbol": "BTCUSDT",
                },
                {
                    "gap_start": _ts("2026-06-13", 12, 0, 10),
                    "gap_end": _ts("2026-06-13", 12, 0, 50),
                    "gap_seconds": 40.0,
                    "symbol": "ETHUSDT",
                },
            ]
        )

        overlapping = check_window.find_overlapping_symbols(gaps)

        assert overlapping == {"BTCUSDT", "ETHUSDT"}

    def test_no_false_overlap(self):
        gaps = pd.DataFrame(
            [
                {
                    "gap_start": _ts("2026-06-13", 12, 0, 0),
                    "gap_end": _ts("2026-06-13", 12, 0, 20),
                    "gap_seconds": 20.0,
                    "symbol": "BTCUSDT",
                },
                {
                    "gap_start": _ts("2026-06-13", 12, 1, 0),
                    "gap_end": _ts("2026-06-13", 12, 1, 30),
                    "gap_seconds": 30.0,
                    "symbol": "ETHUSDT",
                },
            ]
        )

        overlapping = check_window.find_overlapping_symbols(gaps)

        assert overlapping == set()

    def test_find_gaps_all_symbols_concatenates(self, mock_loader):
        mock_loader.register(
            "BTCUSDT",
            "trades",
            _df_with_timestamps(
                [
                    _ts("2026-06-13", 0, 0, 0),
                    _ts("2026-06-13", 0, 0, 45),
                ]
            ),
        )
        mock_loader.register(
            "ETHUSDT",
            "trades",
            _df_with_timestamps(
                [
                    _ts("2026-06-13", 1, 0, 0),
                    _ts("2026-06-13", 1, 0, 30),
                ]
            ),
        )

        result = check_window.find_gaps_all_symbols(
            ["BTCUSDT", "ETHUSDT"],
            "2026-06-13",
            "2026-06-13",
            "trades",
            min_gap_seconds=10.0,
        )

        assert len(result) == 2
        assert set(result["symbol"]) == {"BTCUSDT", "ETHUSDT"}
        assert result["gap_seconds"].tolist() == pytest.approx([45.0, 30.0])
