"""Tests for features/returns.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

_RETURNS_PATH = Path(__file__).resolve().parent.parent / "returns.py"
_SPEC = importlib.util.spec_from_file_location("returns", _RETURNS_PATH)
returns = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules["returns"] = returns
_SPEC.loader.exec_module(returns)


def _bar_df(prices: list[float]) -> pd.DataFrame:
    index = pd.date_range("2026-06-13", periods=len(prices), freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "close": prices,
            "volume": [1.0] * len(prices),
        },
        index=index,
    )


def _tick_series(prices: list[float], seconds: list[int]) -> pd.Series:
    index = pd.to_datetime(seconds, unit="s", utc=True)
    return pd.Series(prices, index=index, name="mid_price")


class TestAddBarReturns:
    def test_backward_return_correctness(self):
        df = _bar_df([100.0, 110.0, 121.0, 133.1])

        result = returns.add_bar_returns(df, horizons=(1,))

        assert result.loc[result.index[1], "ret_1"] == pytest.approx(0.10)
        assert result.loc[result.index[2], "ret_1"] == pytest.approx(0.10)
        assert pd.isna(result.loc[result.index[0], "ret_1"])

    def test_forward_return_correctness(self):
        df = _bar_df([100.0, 110.0, 121.0, 133.1])

        result = returns.add_bar_returns(df, horizons=(1,))

        assert result.loc[result.index[0], "fwd_ret_1"] == pytest.approx(0.10)
        assert result.loc[result.index[1], "fwd_ret_1"] == pytest.approx(0.10)
        assert pd.isna(result.loc[result.index[-1], "fwd_ret_1"])

    def test_input_dataframe_unchanged(self):
        df = _bar_df([100.0, 110.0, 121.0])
        original = df.copy()

        returns.add_bar_returns(df)

        pd.testing.assert_frame_equal(df, original)

    def test_missing_price_column_raises_value_error(self):
        df = _bar_df([100.0, 110.0, 121.0])

        with pytest.raises(ValueError, match="price column not found"):
            returns.add_bar_returns(df, price_col="open")

    def test_non_monotonic_index_raises_value_error(self):
        df = _bar_df([100.0, 110.0, 121.0])
        df = df.iloc[[0, 2, 1]]

        with pytest.raises(ValueError, match="monotonic increasing"):
            returns.add_bar_returns(df)

    def test_output_retains_original_columns(self):
        df = _bar_df([100.0, 110.0, 121.0, 133.1])

        result = returns.add_bar_returns(df, horizons=(1, 5))

        assert "close" in result.columns
        assert "volume" in result.columns
        assert "ret_1" in result.columns
        assert "fwd_ret_5" in result.columns

    def test_output_retains_original_index(self):
        df = _bar_df([100.0, 110.0, 121.0, 133.1])

        result = returns.add_bar_returns(df)

        pd.testing.assert_index_equal(result.index, df.index)


class TestComputeTickForwardReturns:
    def test_correctness_on_irregular_timestamps(self):
        series = _tick_series([100.0, 110.0, 121.0], [0, 5, 10])

        result = returns.compute_tick_forward_returns(series, [5, 10])

        assert result.loc[series.index[0], "fwd_ret_5s"] == pytest.approx(0.10)
        assert result.loc[series.index[0], "fwd_ret_10s"] == pytest.approx(0.21)
        assert result.loc[series.index[1], "fwd_ret_5s"] == pytest.approx(0.10)

    def test_timestamp_based_lookup_is_used(self):
        series = _tick_series([100.0, 200.0, 300.0], [0, 10, 30])

        result = returns.compute_tick_forward_returns(series, [10])

        assert result.loc[series.index[0], "fwd_ret_10s"] == pytest.approx(1.0)

    def test_row_offset_logic_would_produce_different_answer(self):
        series = _tick_series([100.0, 102.0, 120.0], [0, 2, 20])

        result = returns.compute_tick_forward_returns(series, [10])
        row_offset_return = series.shift(-1) / series - 1

        timestamp_return = result.loc[series.index[0], "fwd_ret_10s"]
        offset_return = row_offset_return.iloc[0]

        assert pd.isna(timestamp_return)
        assert offset_return == pytest.approx(0.02)
        assert timestamp_return != offset_return

    def test_tolerance_handling_produces_nan_when_appropriate(self):
        series = _tick_series([100.0, 200.0], [0, 20])

        result = returns.compute_tick_forward_returns(series, [10])

        assert pd.isna(result.loc[series.index[0], "fwd_ret_10s"])

    def test_input_series_unchanged(self):
        series = _tick_series([100.0, 110.0, 121.0], [0, 5, 10])
        original = series.copy()

        returns.compute_tick_forward_returns(series, [5])

        pd.testing.assert_series_equal(series, original)

    def test_non_monotonic_index_raises_value_error(self):
        index = pd.to_datetime([0, 20, 10], unit="s", utc=True)
        series = pd.Series([100.0, 110.0, 121.0], index=index)

        with pytest.raises(ValueError, match="monotonic increasing"):
            returns.compute_tick_forward_returns(series, [5])

    def test_non_datetime_index_raises_value_error(self):
        series = pd.Series([100.0, 110.0, 121.0], index=[0, 1, 2])

        with pytest.raises(ValueError, match="DatetimeIndex"):
            returns.compute_tick_forward_returns(series, [5])

    def test_empty_series_raises_value_error(self):
        series = pd.Series([], dtype=float, index=pd.DatetimeIndex([], tz="UTC"))

        with pytest.raises(ValueError, match="series is empty"):
            returns.compute_tick_forward_returns(series, [5])

    def test_does_not_match_pre_target_tick(self):
        series = _tick_series([100.0, 105.0, 108.0], [0, 9, 12])

        result = returns.compute_tick_forward_returns(series, [10])

        assert result.loc[series.index[0], "fwd_ret_10s"] == pytest.approx(0.08)
        assert result.loc[series.index[0], "fwd_ret_10s"] != pytest.approx(0.05)

    def test_duplicate_timestamps_raise_value_error(self):
        index = pd.to_datetime([0, 0, 10], unit="s", utc=True)
        series = pd.Series([100.0, 101.0, 110.0], index=index)

        with pytest.raises(ValueError, match="duplicate timestamps"):
            returns.compute_tick_forward_returns(series, [5])
