"""Tests for features/ofi.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_OFI_PATH = Path(__file__).resolve().parent.parent / "ofi.py"
_SPEC = importlib.util.spec_from_file_location("ofi", _OFI_PATH)
ofi = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules["ofi"] = ofi
_SPEC.loader.exec_module(ofi)


def _trades_df(
    *,
    seconds: list[int],
    quantities: list[float],
    is_buyer_maker: list[bool],
) -> pd.DataFrame:
    index = pd.to_datetime(seconds, unit="s", utc=True)
    return pd.DataFrame(
        {
            "symbol": ["BTCUSDT"] * len(seconds),
            "price": [100.0] * len(seconds),
            "quantity": quantities,
            "is_buyer_maker": is_buyer_maker,
            "trade_id": list(range(len(seconds))),
            "is_recovered": [False] * len(seconds),
            "received_at": index,
        },
        index=index,
    )


def _utc_series(values: list[float], seconds: list[int]) -> pd.Series:
    index = pd.to_datetime(seconds, unit="s", utc=True)
    return pd.Series(values, index=index)


def _grid_length(start: str, end: str) -> int:
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return int((end_ts - start_ts).total_seconds()) + 1


class TestComputeSignedVolume:
    def test_all_taker_buys_positive(self):
        trades = _trades_df(
            seconds=[0, 1, 2],
            quantities=[1.5, 2.0, 0.5],
            is_buyer_maker=[False, False, False],
        )

        result = ofi.compute_signed_volume(trades)

        assert (result > 0).all()
        pd.testing.assert_series_equal(result, trades["quantity"])

    def test_all_taker_sells_negative(self):
        trades = _trades_df(
            seconds=[0, 1],
            quantities=[3.0, 4.0],
            is_buyer_maker=[True, True],
        )

        result = ofi.compute_signed_volume(trades)

        assert (result < 0).all()
        pd.testing.assert_series_equal(result, -trades["quantity"])

    def test_mixed_signed_values(self):
        trades = _trades_df(
            seconds=[0, 1, 2],
            quantities=[1.0, 2.0, 3.0],
            is_buyer_maker=[False, True, False],
        )

        result = ofi.compute_signed_volume(trades)

        assert result.tolist() == pytest.approx([1.0, -2.0, 3.0])


class TestResampleTo1sGrid:
    def test_absent_seconds_filled_with_zero(self):
        signed = _utc_series([2.0, -1.0], [0, 2])
        start = "1970-01-01"
        end = "1970-01-01"

        result = ofi.resample_to_1s_grid(signed, start, end)

        gap_ts = pd.Timestamp("1970-01-01 00:00:01", tz="UTC")
        assert result.loc[gap_ts] == 0.0
        assert result.loc[signed.index[0]] == 2.0
        assert result.loc[signed.index[1]] == -1.0
        assert not result.isna().any()

    def test_output_covers_full_inclusive_range(self):
        signed = _utc_series([1.0], [0])
        start = "1970-01-01"
        end = "1970-01-01"

        result = ofi.resample_to_1s_grid(signed, start, end)

        expected_index = pd.date_range(
            pd.Timestamp(start, tz="UTC"),
            pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1),
            freq="1s",
        )
        assert len(result) == _grid_length(start, end)
        assert len(result) == 86400
        pd.testing.assert_index_equal(result.index, expected_index)


class TestComputeOfi:
    def test_rolling_sum_matches_hand_calculated_values(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        signed_1s = _utc_series(values, list(range(len(values))))

        result = ofi.compute_ofi(signed_1s, window_seconds=3)

        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == pytest.approx(6.0)
        assert result.iloc[3] == pytest.approx(9.0)
        assert result.iloc[4] == pytest.approx(12.0)

    def test_initial_values_are_nan_for_full_window(self):
        signed_1s = _utc_series([1.0] * 120, list(range(120)))

        result = ofi.compute_ofi(signed_1s, window_seconds=60)

        assert result.iloc[:59].isna().all()
        assert result.iloc[59] == pytest.approx(60.0)


class TestComputeOfiZscore:
    def test_zero_variance_produces_nan(self):
        constant = _utc_series([5.0] * 10, list(range(10)))

        result = ofi.compute_ofi_zscore(constant, lookback_seconds=3)

        assert result.iloc[2:].isna().all()
        assert np.isfinite(result.iloc[:2]).sum() == 0

    def test_zscore_matches_hand_calculated_value(self):
        values = [1.0, 2.0, 3.0, 4.0]
        series = _utc_series(values, list(range(len(values))))

        result = ofi.compute_ofi_zscore(series, lookback_seconds=3)

        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == pytest.approx(1.0)
        assert result.iloc[3] == pytest.approx(1.0)

    def test_initial_values_are_nan_for_full_lookback(self):
        values = list(range(100))
        series = _utc_series([float(v) for v in values], values)

        result = ofi.compute_ofi_zscore(series, lookback_seconds=60)

        assert result.iloc[:59].isna().all()
        assert not pd.isna(result.iloc[59])


class TestValidation:
    def test_non_datetime_index_raises_value_error(self):
        series = pd.Series([1.0, 2.0], index=[0, 1])

        with pytest.raises(ValueError, match="DatetimeIndex"):
            ofi.compute_ofi(series)

    def test_tz_naive_datetime_index_raises_value_error(self):
        index = pd.to_datetime(["2026-06-13 00:00:00", "2026-06-13 00:00:01"])
        series = pd.Series([1.0, 2.0], index=index)

        with pytest.raises(ValueError, match="UTC"):
            ofi.compute_ofi_zscore(series)

    def test_non_monotonic_index_raises_value_error(self):
        series = _utc_series([1.0, 2.0, 3.0], [0, 2, 1])

        with pytest.raises(ValueError, match="monotonic increasing"):
            ofi.resample_to_1s_grid(series, "1970-01-01", "1970-01-01")


class TestComputeNormalizedOfi:
    def test_all_buy_volume_returns_plus_one(self):
        window_seconds = 60
        trades = _trades_df(
            seconds=list(range(window_seconds)),
            quantities=[1.0] * window_seconds,
            is_buyer_maker=[False] * window_seconds,
        )

        result = ofi.compute_normalized_ofi(trades, window_seconds=window_seconds)

        assert result.iloc[window_seconds - 1] == pytest.approx(1.0)

    def test_all_sell_volume_returns_minus_one(self):
        window_seconds = 60
        trades = _trades_df(
            seconds=list(range(window_seconds)),
            quantities=[2.0] * window_seconds,
            is_buyer_maker=[True] * window_seconds,
        )

        result = ofi.compute_normalized_ofi(trades, window_seconds=window_seconds)

        assert result.iloc[window_seconds - 1] == pytest.approx(-1.0)

    def test_equal_buy_and_sell_volume_returns_zero(self):
        window_seconds = 60
        seconds = list(range(window_seconds))
        trades = _trades_df(
            seconds=seconds,
            quantities=[1.0] * window_seconds,
            is_buyer_maker=[False, True] * (window_seconds // 2),
        )

        result = ofi.compute_normalized_ofi(trades, window_seconds=window_seconds)

        assert result.iloc[window_seconds - 1] == pytest.approx(0.0)

    def test_zero_total_volume_returns_nan(self):
        trades = _trades_df(
            seconds=[0],
            quantities=[1.0],
            is_buyer_maker=[False],
        )

        result = ofi.compute_normalized_ofi(trades, window_seconds=60)
        zero_volume_point = result.iloc[119]

        assert pd.isna(zero_volume_point)
        assert not np.isinf(zero_volume_point)
        assert zero_volume_point != 0

    def test_min_periods_enforced(self):
        window_seconds = 60
        trades = _trades_df(
            seconds=list(range(window_seconds)),
            quantities=[1.0] * window_seconds,
            is_buyer_maker=[False] * window_seconds,
        )

        result = ofi.compute_normalized_ofi(trades, window_seconds=window_seconds)

        assert result.iloc[: window_seconds - 1].isna().all()

    def test_output_compatible_with_compute_ofi_zscore(self):
        window_seconds = 60
        seconds = list(range(120))
        trades = _trades_df(
            seconds=seconds,
            quantities=[1.0] * len(seconds),
            is_buyer_maker=[False, True] * (len(seconds) // 2),
        )

        normalized = ofi.compute_normalized_ofi(trades, window_seconds=window_seconds)
        zscore = ofi.compute_ofi_zscore(normalized, lookback_seconds=60)

        assert len(zscore) == len(normalized)
        assert isinstance(zscore.index, pd.DatetimeIndex)

    def test_output_range_respected(self):
        seconds = list(range(120))
        trades = _trades_df(
            seconds=seconds,
            quantities=[1.0, 2.0, 3.0, 4.0] * 30,
            is_buyer_maker=[False, True, False, True] * 30,
        )

        result = ofi.compute_normalized_ofi(trades, window_seconds=60)
        finite_values = result[np.isfinite(result)]

        assert (finite_values >= -1.0).all()
        assert (finite_values <= 1.0).all()
