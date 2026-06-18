"""Order Flow Imbalance (OFI) feature engineering from signed trade volume."""

from __future__ import annotations

from datetime import timezone

import pandas as pd


def _validate_datetime_index(index: pd.Index) -> None:
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("index must be a DatetimeIndex")
    if index.tz is None or index.tz != timezone.utc:
        raise ValueError("index must be timezone-aware in UTC")
    if not index.is_monotonic_increasing:
        raise ValueError("index must be monotonic increasing")


def _inclusive_second_grid(start: str, end: str) -> pd.DatetimeIndex:
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return pd.date_range(start=start_ts, end=end_ts, freq="1s")


def compute_signed_volume(trades: pd.DataFrame) -> pd.Series:
    """Return trade-level signed volume from taker direction."""
    _validate_datetime_index(trades.index)
    return trades["quantity"].where(~trades["is_buyer_maker"], -trades["quantity"])


def resample_to_1s_grid(
    signed_volume: pd.Series,
    start: str,
    end: str,
) -> pd.Series:
    """Resample signed volume to a complete 1-second UTC grid."""
    _validate_datetime_index(signed_volume.index)
    grid = _inclusive_second_grid(start, end)
    resampled = signed_volume.resample("1s").sum()
    return resampled.reindex(grid, fill_value=0.0)


def compute_ofi(
    signed_volume_1s: pd.Series,
    window_seconds: int = 60,
) -> pd.Series:
    """Rolling sum of 1-second signed volume over a fixed window."""
    _validate_datetime_index(signed_volume_1s.index)
    return signed_volume_1s.rolling(
        window=window_seconds,
        min_periods=window_seconds,
    ).sum()


def compute_ofi_zscore(
    ofi: pd.Series,
    lookback_seconds: int = 3600,
) -> pd.Series:
    """Rolling z-score of OFI over a fixed lookback window."""
    _validate_datetime_index(ofi.index)
    rolling_mean = ofi.rolling(
        window=lookback_seconds,
        min_periods=lookback_seconds,
    ).mean()
    rolling_std = ofi.rolling(
        window=lookback_seconds,
        min_periods=lookback_seconds,
    ).std()
    zscore = (ofi - rolling_mean) / rolling_std
    return zscore.where(rolling_std != 0)


def compute_normalized_ofi(
    trades: pd.DataFrame,
    window_seconds: int = 60,
) -> pd.Series:
    """Compute volume-normalized OFI on a 1-second UTC grid.

    Mathematical definition:

        normalized_ofi = rolling_signed_volume_sum / rolling_total_volume_sum

    where signed_volume is +quantity for taker buys and -quantity for taker
    sells, total_volume is quantity regardless of side, and both rolling sums
    use window_seconds with min_periods=window_seconds.

    Expected finite output range is [-1.0, +1.0]. Values are NaN when the
    rolling total volume sum is zero. The returned Series uses the same 1-second
    UTC grid structure as the existing OFI pipeline and is directly compatible
    with compute_ofi_zscore().
    """
    _validate_datetime_index(trades.index)
    if "quantity" not in trades.columns:
        raise ValueError("quantity column is required")
    if "is_buyer_maker" not in trades.columns:
        raise ValueError("is_buyer_maker column is required")

    signed_volume = trades["quantity"].where(~trades["is_buyer_maker"], -trades["quantity"])
    total_volume = trades["quantity"]

    start = trades.index.min().normalize().strftime("%Y-%m-%d")
    end = trades.index.max().normalize().strftime("%Y-%m-%d")
    grid = _inclusive_second_grid(start, end)

    volumes_1s = (
        pd.DataFrame(
            {
                "signed_volume": signed_volume,
                "total_volume": total_volume,
            }
        )
        .resample("1s")
        .sum()
        .reindex(grid, fill_value=0.0)
    )

    signed_sum = volumes_1s["signed_volume"].rolling(
        window=window_seconds,
        min_periods=window_seconds,
    ).sum()
    total_sum = volumes_1s["total_volume"].rolling(
        window=window_seconds,
        min_periods=window_seconds,
    ).sum()

    normalized_ofi = signed_sum / total_sum
    return normalized_ofi.where(total_sum != 0)
