"""Return feature engineering for bar and tick data."""

from __future__ import annotations

import pandas as pd


def add_bar_returns(
    df: pd.DataFrame,
    price_col: str = "close",
    horizons: tuple[int, ...] = (1, 5, 15, 60),
) -> pd.DataFrame:
    """Add backward and forward bar returns for regularly-spaced OHLCV data.

    WARNING:
    This function assumes regularly-spaced bar data.
    Horizons are row offsets, NOT time offsets.
    Do not use on irregular tick data.
    """
    if price_col not in df.columns:
        raise ValueError(f"price column not found: {price_col}")
    if not df.index.is_monotonic_increasing:
        raise ValueError("index must be monotonic increasing")

    result = df.copy()
    price = result[price_col]

    for horizon in horizons:
        result[f"ret_{horizon}"] = price / price.shift(horizon) - 1
        result[f"fwd_ret_{horizon}"] = price.shift(-horizon) / price - 1

    return result


def compute_tick_forward_returns(
    mid_price: pd.Series,
    horizons_seconds: list[int],
) -> pd.DataFrame:
    """Compute timestamp-based forward returns for irregular tick data.

    Forward return uses the first observation at or after target_time
    (t + horizon_seconds), subject to tolerance. If no observation exists
    within tolerance, the corresponding return value must be NaN.
    """
    if mid_price.empty:
        raise ValueError("series is empty")
    if not isinstance(mid_price.index, pd.DatetimeIndex):
        raise ValueError("index must be a DatetimeIndex")
    if not mid_price.index.is_monotonic_increasing:
        raise ValueError("index must be monotonic increasing")
    if mid_price.index.has_duplicates:
        raise ValueError("index must not contain duplicate timestamps")

    index = pd.DatetimeIndex(mid_price.index).as_unit("ns")
    lookup = pd.DataFrame(
        {
            "timestamp": index,
            "price": mid_price.to_numpy(),
        }
    )

    result = pd.DataFrame(index=mid_price.index)
    price_now = mid_price.to_numpy()

    for horizon in horizons_seconds:
        tolerance = pd.Timedelta(seconds=horizon / 2)
        targets = pd.DataFrame(
            {
                "target_time": index + pd.Timedelta(seconds=horizon),
                "price_now": price_now,
                "_orig_order": range(len(mid_price)),
            }
        )

        merged = pd.merge_asof(
            targets.sort_values("target_time"),
            lookup.sort_values("timestamp"),
            left_on="target_time",
            right_on="timestamp",
            direction="forward",
            tolerance=tolerance,
        )
        merged = merged.sort_values("_orig_order")
        result[f"fwd_ret_{horizon}s"] = merged["price"].to_numpy() / merged[
            "price_now"
        ].to_numpy() - 1

    return result
