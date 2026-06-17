"""OFI z-score threshold event detection with per-direction cooldown."""

from __future__ import annotations

from datetime import timezone

import pandas as pd


def _validate_zscore_index(index: pd.Index) -> None:
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("zscore index must be a DatetimeIndex")
    if index.tz is None or index.tz != timezone.utc:
        raise ValueError("zscore index must be a monotonic increasing UTC DatetimeIndex")
    if not index.is_monotonic_increasing:
        raise ValueError("zscore index must be a monotonic increasing UTC DatetimeIndex")


def detect_ofi_events(
    zscore: pd.Series,
    threshold: float,
    cooldown_seconds: int = 300,
    direction: str = "both",
) -> pd.DataFrame:
    """Detect OFI z-score threshold crossings with per-direction cooldown.

    Cooldown is tracked independently for long and short directions because
    cooldown is inherently sequential state. A single loop over timestamps
    is used for that step rather than a vectorized look-back, which is
    easier to verify for correctness.
    """
    _validate_zscore_index(zscore.index)

    if threshold <= 0:
        raise ValueError("threshold must be greater than 0")
    if direction not in ("long", "short", "both"):
        raise ValueError('direction must be one of "long", "short", or "both"')

    detect_long = direction in ("long", "both")
    detect_short = direction in ("short", "both")
    cooldown = pd.Timedelta(seconds=cooldown_seconds)

    events: list[dict[str, object]] = []
    last_long_event: pd.Timestamp | None = None
    last_short_event: pd.Timestamp | None = None
    previous_value: float | None = None

    for timestamp, value in zscore.items():
        if pd.isna(value):
            previous_value = None
            continue

        long_cross = detect_long and value > threshold and (
            previous_value is None or previous_value <= threshold
        )
        short_cross = detect_short and value < -threshold and (
            previous_value is None or previous_value >= -threshold
        )

        if long_cross:
            if last_long_event is None or timestamp - last_long_event >= cooldown:
                events.append(
                    {
                        "timestamp": timestamp,
                        "direction": "long",
                        "zscore_value": float(value),
                    }
                )
                last_long_event = timestamp

        if short_cross:
            if last_short_event is None or timestamp - last_short_event >= cooldown:
                events.append(
                    {
                        "timestamp": timestamp,
                        "direction": "short",
                        "zscore_value": float(value),
                    }
                )
                last_short_event = timestamp

        previous_value = float(value)

    if not events:
        return pd.DataFrame(columns=["timestamp", "direction", "zscore_value"])

    return pd.DataFrame(events).sort_values("timestamp").reset_index(drop=True)


def summarize_events(events: pd.DataFrame) -> dict[str, float | int]:
    """Summarize detected OFI events."""
    if events.empty:
        return {
            "total_events": 0,
            "long_events": 0,
            "short_events": 0,
            "events_per_day": 0.0,
            "median_zscore_magnitude": float("nan"),
            "max_zscore_magnitude": float("nan"),
        }

    magnitudes = events["zscore_value"].abs()
    span_days = (
        events["timestamp"].iloc[-1] - events["timestamp"].iloc[0]
    ).total_seconds() / 86400

    if span_days == 0:
        events_per_day = float(len(events))
    else:
        events_per_day = len(events) / span_days

    return {
        "total_events": int(len(events)),
        "long_events": int((events["direction"] == "long").sum()),
        "short_events": int((events["direction"] == "short").sum()),
        "events_per_day": float(events_per_day),
        "median_zscore_magnitude": float(magnitudes.median()),
        "max_zscore_magnitude": float(magnitudes.max()),
    }
