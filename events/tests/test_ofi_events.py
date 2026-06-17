"""Tests for events/ofi_events.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_OFI_EVENTS_PATH = Path(__file__).resolve().parent.parent / "ofi_events.py"
_SPEC = importlib.util.spec_from_file_location("ofi_events", _OFI_EVENTS_PATH)
ofi_events = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules["ofi_events"] = ofi_events
_SPEC.loader.exec_module(ofi_events)


def _zscore_series(values: list[float], start_second: int = 0) -> pd.Series:
    index = pd.date_range(
        pd.Timestamp("2026-06-13", tz="UTC") + pd.Timedelta(seconds=start_second),
        periods=len(values),
        freq="1s",
    )
    return pd.Series(values, index=index)


class TestDetectOfiEvents:
    def test_simple_threshold_crossing(self):
        values = [0.0] * 5 + [3.0] + [0.0] * 5
        zscore = _zscore_series(values)

        events = ofi_events.detect_ofi_events(zscore, threshold=2.0, cooldown_seconds=60)

        assert len(events) == 1
        assert events.loc[0, "timestamp"] == zscore.index[5]
        assert events.loc[0, "direction"] == "long"
        assert events.loc[0, "zscore_value"] == pytest.approx(3.0)

    def test_cooldown_suppresses_repeated_triggers(self):
        duration = 60 * 3
        values = [0.0] + [3.0] * duration
        zscore = _zscore_series(values)

        events = ofi_events.detect_ofi_events(
            zscore, threshold=2.0, cooldown_seconds=60, direction="long"
        )

        assert len(events) == 1
        assert events.loc[0, "timestamp"] == zscore.index[1]

    def test_cooldown_allows_new_event_after_expiry(self):
        values = [0.0, 3.0] + [0.0] * 120 + [3.5]
        zscore = _zscore_series(values)

        events = ofi_events.detect_ofi_events(
            zscore, threshold=2.0, cooldown_seconds=60, direction="long"
        )

        assert len(events) == 2
        assert events.loc[0, "timestamp"] == zscore.index[1]
        assert events.loc[1, "timestamp"] == zscore.index[-1]

    def test_cooldown_blocks_event_within_window(self):
        values = [0.0, 3.0] + [0.0] * 30 + [3.5]
        zscore = _zscore_series(values)

        events = ofi_events.detect_ofi_events(
            zscore, threshold=2.0, cooldown_seconds=60, direction="long"
        )

        assert len(events) == 1
        assert events.loc[0, "timestamp"] == zscore.index[1]

    def test_opposite_direction_not_blocked_by_cooldown(self):
        values = [0.0, 3.0, -3.0]
        zscore = _zscore_series(values)

        events = ofi_events.detect_ofi_events(
            zscore, threshold=2.0, cooldown_seconds=300, direction="both"
        )

        assert len(events) == 2
        assert events.loc[0, "direction"] == "long"
        assert events.loc[1, "direction"] == "short"

    def test_nan_values_never_trigger_events(self):
        values = [np.nan, 3.0, np.nan, -3.0]
        zscore = _zscore_series(values)

        events = ofi_events.detect_ofi_events(
            zscore, threshold=2.0, cooldown_seconds=60, direction="both"
        )

        assert len(events) == 2
        assert events["timestamp"].tolist() == [zscore.index[1], zscore.index[3]]

    def test_direction_long_only(self):
        values = [3.0, -3.0]
        zscore = _zscore_series(values)

        events = ofi_events.detect_ofi_events(
            zscore, threshold=2.0, cooldown_seconds=60, direction="long"
        )

        assert len(events) == 1
        assert events.loc[0, "direction"] == "long"

    def test_direction_short_only(self):
        values = [3.0, -3.0]
        zscore = _zscore_series(values)

        events = ofi_events.detect_ofi_events(
            zscore, threshold=2.0, cooldown_seconds=60, direction="short"
        )

        assert len(events) == 1
        assert events.loc[0, "direction"] == "short"


class TestSummarizeEvents:
    def test_summary_matches_hand_calculated_values(self):
        timestamps = pd.date_range("2026-06-13", periods=3, freq="1D", tz="UTC")
        events = pd.DataFrame(
            {
                "timestamp": timestamps,
                "direction": ["long", "short", "long"],
                "zscore_value": [2.0, -4.0, 6.0],
            }
        )

        summary = ofi_events.summarize_events(events)

        assert summary["total_events"] == 3
        assert summary["long_events"] == 2
        assert summary["short_events"] == 1
        assert summary["events_per_day"] == pytest.approx(1.5)
        assert summary["median_zscore_magnitude"] == pytest.approx(4.0)
        assert summary["max_zscore_magnitude"] == pytest.approx(6.0)


class TestValidation:
    def test_invalid_threshold_raises_value_error(self):
        zscore = _zscore_series([1.0, 2.0])

        with pytest.raises(ValueError, match="threshold"):
            ofi_events.detect_ofi_events(zscore, threshold=0.0)

    def test_invalid_direction_raises_value_error(self):
        zscore = _zscore_series([1.0, 2.0])

        with pytest.raises(ValueError, match="direction"):
            ofi_events.detect_ofi_events(zscore, threshold=1.0, direction="up")

    def test_non_monotonic_index_raises_value_error(self):
        index = pd.to_datetime([0, 2, 1], unit="s", utc=True)
        zscore = pd.Series([1.0, 2.0, 3.0], index=index)

        with pytest.raises(ValueError, match="monotonic"):
            ofi_events.detect_ofi_events(zscore, threshold=1.0)
