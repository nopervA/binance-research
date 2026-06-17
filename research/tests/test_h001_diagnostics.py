"""Tests for research/h001_diagnostics.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

_H001_PATH = Path(__file__).resolve().parent.parent / "h001_diagnostics.py"
_SPEC = importlib.util.spec_from_file_location("h001_diagnostics", _H001_PATH)
h001_diagnostics = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules["h001_diagnostics"] = h001_diagnostics
_SPEC.loader.exec_module(h001_diagnostics)


def _events_df(
    timestamps: list[pd.Timestamp],
    *,
    directions: list[str] | None = None,
    zscores: list[float] | None = None,
) -> pd.DataFrame:
    directions = directions or ["long"] * len(timestamps)
    zscores = zscores or [2.0] * len(timestamps)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "direction": directions,
            "zscore_value": zscores,
        }
    )


def _mid_price(
    seconds: list[int],
    prices: list[float],
    *,
    base: str = "2026-06-13",
) -> pd.Series:
    base_ts = pd.Timestamp(base, tz="UTC")
    index = base_ts + pd.to_timedelta(seconds, unit="s")
    return pd.Series(prices, index=index, name="mid_price")


class TestGenerateAlignmentDiagnostic:
    def test_known_alignment_matches_hand_calculated_values(self):
        event_ts = pd.Timestamp("2026-06-13 00:00:00", tz="UTC")
        events = _events_df([event_ts], zscores=[2.5])
        mid = _mid_price([0, 300], [100.0, 110.0])

        result = h001_diagnostics.generate_alignment_diagnostic(
            events, mid, horizon_seconds=300, n_events=1
        )

        assert result.loc[0, "time_delta_seconds"] == pytest.approx(0.0)
        assert result.loc[0, "forward_return_used"] == pytest.approx(0.10)
        assert result.loc[0, "price_at_event"] == pytest.approx(100.0)
        assert result.loc[0, "price_at_horizon"] == pytest.approx(110.0)

    def test_missing_match_produces_nan_not_exception(self):
        event_ts = pd.Timestamp("2026-06-13 00:00:00", tz="UTC")
        events = _events_df([event_ts])
        mid = _mid_price([1000], [100.0])

        result = h001_diagnostics.generate_alignment_diagnostic(
            events, mid, horizon_seconds=300, n_events=1
        )

        assert pd.isna(result.loc[0, "price_at_event"])
        assert pd.isna(result.loc[0, "price_at_horizon"])
        assert pd.isna(result.loc[0, "forward_return_used"])

    def test_output_sorted_chronologically(self):
        timestamps = [
            pd.Timestamp("2026-06-13 00:00:10", tz="UTC"),
            pd.Timestamp("2026-06-13 00:00:00", tz="UTC"),
            pd.Timestamp("2026-06-13 00:00:05", tz="UTC"),
        ]
        events = _events_df(timestamps)
        mid = _mid_price(list(range(20)), [100.0 + i for i in range(20)])

        result = h001_diagnostics.generate_alignment_diagnostic(
            events,
            mid,
            horizon_seconds=5,
            n_events=3,
            sample_mode="random",
            random_seed=7,
        )

        assert result["event_timestamp"].is_monotonic_increasing

    def test_price_at_event_uses_forward_matching(self):
        event_ts = pd.Timestamp("2026-06-13 00:00:10", tz="UTC")
        events = _events_df([event_ts])
        mid = _mid_price([9, 11], [99.0, 105.0])

        result = h001_diagnostics.generate_alignment_diagnostic(
            events, mid, horizon_seconds=5, n_events=1
        )

        assert result.loc[0, "price_at_event"] == pytest.approx(105.0)
        assert result.loc[0, "price_at_event_timestamp"] == pd.Timestamp(
            "2026-06-13 00:00:11", tz="UTC"
        )

    def test_first_and_last_sample_mode(self):
        timestamps = [
            pd.Timestamp("2026-06-13", tz="UTC") + pd.Timedelta(seconds=i)
            for i in range(10)
        ]
        events = _events_df(timestamps)
        mid = _mid_price(list(range(400)), [100.0] * 400)

        result = h001_diagnostics.generate_alignment_diagnostic(
            events,
            mid,
            horizon_seconds=5,
            n_events=4,
            sample_mode="first_and_last",
        )

        selected = result["event_timestamp"].tolist()
        assert selected[0] == timestamps[0]
        assert selected[1] == timestamps[1]
        assert selected[2] == timestamps[8]
        assert selected[3] == timestamps[9]

    def test_random_sample_mode_is_reproducible(self):
        timestamps = [
            pd.Timestamp("2026-06-13", tz="UTC") + pd.Timedelta(seconds=i)
            for i in range(20)
        ]
        events = _events_df(timestamps)
        mid = _mid_price(list(range(400)), [100.0] * 400)

        first = h001_diagnostics.generate_alignment_diagnostic(
            events,
            mid,
            horizon_seconds=5,
            n_events=6,
            sample_mode="random",
            random_seed=123,
        )
        second = h001_diagnostics.generate_alignment_diagnostic(
            events,
            mid,
            horizon_seconds=5,
            n_events=6,
            sample_mode="random",
            random_seed=123,
        )

        pd.testing.assert_frame_equal(first, second)
