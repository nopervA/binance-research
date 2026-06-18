"""Tests for research/h002_predictive_power.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_H002_PATH = Path(__file__).resolve().parent.parent / "h002_predictive_power.py"
_H001_PATH = Path(__file__).resolve().parent.parent / "h001_predictive_power.py"

for module_name, module_path in (
    ("h001_predictive_power", _H001_PATH),
    ("h002_predictive_power", _H002_PATH),
):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

h001_predictive_power = sys.modules["h001_predictive_power"]
h002_predictive_power = sys.modules["h002_predictive_power"]


def _trades_df(
    seconds: list[int],
    quantities: list[float],
    is_buyer_maker: list[bool],
    *,
    base: str = "2026-06-13",
) -> pd.DataFrame:
    index = pd.Timestamp(base, tz="UTC") + pd.to_timedelta(seconds, unit="s")
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


def _mid_price(
    seconds: list[int],
    prices: list[float],
    *,
    base: str = "2026-06-13",
) -> pd.Series:
    index = pd.Timestamp(base, tz="UTC") + pd.to_timedelta(seconds, unit="s")
    return pd.Series(prices, index=index, name="mid_price")


def _events_df(timestamps: list[pd.Timestamp]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "direction": ["long"] * len(timestamps),
            "zscore_value": [2.0] * len(timestamps),
        }
    )


class TestSignalPipelines:
    def test_signal_a_matches_h001_bootstrap_mean_on_synthetic_data(self):
        start = "2026-06-13"
        end = "2026-06-13"
        seconds = list(range(4000))
        trades = _trades_df(
            seconds=seconds,
            quantities=[1.0 if i % 2 == 0 else 2.0 for i in seconds],
            is_buyer_maker=[False, True] * (len(seconds) // 2),
        )
        mid = _mid_price(list(range(4500)), [100.0 + i * 0.001 for i in range(4500)])

        raw_zscore = h002_predictive_power.build_signal_a_zscore(trades, start, end)
        raw_events = h002_predictive_power.detect_signal_events(raw_zscore)
        outcomes = h001_predictive_power.compute_event_outcomes(
            raw_events, mid, horizon_seconds=300
        )

        h001_result = h001_predictive_power.analyze_outcomes(outcomes)
        h002_result = h002_predictive_power.analyze_signal(raw_events, mid)

        assert h002_result["metrics"]["bootstrap_mean"] == pytest.approx(
            h001_result["bootstrap"]["bootstrap_mean"]
        )

    def test_signal_b_produces_different_event_timestamps_than_signal_a(self):
        start = "2026-06-13"
        end = "2026-06-13"
        seconds = list(range(5000))
        quantities = [1.0] * len(seconds)
        sides = [i % 2 == 1 for i in seconds]
        for i in range(3800, 3830):
            quantities[i] = 5000.0
            sides[i] = False
        for i in range(4200, 4230):
            quantities[i] = 0.01
            sides[i] = False
        trades = _trades_df(seconds=seconds, quantities=quantities, is_buyer_maker=sides)

        raw_zscore = h002_predictive_power.build_signal_a_zscore(trades, start, end)
        normalized_zscore = h002_predictive_power.build_signal_b_zscore(trades)
        raw_events = h002_predictive_power.detect_signal_events(raw_zscore)
        normalized_events = h002_predictive_power.detect_signal_events(normalized_zscore)

        assert not raw_events.empty
        assert not normalized_events.empty
        assert set(raw_events["timestamp"]) != set(normalized_events["timestamp"])


class TestComparisonMetrics:
    def test_event_count_ratio(self):
        comparison = h002_predictive_power.compute_comparison(
            {
                "metrics": {"mean_signed_return": 0.01, "bootstrap_ci_lower": 0.0, "bootstrap_ci_width": 0.2},
                "events": _events_df([pd.Timestamp("2026-06-13", tz="UTC")] * 4),
            },
            {
                "metrics": {"mean_signed_return": 0.02, "bootstrap_ci_lower": 0.1, "bootstrap_ci_width": 0.3},
                "events": _events_df([pd.Timestamp("2026-06-13", tz="UTC")] * 2),
            },
        )

        assert comparison["event_count_ratio"] == pytest.approx(0.5)

    def test_ci_lower_difference(self):
        comparison = h002_predictive_power.compute_comparison(
            {
                "metrics": {
                    "mean_signed_return": 0.01,
                    "bootstrap_ci_lower": 0.01,
                    "bootstrap_ci_width": 0.2,
                },
                "events": _events_df([pd.Timestamp("2026-06-13", tz="UTC")]),
            },
            {
                "metrics": {
                    "mean_signed_return": 0.02,
                    "bootstrap_ci_lower": 0.05,
                    "bootstrap_ci_width": 0.3,
                },
                "events": _events_df([pd.Timestamp("2026-06-13", tz="UTC")]),
            },
        )

        assert comparison["ci_lower_difference"] == pytest.approx(0.04)

    def test_overlap_count(self):
        base = pd.Timestamp("2026-06-13", tz="UTC")
        raw_events = _events_df([base, base + pd.Timedelta(seconds=100)])
        normalized_events = _events_df(
            [base + pd.Timedelta(seconds=30), base + pd.Timedelta(seconds=500)]
        )

        comparison = h002_predictive_power.compute_comparison(
            {
                "metrics": {
                    "mean_signed_return": 0.01,
                    "bootstrap_ci_lower": 0.0,
                    "bootstrap_ci_width": 0.2,
                },
                "events": raw_events,
            },
            {
                "metrics": {
                    "mean_signed_return": 0.02,
                    "bootstrap_ci_lower": 0.1,
                    "bootstrap_ci_width": 0.3,
                },
                "events": normalized_events,
            },
        )

        assert comparison["overlap_count"] == 1
        assert comparison["overlap_pct"] == pytest.approx(0.5)


class TestReportAndNaNHandling:
    def test_report_generation_on_minimal_synthetic_dataset(self):
        analysis = {
            "sample": {
                "total_events_detected": 1,
                "events_with_valid_outcome": 1,
                "events_dropped_due_to_missing_outcome": 0,
            },
            "metrics": {
                "event_count": 1,
                "mean_signed_return": 0.01,
                "median_signed_return": 0.01,
                "bootstrap_ci_lower": 0.005,
                "bootstrap_ci_upper": 0.015,
                "bootstrap_ci_width": 0.01,
                "direction_accuracy": 1.0,
                "t_test_pvalue": 0.1,
            },
            "events": _events_df([pd.Timestamp("2026-06-13", tz="UTC")]),
        }

        report = h002_predictive_power.generate_h002_report(
            symbol="BTCUSDT",
            start="2026-06-13",
            end="2026-06-13",
            raw_analysis=analysis,
            normalized_analysis=analysis,
            comparison=h002_predictive_power.compute_comparison(analysis, analysis),
        )

        assert "# H002 Result" in report
        assert "## Interpretation" in report
        assert "predictive information content only" in report

    def test_nan_events_counted_and_excluded_for_each_signal(self):
        timestamps = [
            pd.Timestamp("2026-06-13", tz="UTC"),
            pd.Timestamp("2026-06-13 00:00:10", tz="UTC"),
        ]
        events = _events_df(timestamps)
        mid = _mid_price([0], [100.0])

        raw_analysis = h002_predictive_power.analyze_signal(events, mid)
        normalized_analysis = h002_predictive_power.analyze_signal(events, mid)

        for analysis in (raw_analysis, normalized_analysis):
            assert analysis["sample"]["total_events_detected"] == 2
            assert analysis["sample"]["events_with_valid_outcome"] == 0
            assert analysis["sample"]["events_dropped_due_to_missing_outcome"] == 2
            assert analysis["metrics"]["event_count"] == 0
