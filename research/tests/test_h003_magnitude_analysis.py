"""Tests for research/h003_magnitude_analysis.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_H001_PATH = Path(__file__).resolve().parent.parent / "h001_predictive_power.py"
_H003_PATH = Path(__file__).resolve().parent.parent / "h003_magnitude_analysis.py"

for module_name, module_path in (
    ("h001_predictive_power", _H001_PATH),
    ("h003_magnitude_analysis", _H003_PATH),
):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

h001_predictive_power = sys.modules["h001_predictive_power"]
h003_magnitude_analysis = sys.modules["h003_magnitude_analysis"]


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
    index = pd.Timestamp(base, tz="UTC") + pd.to_timedelta(seconds, unit="s")
    return pd.Series(prices, index=index, name="mid_price")


def _synthetic_outcomes_for_events(events: pd.DataFrame) -> pd.DataFrame:
    base = pd.Timestamp("2026-06-13", tz="UTC")
    seconds = list(range(len(events) * 400))
    prices = [100.0 + i * 0.001 for i in seconds]
    mid = _mid_price(seconds, prices, base="2026-06-13")
    return h001_predictive_power.compute_event_outcomes(
        events,
        mid,
        horizon_seconds=300,
    )


class TestBucketAssignment:
    def test_known_abs_z_values_assigned_to_correct_buckets(self):
        cases = [
            (1.75, "low"),
            (-1.9, "low"),
            (2.5, "medium"),
            (-2.1, "medium"),
            (4.0, "high"),
            (-3.5, "high"),
            (6.0, "extreme"),
            (-5.0, "extreme"),
        ]
        for zscore, expected in cases:
            assert h003_magnitude_analysis.assign_event_bucket(zscore) == expected

    def test_boundary_values_go_to_higher_bucket(self):
        assert h003_magnitude_analysis.assign_event_bucket(2.0) == "medium"
        assert h003_magnitude_analysis.assign_event_bucket(-2.0) == "medium"
        assert h003_magnitude_analysis.assign_event_bucket(3.0) == "high"
        assert h003_magnitude_analysis.assign_event_bucket(-3.0) == "high"
        assert h003_magnitude_analysis.assign_event_bucket(5.0) == "extreme"
        assert h003_magnitude_analysis.assign_event_bucket(-5.0) == "extreme"
        assert h003_magnitude_analysis.assign_event_bucket(1.999) == "low"


class TestBucketMetrics:
    def test_share_of_total_events_sums_to_one(self):
        timestamps = [
            pd.Timestamp("2026-06-13", tz="UTC") + pd.Timedelta(seconds=i * 10)
            for i in range(8)
        ]
        zscores = [1.8, 2.2, 2.5, 3.2, 4.0, 5.5, -2.1, -6.0]
        events = _events_df(timestamps, zscores=zscores)
        outcomes = _synthetic_outcomes_for_events(events)

        analysis = h003_magnitude_analysis.analyze_magnitude_buckets(events, outcomes)
        shares = [analysis["buckets"][bucket]["share_of_total_events"] for bucket in h003_magnitude_analysis.BUCKET_ORDER]

        assert sum(shares) == pytest.approx(1.0)

    def test_event_counts_sum_to_total(self):
        timestamps = [
            pd.Timestamp("2026-06-13", tz="UTC") + pd.Timedelta(seconds=i * 10)
            for i in range(10)
        ]
        zscores = [1.8, 1.9, 2.1, 2.4, 3.1, 3.8, 4.5, 5.2, 6.0, -2.2]
        events = _events_df(timestamps, zscores=zscores)
        outcomes = _synthetic_outcomes_for_events(events)

        analysis = h003_magnitude_analysis.analyze_magnitude_buckets(events, outcomes)
        bucket_counts = [
            analysis["buckets"][bucket]["event_count"]
            for bucket in h003_magnitude_analysis.BUCKET_ORDER
        ]

        assert sum(bucket_counts) == analysis["total_events"]
        assert analysis["total_events"] == len(events)


class TestRegressionSummary:
    def test_regression_excludes_buckets_below_threshold(self):
        bucket_metrics = {
            "low": {
                "event_count": 40,
                "mean_abs_zscore": 1.75,
                "mean_signed_return": 0.001,
            },
            "medium": {
                "event_count": 25,
                "mean_abs_zscore": 2.5,
                "mean_signed_return": 0.002,
            },
            "high": {
                "event_count": 35,
                "mean_abs_zscore": 3.5,
                "mean_signed_return": 0.003,
            },
            "extreme": {
                "event_count": 10,
                "mean_abs_zscore": 6.0,
                "mean_signed_return": 0.004,
            },
        }

        regression = h003_magnitude_analysis.compute_regression_summary(bucket_metrics)

        assert regression["buckets_included_in_regression"] == ["low", "high"]
        assert not np.isnan(regression["regression_slope"])
        assert not np.isnan(regression["regression_r_squared"])

    def test_regression_includes_all_buckets_with_sufficient_counts(self):
        bucket_metrics = {
            bucket: {
                "event_count": 30 + index,
                "mean_abs_zscore": 1.75 + index,
                "mean_signed_return": 0.001 * (index + 1),
            }
            for index, bucket in enumerate(h003_magnitude_analysis.BUCKET_ORDER)
        }

        regression = h003_magnitude_analysis.compute_regression_summary(bucket_metrics)

        assert regression["buckets_included_in_regression"] == list(
            h003_magnitude_analysis.BUCKET_ORDER
        )
        assert regression["regression_slope"] > 0


class TestBootstrapPerBucket:
    def test_bootstrap_is_deterministic_with_fixed_seed(self):
        timestamps = [
            pd.Timestamp("2026-06-13", tz="UTC") + pd.Timedelta(seconds=i * 10)
            for i in range(6)
        ]
        zscores = [1.8, 1.9, 2.2, 2.4, 3.2, 5.5]
        events = _events_df(timestamps, zscores=zscores)
        outcomes = _synthetic_outcomes_for_events(events)

        first = h003_magnitude_analysis.analyze_magnitude_buckets(events, outcomes)
        second = h003_magnitude_analysis.analyze_magnitude_buckets(events, outcomes)

        for bucket in h003_magnitude_analysis.BUCKET_ORDER:
            for key in ("bootstrap_ci_lower", "bootstrap_ci_upper", "bootstrap_ci_width"):
                assert first["buckets"][bucket][key] == pytest.approx(
                    second["buckets"][bucket][key]
                )

    def test_different_buckets_can_have_different_bootstrap_results(self):
        timestamps = [
            pd.Timestamp("2026-06-13", tz="UTC") + pd.Timedelta(seconds=i * 10)
            for i in range(8)
        ]
        zscores = [1.8, 1.9, 2.2, 2.4, 3.2, 3.8, 5.5, 6.0]
        events = _events_df(timestamps, zscores=zscores)
        outcomes = _synthetic_outcomes_for_events(events)

        analysis = h003_magnitude_analysis.analyze_magnitude_buckets(events, outcomes)
        low_width = analysis["buckets"]["low"]["bootstrap_ci_width"]
        extreme_width = analysis["buckets"]["extreme"]["bootstrap_ci_width"]

        assert low_width != extreme_width


class TestReportAndAlignment:
    def test_report_generation_on_minimal_synthetic_dataset(self):
        events = _events_df(
            [pd.Timestamp("2026-06-13", tz="UTC")],
            zscores=[2.0],
        )
        outcomes = _synthetic_outcomes_for_events(events)
        analysis = h003_magnitude_analysis.analyze_magnitude_buckets(events, outcomes)

        report = h003_magnitude_analysis.generate_h003_report(
            symbol="BTCUSDT",
            start="2026-06-13",
            end="2026-06-13",
            analysis=analysis,
        )

        assert "# H003 Result" in report
        assert "## Bucket Distribution" in report
        assert "## Predictive Information by Bucket" in report
        assert "## Interpretation" in report
        assert "predictive information content only" in report

    def test_compute_event_outcomes_unchanged_by_bucket_analysis(self):
        timestamps = [
            pd.Timestamp("2026-06-13", tz="UTC"),
            pd.Timestamp("2026-06-13 00:00:10", tz="UTC"),
        ]
        events = _events_df(
            timestamps,
            directions=["long", "short"],
            zscores=[2.0, -3.5],
        )
        mid = _mid_price([0, 10, 310, 320], [100.0, 101.0, 110.0, 108.0])

        direct_outcomes = h001_predictive_power.compute_event_outcomes(
            events,
            mid,
            horizon_seconds=300,
        )
        analysis = h003_magnitude_analysis.analyze_magnitude_buckets(
            events,
            direct_outcomes,
        )

        pd.testing.assert_frame_equal(
            analysis["outcomes"].drop(columns=["bucket"]),
            direct_outcomes,
        )
