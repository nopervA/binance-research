"""Tests for research/h001_predictive_power.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PREDICTIVE_PATH = Path(__file__).resolve().parent.parent / "h001_predictive_power.py"
_SPEC = importlib.util.spec_from_file_location("h001_predictive_power", _PREDICTIVE_PATH)
h001_predictive_power = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules["h001_predictive_power"] = h001_predictive_power
_SPEC.loader.exec_module(h001_predictive_power)


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


class TestComputeEventOutcomes:
    def test_forward_return_and_signed_return(self):
        event_ts = pd.Timestamp("2026-06-13 00:00:00", tz="UTC")
        events = _events_df([event_ts], directions=["long"], zscores=[2.0])
        mid = _mid_price([0, 300], [100.0, 110.0])

        result = h001_predictive_power.compute_event_outcomes(
            events, mid, horizon_seconds=300
        )

        assert result.loc[0, "forward_return"] == pytest.approx(0.10)
        assert result.loc[0, "signed_return"] == pytest.approx(0.10)

        short_events = _events_df([event_ts], directions=["short"], zscores=[-2.0])
        short_result = h001_predictive_power.compute_event_outcomes(
            short_events, mid, horizon_seconds=300
        )
        assert short_result.loc[0, "signed_return"] == pytest.approx(-0.10)

    def test_missing_match_produces_nan_and_is_preserved(self):
        event_ts = pd.Timestamp("2026-06-13 00:00:00", tz="UTC")
        events = _events_df([event_ts, event_ts + pd.Timedelta(seconds=10)])
        mid = _mid_price([0, 5], [100.0, 101.0])

        result = h001_predictive_power.compute_event_outcomes(
            events, mid, horizon_seconds=300
        )

        assert len(result) == 2
        assert pd.isna(result.loc[0, "forward_return"])
        assert pd.isna(result.loc[0, "signed_return"])


class TestBootstrapMeanCi:
    def test_deterministic_with_fixed_seed(self):
        values = np.array([0.01, -0.005, 0.02, 0.015, -0.01])

        first = h001_predictive_power.bootstrap_mean_ci(
            values, n_bootstrap=1000, random_seed=123
        )
        second = h001_predictive_power.bootstrap_mean_ci(
            values, n_bootstrap=1000, random_seed=123
        )

        assert first == second

    def test_sanity_check_on_known_distribution(self):
        values = np.array([1.0, 1.0, 1.0, 1.0])

        result = h001_predictive_power.bootstrap_mean_ci(
            values, n_bootstrap=1000, confidence_level=0.95, random_seed=42
        )

        assert result["bootstrap_mean"] == pytest.approx(1.0)
        assert result["ci_lower"] == pytest.approx(1.0)
        assert result["ci_upper"] == pytest.approx(1.0)


class TestPassCondition:
    def test_positive_ci_passes(self):
        assert h001_predictive_power.evaluate_primary_pass(
            {"bootstrap_mean": 0.01, "ci_lower": 0.001, "ci_upper": 0.02}
        )

    def test_ci_crossing_zero_fails(self):
        assert not h001_predictive_power.evaluate_primary_pass(
            {"bootstrap_mean": 0.01, "ci_lower": -0.001, "ci_upper": 0.02}
        )

    def test_negative_ci_fails(self):
        assert not h001_predictive_power.evaluate_primary_pass(
            {"bootstrap_mean": -0.01, "ci_lower": -0.02, "ci_upper": -0.001}
        )


class TestRobustnessHelpers:
    def test_remove_top_5_abs_signed_returns(self):
        outcomes = pd.DataFrame(
            {
                "signed_return": [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.05],
            }
        )

        trimmed = h001_predictive_power.remove_top_5_abs_signed_returns(outcomes)

        assert len(trimmed) == 2
        assert trimmed["signed_return"].tolist() == pytest.approx([0.1, 0.05])

    def test_direction_accuracy(self):
        signed_returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02])

        metrics = h001_predictive_power.compute_secondary_metrics(signed_returns)

        assert metrics["direction_accuracy"] == pytest.approx(0.6)


class TestReportAndSampleHandling:
    def test_report_generation_on_synthetic_dataset(self, tmp_path, monkeypatch):
        outcomes = pd.DataFrame(
            {
                "signed_return": [0.01, 0.02, -0.01, 0.015, 0.005],
            }
        )
        analysis = h001_predictive_power.analyze_outcomes(outcomes)
        monkeypatch.setattr(h001_predictive_power, "RESULTS_DIR", tmp_path)

        report = h001_predictive_power.generate_h001_report(
            symbol="BTCUSDT",
            start="2026-06-13",
            end="2026-06-13",
            threshold=1.5,
            horizon_seconds=300,
            primary_analysis=analysis,
            robustness={},
        )

        assert "# H001 Result" in report
        assert "predictive information content only" in report
        assert "Secondary Metrics" in report

    def test_nan_events_counted_and_excluded_from_statistics(self):
        outcomes = pd.DataFrame(
            {
                "signed_return": [0.01, np.nan, 0.02, np.nan, 0.03],
            }
        )

        sample = h001_predictive_power.summarize_sample(outcomes)
        analysis = h001_predictive_power.analyze_outcomes(outcomes)

        assert sample["total_events_detected"] == 5
        assert sample["events_with_valid_outcome"] == 3
        assert sample["events_dropped_due_to_missing_outcome"] == 2
        assert analysis["secondary"]["event_count"] == 3
