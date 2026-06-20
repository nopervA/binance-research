"""H003 magnitude-bucket predictive information analysis."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.h001_predictive_power import (
    H001_COOLDOWN_SECONDS,
    H001_CONFIDENCE_LEVEL,
    H001_N_BOOTSTRAP,
    H001_OFI_WINDOW_SECONDS,
    H001_PRIMARY_HORIZON_SECONDS,
    H001_PRIMARY_SYMBOL,
    H001_PRIMARY_THRESHOLD,
    H001_RANDOM_SEED,
    H001_ZSCORE_LOOKBACK_SECONDS,
    bootstrap_mean_ci,
    compute_event_outcomes,
    compute_secondary_metrics,
    run_ofi_event_pipeline,
)

RESULTS_DIR = ROOT / "results"

BUCKET_ORDER = ("low", "medium", "high", "extreme")
MIN_REGRESSION_EVENT_COUNT = 30


def assign_event_bucket(zscore_value: float) -> str:
    """Assign a bucket using abs(zscore_value) and preregistered boundaries."""
    abs_z = abs(float(zscore_value))
    if abs_z >= 5.0:
        return "extreme"
    if abs_z >= 3.0:
        return "high"
    if abs_z >= 2.0:
        return "medium"
    if abs_z >= 1.5:
        return "low"
    raise ValueError(f"abs(zscore_value)={abs_z} is below the minimum bucket bound 1.5")


def assign_buckets_to_events(events: pd.DataFrame) -> pd.DataFrame:
    """Return events with a bucket column derived from zscore_value."""
    if events.empty:
        result = events.copy()
        result["bucket"] = pd.Series(dtype=str)
        return result

    tagged = events.copy()
    tagged["bucket"] = tagged["zscore_value"].map(assign_event_bucket)
    return tagged


def weighted_linear_regression(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    """Return weighted slope and R-squared for y = intercept + slope * x."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    w_arr = np.asarray(weights, dtype=float)

    if x_arr.size < 2:
        return float("nan"), float("nan")

    design = np.column_stack([np.ones(x_arr.size), x_arr])
    xtwx = design.T @ (w_arr[:, np.newaxis] * design)
    xtwy = design.T @ (w_arr * y_arr)
    intercept, slope = np.linalg.solve(xtwx, xtwy)

    y_pred = intercept + slope * x_arr
    y_bar = np.average(y_arr, weights=w_arr)
    ss_res = float(np.sum(w_arr * (y_arr - y_pred) ** 2))
    ss_tot = float(np.sum(w_arr * (y_arr - y_bar) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(r_squared)


def analyze_bucket(
    bucket_events: pd.DataFrame,
    bucket_outcomes: pd.DataFrame,
    total_events: int,
) -> dict[str, Any]:
    """Compute per-bucket distribution and predictive metrics."""
    event_count = len(bucket_events)
    if event_count == 0:
        return {
            "event_count": 0,
            "share_of_total_events": 0.0,
            "mean_abs_zscore": float("nan"),
            "mean_signed_return": float("nan"),
            "median_signed_return": float("nan"),
            "bootstrap_ci_lower": float("nan"),
            "bootstrap_ci_upper": float("nan"),
            "bootstrap_ci_width": float("nan"),
            "direction_accuracy": float("nan"),
            "t_test_pvalue": float("nan"),
            "interpret_with_caution": False,
            "caution_flag": False,
        }

    mean_abs_zscore = float(bucket_events["zscore_value"].abs().mean())
    signed_returns = bucket_outcomes["signed_return"].to_numpy()
    secondary = compute_secondary_metrics(signed_returns)
    bootstrap = bootstrap_mean_ci(
        signed_returns,
        n_bootstrap=H001_N_BOOTSTRAP,
        confidence_level=H001_CONFIDENCE_LEVEL,
        random_seed=H001_RANDOM_SEED,
    )
    caution_flag = event_count < MIN_REGRESSION_EVENT_COUNT

    return {
        "event_count": event_count,
        "share_of_total_events": event_count / total_events if total_events else float("nan"),
        "mean_abs_zscore": mean_abs_zscore,
        "mean_signed_return": secondary["mean_signed_return"],
        "median_signed_return": secondary["median_signed_return"],
        "bootstrap_ci_lower": bootstrap["ci_lower"],
        "bootstrap_ci_upper": bootstrap["ci_upper"],
        "bootstrap_ci_width": bootstrap["ci_upper"] - bootstrap["ci_lower"],
        "direction_accuracy": secondary["direction_accuracy"],
        "t_test_pvalue": secondary["t_test_pvalue"],
        "interpret_with_caution": caution_flag,
        "caution_flag": caution_flag,
    }


def analyze_magnitude_buckets(
    events: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> dict[str, Any]:
    """Assign buckets and compute per-bucket metrics plus regression summary."""
    tagged_events = assign_buckets_to_events(events)
    total_events = len(tagged_events)

    if total_events == 0:
        empty_buckets = {
            bucket: analyze_bucket(
                tagged_events.iloc[0:0],
                outcomes.iloc[0:0],
                total_events=0,
            )
            for bucket in BUCKET_ORDER
        }
        return {
            "events": tagged_events,
            "outcomes": outcomes,
            "buckets": empty_buckets,
            "regression": compute_regression_summary(empty_buckets),
            "total_events": 0,
        }

    outcomes_with_bucket = outcomes.copy()
    outcomes_with_bucket["bucket"] = tagged_events["bucket"].to_numpy()

    bucket_metrics: dict[str, dict[str, Any]] = {}
    for bucket in BUCKET_ORDER:
        bucket_events = tagged_events[tagged_events["bucket"] == bucket]
        bucket_outcomes = outcomes_with_bucket[outcomes_with_bucket["bucket"] == bucket]
        bucket_metrics[bucket] = analyze_bucket(
            bucket_events,
            bucket_outcomes,
            total_events=total_events,
        )

    return {
        "events": tagged_events,
        "outcomes": outcomes_with_bucket,
        "buckets": bucket_metrics,
        "regression": compute_regression_summary(bucket_metrics),
        "total_events": total_events,
    }


def compute_regression_summary(
    bucket_metrics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Weighted regression of mean_signed_return on mean_abs_zscore by bucket."""
    included = [
        bucket
        for bucket in BUCKET_ORDER
        if bucket_metrics[bucket]["event_count"] >= MIN_REGRESSION_EVENT_COUNT
    ]

    if len(included) < 2:
        return {
            "regression_slope": float("nan"),
            "regression_r_squared": float("nan"),
            "buckets_included_in_regression": included,
        }

    x = np.array([bucket_metrics[b]["mean_abs_zscore"] for b in included], dtype=float)
    y = np.array([bucket_metrics[b]["mean_signed_return"] for b in included], dtype=float)
    weights = np.array([bucket_metrics[b]["event_count"] for b in included], dtype=float)

    slope, r_squared = weighted_linear_regression(x, y, weights)
    return {
        "regression_slope": slope,
        "regression_r_squared": r_squared,
        "buckets_included_in_regression": included,
    }


def _format_bucket_distribution_table(buckets: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "| bucket | event_count | share_of_total_events | mean_abs_zscore | caution_flag |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for bucket in BUCKET_ORDER:
        metrics = buckets[bucket]
        lines.append(
            f"| {bucket} | {metrics['event_count']} | "
            f"{metrics['share_of_total_events']:.8f} | "
            f"{metrics['mean_abs_zscore']:.8f} | {metrics['caution_flag']} |"
        )
    return lines


def _format_predictive_table(buckets: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "| bucket | event_count | mean_signed_return | median_signed_return | "
        "bootstrap_ci_lower | bootstrap_ci_upper | direction_accuracy | "
        "t_test_pvalue | caution_flag |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for bucket in BUCKET_ORDER:
        metrics = buckets[bucket]
        lines.append(
            f"| {bucket} | {metrics['event_count']} | "
            f"{metrics['mean_signed_return']:.8f} | "
            f"{metrics['median_signed_return']:.8f} | "
            f"{metrics['bootstrap_ci_lower']:.8f} | "
            f"{metrics['bootstrap_ci_upper']:.8f} | "
            f"{metrics['direction_accuracy']:.8f} | "
            f"{metrics['t_test_pvalue']:.8f} | {metrics['caution_flag']} |"
        )
    return lines


def generate_h003_report(
    *,
    symbol: str,
    start: str,
    end: str,
    analysis: dict[str, Any],
) -> str:
    buckets = analysis["buckets"]
    regression = analysis["regression"]

    lines = [
        "# H003 Result",
        "",
        "## Preregistration Summary",
        "",
        "- Research question: Is there a positive relationship between OFI z-score",
        "  magnitude and predictive information content?",
        f"- Symbol: {symbol}",
        f"- Date range: {start} to {end} (inclusive)",
        f"- threshold: {H001_PRIMARY_THRESHOLD}",
        f"- cooldown_seconds: {H001_COOLDOWN_SECONDS}",
        f"- horizon_seconds: {H001_PRIMARY_HORIZON_SECONDS}",
        f"- ofi_window_seconds: {H001_OFI_WINDOW_SECONDS}",
        f"- zscore_lookback_seconds: {H001_ZSCORE_LOOKBACK_SECONDS}",
        f"- total_events: {analysis['total_events']}",
        "",
        "## Bucket Distribution",
        "",
        *_format_bucket_distribution_table(buckets),
        "",
        "## Predictive Information by Bucket",
        "",
        *_format_predictive_table(buckets),
        "",
        "## Relationship Summary",
        "",
        f"- regression_slope: {regression['regression_slope']:.8f}",
        f"- regression_r_squared: {regression['regression_r_squared']:.8f}",
        f"- buckets_included_in_regression: {regression['buckets_included_in_regression']}",
        "",
        "## Interpretation",
        "",
        "## Conclusion",
        "",
        "This study evaluates predictive information content only. It does not "
        "evaluate tradability, execution feasibility, fees, slippage, or "
        "deployable strategy performance.",
        "",
    ]
    return "\n".join(lines)


def run_h003_magnitude_analysis(
    symbol: str,
    start: str,
    end: str,
    *,
    threshold: float = H001_PRIMARY_THRESHOLD,
    horizon_seconds: int = H001_PRIMARY_HORIZON_SECONDS,
    write_report: bool = True,
) -> dict[str, Any]:
    events, mid_price = run_ofi_event_pipeline(
        symbol,
        start,
        end,
        threshold=threshold,
    )
    outcomes = compute_event_outcomes(events, mid_price, horizon_seconds=horizon_seconds)
    analysis = analyze_magnitude_buckets(events, outcomes)

    report = generate_h003_report(
        symbol=symbol,
        start=start,
        end=end,
        analysis=analysis,
    )

    report_path = None
    if write_report:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_path = RESULTS_DIR / f"H003_{timestamp}.md"
        report_path.write_text(report, encoding="utf-8")

    print(report)
    if report_path is not None:
        print(f"Report written to {report_path}")

    return {
        "events": analysis["events"],
        "outcomes": analysis["outcomes"],
        "buckets": analysis["buckets"],
        "regression": analysis["regression"],
        "total_events": analysis["total_events"],
        "report": report,
        "report_path": report_path,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute H003 magnitude analysis study")
    parser.add_argument("--symbol", default=H001_PRIMARY_SYMBOL)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    run_h003_magnitude_analysis(symbol=args.symbol, start=args.start, end=args.end)


if __name__ == "__main__":
    main()
