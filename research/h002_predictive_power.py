"""H002 predictive information comparison: raw OFI vs normalized OFI."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.loader import load_top_of_book, load_trades
from events.ofi_events import detect_ofi_events
from features.ofi import (
    compute_normalized_ofi,
    compute_ofi,
    compute_ofi_zscore,
    compute_signed_volume,
    resample_to_1s_grid,
)
from research.h001_predictive_power import (
    bootstrap_mean_ci,
    compute_event_outcomes,
    compute_secondary_metrics,
    summarize_sample,
)

RESULTS_DIR = ROOT / "results"

DEFAULT_SYMBOL = "BTCUSDT"
WINDOW_SECONDS = 60
ZSCORE_LOOKBACK_SECONDS = 3600
THRESHOLD = 1.5
COOLDOWN_SECONDS = 300
HORIZON_SECONDS = 300
N_BOOTSTRAP = 10000
CONFIDENCE_LEVEL = 0.95
RANDOM_SEED = 42
OVERLAP_TOLERANCE_SECONDS = 60


def build_signal_a_zscore(
    trades: pd.DataFrame,
    start: str,
    end: str,
    *,
    window_seconds: int = WINDOW_SECONDS,
    lookback_seconds: int = ZSCORE_LOOKBACK_SECONDS,
) -> pd.Series:
    signed_volume = compute_signed_volume(trades)
    signed_volume_1s = resample_to_1s_grid(signed_volume, start, end)
    raw_ofi = compute_ofi(signed_volume_1s, window_seconds=window_seconds)
    return compute_ofi_zscore(raw_ofi, lookback_seconds=lookback_seconds)


def build_signal_b_zscore(
    trades: pd.DataFrame,
    *,
    window_seconds: int = WINDOW_SECONDS,
    lookback_seconds: int = ZSCORE_LOOKBACK_SECONDS,
) -> pd.Series:
    normalized_ofi = compute_normalized_ofi(trades, window_seconds=window_seconds)
    return compute_ofi_zscore(normalized_ofi, lookback_seconds=lookback_seconds)


def detect_signal_events(zscore: pd.Series) -> pd.DataFrame:
    return detect_ofi_events(
        zscore,
        threshold=THRESHOLD,
        cooldown_seconds=COOLDOWN_SECONDS,
        direction="both",
    )


def count_temporal_overlap(
    raw_events: pd.DataFrame,
    normalized_events: pd.DataFrame,
    tolerance_seconds: int = OVERLAP_TOLERANCE_SECONDS,
) -> int:
    if raw_events.empty or normalized_events.empty:
        return 0

    tolerance = pd.Timedelta(seconds=tolerance_seconds)
    raw_times = raw_events["timestamp"].sort_values()
    normalized_times = normalized_events["timestamp"].sort_values()

    overlapping = 0
    for raw_time in raw_times:
        if ((normalized_times - raw_time).abs() <= tolerance).any():
            overlapping += 1
    return overlapping


def analyze_signal(
    events: pd.DataFrame,
    mid_price: pd.Series,
    horizon_seconds: int = HORIZON_SECONDS,
) -> dict[str, Any]:
    outcomes = compute_event_outcomes(events, mid_price, horizon_seconds=horizon_seconds)
    sample = summarize_sample(outcomes)
    signed_returns = outcomes["signed_return"].to_numpy()
    bootstrap = bootstrap_mean_ci(
        signed_returns,
        n_bootstrap=N_BOOTSTRAP,
        confidence_level=CONFIDENCE_LEVEL,
        random_seed=RANDOM_SEED,
    )
    secondary = compute_secondary_metrics(signed_returns)

    metrics = {
        "event_count": secondary["event_count"],
        "mean_signed_return": secondary["mean_signed_return"],
        "median_signed_return": secondary["median_signed_return"],
        "bootstrap_ci_lower": bootstrap["ci_lower"],
        "bootstrap_ci_upper": bootstrap["ci_upper"],
        "bootstrap_ci_width": bootstrap["ci_upper"] - bootstrap["ci_lower"],
        "direction_accuracy": secondary["direction_accuracy"],
        "t_test_pvalue": secondary["t_test_pvalue"],
        "bootstrap_mean": bootstrap["bootstrap_mean"],
    }

    return {
        "sample": sample,
        "bootstrap": bootstrap,
        "secondary": secondary,
        "metrics": metrics,
        "outcomes": outcomes,
        "events": events,
    }


def compute_comparison(
    raw_analysis: dict[str, Any],
    normalized_analysis: dict[str, Any],
) -> dict[str, float | int]:
    raw_metrics = raw_analysis["metrics"]
    normalized_metrics = normalized_analysis["metrics"]
    raw_events = raw_analysis["events"]
    normalized_events = normalized_analysis["events"]

    raw_count = len(raw_events)
    normalized_count = len(normalized_events)
    overlap_count = count_temporal_overlap(raw_events, normalized_events)

    raw_mean = raw_metrics["mean_signed_return"]
    normalized_mean = normalized_metrics["mean_signed_return"]

    return {
        "event_count_ratio": (
            normalized_count / raw_count if raw_count else float("nan")
        ),
        "mean_return_ratio": (
            normalized_mean / raw_mean
            if raw_mean not in (0.0, float("nan")) and pd.notna(raw_mean)
            else float("nan")
        ),
        "ci_lower_difference": (
            normalized_metrics["bootstrap_ci_lower"] - raw_metrics["bootstrap_ci_lower"]
        ),
        "ci_width_difference": (
            normalized_metrics["bootstrap_ci_width"] - raw_metrics["bootstrap_ci_width"]
        ),
        "overlap_count": overlap_count,
        "overlap_pct": (
            overlap_count / min(raw_count, normalized_count)
            if min(raw_count, normalized_count) > 0
            else float("nan")
        ),
    }


def _format_signal_section(title: str, analysis: dict[str, Any]) -> list[str]:
    sample = analysis["sample"]
    metrics = analysis["metrics"]
    return [
        f"## {title}",
        "",
        "### Sample Summary",
        f"- total_events_detected: {sample['total_events_detected']}",
        f"- valid_outcomes: {sample['events_with_valid_outcome']}",
        f"- dropped_outcomes: {sample['events_dropped_due_to_missing_outcome']}",
        "",
        "### Metrics",
        f"- event_count: {metrics['event_count']}",
        f"- mean_signed_return: {metrics['mean_signed_return']:.8f}",
        f"- median_signed_return: {metrics['median_signed_return']:.8f}",
        f"- bootstrap_ci_lower: {metrics['bootstrap_ci_lower']:.8f}",
        f"- bootstrap_ci_upper: {metrics['bootstrap_ci_upper']:.8f}",
        f"- bootstrap_ci_width: {metrics['bootstrap_ci_width']:.8f}",
        f"- direction_accuracy: {metrics['direction_accuracy']:.8f}",
        f"- t_test_pvalue: {metrics['t_test_pvalue']:.8f}",
        "",
    ]


def generate_h002_report(
    *,
    symbol: str,
    start: str,
    end: str,
    raw_analysis: dict[str, Any],
    normalized_analysis: dict[str, Any],
    comparison: dict[str, float | int],
) -> str:
    lines = [
        "# H002 Result",
        "",
        "## Preregistration Summary",
        "",
        "- Research question: Does volume normalization materially change the",
        "  predictive information content of OFI z-score events?",
        f"- Symbol: {symbol}",
        f"- Date range: {start} to {end} (inclusive)",
        f"- threshold: {THRESHOLD}",
        f"- cooldown_seconds: {COOLDOWN_SECONDS}",
        f"- horizon_seconds: {HORIZON_SECONDS}",
        f"- window_seconds: {WINDOW_SECONDS}",
        f"- zscore_lookback_seconds: {ZSCORE_LOOKBACK_SECONDS}",
        "",
        *_format_signal_section("Signal A — Raw OFI", raw_analysis),
        *_format_signal_section("Signal B — Normalized OFI", normalized_analysis),
        "## Comparison",
        "",
        f"- event_count_ratio: {comparison['event_count_ratio']:.8f}",
        f"- mean_return_ratio: {comparison['mean_return_ratio']:.8f}",
        f"- ci_lower_difference: {comparison['ci_lower_difference']:.8f}",
        f"- ci_width_difference: {comparison['ci_width_difference']:.8f}",
        f"- overlap_count: {comparison['overlap_count']}",
        f"- overlap_pct: {comparison['overlap_pct']:.8f}",
        "",
        "## Conclusion",
        "",
        f"- event_count_ratio: {comparison['event_count_ratio']:.8f}",
        f"- mean_return_ratio: {comparison['mean_return_ratio']:.8f}",
        f"- ci_lower_difference: {comparison['ci_lower_difference']:.8f}",
        f"- ci_width_difference: {comparison['ci_width_difference']:.8f}",
        f"- overlap_count: {comparison['overlap_count']}",
        f"- overlap_pct: {comparison['overlap_pct']:.8f}",
        "",
        "This study evaluates predictive information content only. It does not "
        "evaluate tradability, execution feasibility, fees, slippage, or "
        "deployable strategy performance.",
        "",
        "## Interpretation",
        "",
    ]
    return "\n".join(lines)


def run_h002_predictive_power(
    symbol: str,
    start: str,
    end: str,
    *,
    write_report: bool = True,
) -> dict[str, Any]:
    trades = load_trades(symbol, start, end)
    top_of_book = load_top_of_book(symbol, start, end)
    mid_price = top_of_book["mid_price"]

    raw_zscore = build_signal_a_zscore(trades, start, end)
    normalized_zscore = build_signal_b_zscore(trades)

    raw_events = detect_signal_events(raw_zscore)
    normalized_events = detect_signal_events(normalized_zscore)

    raw_analysis = analyze_signal(raw_events, mid_price)
    normalized_analysis = analyze_signal(normalized_events, mid_price)
    comparison = compute_comparison(raw_analysis, normalized_analysis)

    report = generate_h002_report(
        symbol=symbol,
        start=start,
        end=end,
        raw_analysis=raw_analysis,
        normalized_analysis=normalized_analysis,
        comparison=comparison,
    )

    report_path = None
    if write_report:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_path = RESULTS_DIR / f"H002_{timestamp}.md"
        report_path.write_text(report, encoding="utf-8")

    print(report)
    if report_path is not None:
        print(f"Report written to {report_path}")

    return {
        "raw_analysis": raw_analysis,
        "normalized_analysis": normalized_analysis,
        "comparison": comparison,
        "report": report,
        "report_path": report_path,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute H002 predictive power study")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    run_h002_predictive_power(symbol=args.symbol, start=args.start, end=args.end)


if __name__ == "__main__":
    main()
