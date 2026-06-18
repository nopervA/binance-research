"""Exploratory diagnostics for volume-normalized OFI ahead of H002 preregistration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.loader import load_trades
from events.ofi_events import detect_ofi_events
from features.ofi import (
    compute_normalized_ofi,
    compute_ofi,
    compute_ofi_zscore,
    compute_signed_volume,
    resample_to_1s_grid,
)

REPORTS_DIR = ROOT / "qa" / "reports"

DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_WINDOW_SECONDS = 60
DEFAULT_ZSCORE_LOOKBACK_SECONDS = 3600
DEFAULT_COOLDOWN_SECONDS = 300
THRESHOLDS = (1.5, 2.0, 3.0)
OVERLAP_TOLERANCES_SECONDS = (30, 60, 300)


def distribution_summary(series: pd.Series) -> dict[str, float | int]:
    values = series.dropna()
    if values.empty:
        nan = float("nan")
        return {
            "count": 0,
            "mean": nan,
            "std": nan,
            "min": nan,
            "p01": nan,
            "p05": nan,
            "median": nan,
            "p95": nan,
            "p99": nan,
            "max": nan,
        }

    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p01": float(values.quantile(0.01)),
        "p05": float(values.quantile(0.05)),
        "median": float(values.median()),
        "p95": float(values.quantile(0.95)),
        "p99": float(values.quantile(0.99)),
        "max": float(values.max()),
    }


def pearson_correlation(left: pd.Series, right: pd.Series) -> float:
    aligned = pd.concat([left, right], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        return float("nan")
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))


def count_temporal_overlap(
    raw_events: pd.DataFrame,
    normalized_events: pd.DataFrame,
    tolerance_seconds: int,
) -> int:
    if raw_events.empty or normalized_events.empty:
        return 0

    tolerance = pd.Timedelta(seconds=tolerance_seconds)
    raw_times = raw_events["timestamp"].sort_values()
    normalized_times = normalized_events["timestamp"].sort_values()

    overlapping = 0
    for raw_time in raw_times:
        deltas = (normalized_times - raw_time).abs()
        if (deltas <= tolerance).any():
            overlapping += 1
    return overlapping


def overlap_summary(
    raw_events: pd.DataFrame,
    normalized_events: pd.DataFrame,
) -> dict[str, float | int]:
    raw_count = len(raw_events)
    normalized_count = len(normalized_events)
    denominator = min(raw_count, normalized_count)

    summary: dict[str, float | int] = {
        "raw_event_count": raw_count,
        "normalized_event_count": normalized_count,
    }

    for tolerance in OVERLAP_TOLERANCES_SECONDS:
        overlap_count = count_temporal_overlap(
            raw_events, normalized_events, tolerance
        )
        summary[f"overlap_{tolerance}s"] = overlap_count
        if denominator == 0:
            summary[f"overlap_pct_{tolerance}s"] = float("nan")
        else:
            summary[f"overlap_pct_{tolerance}s"] = overlap_count / denominator

    return summary


def run_ofi_feature_pipeline(
    trades: pd.DataFrame,
    start: str,
    end: str,
    window_seconds: int,
) -> dict[str, pd.Series]:
    signed_volume = compute_signed_volume(trades)
    signed_volume_1s = resample_to_1s_grid(signed_volume, start, end)
    raw_ofi = compute_ofi(signed_volume_1s, window_seconds=window_seconds)
    normalized_ofi = compute_normalized_ofi(trades, window_seconds=window_seconds)

    common_index = raw_ofi.index.intersection(normalized_ofi.index)
    raw_ofi = raw_ofi.reindex(common_index)
    normalized_ofi = normalized_ofi.reindex(common_index)

    raw_zscore = compute_ofi_zscore(
        raw_ofi, lookback_seconds=DEFAULT_ZSCORE_LOOKBACK_SECONDS
    )
    normalized_zscore = compute_ofi_zscore(
        normalized_ofi, lookback_seconds=DEFAULT_ZSCORE_LOOKBACK_SECONDS
    )

    return {
        "raw_ofi": raw_ofi,
        "normalized_ofi": normalized_ofi,
        "raw_zscore": raw_zscore,
        "normalized_zscore": normalized_zscore,
    }


def build_threshold_event_summaries(
    raw_zscore: pd.Series,
    normalized_zscore: pd.Series,
) -> dict[str, dict[str, float | int]]:
    summaries: dict[str, dict[str, float | int]] = {}

    for threshold in THRESHOLDS:
        raw_events = detect_ofi_events(
            raw_zscore,
            threshold=threshold,
            cooldown_seconds=DEFAULT_COOLDOWN_SECONDS,
            direction="both",
        )
        normalized_events = detect_ofi_events(
            normalized_zscore,
            threshold=threshold,
            cooldown_seconds=DEFAULT_COOLDOWN_SECONDS,
            direction="both",
        )
        key = f"threshold_{str(threshold).replace('.', '_')}"
        summaries[key] = overlap_summary(raw_events, normalized_events)

    return summaries


def _format_distribution(title: str, stats: dict[str, float | int]) -> list[str]:
    return [
        title,
        f"  count: {stats['count']}",
        f"  mean: {stats['mean']:.8f}",
        f"  std: {stats['std']:.8f}",
        f"  min: {stats['min']:.8f}",
        f"  p01: {stats['p01']:.8f}",
        f"  p05: {stats['p05']:.8f}",
        f"  median: {stats['median']:.8f}",
        f"  p95: {stats['p95']:.8f}",
        f"  p99: {stats['p99']:.8f}",
        f"  max: {stats['max']:.8f}",
        "",
    ]


def generate_h002_feature_report(
    *,
    symbol: str,
    start: str,
    end: str,
    window_seconds: int,
    series: dict[str, pd.Series],
    threshold_summaries: dict[str, dict[str, float | int]],
) -> str:
    lines = [
        "# H002 Feature Diagnostics",
        "",
        f"symbol: {symbol}",
        f"start: {start}",
        f"end: {end}",
        f"window_seconds: {window_seconds}",
        f"zscore_lookback_seconds: {DEFAULT_ZSCORE_LOOKBACK_SECONDS}",
        f"event_cooldown_seconds: {DEFAULT_COOLDOWN_SECONDS}",
        "",
        *_format_distribution("RAW OFI distribution:", distribution_summary(series["raw_ofi"])),
        *_format_distribution(
            "NORMALIZED OFI distribution:",
            distribution_summary(series["normalized_ofi"]),
        ),
        *_format_distribution(
            "RAW OFI Z-SCORE distribution:",
            distribution_summary(series["raw_zscore"]),
        ),
        *_format_distribution(
            "NORMALIZED OFI Z-SCORE distribution:",
            distribution_summary(series["normalized_zscore"]),
        ),
        "CORRELATION",
        f"  raw_ofi vs normalized_ofi: {pearson_correlation(series['raw_ofi'], series['normalized_ofi']):.8f}",
        "  raw_zscore vs normalized_zscore: "
        f"{pearson_correlation(series['raw_zscore'], series['normalized_zscore']):.8f}",
        "",
        "THRESHOLD EVENT COUNTS",
    ]

    for threshold in THRESHOLDS:
        key = f"threshold_{str(threshold).replace('.', '_')}"
        summary = threshold_summaries[key]
        lines.extend(
            [
                f"  threshold={threshold}",
                f"    raw_event_count: {summary['raw_event_count']}",
                f"    normalized_event_count: {summary['normalized_event_count']}",
            ]
        )

    lines.append("")
    lines.append("TEMPORAL OVERLAP")

    for threshold in THRESHOLDS:
        key = f"threshold_{str(threshold).replace('.', '_')}"
        summary = threshold_summaries[key]
        lines.append(f"  threshold={threshold}")
        for tolerance in OVERLAP_TOLERANCES_SECONDS:
            overlap_count = summary[f"overlap_{tolerance}s"]
            overlap_pct = summary[f"overlap_pct_{tolerance}s"]
            lines.append(f"    overlap_{tolerance}s: {overlap_count}")
            lines.append(f"    overlap_pct_{tolerance}s: {overlap_pct:.8f}")

    lines.append("")
    return "\n".join(lines)


def run_h002_feature_diagnostics(
    symbol: str,
    start: str,
    end: str,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
) -> dict[str, Any]:
    trades = load_trades(symbol, start, end)
    series = run_ofi_feature_pipeline(trades, start, end, window_seconds)
    threshold_summaries = build_threshold_event_summaries(
        series["raw_zscore"],
        series["normalized_zscore"],
    )

    report = generate_h002_feature_report(
        symbol=symbol,
        start=start,
        end=end,
        window_seconds=window_seconds,
        series=series,
        threshold_summaries=threshold_summaries,
    )

    print(report)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"H002_feature_diagnostics_{symbol}_{start}_{end}.txt"
    report_path.write_text(report + "\n", encoding="utf-8")

    return {
        "series": series,
        "threshold_summaries": threshold_summaries,
        "report": report,
        "report_path": report_path,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect normalized OFI behavior before H002 preregistration"
    )
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--window-seconds", type=int, default=DEFAULT_WINDOW_SECONDS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    result = run_h002_feature_diagnostics(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        window_seconds=args.window_seconds,
    )
    print(f"Report written to {result['report_path']}")


if __name__ == "__main__":
    main()
