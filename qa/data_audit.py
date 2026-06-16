"""Data quality audit metrics for Binance Futures production datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.loader import load_liquidations, load_top_of_book, load_trades

REPORTS_DIR = Path(__file__).resolve().parent / "reports"


def _percentile_stats(series: pd.Series) -> dict[str, float]:
    if series.empty:
        return {"p50": float("nan"), "p95": float("nan"), "p99": float("nan"), "max": float("nan")}
    return {
        "p50": float(series.quantile(0.50)),
        "p95": float(series.quantile(0.95)),
        "p99": float(series.quantile(0.99)),
        "max": float(series.max()),
    }


def _latency_ms(df: pd.DataFrame) -> pd.Series:
    return (df["received_at"] - df.index.to_series()).dt.total_seconds() * 1000


def _longest_gap_seconds(df: pd.DataFrame) -> float:
    if len(df) < 2:
        return 0.0
    gaps = df.index.to_series().diff().dropna()
    return float(gaps.max().total_seconds())


def _timestamp_bounds(df: pd.DataFrame) -> tuple[str, str]:
    if df.empty:
        return "N/A", "N/A"
    return df.index.min().isoformat(), df.index.max().isoformat()


def audit_trades(symbol: str, start: str, end: str) -> dict[str, Any]:
    df = load_trades(symbol, start, end)
    min_ts, max_ts = _timestamp_bounds(df)
    latency = _latency_ms(df)

    return {
        "row_count": len(df),
        "min_timestamp": min_ts,
        "max_timestamp": max_ts,
        "latency_ms": _percentile_stats(latency),
        "is_buyer_maker_ratio": float(df["is_buyer_maker"].mean()) if len(df) else float("nan"),
        "longest_gap_seconds": _longest_gap_seconds(df),
    }


def audit_top_of_book(symbol: str, start: str, end: str) -> dict[str, Any]:
    df = load_top_of_book(symbol, start, end)
    min_ts, max_ts = _timestamp_bounds(df)
    latency = _latency_ms(df)

    return {
        "row_count": len(df),
        "min_timestamp": min_ts,
        "max_timestamp": max_ts,
        "spread_bps": _percentile_stats(df["spread_bps"]),
        "latency_ms": _percentile_stats(latency),
        "longest_gap_seconds": _longest_gap_seconds(df),
    }


def audit_liquidations(symbol: str, start: str, end: str) -> dict[str, Any]:
    df = load_liquidations(symbol, start, end)
    min_ts, max_ts = _timestamp_bounds(df)
    side_distribution = (
        df["side"].value_counts().astype(int).to_dict() if len(df) else {}
    )

    return {
        "row_count": len(df),
        "min_timestamp": min_ts,
        "max_timestamp": max_ts,
        "side_distribution": side_distribution,
        "notional": _percentile_stats(df["notional"]),
        "longest_gap_seconds": _longest_gap_seconds(df),
    }


def _fmt_stats(stats: dict[str, float], decimals: int = 2) -> str:
    return "/".join(f"{stats[key]:.{decimals}f}" for key in ("p50", "p95", "p99", "max"))


def _fmt_stats_3(stats: dict[str, float], decimals: int = 2) -> str:
    return "/".join(f"{stats[key]:.{decimals}f}" for key in ("p50", "p95", "p99"))


def _fmt_side_distribution(distribution: dict[str, int]) -> str:
    if not distribution:
        return "N/A"
    return ", ".join(f"{side}: {count}" for side, count in sorted(distribution.items()))


def format_report(
    symbol: str,
    start: str,
    end: str,
    trades: dict[str, Any],
    top_of_book: dict[str, Any],
    liquidations: dict[str, Any],
) -> str:
    lines = [
        f"# QA Audit: {symbol} {start} to {end}",
        "",
        "## TRADES",
        f"rows: {trades['row_count']}",
        f"date range: {trades['min_timestamp']} to {trades['max_timestamp']}",
        f"latency p50/p95/p99/max (ms): {_fmt_stats(trades['latency_ms'])}",
        f"is_buyer_maker ratio: {trades['is_buyer_maker_ratio']:.4f}",
        f"longest gap (s): {trades['longest_gap_seconds']:.3f}",
        "",
        "## TOP_OF_BOOK",
        f"rows: {top_of_book['row_count']}",
        f"date range: {top_of_book['min_timestamp']} to {top_of_book['max_timestamp']}",
        f"spread p50/p95/p99 (bps): {_fmt_stats_3(top_of_book['spread_bps'])}",
        f"latency p50/p95/p99/max (ms): {_fmt_stats(top_of_book['latency_ms'])}",
        f"longest gap (s): {top_of_book['longest_gap_seconds']:.3f}",
        "",
        "## LIQUIDATIONS",
        f"rows: {liquidations['row_count']}",
        f"date range: {liquidations['min_timestamp']} to {liquidations['max_timestamp']}",
        f"side distribution: {_fmt_side_distribution(liquidations['side_distribution'])}",
        f"notional p50/p95/p99/max: {_fmt_stats(liquidations['notional'])}",
        f"longest gap (s): {liquidations['longest_gap_seconds']:.3f}",
        "",
    ]
    return "\n".join(lines)


def run_audit(symbol: str, start: str, end: str) -> str:
    trades = audit_trades(symbol, start, end)
    top_of_book = audit_top_of_book(symbol, start, end)
    liquidations = audit_liquidations(symbol, start, end)
    report = format_report(symbol, start, end, trades, top_of_book, liquidations)

    print(report, end="")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"{symbol}_{start}_{end}.txt"
    report_path.write_text(report, encoding="utf-8")

    return report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Binance Futures data quality")
    parser.add_argument("--symbol", required=True, help="Futures symbol, e.g. BTCUSDT")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD, inclusive)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD, inclusive)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    run_audit(args.symbol, args.start, args.end)


if __name__ == "__main__":
    main()
