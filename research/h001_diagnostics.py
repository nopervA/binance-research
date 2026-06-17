"""Pre-study alignment diagnostics for H001 event-to-outcome timestamp matching."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.loader import load_top_of_book, load_trades
from events.ofi_events import detect_ofi_events
from features.ofi import compute_ofi, compute_ofi_zscore, compute_signed_volume, resample_to_1s_grid

REPORTS_DIR = ROOT / "qa" / "reports"

VALID_SAMPLE_MODES = ("first", "first_and_last", "random")


def _select_events(
    events: pd.DataFrame,
    n_events: int,
    sample_mode: str,
    random_seed: int,
) -> pd.DataFrame:
    if sample_mode not in VALID_SAMPLE_MODES:
        raise ValueError(
            f'sample_mode must be one of {VALID_SAMPLE_MODES}, got "{sample_mode}"'
        )
    if events.empty:
        return events.copy()

    ordered = events.sort_values("timestamp").reset_index(drop=True)

    if sample_mode == "first":
        return ordered.head(n_events).copy()

    if sample_mode == "first_and_last":
        first_count = n_events // 2
        last_count = n_events - first_count
        selected = pd.concat(
            [ordered.head(first_count), ordered.tail(last_count)],
            ignore_index=True,
        )
        return selected.drop_duplicates(subset=["timestamp", "direction"]).head(n_events)

    sample_count = min(n_events, len(ordered))
    return ordered.sample(n=sample_count, random_state=random_seed).copy()


def _forward_price_lookup(
    query_times: pd.Series,
    mid_price: pd.Series,
    tolerance: pd.Timedelta,
) -> pd.DataFrame:
    if query_times.empty:
        return pd.DataFrame(
            columns=["query_time", "matched_timestamp", "matched_price", "_orig_order"]
        )

    lookup = pd.DataFrame(
        {
            "timestamp": pd.DatetimeIndex(mid_price.index).as_unit("ns"),
            "price": mid_price.to_numpy(),
        }
    ).sort_values("timestamp")

    queries = pd.DataFrame(
        {
            "query_time": pd.DatetimeIndex(query_times).as_unit("ns"),
            "_orig_order": range(len(query_times)),
        }
    ).sort_values("query_time")

    merged = pd.merge_asof(
        queries,
        lookup,
        left_on="query_time",
        right_on="timestamp",
        direction="forward",
        tolerance=tolerance,
    )
    return merged.sort_values("_orig_order").rename(
        columns={"timestamp": "matched_timestamp", "price": "matched_price"}
    )


def generate_alignment_diagnostic(
    events: pd.DataFrame,
    mid_price: pd.Series,
    horizon_seconds: int,
    n_events: int = 20,
    sample_mode: str = "first",
    random_seed: int = 42,
) -> pd.DataFrame:
    """Inspect event-to-outcome timestamp alignment for selected events."""
    selected = _select_events(events, n_events, sample_mode, random_seed)
    if selected.empty:
        return pd.DataFrame(
            columns=[
                "event_timestamp",
                "event_direction",
                "zscore_value",
                "price_at_event",
                "price_at_event_timestamp",
                "target_time",
                "matched_mid_price_timestamp",
                "time_delta_seconds",
                "price_at_horizon",
                "forward_return_used",
            ]
        )

    selected = selected.sort_values("timestamp").reset_index(drop=True)
    event_times = selected["timestamp"]

    event_matches = _forward_price_lookup(
        event_times,
        mid_price,
        tolerance=pd.Timedelta(seconds=2),
    )
    target_times = event_times + pd.Timedelta(seconds=horizon_seconds)
    horizon_matches = _forward_price_lookup(
        target_times,
        mid_price,
        tolerance=pd.Timedelta(seconds=horizon_seconds / 2),
    )

    price_at_event = event_matches["matched_price"].to_numpy()
    price_at_event_timestamp = event_matches["matched_timestamp"].to_numpy()
    matched_mid_price_timestamp = horizon_matches["matched_timestamp"].to_numpy()
    price_at_horizon = horizon_matches["matched_price"].to_numpy()

    matched_ts = pd.to_datetime(matched_mid_price_timestamp, utc=True)
    target_ts = pd.to_datetime(target_times, utc=True)
    time_delta_seconds = (matched_ts - target_ts).dt.total_seconds().to_numpy()

    forward_return_used = price_at_horizon / price_at_event - 1
    invalid_mask = (
        pd.isna(price_at_event)
        | pd.isna(price_at_horizon)
        | (price_at_event == 0)
    )
    forward_return_used = np.where(invalid_mask, np.nan, forward_return_used)

    result = pd.DataFrame(
        {
            "event_timestamp": event_times.to_numpy(),
            "event_direction": selected["direction"].to_numpy(),
            "zscore_value": selected["zscore_value"].to_numpy(),
            "price_at_event": price_at_event,
            "price_at_event_timestamp": price_at_event_timestamp,
            "target_time": target_times.to_numpy(),
            "matched_mid_price_timestamp": matched_mid_price_timestamp,
            "time_delta_seconds": time_delta_seconds,
            "price_at_horizon": price_at_horizon,
            "forward_return_used": forward_return_used,
        }
    )
    return result.sort_values("event_timestamp").reset_index(drop=True)


def format_diagnostic_table(diagnostic: pd.DataFrame) -> str:
    if diagnostic.empty:
        return "No events selected for alignment diagnostic."
    return diagnostic.to_string(index=False)


def run_h001_alignment_diagnostic(
    symbol: str,
    start: str,
    end: str,
    threshold: float,
    cooldown_seconds: int,
    horizon_seconds: int,
    n_events: int = 20,
    sample_mode: str = "first",
    random_seed: int = 42,
) -> pd.DataFrame:
    trades = load_trades(symbol, start, end)
    top_of_book = load_top_of_book(symbol, start, end)
    mid_price = top_of_book["mid_price"]

    signed_volume = compute_signed_volume(trades)
    signed_volume_1s = resample_to_1s_grid(signed_volume, start, end)
    ofi = compute_ofi(signed_volume_1s, window_seconds=60)
    zscore = compute_ofi_zscore(ofi, lookback_seconds=3600)
    events = detect_ofi_events(
        zscore,
        threshold=threshold,
        cooldown_seconds=cooldown_seconds,
        direction="both",
    )

    diagnostic = generate_alignment_diagnostic(
        events,
        mid_price,
        horizon_seconds=horizon_seconds,
        n_events=n_events,
        sample_mode=sample_mode,
        random_seed=random_seed,
    )

    report = format_diagnostic_table(diagnostic)
    print(report)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"H001_alignment_diagnostic_{symbol}_{start}_{end}.txt"
    report_path.write_text(report + "\n", encoding="utf-8")
    print(f"Alignment diagnostic written to {report_path}. Review before continuing to H001.")

    return diagnostic


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="H001 pre-study event-to-outcome alignment diagnostic"
    )
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--cooldown-seconds", type=int, default=300)
    parser.add_argument("--horizon-seconds", type=int, required=True)
    parser.add_argument("--n-events", type=int, default=20)
    parser.add_argument(
        "--sample-mode",
        choices=list(VALID_SAMPLE_MODES),
        default="first",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    run_h001_alignment_diagnostic(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        threshold=args.threshold,
        cooldown_seconds=args.cooldown_seconds,
        horizon_seconds=args.horizon_seconds,
        n_events=args.n_events,
        sample_mode=args.sample_mode,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
