"""Locate timestamp gaps in Binance Futures production datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.loader import load_liquidations, load_top_of_book, load_trades

STREAM_LOADERS = {
    "trades": load_trades,
    "top_of_book": load_top_of_book,
    "liquidations": load_liquidations,
}

GAP_COLUMNS = ["gap_start", "gap_end", "gap_seconds"]
ALL_SYMBOL_GAP_COLUMNS = [*GAP_COLUMNS, "symbol"]
OVERLAP_TOLERANCE_SECONDS = 2.0


def _load_stream_df(symbol: str, start: str, end: str, stream: str) -> pd.DataFrame:
    loader = STREAM_LOADERS.get(stream)
    if loader is None:
        raise ValueError(
            f"Unknown stream: {stream}. "
            f"Expected one of: {', '.join(sorted(STREAM_LOADERS))}"
        )
    return loader(symbol, start, end)


def find_gaps(
    symbol: str,
    start: str,
    end: str,
    stream: str,
    min_gap_seconds: float = 10.0,
) -> pd.DataFrame:
    df = _load_stream_df(symbol, start, end, stream)
    if len(df) < 2:
        return pd.DataFrame(columns=GAP_COLUMNS)

    timestamps = df.index.to_series()
    diffs = timestamps.diff()

    gaps: list[dict[str, object]] = []
    for i in range(1, len(timestamps)):
        gap_seconds = float(diffs.iloc[i].total_seconds())
        if gap_seconds >= min_gap_seconds:
            gaps.append(
                {
                    "gap_start": timestamps.iloc[i - 1],
                    "gap_end": timestamps.iloc[i],
                    "gap_seconds": gap_seconds,
                }
            )

    if not gaps:
        return pd.DataFrame(columns=GAP_COLUMNS)

    return (
        pd.DataFrame(gaps)
        .sort_values("gap_seconds", ascending=False)
        .reset_index(drop=True)
    )


def find_gaps_all_symbols(
    symbols: list[str],
    start: str,
    end: str,
    stream: str,
    min_gap_seconds: float = 10.0,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        gaps = find_gaps(symbol, start, end, stream, min_gap_seconds)
        if gaps.empty:
            gaps = pd.DataFrame(columns=GAP_COLUMNS)
        gaps = gaps.copy()
        gaps["symbol"] = symbol
        frames.append(gaps)

    if not frames:
        return pd.DataFrame(columns=ALL_SYMBOL_GAP_COLUMNS)

    return pd.concat(frames, ignore_index=True)


def find_overlapping_symbols(
    gaps: pd.DataFrame,
    tolerance_seconds: float = OVERLAP_TOLERANCE_SECONDS,
) -> set[str]:
    if gaps.empty or "symbol" not in gaps.columns:
        return set()

    tolerance = pd.Timedelta(seconds=tolerance_seconds)
    records = gaps.to_dict("records")
    affected: set[str] = set()

    for i, gap_a in enumerate(records):
        for gap_b in records[i + 1 :]:
            if gap_a["symbol"] == gap_b["symbol"]:
                continue
            if (
                gap_a["gap_start"] <= gap_b["gap_end"] + tolerance
                and gap_a["gap_end"] >= gap_b["gap_start"] - tolerance
            ):
                affected.add(gap_a["symbol"])
                affected.add(gap_b["symbol"])

    return affected


def format_gaps(gaps: pd.DataFrame) -> str:
    if gaps.empty:
        return "No gaps detected."

    lines = ["gap_start | gap_end | gap_seconds | symbol"]
    for row in gaps.itertuples(index=False):
        symbol = getattr(row, "symbol", "")
        lines.append(
            f"{row.gap_start} | {row.gap_end} | {row.gap_seconds:.3f} | {symbol}"
        )
    return "\n".join(lines)


def run_check_window(
    symbols: list[str],
    start: str,
    end: str,
    stream: str,
    min_gap_seconds: float = 10.0,
) -> pd.DataFrame:
    gaps = find_gaps_all_symbols(symbols, start, end, stream, min_gap_seconds)
    print(format_gaps(gaps))

    overlapping = find_overlapping_symbols(gaps)
    if overlapping:
        print("OVERLAPPING GAP DETECTED")
        print(f"affected symbols: {', '.join(sorted(overlapping))}")

    return gaps


def _parse_symbols(raw: str) -> list[str]:
    return [symbol.strip() for symbol in raw.split(",") if symbol.strip()]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locate timestamp gaps across Binance Futures streams"
    )
    parser.add_argument(
        "--symbols",
        required=True,
        help="Comma-separated symbols, e.g. BTCUSDT,ETHUSDT,DOGEUSDT",
    )
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--stream",
        required=True,
        choices=sorted(STREAM_LOADERS),
        help="Dataset stream to inspect",
    )
    parser.add_argument(
        "--min-gap-seconds",
        type=float,
        default=10.0,
        help="Minimum gap size to report (default: 10.0)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    run_check_window(
        symbols=_parse_symbols(args.symbols),
        start=args.start,
        end=args.end,
        stream=args.stream,
        min_gap_seconds=args.min_gap_seconds,
    )


if __name__ == "__main__":
    main()
