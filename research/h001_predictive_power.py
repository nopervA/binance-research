"""H001 predictive information study execution."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from math import erfc, sqrt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.loader import load_top_of_book, load_trades
from events.ofi_events import detect_ofi_events
from features.ofi import compute_ofi, compute_ofi_zscore, compute_signed_volume, resample_to_1s_grid

RESULTS_DIR = ROOT / "results"

H001_PRIMARY_SYMBOL = "BTCUSDT"
H001_PRIMARY_THRESHOLD = 1.5
H001_PRIMARY_HORIZON_SECONDS = 300
H001_OFI_WINDOW_SECONDS = 60
H001_ZSCORE_LOOKBACK_SECONDS = 3600
H001_COOLDOWN_SECONDS = 300
H001_MINIMUM_EVENTS = 300
H001_N_BOOTSTRAP = 10000
H001_CONFIDENCE_LEVEL = 0.95
H001_RANDOM_SEED = 42


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


def compute_event_outcomes(
    events: pd.DataFrame,
    mid_price: pd.Series,
    horizon_seconds: int,
) -> pd.DataFrame:
    """Compute aligned forward outcomes for every detected event."""
    if events.empty:
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
                "forward_return",
                "signed_return",
            ]
        )

    ordered = events.sort_values("timestamp").reset_index(drop=True)
    event_times = ordered["timestamp"]

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
    price_at_horizon = horizon_matches["matched_price"].to_numpy()
    matched_ts = pd.to_datetime(horizon_matches["matched_timestamp"], utc=True)
    target_ts = pd.to_datetime(target_times, utc=True)
    time_delta_seconds = (matched_ts - target_ts).dt.total_seconds().to_numpy()

    forward_return = price_at_horizon / price_at_event - 1
    invalid_mask = (
        pd.isna(price_at_event) | pd.isna(price_at_horizon) | (price_at_event == 0)
    )
    forward_return = np.where(invalid_mask, np.nan, forward_return)

    direction_sign = np.where(ordered["direction"].to_numpy() == "long", 1.0, -1.0)
    signed_return = direction_sign * forward_return
    signed_return = np.where(pd.isna(forward_return), np.nan, signed_return)

    return pd.DataFrame(
        {
            "event_timestamp": event_times.to_numpy(),
            "event_direction": ordered["direction"].to_numpy(),
            "zscore_value": ordered["zscore_value"].to_numpy(),
            "price_at_event": price_at_event,
            "price_at_event_timestamp": event_matches["matched_timestamp"].to_numpy(),
            "target_time": target_times.to_numpy(),
            "matched_mid_price_timestamp": horizon_matches[
                "matched_timestamp"
            ].to_numpy(),
            "time_delta_seconds": time_delta_seconds,
            "price_at_horizon": price_at_horizon,
            "forward_return": forward_return,
            "signed_return": signed_return,
        }
    )


def bootstrap_mean_ci(
    values: np.ndarray,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> dict[str, float]:
    """Percentile bootstrap confidence interval for the mean."""
    clean = np.asarray(values, dtype=float)
    clean = clean[~np.isnan(clean)]
    if clean.size == 0:
        return {
            "bootstrap_mean": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
        }

    original_mean = float(np.mean(clean))
    rng = np.random.default_rng(random_seed)
    sample_indices = rng.integers(0, clean.size, size=(n_bootstrap, clean.size))
    bootstrap_means = clean[sample_indices].mean(axis=1)

    alpha = (1.0 - confidence_level) / 2.0
    ci_lower = float(np.percentile(bootstrap_means, 100.0 * alpha))
    ci_upper = float(np.percentile(bootstrap_means, 100.0 * (1.0 - alpha)))

    return {
        "bootstrap_mean": original_mean,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def one_sample_ttest_pvalue(values: np.ndarray) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[~np.isnan(clean)]
    if clean.size < 2:
        return float("nan")

    mean = float(np.mean(clean))
    std = float(np.std(clean, ddof=1))
    if std == 0.0:
        return float("nan")

    t_stat = mean / (std / sqrt(clean.size))
    try:
        from scipy.stats import t as student_t

        return float(2.0 * student_t.sf(abs(t_stat), df=clean.size - 1))
    except ImportError:
        z = abs(t_stat)
        return float(erfc(z / sqrt(2.0)))


def summarize_sample(outcomes: pd.DataFrame) -> dict[str, int]:
    total_events_detected = len(outcomes)
    events_with_valid_outcome = int(outcomes["signed_return"].notna().sum())
    events_dropped_due_to_missing_outcome = (
        total_events_detected - events_with_valid_outcome
    )
    return {
        "total_events_detected": total_events_detected,
        "events_with_valid_outcome": events_with_valid_outcome,
        "events_dropped_due_to_missing_outcome": events_dropped_due_to_missing_outcome,
    }


def compute_secondary_metrics(signed_returns: np.ndarray) -> dict[str, float | int]:
    clean = signed_returns[~np.isnan(signed_returns)]
    if clean.size == 0:
        return {
            "event_count": 0,
            "mean_signed_return": float("nan"),
            "median_signed_return": float("nan"),
            "direction_accuracy": float("nan"),
            "t_test_pvalue": float("nan"),
        }

    return {
        "event_count": int(clean.size),
        "mean_signed_return": float(np.mean(clean)),
        "median_signed_return": float(np.median(clean)),
        "direction_accuracy": float(np.mean(clean > 0)),
        "t_test_pvalue": one_sample_ttest_pvalue(clean),
    }


def evaluate_primary_pass(bootstrap_result: dict[str, float]) -> bool:
    return bootstrap_result["ci_lower"] > 0


def remove_top_5_abs_signed_returns(outcomes: pd.DataFrame) -> pd.DataFrame:
    result = outcomes.copy()
    valid = result[result["signed_return"].notna()].copy()
    if valid.empty:
        return result
    drop_index = valid.assign(
        abs_signed_return=valid["signed_return"].abs()
    ).nlargest(min(5, len(valid)), "abs_signed_return").index
    return result.drop(index=drop_index)


def run_ofi_event_pipeline(
    symbol: str,
    start: str,
    end: str,
    threshold: float,
    cooldown_seconds: int = H001_COOLDOWN_SECONDS,
) -> tuple[pd.DataFrame, pd.Series]:
    trades = load_trades(symbol, start, end)
    top_of_book = load_top_of_book(symbol, start, end)
    mid_price = top_of_book["mid_price"]

    signed_volume = compute_signed_volume(trades)
    signed_volume_1s = resample_to_1s_grid(signed_volume, start, end)
    ofi = compute_ofi(signed_volume_1s, window_seconds=H001_OFI_WINDOW_SECONDS)
    zscore = compute_ofi_zscore(ofi, lookback_seconds=H001_ZSCORE_LOOKBACK_SECONDS)
    events = detect_ofi_events(
        zscore,
        threshold=threshold,
        cooldown_seconds=cooldown_seconds,
        direction="both",
    )
    return events, mid_price


def analyze_outcomes(outcomes: pd.DataFrame) -> dict[str, Any]:
    sample = summarize_sample(outcomes)
    signed_returns = outcomes["signed_return"].to_numpy()
    bootstrap = bootstrap_mean_ci(
        signed_returns,
        n_bootstrap=H001_N_BOOTSTRAP,
        confidence_level=H001_CONFIDENCE_LEVEL,
        random_seed=H001_RANDOM_SEED,
    )
    secondary = compute_secondary_metrics(signed_returns)
    passed = evaluate_primary_pass(bootstrap)

    return {
        "sample": sample,
        "bootstrap": bootstrap,
        "secondary": secondary,
        "pass": passed,
    }


def analyze_robustness_checks(
    start: str,
    end: str,
    primary_outcomes: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    robustness: dict[str, dict[str, Any]] = {}

    for threshold in (2.0, 3.0):
        events, mid_price = run_ofi_event_pipeline(
            H001_PRIMARY_SYMBOL,
            start,
            end,
            threshold=threshold,
        )
        outcomes = compute_event_outcomes(
            events,
            mid_price,
            horizon_seconds=H001_PRIMARY_HORIZON_SECONDS,
        )
        robustness[f"threshold_{str(threshold).replace('.', '_')}"] = analyze_outcomes(
            outcomes
        )

    trimmed = remove_top_5_abs_signed_returns(primary_outcomes)
    robustness["remove_top_5_abs_signed_returns"] = analyze_outcomes(trimmed)

    eth_events, eth_mid = run_ofi_event_pipeline(
        "ETHUSDT",
        start,
        end,
        threshold=H001_PRIMARY_THRESHOLD,
    )
    eth_outcomes = compute_event_outcomes(
        eth_events,
        eth_mid,
        horizon_seconds=H001_PRIMARY_HORIZON_SECONDS,
    )
    robustness["cross_symbol_ethusdt"] = analyze_outcomes(eth_outcomes)

    return robustness


def generate_h001_report(
    *,
    symbol: str,
    start: str,
    end: str,
    threshold: float,
    horizon_seconds: int,
    primary_analysis: dict[str, Any],
    robustness: dict[str, dict[str, Any]],
) -> str:
    sample = primary_analysis["sample"]
    bootstrap = primary_analysis["bootstrap"]
    secondary = primary_analysis["secondary"]
    passed = primary_analysis["pass"]

    limitation = ""
    if sample["events_with_valid_outcome"] < H001_MINIMUM_EVENTS:
        limitation = (
            f"\n\n**Limitation:** events_with_valid_outcome "
            f"({sample['events_with_valid_outcome']}) is below the preregistered "
            f"minimum of {H001_MINIMUM_EVENTS}."
        )

    lines = [
        "# H001 Result",
        "",
        "## Preregistration Summary",
        "",
        f"- Symbol: {symbol}",
        f"- Threshold: {threshold}",
        f"- Cooldown seconds: {H001_COOLDOWN_SECONDS}",
        f"- Horizon seconds: {horizon_seconds}",
        f"- Date range: {start} to {end} (inclusive)",
        "- Research question: Do extreme OFI z-score events contain predictive",
        "  information about 300-second forward mid-price returns?",
        "",
        "## Sample Summary",
        "",
        f"- total_events_detected: {sample['total_events_detected']}",
        f"- valid_outcomes: {sample['events_with_valid_outcome']}",
        f"- dropped_outcomes: {sample['events_dropped_due_to_missing_outcome']}",
        limitation,
        "",
        "## Primary Result",
        "",
        f"- event_count: {secondary['event_count']}",
        f"- bootstrap_mean: {bootstrap['bootstrap_mean']:.8f}",
        f"- bootstrap_ci_lower: {bootstrap['ci_lower']:.8f}",
        f"- bootstrap_ci_upper: {bootstrap['ci_upper']:.8f}",
        f"- pass/fail: {'PASS' if passed else 'FAIL'}",
        "",
        "## Secondary Metrics",
        "",
        f"- mean_signed_return: {secondary['mean_signed_return']:.8f}",
        f"- median_signed_return: {secondary['median_signed_return']:.8f}",
        f"- direction_accuracy: {secondary['direction_accuracy']:.8f}",
        f"- t_test_pvalue: {secondary['t_test_pvalue']:.8f}",
        "",
        "## Robustness Checks",
        "",
    ]

    for name, analysis in robustness.items():
        rb = analysis["bootstrap"]
        sec = analysis["secondary"]
        samp = analysis["sample"]
        lines.extend(
            [
                f"### {name}",
                f"- event_count: {sec['event_count']}",
                f"- bootstrap_mean: {rb['bootstrap_mean']:.8f}",
                f"- bootstrap_ci_lower: {rb['ci_lower']:.8f}",
                f"- bootstrap_ci_upper: {rb['ci_upper']:.8f}",
                f"- valid_outcomes: {samp['events_with_valid_outcome']}",
                "",
            ]
        )

    if passed:
        interpretation = (
            "The primary bootstrap confidence interval lower bound is greater "
            "than zero, consistent with positive predictive information in signed "
            "forward returns following extreme OFI events."
        )
    else:
        interpretation = (
            "The primary bootstrap confidence interval lower bound is not greater "
            "than zero, so the preregistered pass condition is not met."
        )

    lines.extend(
        [
            "## Conclusion",
            "",
            interpretation,
            "",
            "This study evaluates predictive information content only. It does not "
            "evaluate tradability, execution feasibility, fees, slippage, or "
            "deployable strategy performance.",
            "",
        ]
    )
    return "\n".join(lines)


def run_h001_predictive_power(
    symbol: str,
    start: str,
    end: str,
    *,
    threshold: float = H001_PRIMARY_THRESHOLD,
    horizon_seconds: int = H001_PRIMARY_HORIZON_SECONDS,
    write_report: bool = True,
) -> dict[str, Any]:
    events, mid_price = run_ofi_event_pipeline(symbol, start, end, threshold=threshold)
    outcomes = compute_event_outcomes(events, mid_price, horizon_seconds=horizon_seconds)
    primary_analysis = analyze_outcomes(outcomes)

    robustness = {}
    if symbol == H001_PRIMARY_SYMBOL and threshold == H001_PRIMARY_THRESHOLD:
        robustness = analyze_robustness_checks(start, end, outcomes)

    report = generate_h001_report(
        symbol=symbol,
        start=start,
        end=end,
        threshold=threshold,
        horizon_seconds=horizon_seconds,
        primary_analysis=primary_analysis,
        robustness=robustness,
    )

    report_path = None
    if write_report:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_path = RESULTS_DIR / f"H001_{timestamp}.md"
        report_path.write_text(report, encoding="utf-8")

    return {
        "outcomes": outcomes,
        "primary_analysis": primary_analysis,
        "robustness": robustness,
        "report": report,
        "report_path": report_path,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute H001 predictive power study")
    parser.add_argument("--symbol", default=H001_PRIMARY_SYMBOL)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--threshold", type=float, default=H001_PRIMARY_THRESHOLD)
    parser.add_argument("--horizon-seconds", type=int, default=H001_PRIMARY_HORIZON_SECONDS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    result = run_h001_predictive_power(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        threshold=args.threshold,
        horizon_seconds=args.horizon_seconds,
    )
    print(result["report"])
    if result["report_path"] is not None:
        print(f"Report written to {result['report_path']}")


if __name__ == "__main__":
    main()
