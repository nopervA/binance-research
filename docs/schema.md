# Binance Futures Research Data Schema

This document describes the schemas and loading behavior for the first three
production datasets consumed by the research framework.

## Data Root

```
/var/lib/binance-futures-collector/data/{stream}/symbol={SYMBOL}/date={YYYY-MM-DD}/
```

## File Selection

For each requested date partition:

1. If `{stream}.parquet` exists in the date directory, **only** that file is read.
2. If the compacted file does not exist, all `*.segment.*.parquet` files in the
   date directory are read instead.
3. Compacted and segment files are **never** read together for the same day.

Examples:

- `trades/symbol=BTCUSDT/date=2026-06-13/trades.parquet`
- `trades/symbol=BTCUSDT/date=2026-06-13/trades.segment.001.parquet`

## Date Range

`start` and `end` parameters are **inclusive** on both ends.

Example: `load_trades("BTCUSDT", "2026-06-13", "2026-06-15")` loads:

- `2026-06-13`
- `2026-06-14`
- `2026-06-15`

After loading, rows are filtered to the inclusive timestamp window:

- `start_ts = pd.Timestamp(start, tz="UTC")`
- `end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)`

Only rows whose index timestamp falls within `[start_ts, end_ts]` are retained.

## Timestamp Conversion

All raw `timestamp`, `event_time`, and `received_at` fields are stored as
`int64` milliseconds UTC in parquet files.

During loading:

```python
pd.to_datetime(col, unit="ms", utc=True)
```

Returned DataFrames use a UTC `DatetimeIndex` named `timestamp`. The
`received_at` column is converted to `datetime64[ns, UTC]` and kept as a column.

## Trades

**Index:** `timestamp` (UTC `DatetimeIndex`)

| Column           | Type                         | Notes                          |
|------------------|------------------------------|--------------------------------|
| symbol           | str                          |                                |
| price            | float64                      | Must not be NaN                |
| quantity         | float64                      | Must not be NaN                |
| is_buyer_maker   | bool                         |                                |
| trade_id         | int64                        |                                |
| is_recovered     | bool                         |                                |
| received_at      | datetime64[ns, UTC]          | Ingestion time                 |

**Deduplication:** drop duplicates on `(symbol, trade_id)`, keep first.

### Trade Stream Notes

Trades are collected from Binance Futures `@aggTrade`.

Implications:

* Volume-based metrics remain valid.
* Taker-side metrics remain valid.
* OFI-style features remain valid.
* Multiple executions may be aggregated into a single aggTrade record.
* Trade-count-based metrics undercount true execution frequency.
* Researchers must not interpret aggTrade row counts as true trade counts.

## Top Of Book

**Index:** `timestamp` (renamed from raw `event_time`, UTC `DatetimeIndex`)

| Column           | Type                         | Notes                          |
|------------------|------------------------------|--------------------------------|
| symbol           | str                          |                                |
| received_at      | datetime64[ns, UTC]          | Ingestion time                 |
| best_bid_price   | float64                      | Must be positive, not NaN      |
| best_bid_qty     | float64                      |                                |
| best_ask_price   | float64                      | Must be positive, not NaN      |
| best_ask_qty     | float64                      |                                |
| spread           | float64                      |                                |
| spread_bps       | float64                      |                                |
| mid_price        | float64                      |                                |

**Deduplication:** drop fully duplicated rows, keep first.

**Crossed book:** rows where `best_bid_price > best_ask_price` are dropped with
a warning. This does not raise an exception.

## Liquidations

**Index:** `timestamp` (UTC `DatetimeIndex`)

| Column           | Type                         | Notes                          |
|------------------|------------------------------|--------------------------------|
| symbol           | str                          |                                |
| side             | str                          | `"BUY"` or `"SELL"`            |
| price            | float64                      | Must not be NaN                |
| quantity         | float64                      | Must not be NaN                |
| notional         | float64                      |                                |
| order_timestamp  | int64                        | Raw exchange timestamp (ms)    |
| received_at      | datetime64[ns, UTC]          | Ingestion time                 |

**Deduplication:** drop fully duplicated rows, keep first.

## Loader Processing Order

For all streams:

1. Read files (compacted **or** segment, never both)
2. Concatenate across dates
3. Convert timestamps to UTC datetime
4. Set timestamp index
5. Filter rows to inclusive `[start, end]` timestamp window
6. Remove duplicates
7. Apply quality checks
8. Sort ascending by timestamp
9. Return

## Missing Data

If no parquet files are found for the requested symbol and date range, the loader
raises `FileNotFoundError`.

If files exist but required quality checks fail, the loader raises
`DataQualityError`.
