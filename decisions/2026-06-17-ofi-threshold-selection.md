# OFI Threshold Selection

Date: 2026-06-17

## Context

A 9-point sanity check was performed across:

* BTCUSDT
* ETHUSDT
* DOGEUSDT

Dates:

* 2026-06-13
* 2026-06-14
* 2026-06-15

Thresholds evaluated:

* 1.5
* 2.0
* 3.0

## Results summary

**Threshold 1.5:**

approximately 110-143 events/day

**Threshold 2.0:**

approximately 75-100 events/day

**Threshold 3.0:**

approximately 45-57 events/day

## Observations

* Event counts were highly stable across symbols.
* Event counts were highly stable across days.
* DOGEUSDT behaved similarly to BTCUSDT and ETHUSDT despite different liquidity characteristics.
* OFI z-score normalization appears robust across liquidity regimes.

## Decision

Selected threshold = 1.5

Selected cooldown_seconds = 300

### Rationale

* Provides sufficient sample size for H001.
* Produces stable event counts across symbols and dates.
* Minimizes risk of symbol-specific parameter tuning.

## Additional decisions

* Primary evaluation horizon = 300 seconds.
* Secondary exploratory horizon = 60 seconds.
* Event timestamps originate from the OFI 1-second grid.
* Outcome returns are obtained from top_of_book mid-price data using forward timestamp alignment consistent with `features/returns.py`.

## Heavy-tail diagnostic

OFI z-score extremes were investigated.

Extreme positive event:

around 2026-06-13 21:41 UTC.

Extreme negative event:

around 2026-06-13 19:39 UTC.

These were traced to genuine large signed-volume clusters rather than implementation bugs.

### Conclusion

Heavy tails are considered part of the underlying market mechanism and are consistent with the hypothesis under study.

## Future robustness requirement

All H001 results must be re-evaluated after removing the top 5 largest absolute-z-score events.
