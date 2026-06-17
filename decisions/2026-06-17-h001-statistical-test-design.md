# H001 Statistical Test Design

Date: 2026-06-17

## Purpose

Document the final pre-execution statistical design for H001 before any
outcome statistics are computed.

## Design Timeline

1. Initial draft used `p_value_threshold=0.01` and
   `correct_direction_pct=54%`.
2. The 54% direction accuracy threshold was identified as arbitrary.
3. `minimum_effect_pct` was identified as arbitrary.
4. Economic significance was separated from predictive-information testing.
5. Bootstrap confidence interval adopted as the sole pass criterion.
6. Robustness checks changed from pass/fail gates to interpretive context.

## Final Primary Specification

**Research question type:** predictive information (not tradability)

**Primary metric:**

```
signed_return_i = direction_i * forward_return_i
```

where `direction_i = +1` for long events and `-1` for short events.

**Primary test:**

* Bootstrap 95% confidence interval
* `n_bootstrap = 10000`

**Pass condition:**

* `bootstrap_95ci_lower_bound > 0`
* Evaluated only on BTCUSDT, threshold = 1.5, horizon = 300 seconds

**Secondary reporting only (not pass/fail):**

* t-test p-value
* direction accuracy
* median signed return
* mean signed return
* estimated fee impact

**Robustness checks (interpretive context only):**

* remove top 5 absolute signed returns
* threshold 2.0
* threshold 3.0
* cross-symbol ETHUSDT

## Pre-Execution Statement

All revisions documented here occurred **before** H001 execution.

No H001 outcome statistics were observed before this revision:

* No bootstrap results
* No p-values
* No effect sizes
* No signed-return distributions

Prior validation covered pipeline mechanics only:

* OFI computation
* OFI z-score computation
* Event detection and cooldown logic
* Event count sanity checks across BTCUSDT, ETHUSDT, DOGEUSDT
* Event-to-mid-price alignment diagnostic

Alignment diagnostics validated timestamp matching and forward-price lookup
mechanics. They did not inspect aggregate hypothesis outcomes.

## Overall Pass Rule

H001 passes if the primary bootstrap CI lower bound is greater than zero on
the primary specification. Robustness checks are reported alongside the
result but do not determine pass/fail.
