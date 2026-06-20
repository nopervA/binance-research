# H003 Bucket Feasibility Check

Date: 2026-06-19

## Purpose

Pre-registration feasibility check for H003 bucket boundaries before
executing the study.

This check was performed to verify that the proposed magnitude buckets
contain a non-trivial number of events and are suitable for descriptive
and bootstrap-based analysis.

## Dataset

* Symbol: BTCUSDT
* Date range: 2026-06-13 to 2026-06-17
* Threshold: 1.5
* Cooldown: 300 seconds
* Signal: Raw OFI z-score events

## Bucket Distribution

| Bucket  | Range              | Event Count | Share |
| ------- | ------------------ | ----------: | ----: |
| Low     | 1.5 ≤ \|z\| < 2.0  |         433 | 68.3% |
| Medium  | 2.0 ≤ \|z\| < 3.0  |         118 | 18.6% |
| High    | 3.0 ≤ \|z\| < 5.0  |          55 |  8.7% |
| Extreme | \|z\| ≥ 5.0        |          28 |  4.4% |

Total events: 634

## Decision

Bucket boundaries are retained as specified:

* Low: [1.5, 2.0)
* Medium: [2.0, 3.0)
* High: [3.0, 5.0)
* Extreme: [5.0, +∞)

The extreme bucket contains 28 events during the feasibility period.
This is below the conventional 30-event threshold often used for
asymptotic approximations, but is sufficient for descriptive reporting.

Per the H003 registry specification, any bucket with fewer than
30 events should be interpreted cautiously and may be excluded from
regression-based relationship summaries if event counts remain below
that threshold during the final study period.

## Notes

This feasibility check is not part of the H003 result.

No forward-return statistics, bootstrap results, hypothesis outcomes,
or predictive-performance metrics were examined during this check.

The purpose of this document is solely to validate bucket design before
execution.
