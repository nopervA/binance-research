# H003 Bucket Feasibility Check

Pre-registration feasibility check for H003 bucket boundaries.

Conducted on:

* Symbol: BTCUSDT
* Date range: 2026-06-13 to 2026-06-17
* Threshold: 1.5
* Cooldown: 300 seconds

Observed event distribution:

* low (1.5-2.0): 433 events (68.3%)
* medium (2.0-3.0): 118 events (18.6%)
* high (3.0-5.0): 55 events (8.7%)
* extreme (5.0+): 28 events (4.4%)

Total events: 634

Decision:

Bucket boundaries are retained unchanged.

The extreme bucket contains 28 events, slightly below the 30-event guideline for full bootstrap reliability. Therefore:

* The extreme bucket remains part of the study.
* Results for this bucket should be interpreted descriptively if the final study period still contains fewer than 30 events.
* Regression-based relationship summaries may exclude this bucket when event_count < 30.

Notes:

This feasibility check was performed before H003 execution and only to verify that each bucket contains a non-trivial number of events.

These counts are expected to change as additional market data is collected.
