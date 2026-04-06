# data/processed

Processed backtest tables for the Hormuz FlowNet.

Current outputs:

- `brent_wti_daily_prices.csv`
  - daily Brent and WTI history across the one-year backtest window
- `brent_wti_monthly_summary.csv`
  - monthly aggregates derived from the daily price table
- `hormuz_market_event_timeline.csv`
  - dated structural, conflict, market-expectation, and supply-response milestones
- `hormuz_shipping_disruption_daily_signals.csv`
  - source-explicit AIS-based shipping disruption indicators, anchored at `2025-06-15` and then daily
    through the crisis window from `2026-02-28` to `2026-03-17`

These are the canonical history inputs for first-pass model verification.
