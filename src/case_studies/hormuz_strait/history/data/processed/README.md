# data/processed

Processed backtest tables for the Hormuz FlowNet.

Current outputs:

- `brent_wti_daily_prices.csv`
  - daily Brent and WTI history across the one-year backtest window
- `brent_wti_monthly_summary.csv`
  - monthly aggregates derived from the one-year daily price table
- `one_year_brent_wti_price_history_chart.json`
  - chart-ready one-year Brent/WTI payload for downstream clients
- `brent_wti_ten_year_daily_prices.csv`
  - daily Brent and WTI history across the ten-year market context window
- `brent_wti_ten_year_monthly_summary.csv`
  - monthly aggregates derived from the ten-year daily price table
- `ten_year_brent_wti_price_history_chart.json`
  - chart-ready ten-year Brent/WTI payload for downstream clients
- `brent_wti_twenty_year_daily_prices.csv`
  - daily Brent and WTI history across the twenty-year market context window
- `brent_wti_twenty_year_monthly_summary.csv`
  - monthly aggregates derived from the twenty-year daily price table
- `twenty_year_brent_wti_price_history_chart.json`
  - chart-ready twenty-year Brent/WTI payload for downstream clients
- `hormuz_market_event_timeline.csv`
  - dated structural, conflict, market-expectation, and supply-response milestones
- `hormuz_shipping_disruption_daily_signals.csv`
  - source-explicit AIS-based shipping disruption indicators, anchored at `2025-06-15` and then daily
    through the crisis window from `2026-02-28` to `2026-03-17`

These are the canonical history inputs for both first-pass crisis verification and wider market
context.
