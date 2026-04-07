# history

Reproducible history package for the first-degree Hormuz FlowNet.

Scope:

- one-year backtest window: `2025-04-06` to `2026-04-06`
- ten-year market context window: `2016-04-06` to `2026-04-06`
- twenty-year market context window: `2006-04-06` to `2026-04-06`
- crisis calibration window: `2026-02-28` to `2026-04-06`
- primary output target: oil prices

This folder collects the essential inputs needed for both crisis verification and longer-horizon
market context:

- daily Brent and WTI prices across named horizons
- chart-ready JSON payloads for the named price horizons
- monthly price summaries across named horizons
- a dated event timeline
- daily shipping-disruption proxies for the crisis window

It is intentionally not a full oil-market warehouse.

## Structure

- `data/raw/`
  - raw price-history files downloaded for this backtest package
- `data/processed/`
  - backtest tables used by the model
- `sources/`
  - source manifest and checksums
- `scripts/`
  - reproducible fetch/build scripts
- `results/`
  - generated backtest summaries tied to the processed tables
- `notes/`
  - methodology and coverage limitations

## Current processed tables

- `brent_wti_daily_prices.csv`
  - one-year daily price window used in the main backtest
- `brent_wti_monthly_summary.csv`
  - monthly aggregates for the one-year backtest
- `one_year_brent_wti_price_history_chart.json`
  - chart-ready one-year price payload for downstream visualization clients
- `brent_wti_ten_year_daily_prices.csv`
  - ten-year daily price window for medium-horizon context
- `brent_wti_ten_year_monthly_summary.csv`
  - monthly aggregates for the ten-year window
- `ten_year_brent_wti_price_history_chart.json`
  - chart-ready ten-year price payload for downstream visualization clients
- `brent_wti_twenty_year_daily_prices.csv`
  - twenty-year daily price window for long-cycle context
- `brent_wti_twenty_year_monthly_summary.csv`
  - monthly aggregates for the twenty-year window
- `twenty_year_brent_wti_price_history_chart.json`
  - chart-ready twenty-year price payload for downstream visualization clients
- `hormuz_market_event_timeline.csv`
- `hormuz_shipping_disruption_daily_signals.csv`

## Current result summaries

- `one_year_backtest_summary.md` / `.json`
- `ten_year_market_context_summary.md` / `.json`
- `twenty_year_market_context_summary.md` / `.json`

## Coverage note

Price coverage is strong across all three horizons because the package is built from pinned FRED
daily series plus the latest EIA anchor close.

Shipping coverage is strongest for the crisis window after `2026-02-28`, where JMIC/MSCIO provide
daily AIS-derived passage counts. Outside that crisis window, shipping coverage is sparse and should
be treated as baseline anchoring rather than a full daily series.
