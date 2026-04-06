# history

Reproducible backtest dataset for the first-degree Hormuz FlowNet.

Scope:

- backtest window: `2025-04-06` to `2026-04-06`
- crisis calibration window: `2026-02-28` to `2026-04-06`
- primary output target: oil prices

This folder collects only the essential inputs needed for a high-level historical verification:

- daily Brent and WTI prices
- monthly price summaries
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
- `brent_wti_monthly_summary.csv`
- `hormuz_market_event_timeline.csv`
- `hormuz_shipping_disruption_daily_signals.csv`

## Coverage note

Price coverage is strong across the full year.

Shipping coverage is strongest for the crisis window after `2026-02-28`, where JMIC/MSCIO provide
daily AIS-derived passage counts. Outside that crisis window, shipping coverage is sparse and should
be treated as baseline anchoring rather than a full daily series.
