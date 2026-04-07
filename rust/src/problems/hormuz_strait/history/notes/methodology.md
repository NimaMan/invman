# Methodology

## Why this package exists

The Hormuz FlowNet should be validated against history before it is used for forward scenarios.

This history package supports that by giving the model:

- a reliable daily oil-price series over the one-year, ten-year, and twenty-year horizons
- the dated conflict and policy milestones that matter most
- a daily crisis-window proxy for observed passage collapse through the Strait of Hormuz

## Included

- daily Brent and WTI spot history across named horizons
- monthly Brent and WTI summaries across named horizons
- key dated events from official EIA, OPEC, MSCIO/JMIC, and UNCTAD sources
- daily AIS-derived Strait of Hormuz cargo and tanker transits for the acute crisis window from
  `2026-02-28` through `2026-03-17`

## Not yet included

- a complete daily ship-transit series for the whole year
- tanker rates, options prices, or insurance premia as a continuous time series
- refinery-level operations or regional stock draws as a continuous time series

## Interpretation rules

- AIS-derived ship counts are lower-bound observations, not full true traffic counts
- JMIC repeatedly notes that dark transits and GNSS-degraded reporting remain possible
- the shipping-proxy table is therefore a disruption indicator, not a perfect physical throughput
  measurement
- FRED daily prices are used for the long history, while the EIA daily-prices page is used to append
  the latest close available on the analysis date
- the checked-in raw files are the reproducibility anchor; rebuilding the package should not require
  network access unless the user explicitly requests a raw-data refresh
- even on refresh runs, the build falls back to the checked-in raw snapshot if the remote source is
  temporarily unavailable

## First-degree model target

The first-degree Hormuz FlowNet should try to explain:

- broad Brent/WTI direction
- the timing of the crisis break after `2026-02-28`
- the scale of observed traffic collapse
- where the current price regime sits relative to the last ten and twenty years
- the market response to risk and mitigation signals
