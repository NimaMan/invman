# timelines

Working timeline for the Hormuz Strait oil-price program.

## Assessment

The idea is sound.

The right sequence is:

1. model the high-level FlowNet first
2. verify it against recent history
3. only then use it for short-horizon forward scenarios

That keeps the work disciplined. If we skip the history-fit stage, the next-two-month oil-price
outputs will be too assumption-driven to trust.

## Exact Windows

Current analysis date:

- `2026-04-06`

Suggested verification and forecast windows:

- `history_backtest_year_1`
  - `2025-04-06` to `2026-04-06`
  - purpose: test whether the high-level FlowNet can explain the last year of oil-price direction
    and the shipping break after the conflict started

- `conflict_calibration_window`
  - `2026-02-28` to `2026-04-06`
  - purpose: fit the disruption mechanics using the actual start-of-hostilities breakpoint used by
    JMIC

- `forward_month_1`
  - `2026-04-07` to `2026-05-06`
  - purpose: first forecast window for Brent under scenario assumptions

- `forward_month_2`
  - `2026-05-07` to `2026-06-06`
  - purpose: second forecast window with updated supply-response, reserve, and transit assumptions

## What belongs in the first-degree FlowNet

At the first degree, the model should stay coarse and structural:

- exporter supply blocs
- Strait of Hormuz transit capacity
- bypass capacity
- reserve and floating-storage release
- destination demand blocs
- effective shortage / tightness state
- oil-price response layer

This is the right level for the first historical verification run.

## What to verify over the last year

The one-year history should verify at least four things:

- oil-price level and direction
  - Brent and WTI anchors
- physical disruption proxies
  - Hormuz passage counts, especially before and after `2026-02-28`
- response mechanisms
  - reserve release assumptions, bypass usage assumptions, and non-Hormuz supply response
- timing
  - how quickly price responds to traffic collapse and risk escalation

## Practical sequence

1. build a monthly backtest table from `2025-04-06` to `2026-04-06`
2. insert the conflict-break data from `maritime_traffic/`
3. fit a simple high-level price-response layer to history
4. verify the fitted model on the `2026-02-28` to `2026-04-06` crisis window
5. freeze the structure
6. run the two forward monthly windows

## Why this is the right scope

For now, oil prices are the correct output target.

Trying to predict individual voyage paths, exact charter rates, or refinery-by-refinery behavior
 this early would overcomplicate the backbone. The FlowNet should first prove that it can capture:

- transit collapse
- supply tightness
- mitigation response
- price impact

Once that works historically, the model can be pushed down to more detailed maritime or regional
substructure.

