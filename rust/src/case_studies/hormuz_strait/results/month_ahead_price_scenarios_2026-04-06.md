# Hormuz Strait Month-Ahead Brent Scenario Simulation

Analysis date: `2026-04-06`
Latest observed Brent close used as the starting point: `127.61` on `2026-04-02`
EIA month-ahead anchor: Brent stays above `95.00` per barrel over the next two months.
EIA second-quarter average anchor: `91.00` per barrel.

## Scenario Summary

| Scenario | Day 30 Mean | P10 | P50 | P90 | Monthly Mean | Avg Tightness (mb/d) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| De-escalation and gradual reopening | 100.21 | 95.05 | 100.25 | 105.50 | 107.42 | 0.02 |
| Managed partial disruption | 115.66 | 109.59 | 115.59 | 121.86 | 125.30 | 2.17 |
| Severe partial disruption | 153.10 | 145.57 | 153.08 | 160.43 | 150.29 | 7.84 |
| Full closure with coordinated response | 171.87 | 162.71 | 171.99 | 180.96 | 160.48 | 13.03 |
| Full closure with limited response | 192.82 | 182.20 | 192.72 | 203.33 | 173.75 | 21.48 |

## Interpretation

### De-escalation and gradual reopening

Traffic remains disrupted in early April but transit progressively normalizes and coordinated buffers absorb most of the lost flow.

Day 30 mean Brent: `100.21` with an 80% band of `95.05` to `105.50`.
Peak mean Brent in the simulation month: `125.13` on day `1`.
Average effective tightness: `0.02` million b/d.

### Managed partial disruption

Hormuz throughput is materially constrained for a month, but Saudi and UAE bypass capacity plus reserve releases cap the net shortage.

Day 30 mean Brent: `115.66` with an 80% band of `109.59` to `121.86`.
Peak mean Brent in the simulation month: `130.63` on day `6`.
Average effective tightness: `2.17` million b/d.

### Severe partial disruption

A large share of Hormuz throughput stays offline, with meaningful upstream shut-ins and only partial mitigation from rerouting and reserves.

Day 30 mean Brent: `153.10` with an 80% band of `145.57` to `160.43`.
Peak mean Brent in the simulation month: `155.21` on day `22`.
Average effective tightness: `7.84` million b/d.

### Full closure with coordinated response

The strait is effectively closed, but alternative export routes, reserve releases, and emergency coordination partially cushion the market.

Day 30 mean Brent: `171.87` with an 80% band of `162.71` to `180.96`.
Peak mean Brent in the simulation month: `171.87` on day `30`.
Average effective tightness: `13.03` million b/d.

### Full closure with limited response

The strait is closed for the whole month and policy mitigation is weak, leaving a large net supply shock and elevated geopolitical risk premium.

Day 30 mean Brent: `192.82` with an 80% band of `182.20` to `203.33`.
Peak mean Brent in the simulation month: `192.82` on day `30`.
Average effective tightness: `21.48` million b/d.

## Model Notes

- This is a scenario engine, not a market-clearing futures model or a claim about realized prices.
- The model is anchored to official EIA and OPEC figures current as of 2026-04-06 and uses 2024 Hormuz flow weights for physical exposure.
- Brent prices evolve by daily mean reversion from the latest observed close toward a scenario-specific target built from net effective tightness and a persistent risk premium.

The scenario engine uses the checked-in 2024 Hormuz flow weights, the EIA daily prices page dated April 6, 2026, the March 2026 EIA STEO, and the OPEC+ April 5, 2026 release.
