# hormuz_strait

Rust-first home for a source-backed Strait of Hormuz disruption model.

This is a geography-first oil-flow problem, not a company graph.

Why:

- the Strait of Hormuz is a physical maritime chokepoint
- the main disruption mechanism is route closure or capacity loss
- the right first nodes are exporting countries, destination markets, transit assets, and reserve
  buffers

Current scope:

- baseline year: `2024`
- executable month-ahead Brent scenario engine anchored to `2026-04-06`
- disruption scenarios ranging from gradual reopening to full closure with limited response
- first operational node set: `20` nodes

Source-backed facts used in the initial scaffold:

- EIA reported that oil flow through the Strait averaged `20.0` million barrels per day in `2024`,
  about `20%` of global petroleum liquids consumption
- EIA figure data gives `20.261741721311477` million b/d total oil flow in `2024`
- EIA figure data gives `14.318613808743168` million b/d of crude and condensate and
  `5.9431279125683094` million b/d of petroleum products in `2024`
- EIA estimated `2.6` million b/d of available Saudi and UAE bypass capacity in the event of a
  Strait disruption
- EIA estimated that `84%` of crude and condensate moving through Hormuz in `2024` went to Asian
  markets, and that China, India, Japan, and South Korea together accounted for `69%` of Hormuz
  crude and condensate destination flows
- the latest EIA daily prices page available on `2026-04-06` showed a Brent close of `127.61`
  and a WTI close of `113.23` for `2026-04-02`
- the March `2026` EIA STEO said Brent would remain above `$95/b` over the next two months and
  average `$91/b` in `2Q26`
- the OPEC+ statement on `2026-04-05` announced a `206` thousand b/d production adjustment for
  `May 2026`

Folder layout:

- `flownet/`
  - FlowNet formulation and the first source-backed instance
- `scenarios/`
  - month-ahead price-scenario definitions and simulator
- `data/raw/`
  - raw downloaded source files kept as close as possible to the publisher versions
- `data/processed/`
  - deterministic CSV views used to define the node set, scenario parameters, and market anchors
- `sources/`
  - source manifest, checksums, and citation notes
- `scripts/`
  - fetch/build and month-ahead run scripts
- `results/`
  - generated month-ahead scenario reports and path tables

Rebuild the checked-in data artifacts with:

```bash
python rust/src/problems/hormuz_strait/scripts/fetch_and_build.py
```

Run the month-ahead scenario simulation with:

```bash
python rust/src/problems/hormuz_strait/scripts/run_month_ahead_simulation.py
```

The first node set is intentionally mixed:

- `7` exporter nodes from EIA origin-flow data
- `1` chokepoint node
- `1` aggregate bypass asset
- `9` destination-market nodes from EIA destination-flow data
- `1` Gulf refining and storage hub
- `1` strategic reserve and floating storage buffer

That is the minimum structure that lets us represent:

- physical flow through Hormuz
- partial rerouting around Hormuz
- regional demand allocation
- reserve release as an inventory response
