# Literature

This folder documents the public literature rows carried for `multi_echelon`.

## Canonical Anchor

The canonical literature source for this package is the original Van Roy retailer inventory model:

- full report: <https://www.stanford.edu/~bvr/pubs/retail.pdf>
- CDC paper: <https://www.mit.edu/~jnt/Papers/C-97-bvr-retail-CDC.pdf>

The carried published benchmark rows are:

- simple problem
  - constant base-stock `(10, 16) -> 51.7`
  - best reported NDP `52.6`
- complex case study 1
  - constant base-stock `(330, 23) -> 1302`
  - reported NDP rows `1179`, `1181`, `1209`
- complex case study 2
  - constant base-stock `(460, 22) -> 1449`
  - best reported NDP row `1318`

These are the literature-verification targets for the repo heuristic implementation. They are not yet
matched by the current Rust implementation under one stable evaluation protocol.

## Protocol Audit

The Van Roy report is explicit about the benchmark rows but incomplete about the simulation protocol:

- for the heuristic exhaustive search, each plotted point is said to come from a "lengthy simulation"
- the paper does not give a single explicit warm-up ratio for those heuristic averages
- the paper does not give a single explicit initial-state convention for those heuristic averages
- the NDP learning figures are different: they report rolling averages over `10,000` steps in the
  simple problem and `5,000` steps in the two complex case studies during one long simulation run

The current Rust runtime does something more specific:

- every heuristic rollout starts from the zero state
- horizon and warm-up are explicit script parameters
- the audit script `scripts/multi_echelon/audit_literature_protocol.py` sweeps those choices at the
  published heuristic levels and writes machine-readable output to `outputs/multi_echelon/`

Current audit findings:

- the simple problem does not match the published `51.7` row under any horizon in the current
  latent-parameter interpretation; the best current gap is still about `+5.79`
- however, if the simple problem is evaluated with the induced rounded-demand moments
  `(mu, sigma) = (6.2, 6.2)` instead of the latent normal parameters `(5, 8)`, the long-run repo
  cost moves to about `50.99`, which is close to the published row
- the two complex case studies do not behave the same way, so the remaining mismatch there is not
  explained by that simple parameter reinterpretation alone

## Later Gijs Benchmark

Gijsbrechts et al. (2022) reuses the two Van Roy complex case studies and reports later DRL
comparison rows:

- setting 1 A3C savings vs constant base-stock: `8.95% +/- 0.13%`
- setting 2 A3C savings vs constant base-stock: `12.09% +/- 0.39%`

Those are carried as published comparison rows. They are useful for later policy benchmarking, but
they are not the primary absolute heuristic-verification anchor because Van Roy already provides the
stronger absolute constant-base-stock and NDP benchmark numbers.

## Other Benchmark Sources

Other papers that reuse the Van Roy family but do not currently supply stronger heuristic-verification
anchors for this repo include:

- Cheng et al. (2023), Winter Simulation Conference
  - reuses the two 10-retailer Van Roy / Gijs case-study settings
  - reports relative improvements: NDP `10%`, A3C `9%` and `12%`, RBF-DQN `12%`
  - does not provide new absolute constant-base-stock benchmark rows
- Stochastic Optimal Control with Neural Networks and Application to a Retailer Inventory Problem
  (CDC-ECC 2005)
  - reuses the first 10-retailer Van Roy case-study parameters
  - reports learned-controller averages `1176` and `860`
  - states that its learning experiments use a single `5 x 10^5`-step simulation path from random
    initial states
  - is still not a heuristic-verification anchor because it reports controller performance, not a
    new absolute constant-base-stock benchmark row

## Repo Algorithm Status

- `constant_base_stock`
  - target verification rows: the published Van Roy absolute benchmark rows above
  - current status: `literature_verified = false`
- repo exact verifier
  - `literature_verified = false`
  - it is a reduced tractable verifier used to validate the Rust implementation
- published NDP and A3C rows
  - carried as published rows only
  - not tagged as verified repo algorithms
