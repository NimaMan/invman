# Literature

Current literature anchors for `production_assembly_distribution_network`:

- Pirhooshyaran and Snyder 2021 (paper-shaped network formulation; single-node and serial rows)
- Clark and Scarf 1960 / Snyder and Shen "Fundamentals of Supply Chain Theory" Example 6.1 /
  `stockpyl.ssm_serial` (exact serial multi-echelon optimum; `SERIAL_CLARK_SCARF_REFERENCE`)

Current status (re-investigated 2026-05):

- the `env.rs` per-period transition and cost are FAITHFUL to the Pirhooshyaran & Snyder (2021) MDP
  (eq. 1-13 sequence of events; eq. 3 cost), checked equation-by-equation against the paper PDF
- the env is NOT yet anchored to a published benchmark NUMBER, so literature_verified = no
- single-node newsvendor rows are reproduced analytically (closed form) and approximately by env
  simulation (L=1 case ~13 vs published 12.71; residual = integer rounding)
- the serial benchmark optima carried here (Pirhooshyaran Tables 2-3) are the TEXTBOOK Clark-Scarf
  optima; their env-faithful, literature-verified home is the `multi_echelon/serial` family

Why:

- `SINGLE_NODE_BENCHMARK_ROWS` reproduces the paper's Table 1 analytical newsvendor rows exactly
  (closed form)
- `SERIAL_BENCHMARK_ROWS` are the classical periodic-review serial multi-echelon optimum
  (`multi_echelon/serial` problem). The exact solver there reproduces every published serial optimal
  cost within 0.05% relative error. Pirhooshyaran's Table 3 reports that simulating THEIR
  finite-horizon environment with these analytical OULs yields the same cost (case 3: 47.65)
- the Pirhooshyaran network env in THIS family does NOT reproduce those optima when driven with the
  carried OUL levels, but NOT because of a longer effective lead time. CORRECTION: the paper sets
  processing time to zero, and env.rs matches it (effective serial lead time = 2+1+1 = 4, verified);
  holding on in-transit inventory is faithful to eq. 3. The actual cause is a local-vs-echelon
  POLICY/LEVEL-INTERPRETATION mismatch: the carried levels are ECHELON base-stock levels, while
  Pirhooshyaran's pairwise policy (eq. 5) targets the LOCAL raw-material position (excludes finished
  goods), so the levels are the wrong local targets. See `serial_echelon_simulation` and
  `verification/README.md` for the corrected analysis and remaining steps

Use `literature/references.rs` as the source of truth for:

- `SINGLE_NODE_BENCHMARK_ROWS`
- `SERIAL_BENCHMARK_ROWS`
- `SERIAL_CLARK_SCARF_REFERENCE`
- `PRIMARY_REFERENCE_INSTANCE`
- `VERIFICATION_PROBLEM_INSTANCE`
- carried benchmark-policy names and literature notes

Reference hygiene:

- `references.rs` stores literature rows and problem-instance definitions only
- repo-native worked-transition expected values live in verification fixtures, not in literature references

## References

Serial multi-echelon (Clark-Scarf) exact-theory anchor:

- Clark, A. J., and H. Scarf (1960). "Optimal Policies for a Multi-Echelon Inventory Problem."
  *Management Science* 6(4):475-490. Origin of the echelon base-stock optimality result and the
  recursive decomposition this solver implements.
- Federgruen, A., and P. Zipkin (1984). "Computational Issues in an Infinite-Horizon, Multiechelon
  Inventory Model." *Operations Research* 32(4):818-836. Infinite-horizon / average-cost
  stationary echelon base-stock optimality and computation.
- Chen, F., and Y.-S. Zheng (1994). "Lower Bounds for Multi-Echelon Stochastic Inventory Systems."
  *Management Science* 40(11):1426-1443. Echelon cost-function recursion used here.
- Snyder, L. V., and Z.-J. M. Shen. *Fundamentals of Supply Chain Theory* (2nd ed., Wiley 2019),
  Example 6.1 = serial benchmark case 3 (3 stages, echelon holding [2,2,3], lead times [2,1,1],
  stockout 37.12, Normal(5,1) demand; optimal cost 47.65).
- `stockpyl` (Snyder), `stockpyl.ssm_serial.optimize_base_stock_levels` — public reference
  implementation. The repo solver `multi_echelon/serial::exact` matches it to machine precision on
  discrete Poisson instances and within 0.05% relative on every serial Normal-demand row.
  https://stockpyl.readthedocs.io

Paper-shaped network formulation (single-node rows and the carried serial settings):

- Pirhooshyaran, M., and L. V. Snyder (2021). "Simultaneous Decision Making for Stochastic
  Multi-Echelon Inventory Optimization with Deep Neural Networks as Decision Makers."
  arXiv:2006.05608. Table 1 = single-node analytical newsvendor rows; Tables 2-3 = serial
  benchmark settings, analytical order-up-to levels, and optimal average costs.
