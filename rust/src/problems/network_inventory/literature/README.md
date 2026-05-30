# Literature

Current literature anchors for `network_inventory`:

- Pirhooshyaran and Snyder 2021 (paper-shaped network formulation; single-node and serial rows)
- Clark and Scarf 1960 / Snyder and Shen "Fundamentals of Supply Chain Theory" Example 6.1 /
  `stockpyl.ssm_serial` (exact serial multi-echelon optimum; `SERIAL_CLARK_SCARF_REFERENCE`)

Current status:

- literature-verified on the exact serial Clark-Scarf optimal costs (exact-theory anchor)
- single-node newsvendor rows reproduced exactly
- env-simulation reproduction of the serial costs is the remaining (sim) task

Why:

- `SINGLE_NODE_BENCHMARK_ROWS` reproduces the paper's analytical newsvendor rows exactly
- `SERIAL_BENCHMARK_ROWS` are the classical periodic-review serial multi-echelon optimum. The exact
  Clark-Scarf decomposition solver `clark_scarf_serial_exact.rs` reproduces every published serial
  optimal cost within 0.05% relative error, cross-checked against Snyder's `stockpyl.ssm_serial`
  reference implementation (and to machine precision on discrete Poisson instances)
- the discrete env-simulation reproduction of those analytical costs could not be recovered tightly
  from public sources and remains the open (sim) task; it is separate from the exact-theory anchor

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
  implementation. The repo solver `clark_scarf_serial_exact.rs` matches it to machine precision on
  discrete Poisson instances and within 0.05% relative on every serial Normal-demand row.
  https://stockpyl.readthedocs.io

Paper-shaped network formulation (single-node rows and the carried serial settings):

- Pirhooshyaran, M., and L. V. Snyder (2021). "Simultaneous Decision Making for Stochastic
  Multi-Echelon Inventory Optimization with Deep Neural Networks as Decision Makers."
  arXiv:2006.05608. Table 1 = single-node analytical newsvendor rows; Tables 2-3 = serial
  benchmark settings, analytical order-up-to levels, and optimal average costs.
