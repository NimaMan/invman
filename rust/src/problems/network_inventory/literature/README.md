# Literature

Current literature anchors for `network_inventory`:

- Pirhooshyaran and Snyder 2021 (paper-shaped network formulation; single-node and serial rows)
- Clark and Scarf 1960 / Snyder and Shen "Fundamentals of Supply Chain Theory" Example 6.1 /
  `stockpyl.ssm_serial` (exact serial multi-echelon optimum; `SERIAL_CLARK_SCARF_REFERENCE`)

Current status:

- the `env.rs` Pirhooshyaran network model is NOT literature-verified
- single-node newsvendor rows are reproduced analytically (closed form)
- the serial benchmark optima carried here (Pirhooshyaran Tables 2-3) are the TEXTBOOK Clark-Scarf
  optima; their env-faithful, literature-verified home is the `multi_echelon/serial` family

Why:

- `SINGLE_NODE_BENCHMARK_ROWS` reproduces the paper's analytical newsvendor rows exactly (closed
  form, not via the env simulation)
- `SERIAL_BENCHMARK_ROWS` are the classical periodic-review serial multi-echelon optimum, which is
  the `multi_echelon/serial` problem. The exact solver there (`multi_echelon/serial::exact`) reproduces
  every published serial optimal cost within 0.05% relative error, and that family's env simulation
  reproduces it too. These rows are carried here only because Pirhooshyaran's Tables 2-3 report them
- the Pirhooshyaran network env in THIS family does NOT reproduce those optima (per-node production
  + pipeline holding -> longer effective lead time; see `serial_echelon_simulation`), and its own
  published serial protocol could not be recovered tightly from public sources, so this env remains
  not literature-verified

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
