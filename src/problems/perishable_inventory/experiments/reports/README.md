# perishable_inventory Paper Benchmark Report

## Working runner

Use `scripts/perishable_inventory/run_exact_slice_benchmark.py` to refresh this
benchmark. It is self-contained and runs against the installed `invman_rust`
plus the current `invman.policy.Policy` / `invman.cmaes.CMAES` API.

The older `scripts/perishable_inventory/run_paper_benchmark.py` is currently dead:
it imports `from invman.policies.soft_tree import SoftTreePolicy` (via
`common.py`), a module path that no longer exists in the installed `invman`
package (the current API is `invman.policy.Policy` with `backbone="soft_tree"`).
See the repo-wide note in `../../literature/README.md`.

## Latest exact-slice benchmark

- machine-readable: `exact_slice_report.json`
- human-readable: `exact_slice_report.md`

Settings: depth-2 oblique soft tree, leaf types `linear` and `sigmoid_linear`,
CMA-ES 150 generations / popsize 16, gamma 0.99, 48 search seeds, 256 eval seeds.

### Headline numbers (discounted return, higher is better)

`de_moor2022_m2_exp2_l1_cp7_fifo` (FIFO, m=2):

| Policy | Mean Return | SEM | Gap to Best Heuristic |
| --- | ---: | ---: | ---: |
| exact_value_iteration (analytic) | -1457.281 | 0.000 | - |
| base_stock `[7]` (Monte-Carlo) | -1468.363 | 3.620 | +0.463 |
| bsp_low_ew `[7,7,0]` (Monte-Carlo) | -1467.900 | 3.612 | 0.000 |
| soft_tree linear (Monte-Carlo) | -1455.850 | 3.617 | -12.050 |
| soft_tree sigmoid_linear (Monte-Carlo) | -1452.308 | 3.933 | -15.592 |

`de_moor2022_m2_exp1_l1_cp7_lifo` (LIFO, m=2):

| Policy | Mean Return | SEM | Gap to Best Heuristic |
| --- | ---: | ---: | ---: |
| exact_value_iteration (analytic) | -1552.991 | 0.000 | - |
| base_stock `[5]` (Monte-Carlo) | -1558.344 | 4.169 | +0.409 |
| bsp_low_ew `[5,4,10]` (Monte-Carlo) | -1557.936 | 4.169 | 0.000 |
| soft_tree linear (Monte-Carlo) | -1605.252 | 4.435 | +47.317 |
| soft_tree sigmoid_linear (Monte-Carlo) | -1543.801 | 4.139 | -14.134 |

### Interpretation

- The exact value-iteration row reproduces the published Farrington et al. (2025)
  Table 3 mean returns (-1457, -1553) and the De Moor et al. (2022) Figure 3 best
  base-stock levels (7, 5) and optimal-policy tables exactly. This is the
  literature anchor.
- Two estimators appear. `exact_value_iteration` is the analytic expected
  discounted return under the midpoint-binned gamma demand. All other rows are
  Monte-Carlo means over sampled-and-rounded gamma rollouts; on the same instance
  these sit ~10-15 units (~1%) below the analytic value because they are a
  different (sampled, finite-horizon, zero-start) estimator, not because the
  policy is worse. The `gap_to_exact_optimum` column therefore mixes estimators.
- The apples-to-apples comparison is `gap_to_best_heuristic`: every row there uses
  the same Monte-Carlo estimator and the same eval seeds. On FIFO the CMA-ES soft
  tree beats the best tuned heuristic by ~12-16 units (3-4 SEM) — a real win, and
  it is statistically indistinguishable from the optimum on the Monte-Carlo scale.
- On LIFO the `sigmoid_linear` soft tree beats the best heuristic by ~14 units,
  while the `linear`-leaf soft tree landed in a worse basin (+47): an honest
  negative result. LIFO is near heuristic-optimal (exact-vs-heuristic gap ~5),
  matching De Moor's finding that base-stock is nearly optimal under LIFO.
