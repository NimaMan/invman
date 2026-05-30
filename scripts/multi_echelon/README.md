# scripts/multi_echelon

Runnable scripts for the multi-echelon (Gijsbrechts 2022 / Van Roy 1997 one-warehouse,
K-retailer with special delivery) problem implemented in
`rust/src/problems/multi_echelon/divergent_special_delivery/` and exposed to Python through
the `invman_rust` extension.

All scripts are self-contained: they read the env parameters and action grids from the
`invman_rust` reference catalog (the single source of truth) and drive training through the
flat `invman` package (`invman.config`, `invman.experiment_runner`, `invman.policy_registry`,
`invman.rollout_fitness`). There is no local `common.py`.

## Reference catalog (defined in Rust, `references.rs`)

- Paper-faithful **search targets** (`gijs_2022` dynamics, Table-3 demand mean):
  `gijsbrechts2022_setting1` (lw=2, lr=2, mu=5), `gijsbrechts2022_setting2` (lw=5, lr=3, mu=0).
- Van Roy **reproduction** instances (`van_roy_1997` dynamics, reproduce the published
  constant base-stock costs): `van_roy1997_simple_problem` (51.7), `van_roy1997_case_study1`
  (1302), `van_roy1997_case_study2` (1449).

## Scripts

- **`autoresearch_multi_echelon.py`** — CMA-ES policy-search runner for any reference instance.
  Builds env args from the catalog, trains a soft-tree policy via `run_experiment`, and writes
  a results row plus the best constant base-stock baseline (computed in Rust).
  Example: `python autoresearch_multi_echelon.py --reference gijsbrechts2022_setting2 --budget full --description "..."`

- **`train_simple_problem_policy.py`** — end-to-end design + train on the simple Van Roy
  problem. Reports three numbers in one run: the literature reproduction (published levels vs
  51.7), the grid-best constant base-stock, and a soft-tree depth sweep trained by CMA-ES.
  Example: `python train_simple_problem_policy.py --budget full`

## Policy interface

The learned policy mirrors the lost-sales interface: the env emits the **pure decision
state** (`raw_decision_state`) and the **policy** normalizes it (`StateNormalizer`
divide-by-scale, scale = max order-up-to level) before the soft tree produces the
order-up-to action. See the package README under
`rust/src/problems/multi_echelon/divergent_special_delivery/`.

## Verification (Rust)

Literature reproduction and env correctness live in the Rust crate, not here:
`invman_rust.multi_echelon_van_roy_reproduction_summary(...)` (constant base-stock vs the
published rows) and `cargo test --release divergent_special_delivery` (exact-DP, worked
transition, warehouse-order rule).
