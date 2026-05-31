# Vendor-managed-inventory autoresearch

This is the vendor-managed-inventory (VMI) counterpart to the lost-sales, fixed-cost,
dual-sourcing, and multi-echelon autoresearch programs. It is a **policy-design loop on a
problem where the learned soft tree currently LOSES** to the tuned base-stock heuristic on
most instances — the job here is to close (and then beat) those losing margins, not to
re-prove that soft trees win somewhere.

## Benchmark

Trusted env: the **repo-native reduced single-retailer finite-horizon VMI slice**
(`env::step_state`). This is the only VMI env exposed to Python (via the
`invman_rust.vendor_managed_inventory_*` bindings) and the only one validated — by an exact
finite-horizon DP regression (`verification/tests.rs`). It is `self-consistent-only`: the
parameters are repo-chosen with **no published anchor**, so there is **no published number to
beat** here. The continuous-time multi-retailer Sui/Gosavi/Lin (2010) truck-dispatch
simulator is faithful-but-unreproduced and is NOT a valid benchmark anchor (see the problem
README's citation/audit notes).

Instance set (from `scripts/vendor_managed_inventory/benchmark_reduced_single_retailer.py`,
`INSTANCE_SET`) — the `PRIMARY_REFERENCE_INSTANCE` plus four perturbations:

- `giannoccaro2010_style_single_retailer` (primary): periods 24, demand_mean 2.5,
  stockout 5.0, dc_capacity 10, max_shipment 5, discount 0.99
- `low_penalty` (stockout 2.0)
- `high_penalty` (stockout 9.0)
- `low_demand` (demand_mean 1.5)
- `high_demand` (demand_mean 3.5)

**Strongest heuristic = the tuned retailer/DC base-stock control.** Two grid-tuned variants:
`retailer_base_stock` (best base-stock level on a grid) and `dc_reserve_base_stock` (best
level x DC reserve on a grid). The keep/discard target is `min(retailer_base_stock,
dc_reserve_base_stock)` on each instance — on this slice the two are nearly identical.

Fair-comparison protocol (inherited from the benchmark helper): common random numbers —
heuristic grids are tuned on TRAIN seeds, the soft tree is trained on TRAIN seeds via CMA-ES,
and ALL policies are scored on a disjoint HELD-OUT CRN seed block. Lower discounted cost is
better.

Ceiling note: the exact finite-horizon DP optimal (`finite_horizon_dp::solve_optimal_policy`)
is the right ceiling but is NOT a Python binding; exposing it needs a Rust rebuild + a
`bindings.rs` edit, both out of scope. So the gap reported is learned-vs-strongest-heuristic.

## Intended search surface (the editable levers)

The runner routes entirely through the installed
`vendor_managed_inventory_soft_tree_population_rollout` binding (no Rust rebuild). The
editable levers are the soft-tree structure and how CMA is initialized:

- **tree structure**: `--tree_depth`, `--tree_temperature`, `--tree_split_type`
  (`oblique` | `axis_aligned`), `--tree_leaf_type` (`constant` | `linear`). The README
  benchmark that established the losses used `oblique` / `constant` / depth 2 / temp 0.1.
  Linear leaves and a depth/temperature sweep are the first thing to try.
- **action design**: `--action_mode` (`scalar_quantity`) and the action bounds
  (`min`/`max_shipment`) the leaf output is squashed into. The constant leaf maps
  `action = min + sigmoid(leaf) * (max - min)`; the linear leaf maps
  `action = min + softplus(raw)`.
- **CMA-ES warm-start at the base-stock control**: `--warm_start base_stock` seeds the CMA
  mean so the tree's leaf outputs start at the tuned retailer base-stock's per-period
  shipment target, instead of starting at the all-zeros mean. This is the analogue of the
  dual-sourcing "warm-start at the capped-dual-index control" lever that broke the losing
  ties there. `--sigma_init` controls how far CMA explores around that anchor.

Files:

- `scripts/vendor_managed_inventory/autoresearch_vendor_managed_inventory.py` (this runner)
- `scripts/vendor_managed_inventory/benchmark_reduced_single_retailer.py` (reused helpers)
- `rust/src/problems/vendor_managed_inventory/` (env + bindings; read-only here)
- `rust/src/core/policies/soft_tree.rs` (the soft-tree action mapping; read-only here)

## Budgets

Two budgets in the runner (`BUDGETS`):

- `screening` (reject weak ideas fast): popsize 12, iters 40, 16 train seeds, 12 held-out
  seeds, 400 soft-tree eval seeds, 400 heuristic reps.
- `full` (promote promising structures): popsize 24, iters 200, 64 train seeds, 32 held-out
  seeds, 4000 soft-tree eval seeds, 1500 heuristic reps. This matches the protocol that
  produced the README losing-margins table, so a `full` run is directly comparable to it.

HARD CPU CAP for this loop: `RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2` and 1-2 worker processes
(sibling agents run in parallel; the bindings otherwise grab ~27 cores). The runner does not
spawn its own processes; the cap is enforced by the rayon/OMP env vars on the rollout binding.

## Goal

Keep/discard rule: a structure+warm-start configuration is KEPT only if it **beats the
strongest base-stock heuristic** (negative gap%, i.e. lower held-out discounted cost) on the
instances that currently LOSE. Closing a tie to a clean win is the bar; matching the
heuristic is not enough on this convex slice.

Primary metric: relative gap to the strongest heuristic on the same instance,
`gap% = 100 * (learned_cost / best_heuristic_cost - 1)`. Negative is better (learned wins).

Discard immediately: any configuration that widens the loss vs the README baseline at equal
budget, or that only wins by exploiting eval-seed noise (check the SEM the runner logs).

## What we know (from the learned-benchmark phase)

The reduced single-retailer slice is a single-stage lost-sales system: discounted cost is
**convex in the base-stock level with a clean single optimum**, so the tuned base-stock
heuristic is essentially optimal and there is little extra structure for the tree to exploit.
That is exactly why the learned tree loses or ties. The README benchmark (full budget:
oblique / constant / depth 2 / temp 0.1 / 64 train seeds / 200 iters; 32 held-out heuristic
seeds x 1500 reps; 4000 soft-tree held-out seeds; all SEMs < 0.4) reported:

| instance      | best base-stock | soft_tree (d2, oblique, const) | gap%            |
| ------------- | --------------- | ------------------------------ | --------------- |
| primary       | 115.75          | 117.80                         | -1.76% (LOSES)  |
| low_penalty   | 103.01          | 103.18                         | -0.16% (LOSES)  |
| high_penalty  | 124.34          | 127.33                         | -2.40% (LOSES)  |
| low_demand    | 101.63 (rbs) / 101.61 (dcr) | 101.50             | +0.10% (ties/wins) |
| high_demand   | 119.54          | 120.63                         | -0.91% (LOSES)  |

So the learned policy LOSES on 4/5 instances (primary, low_penalty, high_penalty,
high_demand) and only marginally wins on low_demand. The README also notes the residual gap
is partly a **training-budget/temperature artifact**: with a smaller budget (8 train seeds,
temperature 0.25) the tree underfits and the gap widens to ~3-8%. That points the search at:
(1) temperature low enough for the tree to express a sharp base-stock threshold, (2) **linear
leaves** (a linear leaf can represent an exact order-up-to map: ship to close the gap to a
target, which is what base-stock does), and (3) **warm-starting CMA at the tuned base-stock
control** so the optimizer starts on the heuristic it must beat instead of at zero.

The default instance for this runner is `high_penalty` — the **widest current loss (-2.40%)**,
so it is the clearest keep/discard signal.
