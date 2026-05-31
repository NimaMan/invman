# Literature

This folder documents the public literature rows carried for `one_warehouse_multi_retailer`
and the honest current verification status.

## Canonical Reference

- **Kaynov, van Knippenberg, Menkovski, van Breemen & van Jaarsveld (2024)** —
  "Deep Reinforcement Learning for One-Warehouse Multi-Retailer inventory management",
  International Journal of Production Economics 267, 109088.
  <https://doi.org/10.1016/j.ijpe.2023.109088>
  (open-access CC BY; also at the TU/e research portal).

Repo interpretation:

- one upstream warehouse, multiple downstream retailers (divergent two-echelon)
- coupled per-period decisions: warehouse order, retailer orders, and downstream **allocation**
- allocation is part of the action structure, not a post-processing detail
- customer behavior is one of lost-sales, complete-backorder, or partial-backorder
  (the paper's three regimes; matches `CustomerBehaviorModel`)

`references.rs` is the source of truth for the carried Table A.3 instances, the published
benchmark rows, and the `literature_verified` labels.

## What the paper says (used to judge fidelity)

Verified from the abstract and search-accessible text (the full PDF is behind a Cloudflare
bot wall, so the exact per-table demand parameterization could not be byte-checked):

- The paper proposes a DRL action distribution that grows **linearly** in the number of
  retailers, and a **random rationing** allocation used during training to promote feasible
  retailer orders. The repo mirrors this with the `random_sequential` (train) /
  `proportional` (evaluate) protocol and the `AllocationPolicy::RandomSequential` rule.
- Reported DRL gains over the general-purpose benchmark policies: **~1-3% for lost sales**
  and **~12-20% for partial back-ordering**. The carried PPO rows in `references.rs` are
  consistent with this: the partial-backorder instances 11-14 carry PPO gaps of
  -12.58%, -20.21%, -21.63%, -19.72% (i.e. the 12-20% band), and the lost-sales instances
  carry single-digit gaps. This corroborates that the carried published rows are the genuine
  Kaynov Table A.3 / Table B.6 numbers, not invented.
- Benchmarks are echelon base-stock reorder policies paired with **proportional** allocation
  and **min-shortage** (minimize the maximum resulting shortfall) allocation. The repo
  implements both (`allocation.rs`).

## Carried published benchmark rows

`references.rs` carries 14 instances (`TABLE_A3_INSTANCES`). Each row stores the published
**reward** (negative; the paper reports negative cost), the standard error, and the relative
gap percent for:

- `published_proportional_benchmark` (echelon base-stock + proportional)
- `published_min_shortage_benchmark` (echelon base-stock + min-shortage)
- `published_ppo_benchmark` (the paper's PPO learner)

Cost convention: the repo simulator reports a **positive total cost**; the script layer
compares against `-published_reward`. So published reward `-1406.27` ↔ published cost `1406.27`.

## Current Verification Status

**`partial`** — faithful, exact-DP-validated env; published rows carried; published numbers
**approximately** (not bit-) reproduced.

There are two distinct claims, kept separate (as in the sibling `multi_echelon/divergent_special_delivery`):

1. **Env transition + cost are faithful and independently validated.**
   - `tests/verification.rs::worked_transition_matches_expected_accounting` traces a full
     lost-sales period by hand: warehouse_arrival from pipeline head, shipments out of
     on-hand + arrival, demand against (retailer on-hand + arrival), holding charged on
     ending on-hand, penalty on unmet/backordered units; next-state pipelines advance one
     stage and append the new order. The accounting matches the frozen reference exactly.
   - `tests/verification.rs::finite_horizon_dp_dominates_repo_heuristics` runs an exact
     finite-horizon DP on the reduced `VERIFICATION_PROBLEM_INSTANCE` and confirms
     `optimal <= proportional` and `optimal <= min_shortage`. Reproduced live against the
     installed extension via `one_warehouse_multi_retailer_exact_dp_summary()`:
     optimal `8.485`, proportional `9.2225`, min_shortage `9.2225` (both dominated). This is
     a correct-exact-solver validation of the transition/cost, independent of the paper.

2. **Published Kaynov numbers are NOT bit-reproduced.** The repo's echelon base-stock +
   allocation heuristics reproduce the *shape* and *order of magnitude* of the published
   rows but land off by roughly 1-6%, with a direction that flips by customer-behavior regime.
   Measured with `scripts/one_warehouse_multi_retailer/run_heuristic_published_benchmark.py`
   (100-period undiscounted totals, mean-filled pipeline warm start, 1000 evaluation
   trajectories, repo-grid-searched base-stock levels):

   | Instance | CB | repo prop | pub prop | gap% | repo min | pub min | gap% |
   |---|---|---:|---:|---:|---:|---:|---:|
   | `kaynov2024_instance_1`  | backorder         | 1564.53 | 1655.51 | -5.50 | 1551.09 | 1609.47 | -3.63 |
   | `kaynov2024_instance_6`  | lost_sales        | 1340.04 | 1373.91 | -2.47 | 1350.67 | 1366.51 | -1.16 |
   | `kaynov2024_instance_7`  | lost_sales        | 1385.67 | 1406.27 | -1.46 | 1394.82 | 1408.08 | -0.94 |
   | `kaynov2024_instance_8`  | lost_sales        | 1469.62 | 1508.12 | -2.55 | 1477.80 | 1516.67 | -2.56 |
   | `kaynov2024_instance_11` | partial_backorder | 1180.31 | 1111.76 | +6.17 | 1181.55 | 1109.96 | +6.45 |

   - The lost-sales rows (6, 7, 8) reproduce within ~1-2.5% — the closest match.
   - The complete-backorder row (1) is ~3.6-5.5% **below** published.
   - The partial-backorder row (11) is ~6% **above** published.
   - The direction is *not* uniform, which is the signature of a protocol / initial-condition
     difference, **not** a transition bug (the transition passes the worked example and the
     exact-DP dominance check).

   `VERIFICATION_PROBLEM_INSTANCE` itself carries `literature_verified = false`
   (`references.rs:582`): it is a repo-native exact anchor, not a published number.

### Root cause of the residual gap (why approximate, not exact)

The repo benchmark uses a **mean-filled pipeline warm start** (every on-hand and pipeline
slot seeded with rounded one-period mean demand) and **repo-defined base-stock search bounds /
grid resolution**. Kaynov's exact heuristic initial-state convention and base-stock search
grid are not published in the search-accessible text. A warm start near steady state removes
the cold-start transient that a 100-period total would otherwise carry, which plausibly
explains the systematic offset (and its regime-dependent sign, since the transient's cost
direction differs between lost-sales, backorder, and partial-backorder). This is the same
class of "unspecified-protocol residual" documented for the Van Roy reproduction in
`multi_echelon/divergent_special_delivery`.

## Demand Parameterization Note

The carried normal-demand instances use high coefficient-of-variation parameters, stored as
`RoundedNormal(mean, std)`:

- instances 3 / 5: retailer demands `N(1, 5)`, `N(5, 1)`, `Poisson(0.5)`
- instance 13: ten retailers, each `N(5, 14)`
- instance 14: `N(0,20), N(2,16), N(4,12), N(6,8), N(8,4), N(10,0)` plus four Poisson retailers

`N(1, 5)`, `N(5, 14)`, `N(0, 20)` have large mass below zero; after the env's
`round(max(0, ·))` clip the **effective** mean is well above the latent mean (e.g. effective
mean of `N(0,20)` is ~8). This matches the "high-variability" test-instance design common in
this literature, but because the full table could not be byte-checked, **whether the second
parameter is the standard deviation (repo assumption) vs a variance/other convention is the
single most likely cause of any residual demand-side mismatch** and is flagged as a remaining
verification step.

## Benchmark plan / results

- **Heuristics vs published (runnable now, no rebuild):**
  `scripts/one_warehouse_multi_retailer/run_heuristic_published_benchmark.py`. Self-contained,
  imports only `invman_rust`. Produces the table above plus the exact-DP self-consistency
  check. The asymmetric 3-retailer and 10-retailer instances have larger Cartesian grids; pass
  smaller `--search_replications` or a `--instance_names` subset for those.
- **Learned soft-tree vs heuristics (blocked):** the existing
  `scripts/one_warehouse_multi_retailer/run_paper_benchmark.py` + `common.py` import
  `invman.policies.soft_tree.SoftTreePolicy`, a module path that **no longer exists** after the
  repo's policy refactor (`grep` finds no `SoftTreePolicy` anywhere under `invman/`). So the
  soft-tree numbers in `experiments/reports/latest_report.json` cannot be regenerated against
  the current install without first repointing those scripts at the refactored policy builder.
  The Rust rollout bindings themselves (`one_warehouse_multi_retailer_soft_tree_rollout`, etc.)
  are present and working; only the Python policy wrapper import is stale.

## Related Literature on the Same / Closely-Related Problem

- **Van Roy, Bertsekas, Lee & Tsitsiklis (1997)** — NDP for retailer inventory management
  (Proc. 36th IEEE CDC). The classical divergent two-echelon analogue; see the repo's
  `multi_echelon/divergent_special_delivery`.
- **Gijsbrechts, Boute, Van Mieghem & Zhang (2022)** — "Can Deep Reinforcement Learning Improve
  Inventory Management?" (M&SOM 24(3):1349-1368). A3C DRL on divergent settings.
- **Nahmias & Smith (1994)** — two-echelon retailer system with partial lost sales
  (Management Science 40(5):582-596). Closest classical analogue to the partial-backorder regime.
- **Federgruen & Zipkin (1984)** — approximations of dynamic multilocation inventory problems
  (Management Science 30(1):69-84). Classical divergent base-stock + allocation structure;
  the "balance assumption" under which echelon base-stock is optimal.
- **de Kok et al. (2018)** — typology and literature review on stochastic multi-echelon
  inventory models (EJOR 269(3):955-983).

### Policy-design stance

We **design the policy for the problem (the MDP)**, not to reproduce Kaynov's PPO network,
action encoding, or tuning. The published proportional / min-shortage / PPO rows are
**benchmarks to beat**. Only the env (MDP transition + cost) must be faithful — and that is
the part validated against the exact DP. The carried Table A.3 instance parameters define the
benchmark suite; the learned action space (`direct_orders` / `echelon_targets` /
`symmetric_echelon_targets`) is ours to choose.
