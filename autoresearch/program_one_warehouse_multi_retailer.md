# one-warehouse multi-retailer autoresearch (Kaynov 2024 OWMR)

This is the policy-search program for the `one_warehouse_multi_retailer` problem (Kaynov et al.
2024, IJPE 267, 109088): one upstream warehouse replenishing `K` identical capacitated retailers,
with a per-period downstream **allocation** of scarce warehouse stock — a divergent two-echelon
system under one of three customer regimes (`backorder`, `lost_sales`, `partial_backorder`).

The learned-benchmark phase already established a faithful env and a paired held-out comparison;
the learned depth-2 soft-tree **loses narrowly to the tuned heuristic**. This program drives the
follow-up: a single-policy autoresearch loop that searches the policy/control surface to close (and
ideally beat) that margin on the currently-losing instances.

## Benchmark (trusted)

Instance family: the Kaynov Table A.3 instances exposed by `references.rs`
(`one_warehouse_multi_retailer_list_reference_instances`), all symmetric Poisson(3) K=3 cases for
the regimes we screen. The three **currently-losing** instances (the keep/discard set) are one per
regime:

- `kaynov2024_instance_1` — `backorder`
- `kaynov2024_instance_6` — `lost_sales`
- `kaynov2024_instance_11` — `partial_backorder`

**Strongest heuristic** (the number to beat): grid-searched **echelon base-stock** levels (one
warehouse level + one shared retailer level over the demand-moment-derived bounds in
`echelon_base_stock_search_bounds`) evaluated under the **better of `{min_shortage, proportional}`**
allocation. The grid search runs on a search-seed CRN block, and the argmin `(W, R)` is re-scored on
a DISJOINT held-out CRN block; the lower-cost allocation rule is the "best heuristic".

**Published anchor** (Kaynov Table A.3, carried as negative reward → cost via `-mean_cost`):
proportional / min_shortage / PPO rows per instance. The repo heuristic reproduces these within
~1-6% (regime-dependent sign — see `literature/README.md`); the PPO row is the published learned
comparator, reported but not the keep/discard gate (the in-repo tuned heuristic is the gate).

**Env-faithfulness anchor**: `one_warehouse_multi_retailer_exact_dp_summary()` — reduced
finite-horizon exact DP confirms the optimum dominates both allocation heuristics
(`optimal 8.485 <= proportional/min_shortage 9.2225`). The env transition/cost is exact-DP-validated
(worked-transition test); only the *published-number* reproduction is "approximate".

## Intended search surface (editable)

The single-policy runner exposes these levers via CLI (everything else is held fixed by the trusted
benchmark above):

- **tree depth** `{1,2,3}` — depth-2 is the learned-benchmark default; sweep up for the harder rows.
- **tree temperature** — soft-split sharpness (lower = closer to a hard tree).
- **split type** `{oblique, axis_aligned}` — oblique = linear-combination splits; axis_aligned =
  single-feature thresholds (cheaper, often more stable on these symmetric instances).
- **leaf type** `{constant, linear}` (rollout also supports `sigmoid_linear`).
- **action design / policy_action_mode**:
  - `symmetric_echelon_targets` — one warehouse target + one shared retailer target, expanded inside
    the rollout (the natural geometry for the symmetric K=3 instances; the learned-benchmark default).
  - `direct_orders` / `vector_quantity` — a per-decision order vector (`q^w`, `q^r_k`) bounded by the
    physical caps; more expressive, more parameters.
- **allocation policy** used during training and at evaluation (`proportional` / `min_shortage` /
  `random_sequential`); the headline learned cost is scored under each of `{proportional,
  min_shortage}` and reported at the better one, paired with the heuristic.
- **CMA-ES warm-start (`x0`)**: seed the initial mean at the **best base-stock levels** found by the
  heuristic grid search (`--warm_start_at_best_base_stock`). Under `symmetric_echelon_targets` with a
  constant leaf, the leaf bias is set to the argmin `(W, R)` so generation-0 reproduces the strongest
  heuristic and CMA-ES searches *outward* from a known-good point — the single most important lever
  for beating a near-optimal base-stock baseline.

## Budgets

Two presets in the runner (`BUDGETS`):

- **screening** — cheap first pass: small CMA-ES population, few generations, a small held-out block,
  a coarse search block. Used to rank levers, not to certify a win.
- **full** — promotion budget: larger population / generations, 4096 held-out paths, matching the
  learned-benchmark protocol so a screened winner can be re-scored at decision quality.

Hard CPU cap regardless of budget: `RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2`, `mp_num_processors = 1`
(parallelism is rayon inside the population-rollout binding; no Python process pool). Two sibling
autoresearch agents run in parallel, so the ~27-core defaults elsewhere MUST stay overridden here.

## Goal (keep / discard)

For each currently-losing instance, the primary metric is the **held-out relative gap to the best
heuristic** on the same paired CRN block:

    gap% = (best_heuristic_cost - learned_cost) / best_heuristic_cost * 100

(positive = learned beats heuristic). **Keep** a policy design when it flips the sign — i.e. beats
the strongest heuristic out-of-sample on a losing instance, robustly across the held-out block (watch
the held-out stderr; a sub-stderr "win" is not a win). **Discard** designs that stay behind the
heuristic at full budget. Do not lock to one policy class — the job is to find a strong policy, not to
prove soft trees always win; base-stock is near-optimal on these symmetric Poisson(3) instances, so a
flip is a real result.

## What we know (from the learned-benchmark phase)

Full-budget paired held-out result (depth 2, popsize 32, 600 CMA-ES generations, train_seed_batch 12;
4096 held-out paths; 100-period undiscounted total cost; `symmetric_echelon_targets`):

| Instance | CB | Learned | Best Heuristic | Published PPO | Learned vs Heuristic | Winner |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `kaynov2024_instance_1` | `backorder` | 1584.45 | 1558.12 | 1637.20 | **-1.69%** | heuristic |
| `kaynov2024_instance_6` | `lost_sales` | 1370.50 | 1348.05 | 1347.34 | **-1.67%** | heuristic |
| `kaynov2024_instance_11` | `partial_backorder` | 1189.51 | 1184.46 | 971.86 | **-0.43%** | heuristic |

So the learned soft-tree is *competitive but not dominant*: it loses by 0.43%-1.69%. Priors carried
into this search:

1. The tuned echelon base-stock + allocation heuristic is near-optimal on these symmetric Poisson(3)
   instances — the margin to close is small, so the search should start from the heuristic, not from
   random init. **Warm-start CMA-ES at the best base-stock levels** is the leading lever.
2. `symmetric_echelon_targets` is the right action geometry for the symmetric instances; a constant
   leaf at the warm-start point exactly reproduces base-stock, so depth/temperature/leaf only need to
   buy *state-dependent* deviations from it.
3. `partial_backorder` (instance 11) is the closest to flipping (-0.43%) — the emergency-shipment
   option gives a learned policy more room than pure backorder/lost-sales. Prioritize it.
4. Allocation rule is held fixed inside a comparison (learned scored under the same rule the
   heuristic argmin used); both rules are reported so allocation choice is a visible lever.

Next dimensions to sweep, in rough priority: warm-start-at-best-base-stock (on/off); depth {2,3} on
the partial-backorder row; axis_aligned vs oblique with a linear leaf; temperature; then
`direct_orders` as an expressiveness ablation.

## Canonical workspace

- Program file: this file.
- Runner: `scripts/one_warehouse_multi_retailer/autoresearch_one_warehouse_multi_retailer.py`
  (reuses `scripts/one_warehouse_multi_retailer/common.py` +
  `benchmark_learned_vs_heuristic.py` helpers — `build_soft_tree_model`,
  `evaluate_soft_tree_policy`/`*_from_paths`, the heuristic grid search, the CRN path sampler).
- Ledger: `outputs/autoresearch/<run_tag>/results.tsv` (one TSV row per run: cost, best_heuristic,
  gap, gap%, plus the structure flags and instance).
- Problem home / env + bindings: `rust/src/problems/one_warehouse_multi_retailer/`.
