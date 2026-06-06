# spare_parts_inventory — benchmark card

**One-line MDP:** state = (on-hand rotable spares, backlog, procurement pipeline, repair pipeline) for an installed base of repairable units; action = procurement order-up quantity each period; one-period cost = procurement·order + holding·post-failure-on-hand + downtime·post-failure-backlog; objective = minimize (discounted, in the reduced verifier) total cost over a finite horizon.

**Status:** split verdict. **Trainable env = `faithful_unverified`.** The *adjacent* Kranenburg analytical lateral-transshipment module = `verified_rerun` (a structurally DIFFERENT model — does NOT verify the env). The van Oers 2024 Table 1 rows = `snapshot_only_not_rerun` (debt). **Paper:** NOT covered by `learning_inventory_control_policies_es.tex` (no section; this family is benchmark-catalog only, not a paper problem).

## Problem formulation
Repo-native single-echelon **periodic-review repairable** spare-parts MDP (`env.rs`). An installed base of `installed_base` units operates; `operational_units = installed_base − backlog`. Per period, **order-after-demand** timing:

1. `realized_failures` occur among operational units (binomial with per-unit `failure_probability` in `demand.rs`).
2. Failures are met from on-hand spares first; shortfall increases backlog: `post_failure_on_hand = on_hand − min(on_hand, failures)`, `post_failure_backlog = backlog + failures − min(on_hand, failures)`.
3. Arrivals `= procurement_pipeline[0] + repair_pipeline[0]` clear backlog first, then add to on-hand.
4. The chosen `order_quantity` enters the tail of the procurement pipeline; the failed units enter the tail of the repair pipeline and return **deterministically** exactly `repair_lead_time` periods later.
5. `period_cost = procurement_cost·order_quantity + holding_cost·post_failure_on_hand + downtime_cost·post_failure_backlog`.

Long-run objective: minimize total cost over the finite horizon (undiscounted on the 17-period primary instance; discounted at 0.99 on the reduced exact-DP verifier).

## Reference instances
| instance | dimensions covered | key params | literature_verified flag |
|---|---|---|---|
| `single_echelon_repairable_operational_spares` (PRIMARY, trainable) | backorder · periodic-review · deterministic repair-return · finite horizon | horizon=17, installed_base=12, L_proc=3, L_repair=2, p_fail=0.08, holding=0.25, downtime=20.0, procurement=3.0; init on_hand=2, repair_pipe=[1,0]; benchmark S=5, lead-time-mean-cover buffer=1.0 | **false** |
| `VERIFICATION_PROBLEM_INSTANCE` (reduced exact-DP self-consistency) | backorder · periodic-review · repairable · exact-DP-tractable | horizon=4, discount=0.99, installed_base=3, L_proc=2, L_repair=2, p_fail=0.4, holding=0.5, downtime=6.0, procurement=2.0, max_order=4, S=3, buffer=1.0 | **false** |
| `kranenburg2006_table5_2` (35 rows, ADJACENT module) | continuous-review METRIC · multi-location · lateral transshipment · emergency replenishment · analytical-exact | exact R* enumeration; base case (m=0.001) Situation1 R1*=9.09 C1=91.90, Situation3 R3*=6.10 C3=63.00, ratio 1.46 | **true** |
| `van_oers2024_table1` (no_am / upstream_am / downstream_am) | 2-echelon serial · periodic-review · additive-manufacturing · recorded-table-only | sim_horizon=1000, downtime_cost_as_reported=3.75; values RECORDED as constants | **false** (frozen snapshot) |

## Baselines
- **Heuristics:** `base_stock` (order-up-to S; benchmark S=5, best-constant S=6 from grid search) and `lead_time_mean_cover` (safety buffer over expected lead-time failures; buffer=1.0). Searched by constant-S grid / fixed-buffer evaluation on held-out seeds.
- **Exact / optimal:** repo-native `finite_horizon_dp.rs` `solve_optimal_policy` — bounded backward-induction DP, tractable only on the reduced `VERIFICATION_PROBLEM_INSTANCE`. This is a **self-consistency comparator, NOT a published optimum** (matches no paper number). Separately, the Kranenburg analytical exact solver (R* enumeration) provides the exact optimum for the DIFFERENT lateral-transshipment sub-family.
- **Published comparators (CONTEXT only):**
  - Kranenburg 2006 Table 5.2 base case: Situation1 R1*=9.09 C1=91.90; Situation3 R3*=6.10 C3=63.00; ratio 1.46 — an adjacent continuous-review METRIC model, not the env protocol.
  - van Oers 2024 Table 1 no-AM: enumeration 100.0/99.57; newsvendor 117.0/99.08; echelon_separation 105.9/99.36. Downstream-AM: enumeration 71.98; echelon_separation 72.01. RECORDED ONLY, frozen, not re-run.

## Verification
- **Kranenburg module (verified_rerun, ADJACENT only):** Published Table 5.2 base case (m=0.001): Situation1 R1*=9.09 C1=91.90; Situation3 R3*=6.10 C3=63.00; ratio 1.46 (35 rows, TU/e thesis Ch.5 p.107, DOI 10.6100/IR616052). **Re-run reproduced: all 35/35 rows within table-rounding tolerance 0.02; worst abs diff 0.005; base case R1*=9.0900/C1=91.9000, R3*=6.1000/C3=63.0000, ratio=1.4587** via the loop command below. Verdict: verified_rerun. **CAUTION: this is a CONTINUOUS-REVIEW METRIC lateral-transshipment model — structurally DIFFERENT from the trainable `env.rs`; it does NOT verify the environment.**
- **Trainable env (faithful_unverified):** No paper publishes a numeric cost for this exact periodic-review repairable construction (binomial failures + deterministic fixed-lead-time repair return + finite-horizon DP). Source `SPARE_PARTS_REVIEW_REFERENCE` (Zhang, Huang & Yuan 2021, *Spare Parts Inventory Management: A Literature Review*, Sustainability 13(5):2460) is a motivational review with `reported_numbers_available = false`. Only repo-native self-consistency re-ran: **exact DP optimal = 28.39366, base_stock = 28.39366 (gap 0.0), lead_time_mean_cover = 28.91225 (gap 0.519)** — DP weakly dominates both heuristics but matches NO published number. **Debt:** no published cost is reproduced by the trainable env.
- **van Oers 2024 Table 1 (snapshot_only_not_rerun — DEBT, ledger D3):** the two-echelon serial-AM rows are frozen constants in `references.rs`; no executable two-echelon serial env re-runs them. To clear: build the two-echelon serial-AM env to re-run, or drop from the card.

## Results (learned policy)
- **AT RISK — best-of-N, NOT seed-robust.** Carried claim: learned soft-tree (depth 2, oblique, linear, T=0.10) beats best-constant base-stock S=6 by **1.34% out-of-sample (53.06 vs 53.78)** on a 4096-seed holdout (and beats S=5 by 15.77%, lead_time_mean_cover by 42.92%). Manifest `seed_reporting = best_of_n`, `at_risk = true`. **This is a single training seed (best-of-N), NOT yet seed-robust** (no mean±std over ≥5 optimizer seeds) and must not be read as a robust beat.
- **AT RISK — single-seed re-run.** Re-run reproduction: soft_tree=50.72 vs best-constant S=6=54.44 → 6.84% on a 512-seed holdout; the consolidated 4096-seed JSON reproduced 53.06 vs 53.78 → 1.34% exactly. Manifest `seed_reporting = single_seed`, `at_risk = true`. **Single-seed, NOT yet seed-robust.**
- **NOT at risk (self-consistency anchor).** Repo-native exact DP weakly dominates both heuristics on the reduced verifier: optimal 28.394 ≤ base_stock 28.394 ≤ lead_time_mean_cover 28.912. This is an internal anchor, not a published comparison.

## Reproduce
```bash
# Kranenburg Table 5.2 — 35/35 rows reproduced within tolerance (verified_rerun, adjacent module)
python -c "import invman_rust as m; rows=m.spare_parts_inventory_kranenburg_reference_instances(); n=sum(m.spare_parts_inventory_kranenburg_exact_summary(r['name'])['published_table_comparison']['all_within_tolerance'] for r in rows); print(n, '/', len(rows))"

# Reduced exact-DP self-consistency (trainable env, faithful_unverified)
python -c "import invman_rust as m; s=m.spare_parts_inventory_exact_dp_summary(); print(s['optimal_discounted_cost'], s['base_stock_gap_to_optimal'], s['lead_time_mean_cover_gap_to_optimal'])"

# Learned soft-tree vs heuristics on the 17-period primary instance (AT RISK: single training seed)
python scripts/spare_parts_inventory/benchmark_spare_parts_inventory.py --holdout_seeds 4096 --holdout_seed_start 900000
python scripts/spare_parts_inventory/benchmark_spare_parts_inventory.py --holdout_seeds 512 --holdout_seed_start 900000
python scripts/spare_parts_inventory/train_soft_tree_reference.py --seed 123 --depth 2 --temperature 0.10
```

## Pointers & caveats
- code: `src/problems/spare_parts_inventory/env.rs` (trainable MDP), `finite_horizon_dp.rs` (exact DP verifier), `references.rs` (`PRIMARY_REFERENCE_INSTANCE`, `VERIFICATION_PROBLEM_INSTANCE`, `SPARE_PARTS_REVIEW_REFERENCE`, Kranenburg + van Oers rows), `literature/kranenburg_lateral_transshipment.rs` (verified analytical module), `demand.rs`, `rollout.rs`, `tests/`, `verification/`, `practical/`, `experiments/`; scripts: `scripts/spare_parts_inventory/` (`benchmark_spare_parts_inventory.py`, `train_soft_tree_reference.py`, `validate_against_exact_dp.py`, `validate_kranenburg_lateral_transshipment.py`).
- autoresearch: no `autoresearch/program_spare_parts_inventory.md` exists for this family.
- **Caveat — verified module ≠ verified env:** the only peer-reviewed re-run (Kranenburg) is a continuous-review METRIC lateral-transshipment model, structurally DIFFERENT from the trainable periodic-review env; do NOT let it imply the env is verified.
- **Caveat — learned-policy beat is best-of-N / single-seed:** the 1.34% (and 6.84%) beats over best-constant base-stock are NOT seed-robust (no ≥5-seed mean±std); treat as at-risk per the seed-robust reporting standard.
- **Caveat — van Oers 2024 Table 1 is a frozen snapshot** (recorded constants, never re-run); standing verification debt D3.
- **Caveat — exact DP is self-consistency only:** the reduced-instance DP optimum (28.394) is a repo-internal dominance check, not a published optimum.
- Note: an existing `README.md` in this folder already documents the same honest split verdict and does NOT contradict the code/manifest.
