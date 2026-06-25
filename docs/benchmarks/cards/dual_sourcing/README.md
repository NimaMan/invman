# Benchmark card — `dual_sourcing`

**Subfamily:** Gijsbrechts et al. 2022 Figure 9 small-scale family

**Difficulty:** `medium` — Two coupled sourcing actions (regular + expedite) over a pipeline raise action dimensionality; the bounded-DP optimum is only a truncated-box PROXY (unusable as a denominator at l_r=4, where capped-dual-index is the optimal proxy), so the comparator is heuristic_to_beat / bound_gap rather than a clean true optimum.

**Verification tier:** `strict` (re-runs a PEER-REVIEWED printed number)

> Status (manifest, verbatim): verified_rerun

## Reference instances

| Instance | literature_verified | Dimensions |
| --- | --- | --- |
| dual_l2_ce105 | true (experiments/mod.rs grid hardcodes for all 6; references.rs has NO per-instance flag field) | regime:dual_sourcing_backlog, leadtime:Lr2, ce:105, demand:U0-4, cr:100, h:5, b:495, primary_unit_test_instance |
| dual_l2_ce110 | true | regime:dual_sourcing_backlog, leadtime:Lr2, ce:110, demand:U0-4 |
| dual_l3_ce105 | true | regime:dual_sourcing_backlog, leadtime:Lr3, ce:105, demand:U0-4 |
| dual_l3_ce110 | true | regime:dual_sourcing_backlog, leadtime:Lr3, ce:110, demand:U0-4 |
| dual_l4_ce105 | true | regime:dual_sourcing_backlog, leadtime:Lr4, ce:105, demand:U0-4 |
| dual_l4_ce110 | true | regime:dual_sourcing_backlog, leadtime:Lr4, ce:110, demand:U0-4, primary_reference_instance |

## Baselines

**Heuristics**
- single_index
- dual_index
- capped_dual_index (strongest; used as optimal proxy)
- tailored_base_surge

**Exact solver / bound**

bounded DP: solve_bounded_average_cost_optimal_policy (relative value-iteration over truncated inventory box [-12,24]); NOT a proof-level unbounded optimum. For l_r=4 the truncated value sits ~0.2% BELOW the heuristics so it is unusable as a denominator; capped_dual_index is the optimal proxy there.

**Published rows**
- Figure 9 optimality-gap labels (% above bounded-DP optimum): capped_dual_index 0.00-0.11%, dual_index 0.11-0.49%, single_index 0.56-2.44%, tailored_base_surge 0.00-0.99%, A3C 0.51-1.85%
- dual_l2_ce105: CDI 0.00 / DI 0.11 / SI 0.56 / TBS 0.06 / A3C 0.52
- dual_l4_ce110: CDI 0.11 / DI 0.49 / SI 2.44 / TBS 0.58 / A3C 1.33
- NO published absolute-cost table (only Figure 9 gap bar labels)

## Reference results (compare your approach against these)

| seed_reporting | at_risk | seed-robust | Claim |
| --- | --- | --- | --- |
| `multi_seed_mean_std` | False | yes | Learned soft tree MATCHES capped-dual-index optimal proxy on all 6 instances (within CDI's <=0.11% band), thereby clearing published A3C (gaps 0.51-1.85%). Headline match floor, robust. |
| `single_seed` | True | no | Learned soft tree BEATS CDI on 2 of 6 rows: dual_l2_ce110 by -0.009%, dual_l4_ce110 by -0.041%, held-out re-verified on disjoint seeds. |
| `single_seed` | True | no | Factor-screen negative gaps vs best heuristic (dual_l4_ce105 -0.1052%, dual_l2_ce105 axis-linear -0.0621%) presented as learned beating heuristic. |

## How to reproduce & compare

**Expected (published) value:** Figure 9 optimality gaps for dual_l2_ce105: CDI 0.00%, DI 0.11%, SI 0.56%, TBS 0.06% (and dual_l2_ce110: 0.03/0.18/1.03/0.99%)

**Reproduced value (this audit):** dual_l2_ce105 (re-run, 10.4s): optimal_avg=216.770; CDI gap=0.0058% (pub 0.00), TBS=0.0615% (pub 0.06), DI=0.1164% (pub 0.11), SI=0.5675% (pub 0.56). dual_l2_ce110: optimal_avg=219.733; CDI=0.0301% (pub 0.03), DI=0.1784% (pub 0.18), TBS=0.9874% (pub 0.99), SI=1.0316% (pub 1.03). All within <=0.0075pp. l_r=3,4 rows NOT re-run this audit (minutes-scale, #[ignore]d) but externally validated 2026-06-04.

**Rerun method / tolerance:** python -c invman_rust.dual_sourcing_reference_benchmark_summary('dual_l2_ce105', inventory_lower=-12, inventory_upper=24, tolerance=1e-8, max_iterations=250, search_seed=123, search_horizon=6000, warm_up_periods_ratio=0.2) and same for 'dual_l2_ce110'.

**Reproduce command(s):**

```bash
python -c "import invman_rust; r=invman_rust.dual_sourcing_reference_benchmark_summary('dual_l2_ce105', inventory_lower=-12, inventory_upper=24, tolerance=1e-8, max_iterations=250, search_seed=123, search_horizon=6000, warm_up_periods_ratio=0.2); print(r['optimal']['average_cost']); [print(h['policy_name'], h['optimality_gap_pct'], h['published_optimality_gap_pct']) for h in r['heuristics']]"
python -c "import invman_rust; print(invman_rust.dual_sourcing_reference_benchmark_summary('dual_l2_ce110', inventory_lower=-12, inventory_upper=24, tolerance=1e-8, max_iterations=250, search_seed=123, search_horizon=6000, warm_up_periods_ratio=0.2))"
python /home/nima/code/ml/invman/scripts/dual_sourcing/validate_reference_grid.py
python /home/nima/code/ml/invman/scripts/dual_sourcing/validate_reference_grid.py --references dual_l2_ce105 dual_l2_ce110
cargo test -p invman_rust dual_sourcing -- --ignored
python /home/nima/code/ml/invman/policy_search/studies/dual_sourcing_policy_search/run_factor_screen.py
```

To compare your own policy: run the command(s) above to regenerate the baseline on the named instance(s), evaluate your policy under the SAME instance + eval protocol (seeds / horizon / tolerance shown above), and report mean±std over ≥5 optimizer seeds vs the strongest baseline.

_Generated from `docs/benchmarks/BENCHMARK_MANIFEST.json` via `invman.benchmarks.catalog.render_card`. Do not edit by hand._
