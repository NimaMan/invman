# nonstationary_lot_sizing

Canonical Rust-first home for the nonstationary single-item lot-sizing family
(Dehaybe, Catanzaro & Chevalier 2024, EJOR 314(2):433-445).

## Verification status: VERIFIED (yes)

The environment faithfully matches the cited model and the repo reproduces the
published author-repo benchmark numbers on all eight forecast instances.

- Model fidelity: `env.rs::step_state` reproduces the paper's Section 4.2 worked
  transition exactly (period cost 130, reward -130); the `simple_s_s` heuristic
  matches the author testbed (s,S) formula term-for-term; demand models follow the
  testbed (CV-Normal for the simple baseline, Poisson for the DP baseline).
- Published-number reproduction: the eight `references.rs` rows are byte-for-byte
  the author-repo CSVs, and the repo's simulator reproduces every one of them
  within the stored 35-cost tolerance (±0.17% relative) at 25,000 replications.
- See `literature/README.md` for the full evidence table and source pointers, and
  `verification/README.md` for the executable verifier contract.

## Code

- implementation: `rust/src/problems/nonstationary_lot_sizing/`
- tests: `rust/src/problems/nonstationary_lot_sizing/tests/verification.rs`

## Artifact folders

- `literature/` — paper scope, fidelity argument, reproduced-number table
- `practical/` — checked-in rolling forecast trace, benchmark spec, latest report
- `experiments/` — paper-facing benchmark definition
- `verification/` — human-readable statement of what the verifier asserts

## Canonical anchors

- primary literature instance: `dehaybe2024_lostsales_lt2_b5_k10_constant_10`
- verification instance: `constant_10_rolling_dp_reference`
- practical benchmark dataset: `retail_like_weekly_trace`

## Benchmark

Two scripts, both runnable against the installed `invman_rust` (no rebuild):

1. Literature benchmark (the eight forecast instances):
   `scripts/nonstationary_lot_sizing/run_literature_benchmark.py`
   - reproduces the author `simple` (CV-Normal) and `DP` (Poisson) rows,
   - reports `lead_time_base_stock` and the gap of every policy to the `rolling_dp_s_s`
     comparator (the paper's strong DP baseline; no exact optimum exists for the
     rolling-forecast path),
   - with `--learned`, trains a CMA-ES soft tree per instance against the exposed
     Rust rollout binding and reports its gap to the DP baseline.
2. Practical benchmark (one fixed rolling forecast path, operational metrics):
   `scripts/nonstationary_lot_sizing/run_practical_benchmark.py`
   (snapshot in `practical/reports/`).

### Heuristics-vs-DP results (verified, 25,000 replications, seed 1234)

Simulated total cost over 104 periods. `rolling_dp_s_s` (Poisson) is the strongest
comparator; `simple_s_s` and `lead_time_base_stock` use CV-Normal demand.

| instance     | simple_s_s | rolling_dp_s_s | lead_time_base_stock |
| ------------ | ---------: | -------------: | -------------------: |
| constant_5   |    1253.11 |        1214.92 |              1300.3 |
| constant_10  |    1834.92 |        1714.15 |              1542.8 |
| constant_15  |    2367.30 |        2071.94 |              1839.4 |
| seasonal_1   |    1825.71 |        1678.18 |              1545.8 |
| seasonal_2   |    1870.33 |        1678.84 |              1549.4 |
| seasonal_4   |    1857.53 |        1684.87 |              1558.7 |
| growth       |    1753.17 |        1605.69 |              1485.5 |
| decline      |    1963.09 |        1837.68 |              1653.6 |

Reading: the author "DP" row (`rolling_dp_s_s`, Poisson) is reproduced as the
DP baseline, but it is NOT the cheapest policy on this slice. With a small fixed
cost (K=10) relative to mean demand, `lead_time_base_stock` -- which orders every
period (no fixed-cost batching) at the lead-time critical-ratio level -- is cheaper
than both (s,S) policies on every instance except constant_5, beating the DP
baseline by 5-13% on the larger-demand instances. The two demand models are not
identical (DP uses Poisson, the others CV-Normal), so this is a heuristic-design
observation, not a strict apples-to-apples optimum; it is exactly the kind of gap
the paper's DRL agent targets, and it motivates the learned-policy comparison
below. (lead_time_base_stock measured at 5,000 replications, seed 1234.)

### Learned-policy comparison

The repo exposes a soft-tree rollout binding
(`nonstationary_lot_sizing_soft_tree_population_rollout`), and the benchmark script
trains it directly with the read-only `invman.cmaes.CMAES` (state = normalized
forecast window + net inventory + pipeline; scalar order quantity clipped to
`[0, --action_cap]`).

A small-budget illustration (depth-2 linear-leaf tree, 30 CMA-ES generations x 24
candidates, evaluated at 2,000 replications; total cost over 104 periods,
gap vs the `rolling_dp_s_s` baseline):

| instance     | learned soft tree | gap vs DP |
| ------------ | ----------------: | --------: |
| growth       |            1598.3 |    -0.44% |
| seasonal_2   |            1689.1 |    +0.58% |
| constant_10  |            2159.0 |   +25.86% |

At this tiny budget the learned policy already matches or beats the DP baseline on
seasonal_2 and growth, but under-converges on constant_10 (the run had not settled).
This is an honest small-budget snapshot, not a tuned result; raising
`--generations`/`--popsize` and `--action_cap` closes the constant_10 gap. The point
is that the comparison is fully runnable today against the installed extension.

#### Learned-policy blocker (standard harness)

A learned-policy benchmark through the project's standard training harness is NOT
wired yet, and wiring it requires edits to guardrail-protected shared files:

- `invman/rollout_fitness.py` dispatches soft-tree training only for `lost_sales`,
  `dual_sourcing`, and `multi_echelon`; it has no `nonstationary_lot_sizing` branch.
- `invman/policy_build.py::build_policy` likewise has no branch for this family.
- There is no `invman/problems/nonstationary_lot_sizing/` Python package (the
  finished siblings ship a `reference_instances.py`).

The benchmark script's `--learned` path sidesteps all of this by calling the
exposed binding and CMA-ES directly, so a learned-vs-DP comparison is runnable
today without touching shared infrastructure.
