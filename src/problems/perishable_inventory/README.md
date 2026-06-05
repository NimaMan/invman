# perishable_inventory

Canonical Rust-first home for the perishable-inventory family.

Code:

- implementation: `src/problems/perishable_inventory/`
- tests: `src/problems/perishable_inventory/tests/verification.rs`

Artifact folders:

- `literature/`
  - paper scope and benchmark interpretation
- `practical/`
  - checked-in practical trace, benchmark spec, and latest report snapshot
- `experiments/`
  - paper-facing benchmark definition
- `verification/`
  - human-readable statement of what the exact verifier asserts

Verification status:

- LITERATURE-VERIFIED on the `m = 2`, lead-time-1 slice ONLY (four 121-state
  instances). The exact value-iteration MDP re-derives, in-repo at test time, the
  De Moor et al. (2022, EJOR 301(2):535-545) optimal-policy tables and best
  base-stock levels (5 LIFO, 7 FIFO) and the Farrington, Wong, Li, Utley (2025,
  Ann. Oper. Res. 349(3):1609-1638) Table 3 value-iteration returns (-1553 LIFO,
  -1457 FIFO) exactly. The repo labels the De Moor policy tables "Figure 3"; the
  exact published figure number was not independently confirmed (paywalled EJOR
  full text). The other 28 Scenario A rows are TABLE-ONLY anchors (stored, not
  re-derived). Details, citation-correctness notes, and the estimator caveat are
  in `literature/README.md`.

Current anchors:

- primary literature instance: `de_moor2022_m2_exp2_l1_cp7_fifo`
- exact verification instances:
  - `de_moor2022_m2_exp1_l1_cp7_lifo`
  - `de_moor2022_m2_exp2_l1_cp7_fifo`
- practical benchmark instance: `de_moor2022_m4_exp6_l2_cp7_fifo`

Benchmark:

- working runner: `scripts/perishable_inventory/run_exact_slice_benchmark.py`
  (exact optimum vs tuned `base_stock` / `bsp_low_ew` vs CMA-ES soft tree)
- latest report: `experiments/reports/exact_slice_report.md`
- NOTE: the older `scripts/perishable_inventory/run_paper_benchmark.py` is dead
  (imports the removed `invman.policies.soft_tree`); use the runner above.

State interface:

- `env.rs` exposes raw inventory and pipeline quantities in observation order
- any scaling used by a learned policy belongs in `rollout.rs` or the policy itself
- environment code must not silently normalize policy inputs
