# spare_parts_inventory

Rust-first problem home for `spare_parts_inventory`.

Repo interpretation:

- repairable spare-parts control
- installed-base failures create demand
- procurement and repair pipelines jointly determine service and downtime
- this folder may also catalog adjacent spare-parts literature benchmarks when a paper publishes
  reusable numeric benchmark rows, even if the repo-native executable primary instance is a
  different spare-parts subfamily

Current benchmark split (three blocks, with distinct verification status each):

1. Executable literature-verified EXACT benchmark: Kranenburg (2006) Chapter 5
   lateral-transshipment comparison. The Rust analytical solver
   (`literature/kranenburg_lateral_transshipment.rs`) reproduces all 35 rows of
   the published Table 5.2 — situation-1 (separate stock points) vs situation-3
   (lateral transshipment), optimal randomized stock `R*`, cost `C(R*)`, and the
   cost ratio. This is the load-bearing literature verification: numbers are
   recomputed from Kranenburg's Chapter 5 model, not stored.
2. Repo-native EXACT finite-horizon DP on a reduced verification instance for the
   single-echelon repairable spare-parts MDP. This is NOT literature-verified; it
   is an internal self-consistency anchor (the exact DP must weakly dominate both
   carried heuristics).
3. Literature catalog (table-only): van Oers et al. (2024) Table 1 two-echelon
   periodic-review serial spare-parts benchmark with optional additive
   manufacturing. The repo stores the published rows exactly; there is no repo
   solver that re-derives them yet.

Verification status (honest, verified 2026-05-31 against the installed
`invman_rust`):

- Kranenburg (2006) Table 5.2: LITERATURE-VERIFIED. All 35 rows reproduced within
  worst absolute deviation 0.005 (situation-1 `R*`), 0.005 (situation-3 `R*`/cost),
  0.005 (ratio), all well under the 0.02 table-rounding tolerance.
- van Oers et al. (2024) Table 1: recorded-as-published only (table-only catalog),
  not reproduced by a repo solver.
- Single-echelon repairable MDP (primary + verification instances): repo-native,
  NOT literature-verified; flagged in code as
  `repo_exact_solver_not_verified_against_literature`.

Benchmark results (block 3, learned policy on the 17-period primary instance,
discount 0.99, evaluated on a held-out block of 4096 fresh seeds 900000..904096):

| Policy | Params | Mean discounted cost | vs soft-tree |
| --- | --- | ---: | ---: |
| `soft_tree` (depth 2, oblique, linear, T=0.10) | trained CMA-ES artifact | 53.06 | — |
| best constant `base_stock` | S=6 | 53.78 | soft-tree 1.34% better |
| benchmark `base_stock` | S=5 | 62.99 | soft-tree 15.77% better |
| `lead_time_mean_cover` | buffer=1.0 | 92.95 | soft-tree 42.92% better |

The soft-tree weights are loaded from the saved CMA-ES artifact
`outputs/spare_parts_inventory/retry_d2_t010_e300_s123.json` and re-rolled out of
sample, so the comparison is held out. Reproduce all three blocks with
`scripts/spare_parts_inventory/benchmark_spare_parts_inventory.py`.

Code lives under `rust/src/problems/spare_parts_inventory/`.

Verification and benchmark anchors live in:

- `references.rs`
- `tests/verification.rs`
- `literature/`
- `practical/`
- `experiments/`
- `verification/`

State interface rule:

- `env.rs` exposes raw state quantities only
- any normalization, scaling, or derived inventory-position features for learned policies must live outside the environment layer
- `rollout.rs` is the right place to convert raw state into the feature vector expected by a specific policy family
