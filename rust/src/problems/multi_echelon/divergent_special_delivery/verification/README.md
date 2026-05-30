# Verification

`multi_echelon` has two verification layers.

## Literature Verification

- The executable literature reference is the original Van Roy formulation.
- The carried published absolute rows are:
  - simple problem: constant base-stock `(10, 16) -> 51.7`, best reported NDP `52.6`
  - case study 1: constant base-stock `(330, 23) -> 1302`, reported NDP rows `1179`, `1181`, `1209`
  - case study 2: constant base-stock `(460, 22) -> 1449`, best reported NDP `1318`
- The later Gijs settings are the two Van Roy case studies reused as DRL benchmarks with published
  relative A3C gains over constant base-stock:
  - setting 1: `8.95% +/- 0.13%`
  - setting 2: `12.09% +/- 0.39%`
- The current validation script checks repo reproduction of the published Van Roy constant
  base-stock rows directly. Gijs is used later for relative-gain comparison, not as the primary
  absolute heuristic-verification reference.
- The executable Rust check is
  `invman_rust.multi_echelon_van_roy_reproduction_summary(...)`. It evaluates the published
  constant base-stock levels and reports the repo-vs-published gap row by row.
- The carried Gijs relative rows are now a Rust-side verification artifact via
  `verification::gijs_relative_verification_summary` and the binding
  `invman_rust.multi_echelon_gijs_relative_verification_summary(...)`.
  This records a literature reference with the published relative metric
  `published_relative_a3c_savings_vs_constant_base_stock_pct`; it does not mark the implementation
  as verified.
- The main failure mode we found was benchmark framing, not just tuning:
  - the exploratory soft-tree benchmark is a different algorithm family from the paper's A3C row
  - its reduced-grid repo comparator is not the published Van Roy constant base-stock benchmark row
- The protocol sweep (horizon / warm-up / allocation / base-stock mode sensitivity at the published
  levels, keeping the zero-state initialization fixed) is reproducible in Rust via
  `invman_rust.multi_echelon_van_roy_reproduction_summary(...)` and
  `multi_echelon_search_stationary_policy(...)`.
- Current status:
  - the literature rows are present and checked
  - the strict Van Roy reproduction summary is available through the Python binding
  - the carried Gijs relative rows are frozen and auditable inside the Rust verification module
  - the repo heuristic implementation is still `literature_verified = false`
  - current comparisons do not yet reproduce all published Van Roy rows under one stable protocol
  - the repo still does not generate the published A3C row, so the Gijs relative row remains a
    carried literature benchmark rather than a verified repo policy result

## Repo Exact Verification

- [`VERIFICATION_PROBLEM_INSTANCE`](../references.rs) is a reduced finite-horizon problem with a repo-native exact DP solution.
- The reference stores only the problem instance.
- Verification generates the exact optimal policy and the best stationary heuristic costs at runtime in Rust.
- The Rust test suite checks:
  - exact DP against an independently implemented Bellman oracle on the reduced instance
  - exact stationary-policy search against a brute-force grid search over the action candidates
  - one worked transition under the literature-style `regular` plus `min_shortage` semantics
