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
- The protocol audit script `scripts/multi_echelon/audit_literature_protocol.py` isolates horizon
  and warm-up sensitivity while keeping the current zero-state initialization fixed.

### Why no published absolute number is reproduced (the structural reason)

The two complex case studies expose two different warehouse-order conventions, and only one of
them is even close to the published Van Roy absolute cost:

- `van_roy_1997` mode = **post-shipment** warehouse order. Van Roy's heuristic computes the
  warehouse order AFTER store orders are deducted (full report Section 4, p.10-11). This is the
  convention that produced the published `1302` / `1449`.
- `gijs_2022` mode = **pre-shipment** warehouse order. Gijsbrechts et al. (2022) Eq. (2)
  (MSOM 24(3), p.1365-1366) raise the warehouse inventory position to its base-stock level FIRST,
  and only then "After the warehouse has ordered, each retailer places its order." This is the
  faithful policy-search target.

Re-running this family's environment + constant base-stock heuristic at the published Van Roy
levels (executing test
`verification::tests::neither_dynamics_mode_reproduces_published_absolute_cost_within_tolerance`)
gives, at horizon 20,000 (stable to long-run within ~0.5):

| setting | levels | `van_roy_1997` (post-shipment) | `gijs_2022` (pre-shipment, faithful) | published |
| --- | --- | --- | --- | --- |
| setting 1 | (330, 23) | ~1285 (-1.3%) | ~1052 (-19.2%) | 1302 |
| setting 2 | (460, 22) | ~1345 (-7.2%) | ~1139 (-21.4%) | 1449 |

Consequences for the honest status:

- The faithful `gijs_2022` MDP is a **structurally different transition** from the model that
  produced the published numbers; it lands ~19%-21% below them and is NOT expected to reproduce
  any published absolute anchor. Gijsbrechts et al. (2022) print **no absolute cost** for this
  setting at all -- only the ~8.95% / ~12.09% relative A3C savings.
- The `van_roy_1997` reproduction mode only *approaches* the published numbers and does not match
  them within the repo's 1% literature tolerance (and Van Roy's "lengthy simulation" protocol is
  under-specified about the initial-state / warm-up convention).
- Therefore **neither executable mode reproduces a paper-printed absolute number within tolerance**,
  so every divergent special-delivery row keeps `literature_verified = false`. This is not a tuning
  gap that a stable protocol would close; it is the correct, honest status.

- Current status:
  - the literature rows are present and checked
  - the strict Van Roy reproduction summary is available through the Python binding
  - the carried Gijs relative rows are frozen and auditable inside the Rust verification module
  - the repo heuristic implementation is still `literature_verified = false`, and this is correct:
    no executable mode reproduces a published absolute number within tolerance
  - `neither_dynamics_mode_reproduces_published_absolute_cost_within_tolerance` is the executing
    drift guard that pins this finding and would trip if a future change made the env silently
    "match" 1302/1449 or collapsed the pre-/post-shipment structural separation
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
