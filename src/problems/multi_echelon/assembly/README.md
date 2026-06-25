# multi_echelon / assembly — textbook assembly system (Rosling 1989)

The `assembly` *version* of the multi-echelon problem: several components are procured from
outside suppliers and assembled into one finished product that faces customer demand. Sibling of
`serial` and `production_assembly_distribution_network` under `multi_echelon/`.

## Problem

- Components `1..M`, each procured from an ample external supplier with a shared lead time `L_c`.
- An assembly/finished stage consumes one of each component per finished unit (lead time `L_a`),
  faces i.i.d. customer demand, installation holding cost, and a backorder penalty.
- Optimal policy: echelon base-stock (the assembly system is equivalent to a serial system).

## Verification — Rosling (1989) equivalence

Rosling (1989) proved an assembly system is equivalent to a serial system. For the equal
component-lead-time case, a balanced echelon base-stock policy stocks every component identically,
so the components collapse into a single "kit" stage and the system is exactly the 2-stage serial
system **kit → finished**. `rosling.rs` performs that reduction, and the optimal echelon base-stock
levels + optimal cost come from the literature-verified `multi_echelon::serial::exact` solver.

`verification.rs` checks that the assembly **`env.rs` simulation** under those echelon base-stock
levels reproduces the exact serial optimum (within Monte-Carlo error), for:

- 2 identical components, `L_c=1`, finished `L_a=1`;
- 3 identical components, `L_c=2` (component/upstream lead time 2 is fully supported), `L_a=1`;
- heterogeneous component holding costs (kit holding = their sum), `L_c=2`, `L_a=1`.

This is the assembly literature anchor: Rosling's equivalence + the serial Clark–Scarf anchor.

## Verification status — VERIFIED-BY-EQUIVALENCE, NOT a published assembly number

**`literature_verified = false` for every carried assembly instance** (encoded in `references.rs`
and guarded by `references::tests::no_assembly_instance_is_literature_verified`). Per the repo rule
in `docs/rust/README.md`, a family is literature-verified only when an in-crate test re-runs the
env/solver and reproduces a number PRINTED IN A PAPER. This family has **no directly reproducible
published assembly number**, for two structural reasons:

1. **Rosling (1989) is a structural result, not a worked benchmark.** It proves an assembly system
   is equivalent to a serial system (with lead-time reordering in the general case) and
   characterizes the optimal policy as a balanced echelon base-stock policy under "long-run
   balance." It does **not** tabulate an assembly optimal cost or base-stock vector that this
   equal-lead-time, 2-stage-reducible env can reproduce. (Re-checked 2026-06-04 against the
   RePEc/IDEAS abstract and secondary characterizations — e.g. Chen & Muharremoglu, "Completing
   Rosling's Characterization" — no paper-printed assembly cost/base-stock table is available.)
2. **The only published number in the chain is a 3-stage serial system the reduction cannot reach.**
   Snyder & Shen Example 6.1 optimal cost **47.65** is the one genuinely paper-printed anchor, and
   it is re-derived in `multi_echelon/serial`. But it is a **3-stage** serial system, while the
   Rosling reduction of an equal-lead-time assembly system yields a **2-stage** serial system
   (kit → finished). A 2-stage assembly path cannot produce the 3-stage 47.65.

**What IS verified (the honest basis), strictly stronger than "self-consistent only":**

- The equivalence is **structurally anchored by Rosling (1989)**: the
  equal-lead-time reduction in `rosling.rs` is exactly the collapse to a 2-stage `kit → finished`
  serial system.
- The serial system the assembly reduces to is the same Clark & Scarf model whose published anchor
  (Snyder & Shen 47.65) and `stockpyl` reference optima **are** verified in `multi_echelon/serial`.
- The assembly **`env.rs` simulation** reproduces (within Monte-Carlo error) the exact serial
  optimum that the Rosling reduction + the literature-verified serial solver produce
  (`verification.rs`, finished lead time 1).
- A cross-family **drift guard** (`verification::tests::rosling_reduction_matches_serial_reference_
  instance_not_a_published_assembly_number`) pins the reduction mechanism: a single-component
  assembly instance (kit holding 1, finished holding 3, `L_c=L_a=1`, `p=10`, Poisson(5)) reduces
  EXACTLY to the serial 2-stage Poisson reference instance (echelon `[2,1]`, `S*=[7,13]`,
  `C*=16.797779`) — which is itself **reference-implementation-verified against `stockpyl`, NOT a
  paper-printed number**, so reproducing it through the reduction is a consistency check, not
  literature verification.

The assembly *instance numbers themselves* (22.759 / 52.536 / 27.530) are **solver-derived**, not
published. Net: structurally anchored by the Rosling equivalence + reproduction of the verified serial
solver's optima by the env — but **not** a published assembly anchor, and every carried assembly
instance remains `literature_verified=false`.

### Earlier note (retained for the metadata + independent-reproduction record)

Read this precisely — what is and is not anchored:

- The two **citations** (Rosling 1989; Clark & Scarf 1960) are correct in every metadata field
  (independently confirmed against RePEc/IDEAS and the INFORMS/ACM DOIs — see References).
- The structural claim is **literature-verified**: Rosling (1989) proves the assembly system is
  equivalent to a serial system, and the equal-lead-time reduction in `rosling.rs` is exactly that
  collapse to a 2-stage `kit → finished` serial system.
- The env is **verified by reproduction against a literature-verified solver** (the Clark–Scarf
  exact serial optimizer), NOT against a number published in Rosling (1989) or Clark & Scarf (1960).
  Those two papers do **not** tabulate the three instance costs below (22.759 / 52.536 / 27.530);
  those costs are produced by the repo's own exact serial solver and then reproduced by the env
  simulation. The only number that traces to a **published table** is the serial-family anchor
  Snyder & Shen Example 6.1 = 47.65 (re-derived in `multi_echelon/serial`, not an assembly instance).
  The 3-component instance is *constructed* to share Example 6.1's two downstream stages, but its
  cost 52.536 is solver-derived, not a published value.
- Net: this is **literature-verified at the structural/equivalence level + self-consistent
  reproduction of the (literature-verified) serial solver's optima** — strictly stronger than
  "self-consistent only", but the assembly *instance numbers themselves are not published anchors*.

The three `verification.rs` instances were independently reproduced from scratch — a pure-Python
reimplementation of both the Clark–Scarf exact solver (`serial/exact.rs`) and the assembly env
(`env.rs`), run side-by-side, NOT importing the Rust extension (assembly has no Python binding).
Independent results:

| instance | serial-equivalent (kit→finished) | exact optimum | env-sim cost | rel. error |
|---|---|---|---|---|
| 2-comp, `L_c=1`, `h_fin=3`, `p=10`, Poisson(5)        | local `[2,3]`, lead `[1,1]` | 22.759 | 22.80  | 0.18% |
| 3-comp, `L_c=2`, `h_fin=7`, `p=37.12`, Poisson(5)     | local `[3,7]`, lead `[2,1]` | 52.536 | 52.49  | 0.08% |
| heterogeneous `[0.5,1.5]`, `L_c=2`, `h_fin=4`, `p=20`, Poisson(4) | local `[2,4]`, lead `[2,1]` | 27.530 | 27.66  | 0.46% |

The Clark–Scarf anchor itself reproduces the published numbers independently: Snyder & Shen
Example 6.1 → 47.665 (vs 47.65); stockpyl Poisson optima 4.2211 / 16.7983 / 72.0467 with
`S* = [8] / [7,13] / [9,15,26]`. So the assembly verification chain holds independently of
the Rust unit tests (which this agent could not run: `cargo test` is out of scope and there
is no assembly Python binding). Reproduce with
`scripts/assembly/verify_assembly_rosling_independent.py`.

## Policy benchmark (in-scope: finished lead time 1)

For this problem the OPTIMAL policy is known analytically (echelon base-stock at the
Rosling/Clark–Scarf levels), so the benchmark measures how far natural heuristics fall short
of that optimum. `scripts/assembly/benchmark_assembly_policies.py` runs, on the three anchor
instances plus two extra in-scope instances, the optimal echelon base-stock policy vs two
base-stock heuristics. Gap vs the analytic optimum (200k-period MC; lower is better):

| instance | OPTIMAL (echelon base-stock) | myopic-newsvendor | mean-cover (no safety stock) |
|---|---|---|---|
| V1 2-comp Poisson(5)            | +0.01% | +53.0% | +12.2% |
| V2 3-comp `L_c=2` Poisson(5)    | −0.06% | +63.3% | +4.2%  |
| V3 heterogeneous Poisson(4)     | +0.07% | +59.2% | +1.7%  |
| E1 4-comp Poisson(8)            | +0.05% | +82.9% | +19.9% |
| E2 2-comp `L_c=3` Poisson(3)    | +0.21% | +36.9% | +6.0%  |

The optimal echelon base-stock recovers the analytic optimum to within MC noise on every
instance; the decentralized myopic-newsvendor heuristic (which double-counts safety stock by
treating stages independently) is 37–83% worse, confirming the optimal policy is the
meaningful target. A LEARNED soft-tree benchmark is **not runnable today**: assembly is not
registered in `multi_echelon/bindings.rs`, so the installed `invman_rust` exposes no
`assembly_*` rollout (see "Training / binding status" below).

## Scope and known limitation

- **Equal component lead time** (the clean Rosling reduction). Distinct component lead times need
  Rosling's lead-time reordering and are out of scope here.
- **Verified for finished (demand-facing) lead time 1.** The shared serial/assembly env has a known
  open discrepancy when the demand-facing stage has lead time ≥ 2 (the multi-stage simulation
  under-counts cost vs the exact solver; single-stage is correct at every lead time, and
  component/upstream lead times ≥ 2 are fine). Must be resolved before training on
  finished-lead-time ≥ 2 instances. See the `env.rs` module docs.

  Independently re-measured (2026-05-31, same 2-comp instance, varying only `L_a`):
  `L_a=1` → 0.28% error (correct); `L_a=2` → **27.9% under-count**; `L_a=3` → **41.4%
  under-count**. By contrast, varying only the component lead time with `L_a=1` stays at
  ~0.29% for `L_c∈{1,2,3}` — confirming component/upstream lead times ≥ 2 are fine.

  Root cause (mechanism): cost is `finished_holding * max(finished_on_hand,0)` on PHYSICAL
  on-hand only, with the in-transit finished pipeline uncharged. The Clark–Scarf optimum is
  defined on ECHELON inventory. For `L_a=1` the physical on-hand at cost-assessment time
  coincides with the echelon quantity the optimal cost is charged on, so the conventions
  agree; for `L_a≥2` the finished units already assembled and in transit carry real value
  the cost function ignores, so the simulated cost under-counts. The fix is NOT a one-line
  "charge the full finished pipeline" — that over-counts (tested: `L_a=2` then over-counts
  to +14.3%, `L_a=3` to +21.1%). A correct fix needs the echelon-holding accounting
  re-derived for `L_a≥2` (charge the echelon holding increment on in-transit echelon
  inventory weighted by remaining lead time), validated against the exact solver. Deferred
  as a next step; the verified scope (`L_a=1`) is unaffected and is what the anchor instances
  and the benchmark use.

## Training / binding status

The assembly module is **not exposed to Python**: it is not registered in
`src/problems/multi_echelon/bindings.rs`, and `env.rs`/`echelon_base_stock.rs` carry no
`#[pyfunction]` rollout entry point. The installed `invman_rust` therefore exposes no
`assembly_*` symbol (verified: `dir(invman_rust)` has none). Consequences:

- The repo's learned soft-tree / population rollout cannot be trained or benchmarked on this
  env without a Rust rebuild plus new bindings (out of scope for the verification pass here).
- The "training-ready" claim is true at the Rust API level (`consume`/`replenish`/
  `raw_state_vector` exist) but the Python wiring is missing. See the repo report's blockers
  for the exact bindings/registration diff that would expose an `assembly_soft_tree_*` rollout
  in the pattern of the sibling `multi_echelon_*` / `one_warehouse_multi_retailer_*` families.

## Package layout

- `env.rs` — clean assembly environment (training-ready Rust API; `consume`/`replenish`
  two-phase; no Python binding yet).
- `rosling.rs` — Rosling reduction of the assembly instance to its equivalent serial instance.
- `echelon_base_stock.rs` — optimal echelon base-stock policy + Monte-Carlo evaluator.
- `references.rs` — literature citations (Rosling 1989, Clark & Scarf 1960, the Snyder & Shen
  serial anchor) and the honest per-instance `literature_verified = false` flags, with a guard test
  (`no_assembly_instance_is_literature_verified`) that fails if any flag is flipped without a real
  paper-printed assembly anchor.
- `verification.rs` — env simulation reproduces the Rosling serial optimum (Rust unit tests,
  `L_a=1`), plus the cross-family drift guard
  (`rosling_reduction_matches_serial_reference_instance_not_a_published_assembly_number`).
  Independently re-confirmed by `scripts/assembly/verify_assembly_rosling_independent.py`.
- `scripts/assembly/` (repo-level, outside this dir) — `verify_assembly_rosling_independent.py`
  (independent reproduction of verification.rs) and `benchmark_assembly_policies.py`
  (optimal-vs-heuristic benchmark).

## References

- Rosling, K. (1989). "Optimal Inventory Policies for Assembly Systems Under Random Demands."
  *Operations Research* 37(4):565–579. DOI 10.1287/opre.37.4.565.
  (Verified: RePEc/IDEAS and the INFORMS DOI; author "Kaj Rosling", venue/volume/issue/pages/year
  all confirmed.)
- Clark, A. J., and H. Scarf (1960). "Optimal Policies for a Multi-Echelon Inventory Problem."
  *Management Science* 6(4):475–490. DOI 10.1287/mnsc.6.4.475 (the serial equivalent and its solver).
  (Verified: INFORMS/ACM DOI and RePEc/IDEAS; venue/volume/issue/pages/year confirmed.)
- Serial Clark–Scarf anchor numbers used downstream (Snyder & Shen *Fundamentals of Supply Chain
  Theory*, 2nd ed., Wiley 2019, ISBN 9781119024842, Example 6.1; and `stockpyl.ssm_serial`) live in
  `multi_echelon/serial/README.md`; this assembly module re-uses the serial solver rather than
  citing those numbers directly.
