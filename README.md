# invman

`invman` is an Inventory Management Learning Benchmark: an ImageNet-style problem
suite for inventory-control research. The repository keeps a Rust-first core for
problem dynamics, reference instances, heuristics, exact or bounded solvers, and
high-throughput rollout kernels, with Python bindings and scripts for experiment
orchestration, policy construction, and reporting.

The intended reader is a coding or research agent that needs to understand the
project quickly, choose the right problem folder, and then continue in the
detailed READMEs for that family.

Public write-up: [Learning to Control Inventory Management Systems](https://nimamanaf.com/posts/learning-to-control-inventory-management-systems/).

## Where To Start

- `src/problems/README.md`: canonical map of Rust-first benchmark families,
  verification rules, and cross-problem conventions.
- `src/problems/<problem>/README.md`: the first stop before editing or running a
  family-specific workflow.
- `src/problems/<problem-or-subproblem>/instances/README.md`:
  machine-readable instance catalog notes for that problem. Multi-echelon
  variants usually own their own subproblem catalogs.
- `AGENTS.md`: repo-local setup, sanity checks, and ready experiment-suite
  commands.
- `policy_search/README.md` and `policy_search/POLICY_DESIGN_GUIDELINES/README.md`:
  learned-policy and decoder design workflow.
- `presentation/index.html`: current deck framing for learned policies,
  heuristics, action parameterizations, and reporting language.
- `docs/benchmarks/`: historical benchmark audits and manifest-era notes. Use
  `src/problems/**/instances/` and the problem READMEs for the active instance
  catalog convention.

## Repository Layout

```text
Cargo.toml, src/                  Rust crate and problem implementations
src/problems/                     canonical Rust-first benchmark families
src/problems/<problem-or-subproblem>/instances/ per-problem JSON instance catalogs
invman/                           Python policy, rollout, optimizer, and glue modules
scripts/                          validation, benchmark, training, and reporting scripts
numerical_experiments/            curated suite catalog and launcher
policy_search/                    policy-structure search programs and studies
docs/                             benchmark, literature, and Rust notes
tests/                            Python regression and catalog tests
outputs/                          generated experiment outputs, logs, models, reports
```

Problem folders normally contain local `README.md`, `literature/`,
`practical/`, `experiments/`, `verification/`, `heuristics/`, `env.rs`, and
`rollout.rs` material. Treat those READMEs as routing documents; they explain
which claims are literature-backed, repo-native, provisional, or only useful as
context.

## Problem Families

Current Rust-first families and important subfamilies include:

- `lost_sales`: vanilla lost-sales inventory with lead times, holding and
  lost-sales costs, Poisson/geometric/correlated demand support, plus
  `lost_sales/fixed_order_cost`.
- `dual_sourcing`: regular and expedited suppliers, dual-index/capped-dual-index
  style baselines, and bounded-DP benchmark checks.
- `multi_echelon`: serial systems, assembly, divergent special delivery, general
  backorder fixed cost, and production-assembly-distribution-network variants.
- `one_warehouse_multi_retailer`: OWMR / divergent allocation settings.
- `perishable_inventory`: finite shelf-life inventory with FIFO/LIFO style
  slices and practical traces.
- `joint_replenishment`: multi-item joint ordering with shared fixed cost.
- `nonstationary_lot_sizing`: time-varying demand lot-sizing instances.
- `random_yield_inventory`: stochastic yield and reduced exact verifiers.
- `procurement_removal_inventory`: procurement/removal and returnability slices.
- `spare_parts_inventory`: repairable spare-parts benchmark instances.
- `vendor_managed_inventory`: VMI and worked newsvendor/truck-dispatch anchors.
- `decentralized_inventory_control`: beer-game and decentralized supply-chain
  control settings.
- `ameliorating_inventory`: inventory whose value/quality can improve over time.
- `joint_pricing_inventory`: coupled pricing and inventory decisions.

See `src/problems/README.md` for the current verification status of each family.
Unless a problem README explicitly says otherwise, do not assume a family is
literature-verified.

## Instance Catalogs

Machine-readable benchmark instances should use this convention:

```text
src/problems/<problem-or-subproblem>/instances/
  README.md
  <instance_id>.json
```

For top-level families this is usually `src/problems/<problem>/instances/`.
For umbrella families such as `multi_echelon`, each formulation can own a
subfolder catalog such as `src/problems/multi_echelon/serial/instances/`.

Do not add new `BENCHMARK.md` files. Instance documentation belongs in
`instances/README.md` and the problem `README.md`; executable checks belong in
`verification/`, `scripts/`, or tests.

The cross-family validator is intentionally lightweight:

```bash
python scripts/instances/validate_problem_instances.py
```

Each JSON instance must have `schema_version: 1`, an `instance_id` matching the
filename, `problem_family`, `classification`, `source` or `provenance`,
`parameters`/`model`/`network`, and a `verification` object.

Classification meanings:

- `strict_literature`: parameters and benchmark numbers are reproduced directly
  from public literature.
- `companion_code`: derived from public companion code or data rather than only
  the paper text.
- `table_only`: transcribed from a public table, but with missing implementation
  details or no executable reproduction path.
- `faithful_unverified`: a faithful implementation of a published model where
  public row-level numbers are insufficient for a strict check.
- `generated`: repo-generated stress, practical, or verifier instance, not a
  literature row.

Keep provenance honest. If a number is produced by this repo, label it as
repo-generated or verification output; do not present it as a published
literature benchmark.

## Verification Expectations

Benchmark claims require targeted reproduction before they are reported.
Verification material can live in:

- `src/problems/**/verification/`
- family scripts under `scripts/**`
- Rust tests under `cargo test`
- Python tests under `tests/`

Run the smallest targeted check that exercises the claim before claiming a
benchmark number. Good starting commands are:

```bash
cargo test --manifest-path Cargo.toml -q
python -m pytest tests/test_lost_sales_reference_grid.py tests/test_fixed_order_cost_reference_grid.py -q
python -m pytest tests/test_dual_sourcing_problem.py tests/test_multi_echelon_problem.py -q
python -m pytest tests/test_numerical_experiments_catalog.py tests/test_problem_verification_files.py -q
python scripts/instances/validate_problem_instances.py
```

Useful family-specific script entry points include:

```bash
python scripts/lost_sales/validate_reference_instance.py
python scripts/lost_sales_fixed_order_cost/validate_known_optimum.py
python scripts/dual_sourcing/validate_reference_grid.py
python scripts/one_warehouse_multi_retailer/validate_reference_instance.py
python scripts/joint_replenishment/validate_against_exact_dp.py
python scripts/perishable_inventory/validate_against_papers.py
```

Some verification is slow or bounded by truncation choices. Report that scope
explicitly, especially for bounded DP, simulation tolerances, seed-robust runs,
or table-only literature sources.

## Learned Policies

The benchmark compares learned policies against classical heuristics and exact
or bounded references where available. A learned policy should not be described
as `literature_verified`; published PPO/A3C rows from papers are context rows,
while repo learned policies are repo-generated results.

The current policy-learning frame is:

- environment dynamics and raw state live in Rust problem modules;
- Python builds policy descriptors and launches CMA-ES / evaluation workflows;
- the policy owns any feature scaling, decoder, action parameterization, gates,
  caps, thresholds, residuals, or order-up-to transforms;
- heuristics provide strong coordinates and same-protocol gates;
- policy-search programs document the trusted instance, heuristic gate,
  editable surface, Rust binding, and known outcomes.

Read `policy_search/POLICY_DESIGN_GUIDELINES/README.md` before designing a new decoder
or policy surface. Read `policy_search/programs/<problem>/README.md` before
running a family-specific policy search when that program exists. The deck in
`presentation/index.html` shows the intended learned-policy versus heuristic
framing: match optima, beat same-protocol heuristics when supported, and keep
cross-protocol DRL numbers as context.

Experiment and training scripts generally live in:

- `scripts/<problem>/`
- `policy_search/programs/`
- `policy_search/studies/`
- `policy_search/agentic/`
- `numerical_experiments/catalog.py`

## Setup And Common Commands

Create a Python environment and install the package:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip install maturin
```

Build the Rust Python extension into the active environment:

```bash
python scripts/rust/build_extension.py
```

Run broad health checks:

```bash
cargo test --manifest-path Cargo.toml -q
python -m pytest tests -q
```

List curated experiment suites:

```bash
python numerical_experiments/run.py --list
python numerical_experiments/run.py --list --status ready
```

Dry-run or launch a curated suite:

```bash
python numerical_experiments/run.py --suite lost_sales_single_instance_check --dry-run
python numerical_experiments/run.py --suite fixed_cost_single_instance_check
```

Run the generic small experiment path:

```bash
python scripts/experiments/run_experiment.py --training_episodes 20 --horizon 200 --eval_horizon 2000 --eval_seeds 5
```

Outputs from ad hoc and benchmark runs are written under `outputs/`, especially
`outputs/logs/`, `outputs/models/`, `outputs/results/`, and
`outputs/benchmarks/`.

## Contributor And Agent Workflow

1. Start with the relevant folder `README.md`; do not infer conventions from
   filenames alone.
2. Check `instances/README.md` and JSON provenance before using an instance in a
   benchmark table or experiment claim.
3. Run targeted verification before reporting benchmark numbers.
4. Keep literature rows, repo-generated rows, learned-policy rows, and published
   external learned-policy rows separate.
5. Do not overwrite unrelated local work. This repo often has generated outputs
   and concurrent edits in flight.
6. Use READMEs as routing docs. Put detailed family-specific instructions in the
   family folder, not in this root README.
