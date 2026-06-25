# `scripts/benchmark_baselines/` — executable baseline reports

End-to-end experiment scripts that USE the executable baseline layer
(`invman/benchmarks/runners/`) to produce a per-family comparison table: the
published literature baselines next to the same numbers re-run on the live env,
so a consumer can see the reference and the reproduction side by side and drop
their own result into the comparison.

These are the worked examples for the three families the build-out targeted —
`lost_sales` (+ its fixed-order-cost version), `dual_sourcing` (the Gijsbrechts
Figure-9 instances), and `multi_echelon` (the Van Roy / Gijs divergent settings).

## Files (functionality)

| File | What it does |
|---|---|
| `benchmark_baseline_report.py` | Shared harness: `collect_rows` (load instances + published baselines, optionally re-run on the live env), `render_markdown` / `write_outputs` (published vs recomputed table + JSON sidecar), `evaluate_zero_policy` (proves the evaluate seam runs), and the common `--simulate / --full / --instances / --evaluate-zeros / --out` CLI. Trains nothing. |
| `check_registry.py` | Validates `src/problems/**/literature/baselines.yaml` coverage, schema fields, citation keys, and local path references. |
| `run_family_baselines.py` | **Generic dispatcher — works for ALL 14 catalog families.** First arg is the problem name: `run_family_baselines.py <problem> [--simulate ...]`. |
| `run_lost_sales_baselines.py` | `lost_sales` report — 33 vanilla Zipkin cells + the Bijvank fixed-order-cost instance. |
| `run_dual_sourcing_gijs_baselines.py` | `dual_sourcing` report — the 6 Gijs Figure-9 instances (published gaps + recomputed absolute heuristic costs). |
| `run_multi_echelon_baselines.py` | `multi_echelon` report — the 5 divergent Van Roy / Gijs instances (published + recomputed best constant base-stock). |

## Usage

```bash
# Read-only: published baselines for every instance of a family.
python scripts/benchmark_baselines/run_lost_sales_baselines.py

# Re-run the shipped baselines on the live env (fast smoke protocol).
python scripts/benchmark_baselines/run_dual_sourcing_gijs_baselines.py --simulate

# Literature-faithful protocol + write the md/json report.
python scripts/benchmark_baselines/run_multi_echelon_baselines.py \
    --simulate --full --out outputs/benchmark_baselines

# Restrict to specific instances and probe the evaluate seam.
python scripts/benchmark_baselines/run_lost_sales_baselines.py \
    --instances lit_poisson_p4_l4 bijvank2015_table1_l2_p14_k5 --simulate --evaluate-zeros
```

Set `RAYON_NUM_THREADS=2` (and keep `--full` for deliberate runs) to respect the
repo's CPU-budget convention.

## How to compare YOUR approach

```python
from invman.benchmarks import catalog

inst = catalog.get("lost_sales").load_instance("lit_poisson_p4_l4")
n = inst.policy_param_count(depth=2, leaf_type="linear")  # size your weight vector
my_cost = inst.evaluate(my_trained_params)                # same seam as CMA-ES
print(inst.compare(my_cost))   # {'reference': 'optimal', 'gap_pct': ..., 'beats': ...}
```

Training a soft-tree to actually beat a baseline is the job of the per-family
CMA-ES scripts under `scripts/<family>/`; these reports establish the reference
numbers and the comparison harness, not a trained result.
