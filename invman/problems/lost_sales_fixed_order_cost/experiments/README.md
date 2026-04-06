# Paper Benchmark

Paper objective for this family:

- run the literature-aligned fixed-order-cost lost-sales benchmark subset
- compare a small stable learned-policy shortlist against the benchmark heuristics
- keep the fixed-cost paper surface focused on policies that have already worked on the canonical `L=4, p=4, K=5` instance

## Reported Instances

Use the repo's literature-aligned fixed-cost subset:

- grid name: `literature_subset_poisson_mu5`
- lead times: `1, 2, 3, 4`
- shortage costs: `4, 19`
- fixed order costs: `5, 25`
- demand: `Poisson(5)`
- holding cost: `1`

This is the current full paper instance set for fixed-order-cost lost sales in `invman`.

## Report Table Shape

The intended fixed-cost paper presentation is again an instance-grid table.

A typical table layout is:

- rows:
  - benchmark heuristics `s,S`, `s,nQ`, modified `s,S,q`
  - selected learned policies
- columns:
  - lead time `L`
  - grouped by shortage cost `p` and fixed setup cost `K`

So one reported fixed-cost table block corresponds to:

- fixed shortage cost `p`
- fixed setup cost `K`
- varying lead time `L`

and the full paper section stacks those blocks across:

- `p in {4, 19}`
- `K in {5, 25}`

## Learned Policy Families

Report the selected stable shortlist:

- `linear_soft_gated_direct_quantity`
- `linear_soft_gated_ordinal_quantity`
- `soft_tree_depth1_linear_leaf`
- `soft_tree_depth2_linear_leaf`

These are the retained policies after the canonical `L=4, p=4, K=5` reruns with policy-side state scaling.

## Heuristic Comparators

Report against:

- `(s,S)`
- `(s,nQ)`
- modified `(s,S,q)`

## Reported Metrics

Per instance:

- mean cost
- standard deviation across evaluation seeds
- gap to best heuristic

Aggregate:

- mean relative gap to best heuristic across instances
- count of instances where the learned policy beats the best heuristic

## Executable Benchmark

Use the existing full-suite runner with the selected shortlist:

```bash
python scripts/lost_sales_fixed_order_cost/benchmark_full_suite.py \
  --grid_name literature_subset_poisson_mu5 \
  --run_tag fixed_cost_selected_paper_suite_scale20_rust_seed42 \
  --seed 42 \
  --mp_num_processors 4 \
  --instance_jobs 2 \
  --eval_horizon 1000000 \
  --eval_seeds 10 \
  --state_scale 20 \
  --only \
    linear_soft_gated_direct_quantity \
    linear_soft_gated_ordinal_quantity \
    soft_tree_depth1_linear_leaf \
    soft_tree_depth2_linear_leaf
```

Default report locations:

- `outputs/benchmarks/fixed_cost_selected_paper_suite_scale20_rust_seed42/fixed_cost_full_suite.json`
- `outputs/benchmarks/fixed_cost_selected_paper_suite_scale20_rust_seed42/fixed_cost_full_suite.md`
