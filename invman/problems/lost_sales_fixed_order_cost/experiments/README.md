# Paper Benchmark

Paper objective for this family:

- run the fixed-order-cost lost-sales paper benchmark grid
- compare a small stable learned-policy shortlist against the benchmark heuristics
- keep the fixed-cost paper surface focused on policies that have already worked on the canonical `L=4, p=4, K=5` instance

## Reported Instances

Use the repo's lost-sales-style fixed-cost benchmark grid:

- grid name: `lost_sales_style_full_grid_mu5`
- demand families: `Poisson`, `Geometric`, `MMPP2 positive`, `MMPP2 negative`
- lead times: `4, 6, 8, 10`
- shortage costs: `4, 19`
- fixed order costs: `5, 25`
- mean demand: `5`
- holding cost: `1`

This is the current full paper instance set for fixed-order-cost lost sales in `invman`.

## Report Table Shape

The intended fixed-cost paper presentation is again an instance-grid table.

A typical table layout is:

- rows:
  - benchmark heuristics `s,S`, `s,nQ`, modified `s,S,q`
  - selected learned policies
- columns:
  - lead time `L in {4, 6, 8, 10}`
  - grouped by shortage cost `p` and fixed setup cost `K`

So one reported fixed-cost table block corresponds to:

- fixed demand family
- fixed shortage cost `p`
- fixed setup cost `K`
- varying lead time `L`

and the full paper section stacks those blocks across:

- demand families `{Poisson, Geometric, MMPP2 positive, MMPP2 negative}`
- `p in {4, 19}`
- `K in {5, 25}`

## Learned Policy Families

Report the selected comparison set:

- `linear_soft_gated_direct_quantity`
- `nn_soft_gated_direct_quantity_h8_selu`
- `linear_soft_gated_ordinal_quantity`
- `nn_soft_gated_ordinal_quantity_h8_selu`
- `soft_tree_depth1_linear_leaf`
- `soft_tree_depth2_linear_leaf`

The `nn_*_h8_selu` rows are one-hidden-layer neural variants of the dense direct and ordinal heads.
The tree rows stay as tree policies; they do not have a hidden-layer width parameter.

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
  --grid_name lost_sales_style_full_grid_mu5 \
  --run_tag fixed_cost_paper_suite_2k_scale20_seed42 \
  --seed 42 \
  --mp_num_processors 4 \
  --instance_jobs 2 \
  --training_episodes 2000 \
  --eval_horizon 1000000 \
  --eval_seeds 10 \
  --state_scale 20 \
  --only \
    linear_soft_gated_direct_quantity \
    nn_soft_gated_direct_quantity_h8_selu \
    linear_soft_gated_ordinal_quantity \
    nn_soft_gated_ordinal_quantity_h8_selu \
    soft_tree_depth1_linear_leaf \
    soft_tree_depth2_linear_leaf
```

Default report locations:

- `outputs/benchmarks/fixed_cost_paper_suite_2k_scale20_seed42/fixed_cost_full_suite.json`
- `outputs/benchmarks/fixed_cost_paper_suite_2k_scale20_seed42/fixed_cost_full_suite.md`
