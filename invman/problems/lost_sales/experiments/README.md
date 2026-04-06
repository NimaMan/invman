# Paper Benchmark

Paper objective for this family:

- run the literature-aligned vanilla lost-sales benchmark grid
- compare a small stable learned-policy shortlist against the standard heuristic baselines
- keep the reported benchmark surface small enough to be reproducible

## Reported Instances

Use the repo's literature-aligned vanilla grid:

- grid name: `xin2020_extended_lost_sales`
- demand families: `Poisson`, `Geometric`
- lead times: `2, 4, 6, 8, 10`
- shortage costs: `4, 19`
- mean demand: `5`
- holding cost: `1`

This is the current full paper instance set for vanilla lost sales in `invman`.

## Learned Policy Families

Report the selected stable shortlist:

- `linear_soft_gated_direct_quantity`
- `linear_soft_gated_ordinal_quantity`
- `soft_tree_depth1_linear_leaf`
- `soft_tree_depth2_linear_leaf`

These are the currently retained policies after the focused `L=4, p=4` reruns.

## Heuristic Comparators

Report against:

- `myopic1`
- `myopic2`
- `svbs`
- literature `capped_base_stock` reference when available

## Reported Metrics

Per instance:

- mean cost
- standard deviation across evaluation seeds
- gap to best heuristic
- gap to literature capped-base-stock reference when available

Aggregate:

- mean relative gap to best heuristic across instances
- count of instances where the learned policy beats the best heuristic

## Executable Benchmark

Use the existing full-suite runner with the selected shortlist:

```bash
python scripts/lost_sales/benchmark_full_suite.py \
  --grid_name xin2020_extended_lost_sales \
  --run_tag lost_sales_selected_paper_suite_scale20_rust_seed42 \
  --seed 42 \
  --mp_num_processors 4 \
  --eval_horizon 1000000 \
  --eval_seeds 10 \
  --only \
    linear_soft_gated_direct_quantity \
    linear_soft_gated_ordinal_quantity \
    soft_tree_depth1_linear_leaf \
    soft_tree_depth2_linear_leaf
```

Default report locations:

- `outputs/benchmarks/lost_sales_selected_paper_suite_scale20_rust_seed42/lost_sales_full_suite.json`
- `outputs/benchmarks/lost_sales_selected_paper_suite_scale20_rust_seed42/lost_sales_full_suite.md`
