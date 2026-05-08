# Paper Benchmark

Paper objective for this family:

- run the vanilla lost-sales paper benchmark grid
- compare a small stable learned-policy shortlist against the standard heuristic baselines
- keep the reported benchmark surface small enough to be reproducible

## Reported Instances

Use the repo's literature-aligned vanilla grid:

- grid name: `xin2020_extended_lost_sales`
- demand families: `Poisson`, `Geometric`, `MMPP2 positive`, `MMPP2 negative`
- lead times: `4, 6, 8, 10`
- shortage costs: `4, 19`
- mean demand: `5`
- holding cost: `1`

This is the current full paper instance set for vanilla lost sales in `invman`. The Poisson and
Geometric rows are literature-aligned; the MMPP2 rows are mean-preserving repo extensions used for
demand robustness.

## Report Table Shape

The intended paper presentation is an instance-grid table, not only the single canonical `L=4`,
`p=4` slice.

A typical table layout is:

- rows:
  - heuristic baselines
  - selected learned policies
- columns:
  - lead times `L in {4, 6, 8, 10}`
  - grouped by demand family and shortage cost

So one reported vanilla table block corresponds to:

- fixed shortage cost `p`
- fixed demand family
- varying lead time `L`

and the full paper section then stacks those blocks across:

- `p in {4, 19}`
- demand families `{Poisson, Geometric, MMPP2 positive, MMPP2 negative}`

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
  --run_tag lost_sales_paper_suite_2k_scale20_seed42 \
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

- `outputs/benchmarks/lost_sales_paper_suite_2k_scale20_seed42/lost_sales_full_suite.json`
- `outputs/benchmarks/lost_sales_paper_suite_2k_scale20_seed42/lost_sales_full_suite.md`
