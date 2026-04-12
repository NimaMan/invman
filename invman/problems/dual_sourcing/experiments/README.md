# Paper Benchmark

Paper objective for this family:

- run the six Gijsbrechts Figure 9 benchmark rows
- compare the benchmark heuristics against a small learned-policy shortlist
- keep the learned-policy surface tied to the same structural families used by the literature heuristics

## Reported Instances

Use the repo's literature-aligned dual-sourcing grid:

- grid name: `gijsbrechts2022_figure9_family`
- regular lead times: `2, 3, 4`
- expedited order costs: `105, 110`
- regular order cost: `100`
- holding cost: `5`
- shortage cost: `495`
- demand: `U{0,1,2,3,4}`

This is the current full paper instance set for dual sourcing in `invman`.

## Report Table Shape

The intended dual-sourcing paper presentation is a small instance-grid table.

A typical table layout is:

- rows:
  - benchmark heuristics `single-index`, `dual-index`, `capped dual-index`, `tailored base-surge`
  - selected learned soft-tree policies
- columns:
  - regular lead times `l_r in {2, 3, 4}`
  - grouped by expedited order cost `c_e in {105, 110}`

## Learned Policy Families

Report the structured shortlist:

- `soft_tree_single_index_targets`
- `soft_tree_dual_index_targets`
- `soft_tree_capped_dual_index_targets`
- `soft_tree_base_surge_targets`
- `soft_tree_capped_dual_index_delta_smallcap_targets`
- `soft_tree_axis_constant_capped_dual_index_delta_smallcap_targets`

These are learned state-dependent variants that mirror the main Gijs heuristic families.

Current `l_r=3` finding:

- the main failure mode was not missing state information, but poor regular-side inductive bias
- the oblique linear small-cap tree fixed `dual_l3_ce105` by escaping the `cap_r=0` collapse
- the strongest current `dual_l3_ce105`/`dual_l3_ce110` policy is the axis-aligned constant-leaf
  small-cap capped-dual-index tree, which keeps the learned controls on a tight heuristic-like grid
  and materially improves both rows

## Heuristic Comparators

Report against:

- `single_index`
- `dual_index`
- `capped_dual_index`
- `tailored_base_surge`

## Reported Metrics

Per instance:

- mean cost
- standard deviation across evaluation seeds
- gap to best heuristic
- heuristic optimality-gap reproduction against the published Figure 9 labels

Aggregate:

- mean relative gap to best heuristic across instances
- count of instances where a learned policy beats the best heuristic

## Executable Benchmark

Use the suite runner:

```bash
python scripts/dual_sourcing/benchmark_full_suite.py \
  --grid_name gijsbrechts2022_figure9_family \
  --run_tag dual_sourcing_gijs_structured_screening \
  --budget screening \
  --seed 123 \
  --mp_num_processors 4 \
  --eval_horizon 5000 \
  --eval_seeds 2
```
