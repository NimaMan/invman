# Dual-Sourcing Policy Search

This folder reframes dual sourcing as a six-row policy-design problem rather than a single-instance smoke test.

The benchmark family is the Gijs Figure 9 grid:

- `dual_l2_ce105`
- `dual_l2_ce110`
- `dual_l3_ce105`
- `dual_l3_ce110`
- `dual_l4_ce105`
- `dual_l4_ce110`

The search objective is simple:

- find learned policies that perform extremely well on these six literature-aligned rows
- understand which design choices consistently help
- keep the search surface open rather than assuming soft trees are the answer

## Main Questions

The canonical factor screen asks:

1. Is control geometry more important than raw model flexibility?
2. Does factorizing the regular target as `s_r = s_e + delta_r` help?
3. Does a small discrete regular-cap grid help more than a wide continuous cap?
4. On the same small-cap control family, do tighter trees outperform wider oblique trees?
5. Once the control family is right, does a dense linear or neural backbone compete with trees?

## Canonical Sweep

Run the factor screen with:

```bash
python policy_search/studies/dual_sourcing_policy_search/run_factor_screen.py
```

This writes artifacts under:

- `outputs/autoresearch/dual_sourcing_factor_screen_v1/`

Then render the markdown note with:

```bash
python policy_search/studies/dual_sourcing_policy_search/summarize_factor_screen.py
```

The summary lands in:

- `policy_search/studies/dual_sourcing_policy_search/factor_screen_results.md`

## Candidate Surface

The runner exposes a wider candidate catalog, but the completed `dual_sourcing_factor_screen_v1`
summary used this focused five-policy set:

- dense linear small-cap capped-delta
- soft-tree capped dual-index
- soft-tree capped delta
- soft-tree oblique small-cap capped-delta
- soft-tree axis-aligned constant small-cap capped-delta

This is intentionally not a final policy set. It is a controlled screen for identifying the dominant
factors before spending more budget on larger families or hybrid policies.

The next expansion candidates after `v1` are:

- dense NN small-cap capped-delta
- axis-aligned linear-leaf small-cap delta trees
- hybrid mixtures between the capped-delta and axis-constant small-cap families

## Current Direction

The factor screen plus the completed axis-linear follow-ups now point to a row-dependent policy family:

- common base: stay in the factorized capped-delta coordinates
- `l_r = 2`: axis-aligned linear leaves can beat the best heuristic
- `l_r in {3,4}`: the tighter axis-constant small-cap tree remains clearly better

Concrete follow-up evidence:

- `dual_l2_ce105`: `tree_axis_linear_smallcap_delta` reaches `-0.0621%` vs best heuristic
- `dual_l2_ce110`: `tree_axis_linear_capped_delta` reaches `-0.0831%` vs best heuristic
- `dual_l3_ce105` to `dual_l4_ce110`: axis-linear probes remain worse than `tree_axis_constant_smallcap_delta`

So the best next design is not "linear leaves everywhere." It is:

- keep the factorized capped-delta control family
- branch the geometry by row difficulty or lead time
- if a single default is needed, keep `tree_axis_constant_smallcap_delta`
- if instance-conditioned family selection is allowed, use an axis-linear branch for `l_r = 2`

See also:

- `policy_search/studies/dual_sourcing_policy_search/factor_screen_results.md`
- `outputs/autoresearch/dual_l2_ce110_axis_family_probe/`
- `outputs/autoresearch/dual_l3_axis_linear_cappeddelta_probe/`
- `outputs/autoresearch/dual_hard_axis_linear_smallcap_probe/`

## Post-migration scripts (Rust-routed)

After the Python-cleanup migration the dual-sourcing problem/policies live entirely in
`invman_rust`; the old `invman.problems.dual_sourcing.*` and `invman.policies.registry`
imports are gone. The `scripts/dual_sourcing/` tools were repointed at:

- `invman.policy_registry` (`apply_policy_name`, `make_soft_tree_policy_name`)
- `invman_rust` for the grid / reference instances / heuristic + optimal baselines
- a shared, Rust-backed glue module `scripts/dual_sourcing/dual_sourcing_benchmark_lib.py`
  (`build_reference_args`, `get_benchmark_grid`/`build_grid_instances`,
  `evaluate_default_heuristics` via the `*_search_from_demands` bindings,
  `bounded_dp_optimal` opt-in, `EXPERIMENT_SPECS`/`configure_run_args`, budgets)

Dual sourcing is soft_tree-ONLY (see `invman.rollout_fitness._dual_sourcing_kwargs`), so
the deleted dense linear/nn variants were dropped from the rosters. The experiment payload's
`evaluation.heuristics` block is empty for dual sourcing, so every script computes heuristics
itself from the Rust search bindings. Launch the full benchmark via:

```bash
python scripts/dual_sourcing/benchmark_full_suite.py \
    --run_tag dual_sourcing_paper_suite --budget full \
    --mp_num_processors 4 --instance_jobs 1 --reuse_existing
```

`run_factor_screen.py` / `summarize_factor_screen.py` in this folder still carry the old
broken imports and were out of scope for the migration fix; use the `scripts/dual_sourcing/`
tools above.

## Working Rule

Do not optimize around one favorite architecture.

If a simple dense policy on the right control family wins, keep it.
If a tiny axis-aligned tree wins, keep it.
If a hybrid or entirely new family is needed, add it.

The job of this folder is to keep that search organized and evidence-driven.
