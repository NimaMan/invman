# Baseline Registry Schema

This schema records which problem instances and baselines are safe to use for
paper claims, regression gates, and repo-native comparisons. It is intentionally
human-readable YAML, not generated Rust metadata.

Per-problem registry path:

- `src/problems/<problem>/literature/baselines.yaml`
- for nested problem variants, `src/problems/<problem>/<variant>/literature/baselines.yaml`

## Verification Status

Use `status.verification` for the entry-level status.

| Status | Meaning |
| --- | --- |
| `strict_literature_verified` | Every promoted published number in the entry is re-derived by a named repo test or command. |
| `partial` | At least one promoted published number is re-derived, but one or more carried published rows are table-only or otherwise not executable. |
| `table_only` | Published numbers are stored and sourced, but no repo test currently re-derives them. |
| `repo_native` | The baseline is generated inside the repo and is not a published literature number. |
| `not_verified` | The entry is intended as a baseline or anchor, but provenance or execution is not established enough for claims. |

Row-level `verification_status` uses the same vocabulary. This matters for mixed
entries, such as an instance where heuristic rows are executable but an optimal
row is only transcribed from a paper.

## Top-Level Shape

```yaml
schema_version: 1
problem: lost_sales_vanilla
registry_owner: src/problems/lost_sales/vanilla
source_of_truth:
  - src/problems/lost_sales/vanilla/literature/references.rs
entries:
  - id: stable_snake_case_id
    problem: lost_sales_vanilla
    instance_name: vanilla_l4_p4_poisson5
    roles:
      - primary_reference_instance
      - verification_problem_instance
    status:
      verification: partial
      comparator_type: mixed_published_optimum_and_heuristics
      paper_status: usable_with_caveat
      last_reviewed: "2026-06-05"
```

Required top-level fields:

- `schema_version`: currently `1`.
- `problem`: problem-family id used by the registry.
- `registry_owner`: problem directory that owns the registry.
- `source_of_truth`: Rust/doc files that should be checked before editing rows.
- `entries`: list of baseline/problem-instance entries.

## Entry Fields

Required fields:

- `id`: stable, unique snake-case id within the registry.
- `problem`: repeats the problem id so entries can be merged across registries.
- `instance_name`: exact repo reference instance name, grid name, or dataset name.
- `roles`: why the entry exists.
- `status`: entry-level verification and paper-readiness status.
- `source`: citation, table, URL, and source notes.
- `instance`: reference path, reference const, verification const, and parameters.
- `published_numbers`: objective, sign convention, and per-row published values.
- `repo_verification`: executable test/command, tolerance, artifact, and row coverage.
- `repo_baseline_gate`: script/report used to gate repo-native comparator claims.
- `paper_link`: paper section/table claim and caveats.

Recommended `roles` values:

- `primary_reference_instance`
- `primary_literature_validation`
- `verification_problem_instance`
- `published_optimum`
- `published_heuristic_validation`
- `repo_native_benchmark`
- `repo_native_benchmark_grid`
- `practical_benchmark_instance`
- `paper_benchmark_anchor`
- `paper_exact_slice`
- `paper_medium_slice`
- `table_only_literature_anchor`

Recommended `status.comparator_type` values:

- `published_exact_optimum`
- `published_heuristics`
- `mixed_published_optimum_and_heuristics`
- `repo_native_heuristics`
- `repo_native_learned_policy`
- `repo_native_practical_trace`
- `none`

Recommended `status.paper_status` values:

- `ready`
- `usable_with_caveat`
- `context_only`
- `not_for_claim`

## Published Numbers

`published_numbers.rows` should not contain guessed values. If the repo does not
state a precise number, set the value to `null` and explain the gap in `notes` or
`unknowns`.

```yaml
published_numbers:
  objective: steady_state_mean_cost
  sign: lower_is_better
  rows:
    - label: Myopic-2
      policy_id: myopic2
      metric: mean_cost
      value: 4.82
      params: null
      verification_status: strict_literature_verified
      notes: Re-derived by rollout test.
```

Use `source_numbers_not_promoted` when a paper reports additional numbers that
are intentionally not part of the promoted comparator claim, such as a standard
deviation carried only for provenance.

## Verification

`repo_verification` must distinguish exact executable coverage from table-only
coverage.

```yaml
repo_verification:
  test: src/problems/.../tests.rs::test_name
  command: cargo test test_name
  tolerance: absolute cost tolerance 0.01
  reproduced_rows:
    - optimal
    - s_s
  table_only_rows: []
  artifact: src/problems/.../verification/README.md
```

A registry entry is not `strict_literature_verified` just because a Rust
reference struct has `literature_verified = true`. The registry status is based
on named executable coverage for the rows promoted in `published_numbers.rows`.

## Baseline Gate

`repo_baseline_gate` records what the repo should use for current comparisons.
This can be a literature verifier, a canonical suite, or a practical trace
runner.

```yaml
repo_baseline_gate:
  policy: best published heuristic
  script: scripts/problem/run_benchmark.py
  eval: horizon=100000; seeds=3
  latest_report: docs/benchmarks/problem_refresh.md
  notes: Larger grid is repo-native, not a published table.
```

## Unknowns

Every incomplete point should be explicit:

```yaml
unknowns:
  - No current repo command re-derives the published optimal row.
  - Exact paper figure number is not independently confirmed.
```
