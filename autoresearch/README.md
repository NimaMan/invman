# Lost-Sales Autoresearch

This directory adapts the `karpathy/autoresearch` idea to the inventory-management repo.

Primary references:

- repository: https://github.com/karpathy/autoresearch
- agent program: https://raw.githubusercontent.com/karpathy/autoresearch/master/program.md

The first target is the trusted vanilla lost-sales benchmark:

- reference instance: `vanilla_l4_p4_poisson5`
- learned metric: mean long-run cost after warm-up
- baseline heuristics: Myopic-1, Myopic-2, SVBS

Unlike Karpathy's single-GPU training setup, the initial inventory-management loop uses a fixed
simulation budget instead of a fixed wall-clock budget. This keeps policy quality comparisons fair
across policy classes while the Rust rollout path is still being expanded.

The idea we keep:

- one trusted benchmark
- one narrow editable surface
- one fixed experiment budget
- automatic result logging
- keep/discard decisions against a running baseline

The main adaptation:

- Karpathy fixes wall-clock time because all experiments share one training script on one GPU.
- Here we fix the rollout/training budget because policy families currently have different runtime
  backends (`python` vs `rust`) and we care first about policy quality on the benchmark.

Key files:

- `program_lost_sales.md`: agent instructions for the autonomous loop
- `../scripts/autoresearch_lost_sales.py`: fixed-budget experiment runner and logger
- `../scripts/build_rust_extension.py`: helper to rebuild the Rust extension in the shared virtualenv
