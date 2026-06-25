# Experiment Scripts

This folder contains generic experiment entry points that are not owned by a
single inventory problem.

- `run_experiment.py` is the thin CLI wrapper around `invman.experiment_runner`.
  Problem-specific benchmark suites should live in `scripts/<problem>/`.

