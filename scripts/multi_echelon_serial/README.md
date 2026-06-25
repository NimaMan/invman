# scripts/multi_echelon_serial — serial Clark-Scarf family

Scripts for the SERIAL multi-echelon (Clark-Scarf) problem family
(`src/problems/multi_echelon/serial`; srp problem_id `multi_echelon_serial`).
The comparator on this family is the EXACT echelon base-stock optimum from the
in-repo recursive-newsvendor solver (verified against stockpyl; Snyder & Shen
Example 6.1 published optimum 47.65). Because that optimum is PROVEN and the
echelon base-stock class is the optimal policy class, the honest ceiling for
any learned policy is MATCH — these scripts never claim a beat.

## Files

### `autoresearch_multi_echelon_serial.py` — single-seed training entry point
Trains one warm-started direct-level soft tree per optimizer seed:
- Instances: `snyder_shen_example_6_1` (published anchor, 3-stage Normal),
  `serial_2stage_normal`, `serial_5stage_normal`, `serial_5stage_poisson`
  (stockpyl-derived; comparator = exact solver optimum).
- Solves the exact Clark-Scarf optimum, simulates the exact-levels warm-start
  anchor on a held-out CRN eval block (`warm_start_gen0_mean_cost`),
  warm-starts CMA-ES at the exact levels, then deploys the cheapest of
  {warm-start anchor, CMA incumbent, generation best} on the same block.
- Budgets: `smoke` / `screening` / `full`. Output: per-run JSON +
  `outputs/autoresearch/<run_tag>/results.tsv`.

### `seed_robust_multi_echelon_serial.py` — seed-robust headline runner (srp)
Holds the family headline to the optimizer-seed robustness standard
(`invman/optimizer_seed_robustness_policy.py`): loops the EXISTING entry point
above over >= 5 independent optimizer seeds (canonical 9001..9005) via
`srp.run_over_seeds` and reports the standardized summary (learned/gate
seed-mean +/- sample std, savings-% mean +/- std, frac beating gate, shared
verdict). Gate = the exact Clark-Scarf policy SIMULATED on the same held-out
CRN block as the learned policy (same-protocol, paired per seed); the analytic
optimum is context only. Expected honest verdict: PARITY (match-the-optimum).

- REAL artifact: `outputs/multi_echelon_serial/seed_robust_report.json`
  (instance-suffixed for non-default instances), per-seed JSONs under
  `outputs/multi_echelon_serial/per_seed/`.
- `--smoke`: tiny budget (smoke preset + popsize 6 / 4 gens / 4 eval seeds),
  `mp_num_processors` forced to 1, writes ONLY under
  `outputs/multi_echelon_serial/smoke_seed_robust/` — never the real path.

Full-budget usage (CPU-capped per project convention):

    RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=/home/nima/code/ml/invman \
    python3 scripts/multi_echelon_serial/seed_robust_multi_echelon_serial.py \
        --instance snyder_shen_example_6_1 --budget full \
        --seeds 9001 9002 9003 9004 9005 --mp_num_processors 2

### `benchmark_serial_clark_scarf.py` — baselines vs the exact optimum
Compares simple base-stock heuristics (`lead_time_mean_cover`,
`newsvendor_per_echelon`) against the exact Clark-Scarf echelon base-stock
optimum on the verified serial instance set, via a faithful line-for-line
Python port of the serial env + recursive-newsvendor solver (the serial env
has no Python binding in the installed `invman_rust`); the port is validated
to reproduce the Rust/stockpyl optima.
