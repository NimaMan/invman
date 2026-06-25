# scripts/production_assembly_distribution_network

Python drivers for the `production_assembly_distribution_network` problem (Pirhooshyaran &
Snyder 2021, arXiv:2006.05608 — a finite-horizon stochastic multi-echelon inventory MDP on a
directed acyclic supply network). All scripts drive the installed `invman_rust` bindings; none
re-implement the env in Python.

HONEST STATUS: this env is faithful to the paper's MDP but is NOT literature-verified — there
is no published optimum for THIS network env, and the serial textbook optimum 47.65 is
structurally unreachable here (echelon-vs-local level mismatch; see the env README). Results
here are RESEARCH comparisons against the env's OWN best heuristic, not literature reproductions.

## Files

- `reproduce_pirhooshyaran_serial_case3.py`
  Diagnostic: simulates the pairwise base-stock policy on the serial case3 instance with the
  carried analytical Clark-Scarf OUL levels and shows the env does NOT reach the paper's 47.65
  (the gap is a local-vs-echelon level-interpretation mismatch, not a dynamics bug). Establishes
  the case3 instance mapping (relations = edges first, then external suppliers).

- `autoresearch_production_assembly_distribution_network.py`
  Single-policy autoresearch runner. Trains ONE soft-tree CMA-ES policy on the case3 instance
  via `production_assembly_distribution_network_soft_tree_population_rollout`, evaluates its
  held-out paired-CRN mean per-period cost, and compares it to the env's OWN best pairwise
  base-stock (grid-searched per-relation OUL levels, re-scored on a disjoint held-out block).
  Appends a TSV ledger row and dumps a JSON results artifact under
  `outputs/autoresearch/production_assembly_distribution_network_autoresearch/`.

  Action design: `vector_quantity` direct order per supply relation (action_dim = 3 for case3);
  the LINEAR leaf reads per-relation raw inventory + in-transit features so it can express
  inventory-position (base-stock-like) feedback; warm-started at the steady-state flow rate.

  Budgets: `smoke` (validate only), `screening`, `full`. CPU cap is HARD — launch with
  `RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2` (parallelism is rayon inside the binding; no Python
  process pool). See `policy_search/programs/production_assembly_distribution_network/README.md` for the
  search direction and the headline result vs the baseline.

  Usage (smoke):
      RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 \
      python scripts/production_assembly_distribution_network/autoresearch_production_assembly_distribution_network.py \
          --description "smoke" --budget smoke

- `autoresearch_pure_assembly_network.py`
  Same single-policy autoresearch loop on the PURE ASSEMBLY SCN (1) topology (Pirhooshyaran
  Fig 6 left / Supplement Table 3 instance 1; 7 nodes, ACTION_DIM = 10, demand N(13,1.2)).
  Same gate (env-own best pairwise base-stock, echelon-tied grid) and protocol as case 3.

- `autoresearch_mixed_distribution_assembly_network.py`
  Same loop on the MIXED distribution-and-assembly SCN (Pirhooshyaran Fig 1 / Table 5;
  6 nodes, ACTION_DIM = 8, two customer nodes). Owns the instance constants, CRN blocks,
  budgets and gate search that the seed-robust mixed runner imports verbatim.

- `seed_robust_mixed_distribution_assembly_network.py`
  Seed-ROBUST (>= 5 optimizer seeds) learned-vs-gate runner for the MIXED topology:
  gate-flow warm start, honest deployment floor (argmin over {trained, anchor, gate}),
  gentle sigma. MIGRATED (2026-06-12) onto the central
  `invman/optimizer_seed_robustness_policy.py` (srp) aggregator: the artifact carries the
  standardized keys (`n_optimizer_seeds`, `learned/gate_seed_mean/std`,
  `savings_pct_seed_mean/std`, `frac_seeds_beating_gate`,
  `verdict_vs_same_protocol_gate`) plus `per_seed`; sample (n-1) std (numerically the same
  convention the script already used); default seeds = canonical 9001..9005 (the historical
  run used `--seeds 11 22 33 44 55`). The legacy `ROBUST_GATE_MATCH_ONLY` label is now srp
  `PARITY` + `gate_pinned_all_seeds: true`.
  Real artifact: `outputs/production_assembly_distribution_network/seed_robust_report_mixed.json`
  (budget-suffixed when not `full`); `--smoke` writes ONLY under
  `outputs/production_assembly_distribution_network/smoke_seed_robust/` with a tiny budget
  and 1 CPU worker.

- `seed_robust_serial_and_pure_assembly_networks.py`  (NEW 2026-06-12, paper review issue M2)
  The pending-audit infrastructure: 5-seed robustness audit for the SERIAL (case 3) and
  PURE-ASSEMBLY topologies, whose paper rows were single-/two-seed. Reuses each topology's
  EXISTING autoresearch module verbatim (`train_soft_tree` + gate grid search; nothing
  duplicated), computes the deterministic gate once, trains the canonical >= 5 optimizer
  seeds via `srp.run_over_seeds`, and writes the standardized seed-robust report to
  `outputs/production_assembly_distribution_network/seed_robust_report_<topology>.json`
  (`--smoke` -> `smoke_seed_robust/`, never the real path). No honest floor here — the
  audit deploys the trained xbest exactly like the existing single-seed protocol.
  Usage (full audit):
      RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 \
      python scripts/production_assembly_distribution_network/seed_robust_serial_and_pure_assembly_networks.py \
          --topology serial_case3 --budget full --mp_num_processors 2
      (then `--topology pure_assembly`)

## Seed-robustness standard

All seed-robust runners in this folder aggregate through
`invman/optimizer_seed_robustness_policy.py` (problem_id
`production_assembly_distribution_network`, seeds-mode, >= 5 optimizer seeds, canonical
list 9001..9005, sample (n-1) std, shared verdict rule). Savings sign convention:
positive = learned cheaper than the gate. The mixed topology's committed agentic result
(291.136, residual gate-backbone head) lives in
`policy_search/agentic/RESULTS_padn_mixed/README.md`; the serial and pure-assembly 5-seed audits
are produced by the new runner above.
