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
  process pool). See `autoresearch/program_production_assembly_distribution_network.md` for the
  search direction and the headline result vs the baseline.

  Usage (smoke):
      RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 \
      python scripts/production_assembly_distribution_network/autoresearch_production_assembly_distribution_network.py \
          --description "smoke" --budget smoke
