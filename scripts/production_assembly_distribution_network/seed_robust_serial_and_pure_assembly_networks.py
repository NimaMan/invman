"""
Seed-ROBUST learned-vs-gate AUDIT runner for the SERIAL (case 3) and PURE-ASSEMBLY
(assembly SCN 1) topologies of production_assembly_distribution_network
(Pirhooshyaran & Snyder 2021, arXiv:2006.05608) -- the paper review issue M2 fix.

OBJECTIVE
---------
The paper's serial case-3 and pure-assembly PADN rows are single-/two-seed results
(seed=123 best-of runs from the autoresearch scripts), which violates the central
optimizer-seed robustness standard (invman/optimizer_seed_robustness_policy.py, "srp":
every seeds-mode headline needs >= 5 independent CMA-ES optimizer seeds, reported as
seed-mean +/- SAMPLE (n-1) std vs the same-protocol gate). This runner provides the
pending-audit infrastructure: it re-runs each topology's EXISTING single-seed training
entry point over the canonical >= 5 optimizer seeds and emits the standardized
seed-robust report. The MIXED topology has its own dedicated runner
(seed_robust_mixed_distribution_assembly_network.py, already 5-seed + srp-standardized);
this sibling intentionally covers only the two topologies still pending audit.

ALGORITHM (full description)
----------------------------
1. Resolve --topology to its EXISTING autoresearch module (reused verbatim via import;
   no env, policy, or training code is duplicated or modified):
       serial_case3  -> autoresearch_production_assembly_distribution_network.py
                        (3-node serial case3, ACTION_DIM=3, demand N(5,1), T=10)
       pure_assembly -> autoresearch_pure_assembly_network.py
                        (7-node assembly SCN 1, ACTION_DIM=10, demand N(13,1.2), T=10)
2. Build the module's disjoint CRN demand-path blocks (search + held-out; fixed
   SEARCH_SEED/HOLDOUT_SEED samplers -- identical protocol to the single-seed runs, so
   costs are directly comparable to the existing paper rows).
3. Gate (computed ONCE, outside the seed loop: the grid search is deterministic given
   the CRN blocks, so it is identical for every optimizer seed): the env's OWN best
   pairwise base-stock via module.search_best_pairwise_base_stock, grid-searched on the
   search block and re-scored held-out. This is a RESEARCH comparison, NOT a literature
   optimum (the env is faithful but not literature-verified; see the module docstrings).
4. For each optimizer seed: call module.train_soft_tree(ns, budget, holdout_paths) with
   the topology's OWN single-seed protocol defaults (depth-2 oblique soft tree, linear
   leaves, vector_quantity action, flat flow warm-start at the demand mean, sigma 0.8)
   and the per-seed CMA-ES seed. Deployment = the trained xbest scored on the held-out
   block (NO honest floor, matching the existing single-seed protocol: a seed CAN land
   above the gate -- that is exactly what the audit measures). Per-seed record =
   {seed, gate_cost, best_learned_cost, savings_pct_vs_gate, gen0/best_train costs,
   train_seconds}; the trained flat params are deliberately NOT stored (lean artifact).
5. Aggregate with srp.run_over_seeds -> the standardized summary keys
   (n_optimizer_seeds, learned_seed_mean/std, gate_seed_mean/std,
   savings_pct_seed_mean/std, frac_seeds_beating_gate, verdict_vs_same_protocol_gate;
   shared verdict rule ROBUST_BEAT_VS_GATE / BEAT_WITHIN_STD / PARITY /
   ROBUST_LOSS_VS_GATE; SAMPLE (n-1) std; >=5-seed enforcement is srp's, fail-closed).
   Savings sign convention: positive = learned cheaper than the gate.

ARTIFACTS
---------
REAL run (no --smoke, --budget full):
    outputs/production_assembly_distribution_network/seed_robust_report_<topology>.json
REAL run at a non-full budget gets a budget-suffixed name (screening can never clobber
the audited full artifact). --smoke ALWAYS writes under
    outputs/production_assembly_distribution_network/smoke_seed_robust/
and never the real path.

CPU CAP / USAGE
---------------
CPU is capped before NumPy/Rust import via --mp_num_processors (default 2 rayon/omp
threads; --smoke forces 1). Seeds run serially in one process (population rollouts are
already rayon-parallel inside the binding).
USAGE (full audit, one topology per invocation):
  RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  python scripts/production_assembly_distribution_network/seed_robust_serial_and_pure_assembly_networks.py \
      --topology serial_case3 --budget full --seeds 9001 9002 9003 9004 9005 --mp_num_processors 2
USAGE (smoke, tiny budget, separate artifact):
  python scripts/production_assembly_distribution_network/seed_robust_serial_and_pure_assembly_networks.py \
      --topology serial_case3 --smoke --mp_num_processors 1
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from invman.cpu_limits import configure_process_cpu_limits, configure_process_cpu_limits_from_argv

configure_process_cpu_limits_from_argv(sys.argv[1:], default=2)

from invman import optimizer_seed_robustness_policy as srp  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

PROBLEM_ID = "production_assembly_distribution_network"

TOPOLOGY_MODULES = {
    "serial_case3": "autoresearch_production_assembly_distribution_network",
    "pure_assembly": "autoresearch_pure_assembly_network",
}
TOPOLOGY_REFERENCES = {
    "serial_case3": "pirhooshyaran2021_serial_case3",
    "pure_assembly": "pirhooshyaran2021_assembly_scn1_instance1_supp_table3",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="5-seed robustness audit for the PADN serial_case3 / pure_assembly topologies."
    )
    p.add_argument("--topology", required=True, choices=sorted(TOPOLOGY_MODULES))
    p.add_argument("--budget", choices=["smoke", "screening", "full"], default="full")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny end-to-end validation: forces --budget smoke, caps CPU at 1 worker, "
                        "and writes ONLY under outputs/.../smoke_seed_robust/ (never the real artifact).")
    p.add_argument("--seeds", type=int, nargs="+",
                   default=list(srp.seeds_for(PROBLEM_ID)),
                   help="Optimizer seeds (default = canonical srp list 9001..9005).")
    p.add_argument("--mp_num_processors", type=int, default=2,
                   help="Rayon/BLAS worker cap (read pre-import by invman.cpu_limits).")
    # Protocol knobs: defaults mirror the topology's single-seed autoresearch runner.
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--temperature", type=float, default=None,
                   help="Soft-tree split temperature (default = module TEMPERATURE_DEFAULT).")
    p.add_argument("--split_type", choices=["oblique", "axis_aligned"], default="oblique")
    p.add_argument("--leaf_type", choices=["constant", "linear", "sigmoid_linear"], default="linear")
    p.add_argument("--warm_start_flow", type=float, default=None,
                   help="Flat flow warm-start level (default = module DEMAND_MEAN, the "
                        "single-seed protocol's anchor).")
    p.add_argument("--sigma_init", type=float, default=0.8)
    p.add_argument("--description", default="seed-robust PADN topology audit (review issue M2)")
    return p.parse_args()


def _artifact_path(parsed) -> Path:
    out_dir = PACKAGE_ROOT / "outputs" / "production_assembly_distribution_network"
    if parsed.smoke:
        out_dir = out_dir / "smoke_seed_robust"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if (parsed.smoke or parsed.budget == "full") else f"_{parsed.budget}"
    return out_dir / f"seed_robust_report_{parsed.topology}{suffix}.json"


def main():
    parsed = parse_args()
    if parsed.smoke:
        parsed.budget = "smoke"
        configure_process_cpu_limits(1)  # before the first rollout initializes rayon

    module = importlib.import_module(TOPOLOGY_MODULES[parsed.topology])
    budget = module.BUDGETS[parsed.budget]
    temperature = parsed.temperature if parsed.temperature is not None else module.TEMPERATURE_DEFAULT
    flow = parsed.warm_start_flow if parsed.warm_start_flow is not None else module.DEMAND_MEAN

    # CRN blocks + gate: deterministic given the budget, shared by every optimizer seed.
    search_paths = module.make_paths(budget["search_paths"], module.SEARCH_SEED)
    holdout_paths = module.make_paths(budget["holdout_paths"], module.HOLDOUT_SEED)
    t_gate = time.time()
    heuristic = module.search_best_pairwise_base_stock(search_paths, holdout_paths, budget["grid"])
    gate_cost = float(heuristic["holdout_mean_cost"])
    print(f"[gate] {parsed.topology} best pairwise base-stock held-out {gate_cost:.3f} "
          f"+/- {heuristic['holdout_stderr']:.3f}  OUL {heuristic['oul_levels']}  "
          f"({time.time() - t_gate:.1f}s)")

    def train_one_seed(seed: int) -> dict:
        ns = argparse.Namespace(
            depth=parsed.depth, temperature=temperature, split_type=parsed.split_type,
            leaf_type=parsed.leaf_type, warm_start_flow=flow,
            sigma_init=parsed.sigma_init, seed=seed,
        )
        learned = module.train_soft_tree(ns, budget, holdout_paths)
        cost = float(learned["holdout_mean_cost"])
        rec = {
            "seed": int(seed),
            "gate_cost": gate_cost,
            "best_learned_cost": cost,
            "savings_pct_vs_gate": 100.0 * (gate_cost - cost) / gate_cost,
            "gap_pct_vs_gate": (cost / gate_cost - 1.0) * 100.0,
            "learned_holdout_stderr": float(learned["holdout_stderr"]),
            "gen0_holdout_mean": float(learned["gen0_holdout_mean_cost"]),
            "best_train_cost": float(learned["best_train_cost"]),
            "train_seconds": float(learned["train_seconds"]),
        }
        print(f"[seed {seed}] gate {gate_cost:.3f}  learned {cost:.3f}  "
              f"savings {rec['savings_pct_vs_gate']:+.2f}%  ({rec['train_seconds']:.1f}s)")
        return rec

    # Centralized loop + aggregation + >=5-seed enforcement (single source of truth).
    result = srp.run_over_seeds(PROBLEM_ID, train_one_seed, seeds=parsed.seeds)

    payload = {
        "topology": parsed.topology,
        "reference": TOPOLOGY_REFERENCES[parsed.topology],
        "literature_verified": False,
        "baseline_kind": "env_own_best_pairwise_base_stock (RESEARCH comparison, NOT a literature optimum)",
        "policy_architecture": (
            f"soft_tree_d{parsed.depth}_{parsed.split_type}_{parsed.leaf_type}"
            f"_temp{temperature}_vector_quantity_warmstart_flow{flow}_sigma{parsed.sigma_init}"
        ),
        "budget": parsed.budget,
        "smoke": bool(parsed.smoke),
        "description": parsed.description,
        "commit": module._git_short_commit(PACKAGE_ROOT),
        "gate": heuristic,
        # seeds, per_seed, and the standardized srp summary keys (n_optimizer_seeds,
        # learned/gate seed mean+/-std, savings_pct_seed_mean/std,
        # frac_seeds_beating_gate, verdict_vs_same_protocol_gate).
        **result,
    }
    json_path = _artifact_path(parsed)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=" * 78)
    print(f"{parsed.topology}  budget={parsed.budget}  arch={payload['policy_architecture']}")
    print(f"GATE seed-mean    {payload['gate_seed_mean']:.3f} +/- {payload['gate_seed_std']:.3f}")
    print(f"LEARNED seed-mean {payload['learned_seed_mean']:.3f} +/- {payload['learned_seed_std']:.3f}")
    print(f"SAVINGS vs gate   {payload['savings_pct_seed_mean']:+.2f}% +/- "
          f"{payload['savings_pct_seed_std']:.2f}%  (beating gate {payload['frac_seeds_beating_gate']})")
    print(f"VERDICT (srp, vs same-protocol gate): {payload['verdict_vs_same_protocol_gate']}")
    print(f"WROTE_JSON: {json_path}")


if __name__ == "__main__":
    main()
