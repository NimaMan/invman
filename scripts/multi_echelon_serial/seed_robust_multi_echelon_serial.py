"""
Seed-ROBUST learned-vs-optimum runner for the SERIAL Clark-Scarf multi-echelon
family (multi_echelon_serial; Snyder & Shen Example 6.1 and the stockpyl-derived
serial instances).

OBJECTIVE
---------
Hold the multi_echelon_serial headline to the project-wide optimizer-seed
robustness standard (invman/optimizer_seed_robustness_policy.py, "srp"):
mean +/- sample std over >= 5 independent CMA-ES optimizer seeds, never a single
seed. On this family the comparator is the EXACT Clark-Scarf echelon base-stock
optimum (recursive newsvendor; a PROVEN optimum and the optimal policy CLASS),
so the honest ceiling is MATCH/PARITY -- this runner never manufactures a
"beat"; it certifies that matching the optimum is robust to optimizer
randomness.

GATE SEMANTICS (same-protocol, paired per seed)
-----------------------------------------------
gate_cost(seed) = the exact Clark-Scarf echelon base-stock policy SIMULATED on
the SAME held-out CRN eval block the learned policy is scored on (the existing
runner's `warm_start_gen0_mean_cost`). This is the same-protocol gate: both
sides share the eval seeds derived from the optimizer seed, so the comparison
is paired and free of analytic-vs-simulation bias. The analytic optimum
(published 47.65 for Example 6.1 / the exact-solver value otherwise) is carried
as CONTEXT per seed (gap_vs_published_pct, match_pct), never as the paired
gate. NOTE: the existing entry point always includes the warm-start anchor in
its deployment candidate set, so best_learned_cost <= gate_cost by construction
and per-seed savings_pct_vs_gate >= 0; any positive savings is simulation/
selection noise around a true optimum, and the honest headline remains MATCH.

ALGORITHM (full description)
----------------------------
1. Resolve the optimizer-seed list from srp (canonical 9001..9005; >=5 distinct
   seeds enforced for this seeds-mode problem).
2. For each optimizer seed s, invoke the EXISTING single-seed entry point
   scripts/multi_echelon_serial/autoresearch_multi_echelon_serial.py as a
   subprocess (zero logic duplication; no env/policy/Rust code touched) with
   --seed s and --output_json <per-seed path>. That script, per its own design:
   a. solves the exact Clark-Scarf optimum (cost + echelon levels) with the
      in-repo recursive-newsvendor solver (Normal or Poisson variant);
   b. evaluates the exact-levels warm-start anchor on the held-out CRN eval
      block keyed off seed s -> gate_cost(s);
   c. warm-starts CMA-ES at the exact levels, trains, and deploys the cheapest
      of {warm_start_anchor, cma_incumbent, gen-best} on the SAME held-out
      block -> best_learned_cost(s).
3. Parse each per-seed JSON into the srp per-seed record {seed, gate_cost,
   best_learned_cost, savings_pct_vs_gate, + context: published/exact optimum,
   gap_vs_published_pct, match_pct, the single-seed match verdict, timings}.
4. srp.run_over_seeds aggregates: learned/gate seed-mean +/- SAMPLE std,
   savings-% seed-mean +/- std, frac_seeds_beating_gate, and the shared verdict
   (ROBUST_BEAT_VS_GATE / BEAT_WITHIN_STD / PARITY / ROBUST_LOSS_VS_GATE).
   Expected honest verdict here: PARITY (match-the-optimum).
5. Write the report JSON. REAL artifact ->
   outputs/multi_echelon_serial/seed_robust_report.json (instance-suffixed for
   non-default instances). --smoke NEVER touches that path: it forces the tiny
   "smoke" budget preset further shrunk (popsize 6, 4 generations, 4 eval
   seeds), forces mp_num_processors 1, and writes ONLY under
   outputs/multi_echelon_serial/smoke_seed_robust/.

CPU CAP / USAGE
---------------
  RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  python scripts/multi_echelon_serial/seed_robust_multi_echelon_serial.py \
      --instance snyder_shen_example_6_1 --budget full \
      --seeds 9001 9002 9003 9004 9005 --mp_num_processors 2

  smoke test (tiny budget, separate artifact path, 1 CPU):
  python scripts/multi_echelon_serial/seed_robust_multi_echelon_serial.py --smoke
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from invman.cpu_limits import configure_process_cpu_limits_from_argv, cpu_limited_environ  # noqa: E402

configure_process_cpu_limits_from_argv(sys.argv[1:], default=2)

from invman import optimizer_seed_robustness_policy as _srp  # noqa: E402

PROBLEM_ID = "multi_echelon_serial"
DEFAULT_INSTANCE = "snyder_shen_example_6_1"

# Load the existing single-seed entry point as a module ONLY to share its
# INSTANCES / BUDGETS registries (single source of truth for choices); the
# per-seed training itself runs through its __main__ entry point verbatim.
_RUNNER = PACKAGE_ROOT / "scripts" / "multi_echelon_serial" / "autoresearch_multi_echelon_serial.py"
_spec = importlib.util.spec_from_file_location("autoresearch_multi_echelon_serial", _RUNNER)
_tmp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tmp)

# Tiny --smoke overrides layered on the existing "smoke" budget preset
# (popsize 8 / 10 gens / 20k train periods / 8x60k eval) to keep a 5-seed smoke
# in the seconds-per-seed range.
SMOKE_OVERRIDES = {"popsize": 6, "generations": 4, "eval_seeds": 4}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instance", choices=sorted(_tmp.INSTANCES), default=DEFAULT_INSTANCE)
    p.add_argument("--budget", choices=sorted(_tmp.BUDGETS), default="full")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help=f"optimizer seeds; default = srp canonical {list(_srp.CANONICAL_SEEDS_5)}")
    p.add_argument("--depth", type=int, default=1,
                   help="soft-tree depth passed through to the single-seed runner")
    p.add_argument("--mp_num_processors", type=int, default=2)
    p.add_argument("--smoke", action="store_true",
                   help="tiny budget (smoke preset + popsize 6 / 4 gens / 4 eval seeds), "
                        "mp_num_processors forced to 1, output ONLY under smoke_seed_robust/")
    return p.parse_args()


def run_single_seed_entry_point(seed: int, *, instance: str, budget: str, depth: int,
                                smoke: bool, mp: int, per_seed_dir: Path, run_tag: str) -> dict:
    """Run the EXISTING autoresearch entry point for one optimizer seed and parse
    its JSON payload into the srp per-seed record (gate = warm-start anchor cost
    on the seed's held-out CRN block; learned = its deployed candidate cost)."""
    per_seed_json = per_seed_dir / f"{instance}_d{depth}_{budget}_seed{seed}.json"
    cmd = [
        sys.executable, str(_RUNNER),
        "--instance", instance,
        "--budget", budget,
        "--seed", str(seed),
        "--depth", str(depth),
        "--run_tag", run_tag,
        "--output_json", str(per_seed_json),
    ]
    if smoke:
        for key, val in SMOKE_OVERRIDES.items():
            cmd += [f"--{key}", str(val)]
    t0 = time.time()
    # The child caps its own thread pools from these env vars (cpu_limits is
    # imported before invman_rust in the entry point).
    subprocess.run(cmd, check=True, env=cpu_limited_environ(mp),
                   stdout=subprocess.DEVNULL, stderr=None)
    payload = json.loads(per_seed_json.read_text(encoding="utf-8"))

    gate_cost = float(payload["baselines"]["warm_start_gen0_mean_cost"])
    learned_cost = float(payload["learned"]["mean_cost"])
    record = {
        "seed": seed,
        # SAME-PROTOCOL gate: exact Clark-Scarf policy simulated on this seed's
        # held-out CRN eval block (paired with the learned evaluation).
        "gate_cost": gate_cost,
        "best_learned_cost": learned_cost,
        "savings_pct_vs_gate": 100.0 * (gate_cost - learned_cost) / gate_cost,
        # context (NOT the paired gate): the analytic optimum and the
        # single-seed runner's own match bookkeeping.
        "published_optimum_context": payload["baselines"]["published_optimum"],
        "exact_solver_optimum_context": payload["baselines"]["exact_solver_optimum"],
        "gap_vs_published_pct": payload["result"]["gap_vs_published_pct"],
        "match_pct": payload["result"]["match_pct"],
        "single_seed_verdict": payload["result"]["verdict"],
        "learned_source": payload["learned"]["source"],
        "learned_sem_cost": payload["learned"]["sem_cost"],
        "gate_sem_cost": payload["baselines"]["warm_start_gen0_sem"],
        "train_seconds": payload["config"]["train_seconds"],
        "seconds": round(time.time() - t0, 1),
        "per_seed_json": str(per_seed_json),
    }
    print(f"[seed {seed}] gate(sim exact policy) {gate_cost:.4f}  "
          f"learned {learned_cost:.4f} ({record['learned_source']})  "
          f"savings {record['savings_pct_vs_gate']:+.4f}%  "
          f"gap vs published {record['gap_vs_published_pct']:+.4f}%  "
          f"[{record['single_seed_verdict']}]  ({record['seconds']}s)")
    return record


def main():
    parsed = parse_args()
    smoke = bool(parsed.smoke)
    budget = "smoke" if smoke else parsed.budget
    mp = 1 if smoke else parsed.mp_num_processors
    if smoke and parsed.budget != "smoke":
        print(f"[--smoke] forcing budget 'smoke' (ignoring --budget {parsed.budget}) "
              f"and mp_num_processors 1")

    out_root = PACKAGE_ROOT / "outputs" / PROBLEM_ID
    if smoke:
        # HARD separation: a smoke run may NEVER write the real artifact path.
        report_dir = out_root / "smoke_seed_robust"
        suffix = "" if parsed.instance == DEFAULT_INSTANCE else f"_{parsed.instance}"
        report_path = report_dir / f"seed_robust_report_smoke{suffix}.json"
        run_tag = "multi_echelon_serial_seed_robust_smoke"
    else:
        report_dir = out_root
        suffix = "" if parsed.instance == DEFAULT_INSTANCE else f"_{parsed.instance}"
        report_path = report_dir / f"seed_robust_report{suffix}.json"
        run_tag = "multi_echelon_serial_seed_robust"
    per_seed_dir = report_dir / "per_seed"
    per_seed_dir.mkdir(parents=True, exist_ok=True)

    def train_one_seed(seed: int) -> dict:
        return run_single_seed_entry_point(
            seed, instance=parsed.instance, budget=budget, depth=parsed.depth,
            smoke=smoke, mp=mp, per_seed_dir=per_seed_dir, run_tag=run_tag,
        )

    # srp owns the loop, the >=5-distinct-seed enforcement, the sample-std
    # aggregation, and the shared verdict rule.
    result = _srp.run_over_seeds(PROBLEM_ID, train_one_seed, seeds=parsed.seeds)

    inst = _tmp.INSTANCES[parsed.instance]
    out = {
        "family": PROBLEM_ID,
        "benchmark": "seed_robust_multi_echelon_serial",
        "instance": parsed.instance,
        "demand_kind": inst["demand_kind"],
        "num_stages": len(inst["lead_time"]),
        "budget": budget,
        "smoke": smoke,
        "depth": parsed.depth,
        "mp_num_processors": mp,
        "gate_definition": (
            "warm_start_gen0_mean_cost: the exact Clark-Scarf echelon base-stock "
            "policy (recursive-newsvendor optimum) SIMULATED on the same held-out "
            "CRN eval block as the learned policy (same-protocol, paired per "
            "optimizer seed). The analytic optimum (published / exact solver) is "
            "context only."
        ),
        "honest_interpretation": (
            "The gate is a PROVEN optimum and the optimal policy class; the honest "
            "ceiling is MATCH (PARITY). The entry point deploys the cheapest of "
            "{warm-start anchor, CMA candidates} on the held-out block, so per-seed "
            "savings >= 0 by construction; any positive savings is simulation/"
            "selection noise, never a claimed beat of the optimum."
        ),
        "published_optimum_context": inst["published_optimum"],
        **result,
    }
    report_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("=" * 78)
    print(f"{parsed.instance}  budget={budget}  depth={parsed.depth}  smoke={smoke}")
    print(f"GATE (sim exact policy) seed-mean  {out['gate_seed_mean']:.4f} +/- {out['gate_seed_std']:.4f}")
    print(f"LEARNED seed-mean                  {out['learned_seed_mean']:.4f} +/- {out['learned_seed_std']:.4f}")
    print(f"SAVINGS vs gate                    {out['savings_pct_seed_mean']:+.4f}% +/- "
          f"{out['savings_pct_seed_std']:.4f}%  (beating gate {out['frac_seeds_beating_gate']})")
    print(f"Published optimum (context): {inst['published_optimum']}")
    print(f"VERDICT (vs same-protocol gate): {out['verdict_vs_same_protocol_gate']}")
    print(f"WROTE_JSON: {report_path}")


if __name__ == "__main__":
    main()
