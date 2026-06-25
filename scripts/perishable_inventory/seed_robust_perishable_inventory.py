"""
Seed-ROBUST learned-vs-gate runner for the perishable-inventory family
(De Moor, Gijsbrechts & Boute 2022 / Farrington et al. 2025 Scenario A).

OBJECTIVE
---------
The paper's perishable headline is the De Moor m=2 / L=1 FIFO anchor
(`de_moor2022_m2_exp2_l1_cp7_fifo`), where the warm-started depth-2 oblique-linear
soft tree BEATS the best tuned base-stock, claimed over 5 independent CMA-ES
optimizer seeds with model selection on a DISJOINT validation block. Per the
project mandate (invman/optimizer_seed_robustness_policy.py, problem_id
"perishable_inventory", mode="seeds"), that claim must be backed by a single
artifact reporting the cross-seed mean +/- sample std and the standardized verdict
-- not by ad-hoc ledger rows. This runner produces that artifact:
outputs/perishable_inventory/seed_robust_report.json, with one per-instance block
per anchor instance and the standardized summary keys from
optimizer_seed_robustness_policy (srp).

WHAT IS REUSED (no new training code, no env/policy/Rust changes)
-----------------------------------------------------------------
The EXISTING single-seed entry point
scripts/perishable_inventory/autoresearch_perishable_inventory.py is loaded via
importlib and its main() is invoked VERBATIM once per (instance, optimizer seed)
with a patched sys.argv (the same reuse device as the exemplar
scripts/multi_echelon/seed_robust_divergent_multi_echelon.py, taken one step
further: calling main() itself guarantees zero drift from the published
training/selection protocol -- warm start at the encoded best base-stock,
CMA-ES, candidate selection on the disjoint validation block, report on the
held-out eval block). Each call writes its own per-seed JSON under this runner's
output directory; this runner only aggregates.

GATE AND METRIC SEMANTICS (sign convention)
-------------------------------------------
The environment is reward-style: mean DISCOUNTED RETURN = negated discounted cost
(higher / less negative is better). The autoresearch script's base-stock GATE is
the better of {exact-MDP best base-stock level, per-seed stochastic-search argmin
level}, re-scored by the SAME Monte-Carlo estimator on the SAME held-out CRN eval
block as the learned policy (the apples-to-apples comparator; the analytic VI
optimum is context only, different estimator). For the srp cost-style keys we
NEGATE returns into positive costs:
    gate_cost          = -base_stock_gate_return        (per-seed, paired)
    best_learned_cost  = -learned mean_return            (per-seed, eval block)
    savings_pct_vs_gate = 100*(gate_cost - best_learned_cost)/gate_cost
                        == the script's gap_vs_base_stock_gate_pct (sign-checked
                           per seed; positive == learned beats the gate).
The eval block is seeded from the optimizer seed (seed + 1_000_000 + i), so the
gate moves slightly per seed exactly like the multi-echelon exemplar's per-seed
gate -- learned and gate are PAIRED within each seed.

ALGORITHM (full description)
----------------------------
1. Resolve the optimizer-seed list via srp.seeds_for("perishable_inventory",
   --seeds) (canonical 9001..9005; >=5 distinct seeds enforced, fail-closed).
2. For each anchor instance (--references, default = the paper's two exact-MDP
   anchors: m2_exp2 FIFO primary + m2_exp1 LIFO sibling):
   for each optimizer seed, run the autoresearch main() at --budget
   (default "full": popsize 24, 120 generations, 64 search / 512 validation /
   2048 eval CRN seeds, horizon 465, gamma .99) writing a per-seed JSON; extract
   {gate_cost, best_learned_cost, savings_pct_vs_gate, gate level, selected
   endpoint, single-seed verdict, VI-gap context}.
3. Aggregate each instance's 5 per-seed records with srp.run_over_seeds ->
   standardized summary keys (n_optimizer_seeds, learned/gate_seed_mean/std,
   savings_pct_seed_mean/std, frac_seeds_beating_gate,
   verdict_vs_same_protocol_gate; sample n-1 std; shared verdict rule).
4. Write ONE report JSON with the per-instance blocks; the PRIMARY anchor's
   standardized summary keys are also hoisted to the top level as the headline.
   Real artifact: outputs/perishable_inventory/seed_robust_report.json.
   SMOKE artifact: outputs/perishable_inventory/smoke_seed_robust/
   seed_robust_report_smoke.json (a --smoke run can NEVER touch the real path;
   it also forces the tiny "smoke" budget preset -- popsize 8, 10 generations,
   128 eval seeds -- and mp_num_processors=1).

CPU CAP / USAGE
---------------
  # smoke (tiny budget, ~seconds/seed, separate artifact path):
  RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  python scripts/perishable_inventory/seed_robust_perishable_inventory.py \
      --smoke --seeds 9001 9002 9003 9004 9005 --mp_num_processors 1
  # real full-budget 5-seed report (the paper artifact):
  RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  python scripts/perishable_inventory/seed_robust_perishable_inventory.py \
      --budget full --seeds 9001 9002 9003 9004 9005 --mp_num_processors 2
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import sys
import time
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from invman.cpu_limits import configure_process_cpu_limits, configure_process_cpu_limits_from_argv  # noqa: E402

# CPU cap MUST be set before numpy/invman_rust are imported (the autoresearch
# module imports both at exec time). A --smoke run is pinned to 1 worker.
if "--smoke" in sys.argv[1:]:
    configure_process_cpu_limits(1)
else:
    configure_process_cpu_limits_from_argv(sys.argv[1:], default=2)

from invman import optimizer_seed_robustness_policy as _srp  # noqa: E402

# Load the EXISTING single-seed autoresearch runner and reuse its main() verbatim.
_RUNNER = PACKAGE_ROOT / "scripts" / "perishable_inventory" / "autoresearch_perishable_inventory.py"
_spec = importlib.util.spec_from_file_location("autoresearch_perishable_inventory", _RUNNER)
_aut = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_aut)

PROBLEM_ID = "perishable_inventory"
# The paper's two exact-MDP anchor instances (Table tab:perish-results):
# primary FIFO headline + LIFO sibling. Further Scenario-A instances may be
# appended via --references.
DEFAULT_ANCHOR_REFERENCES = (
    "de_moor2022_m2_exp2_l1_cp7_fifo",
    "de_moor2022_m2_exp1_l1_cp7_lifo",
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--references", default=",".join(DEFAULT_ANCHOR_REFERENCES),
                   help="comma-separated De Moor reference-instance names (anchor instances)")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help=f"optimizer seeds (default canonical {list(_srp.CANONICAL_SEEDS_5)}; "
                        f">= {_srp.MIN_OPTIMIZER_SEEDS} distinct enforced)")
    p.add_argument("--budget", choices=sorted(_aut.BUDGETS), default="full")
    p.add_argument("--smoke", action="store_true",
                   help="tiny-budget shakeout: forces budget='smoke', mp_num_processors=1, and a "
                        "SEPARATE smoke artifact path (never the real report JSON)")
    p.add_argument("--mp_num_processors", type=int, default=2)
    # pass-through policy/optimizer knobs (defaults == the autoresearch defaults
    # used for the paper rows: depth-2 oblique-linear, sigma .75, floor endpoint)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.25)
    p.add_argument("--split_type", choices=["oblique", "axis_aligned"], default="oblique")
    p.add_argument("--sigma_init", type=float, default=0.75)
    p.add_argument("--deploy_endpoint", choices=["floor", "xbest", "xfavorite"], default="floor")
    return p.parse_args()


def run_autoresearch_once(*, reference: str, seed: int, budget: str, parsed,
                          run_tag: str, per_seed_json: Path) -> dict:
    """Invoke the existing autoresearch main() verbatim under a patched argv.

    main() trains (warm-started CMA-ES), selects on the disjoint validation
    block, reports on the held-out eval block, and writes `per_seed_json`; we
    reload that payload. Its stdout (the full payload dump) is captured to keep
    this runner's log one line per seed.
    """
    argv = [
        str(_RUNNER),
        "--reference", reference,
        "--budget", budget,
        "--seed", str(seed),
        "--depth", str(parsed.depth),
        "--temperature", str(parsed.temperature),
        "--split_type", parsed.split_type,
        "--sigma_init", str(parsed.sigma_init),
        "--deploy_endpoint", parsed.deploy_endpoint,
        "--run_tag", run_tag,
        "--output_json", str(per_seed_json),
        "--mp_num_processors", str(parsed.mp_num_processors),
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            _aut.main()
    finally:
        sys.argv = old_argv
    return json.loads(per_seed_json.read_text(encoding="utf-8"))


def per_seed_record(seed: int, payload: dict, seconds: float, per_seed_json: Path) -> dict:
    """Map one autoresearch payload to the srp per-seed record (cost convention).

    Returns are negated discounted costs (higher better); srp keys are
    cost-style (lower better), so gate_cost / best_learned_cost = -return.
    """
    gate_return = float(payload["baselines"]["base_stock_gate_return"])
    learned_return = float(payload["learned"]["mean_return"])
    gate_cost = -gate_return
    best_learned_cost = -learned_return
    savings = 100.0 * (gate_cost - best_learned_cost) / gate_cost
    # sign-convention self-check vs the script's own gap (identical up to fp noise)
    script_pct = float(payload["result"]["gap_vs_base_stock_gate_pct"])
    if abs(savings - script_pct) > 1e-6:
        raise AssertionError(
            f"savings sign/derivation mismatch: derived {savings} vs script {script_pct}")
    return {
        "seed": seed,
        "gate_cost": gate_cost,
        "best_learned_cost": best_learned_cost,
        "savings_pct_vs_gate": savings,
        "base_stock_gate_return": gate_return,
        "learned_mean_return": learned_return,
        "base_stock_gate_level": int(payload["baselines"]["base_stock_gate_level"]),
        "learned_source": payload["learned"]["source"],
        "single_seed_verdict_vs_gate": payload["result"]["verdict"],
        "gap_vs_vi_optimum_pct_CONTEXT_ONLY": float(payload["result"]["gap_vs_vi_optimum_pct"]),
        "paired_sem": float(payload["result"]["paired_sem"]),
        "per_seed_json": str(per_seed_json),
        "seconds": round(seconds, 1),
    }


def main():
    parsed = parse_args()
    budget = "smoke" if parsed.smoke else parsed.budget
    if parsed.smoke:
        parsed.mp_num_processors = 1
    references = [r.strip() for r in parsed.references.split(",") if r.strip()]
    seed_list = _srp.seeds_for(PROBLEM_ID, parsed.seeds)

    out_root = PACKAGE_ROOT / "outputs" / PROBLEM_ID
    if parsed.smoke:
        # HARD separation: a smoke run can never write the real artifact path.
        report_dir = out_root / "smoke_seed_robust"
        report_path = report_dir / "seed_robust_report_smoke.json"
        run_tag = "perishable_seed_robust_smoke"
    else:
        report_dir = out_root
        report_path = report_dir / "seed_robust_report.json"
        run_tag = "perishable_seed_robust"
    runs_dir = report_dir / ("runs_smoke" if parsed.smoke else "seed_robust_runs")
    runs_dir.mkdir(parents=True, exist_ok=True)

    instances: dict[str, dict] = {}
    for reference in references:
        def train_one_seed(seed: int, _reference: str = reference) -> dict:
            t0 = time.time()
            per_seed_json = runs_dir / f"{_reference}_seed{seed}_{budget}.json"
            payload = run_autoresearch_once(
                reference=_reference, seed=seed, budget=budget, parsed=parsed,
                run_tag=run_tag, per_seed_json=per_seed_json,
            )
            rec = per_seed_record(seed, payload, time.time() - t0, per_seed_json)
            print(f"[{_reference} seed {seed}] gate {rec['gate_cost']:.2f} "
                  f"(S={rec['base_stock_gate_level']})  learned {rec['best_learned_cost']:.2f} "
                  f"({rec['learned_source']})  savings {rec['savings_pct_vs_gate']:+.3f}%  "
                  f"({rec['seconds']}s)")
            return rec

        # Shared seeds-mode driver: loops the seeds, enforces >=5 distinct, and
        # appends the standardized summary keys (sample std, shared verdict rule).
        instances[reference] = _srp.run_over_seeds(PROBLEM_ID, train_one_seed, seeds=seed_list)

    primary = references[0]
    summary_keys = ("problem_id", "n_optimizer_seeds", "learned_seed_mean", "learned_seed_std",
                    "gate_seed_mean", "gate_seed_std", "savings_pct_seed_mean",
                    "savings_pct_seed_std", "frac_seeds_beating_gate",
                    "verdict_vs_same_protocol_gate")
    headline = {k: instances[primary][k] for k in summary_keys}

    out = {
        "family": PROBLEM_ID,
        "benchmark": "seed_robust_perishable_inventory",
        "budget": budget,
        "smoke": parsed.smoke,
        "seeds": list(seed_list),
        "references": references,
        "config": {
            "depth": parsed.depth, "temperature": parsed.temperature,
            "split_type": parsed.split_type, "leaf_type": "linear",
            "sigma_init": parsed.sigma_init, "deploy_endpoint": parsed.deploy_endpoint,
            "mp_num_processors": parsed.mp_num_processors,
            "budget_preset": dict(_aut.BUDGETS[budget]),
        },
        "sign_convention": "gate_cost/best_learned_cost = NEGATED mean discounted return "
                           "(cost-style, lower better); savings_pct_vs_gate > 0 == learned "
                           "beats the best base-stock gate on the paired held-out eval block.",
        # headline = the primary anchor's standardized summary keys, hoisted.
        "headline_instance": primary,
        **headline,
        # one block per anchor instance: {seeds, per_seed, **standardized summary}.
        "instances": instances,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("=" * 78)
    for reference in references:
        blk = instances[reference]
        print(f"{reference}  [{budget}]")
        print(f"  GATE seed-mean    {blk['gate_seed_mean']:.2f} +/- {blk['gate_seed_std']:.2f}  (cost)")
        print(f"  LEARNED seed-mean {blk['learned_seed_mean']:.2f} +/- {blk['learned_seed_std']:.2f}  (cost)")
        print(f"  SAVINGS vs gate   {blk['savings_pct_seed_mean']:+.3f}% +/- {blk['savings_pct_seed_std']:.3f}%  "
              f"(beating gate {blk['frac_seeds_beating_gate']})")
        print(f"  VERDICT (vs same-protocol gate): {blk['verdict_vs_same_protocol_gate']}")
    print(f"WROTE_JSON: {report_path}")


if __name__ == "__main__":
    main()
