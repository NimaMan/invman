"""Seed-ROBUST dual-sourcing runner: warm-start-at-CDI soft tree vs capped-dual-index,
paired-CRN, on all six Gijsbrechts (2022) Figure-9 reference rows.

OBJECTIVE
---------
The paper's dual-sourcing headline ("learned soft-tree matches CDI on all 6 Fig-9 rows,
beats it on 2") is currently backed by outputs/dual_sourcing_policy_search/final_report.json,
which is a BEST-OF-CANDIDATES report at a single optimizer seed (123). Per the project
mandate (invman/optimizer_seed_robustness_policy.py: dual_sourcing is a "seeds"-mode
problem), the headline must be a mean +/- sample-std over >= 5 independent CMA-ES
optimizer seeds. This runner retrains THE chosen protocol -- the warm-start-at-CDI
depth-2 axis-aligned/constant soft tree in the factorized small-cap capped-dual-index
coordinate (adapter capped_dual_index_delta_smallcap_targets, sigma_init 0.5; exactly
train_warmstart.py's winning configuration, "ws_smallcap_axisconst" in final_report.py)
-- once per optimizer seed on every row, evaluates each trained policy on the paired
common-random-number (CRN) protocol against CDI, and writes the standardized seed-robust
report that paper/generate_results_tables.py's fail-loud gate reads at
outputs/dual_sourcing_policy_search/seed_robust_report.json.

RELATION TO seed_robust_learned_vs_cdi_tier_c.py: that script covers ONE synthetic
taxonomy cell (dual_l2_ce110_b50_u08_catC) on the suite eval protocol; this one covers
the six published reference rows on the high-precision paired-CRN protocol and produces
the gate artifact. Both stay.

GATE SEMANTICS / SIGN CONVENTION
--------------------------------
gate_cost = the CDI (capped-dual-index) control -- the verified optimal STATIC policy and
the published ~0% optimality proxy -- encoded as a depth-1 constant tree and rolled out
through the SAME invman_rust.dual_sourcing_soft_tree_rollout on the SAME CRN eval seeds
as the learned policy (apples-to-apples, final_report.py protocol: eval seeds
3000..3000+N-1, horizon 100000, warm-up 0.2). gate_cost is optimizer-seed-INDEPENDENT,
so within an instance its seed-std is 0 by construction. best_learned_cost = the learned
policy's CRN mean cost at that optimizer seed. savings_pct_vs_gate =
100*(gate-learned)/gate (positive = learned cheaper); the familiar gap%-vs-CDI is its
negation and is also recorded, together with the paired per-seed difference and its SEM.
Expected honest outcome: PARITY (CDI is ~optimal; the single-seed "beats" on 2 rows were
best-of-candidates margins of -0.009%/-0.041%).

ALGORITHM (full description)
----------------------------
1. For each of the 6 reference rows (l_r in {2,3,4} x c_e in {105,110}), compute the CDI
   CRN cost vector once (eval_artifacts_highprec.cdi_costs_for; CDI params read from the
   benchmark suite instance JSON).
2. For each optimizer seed (srp.run_over_seeds enforces >= 5 distinct seeds; canonical
   9001..9005):
   a. Build run args via train_warmstart.make_args (budget "full" = the paper protocol:
      1500 CMA generations, categorical population base 128, train horizon 2000), then
      override seed, sigma_init=0.5, CPU caps, and a SEED-DISTINCT experiment name (the
      checkpoint dir name would otherwise collide across seeds).
   b. Warm-start CMA-ES at the encoded CDI control: build_policy -> warmstart_x0 with the
      row's (s_e, delta_r, cap_r) targets; set args.cma_x0 (train_warmstart protocol).
   c. Train via invman.experiment_runner.run_experiment. The final checkpoint (episode ==
      training_episodes, asserted divisible by save_every) holds CMA-ES xbest == the
      deployed policy_artifact.json. If both the result JSON and the final checkpoint
      already exist the run is REUSED (resume support for the long full-budget run).
   d. Evaluate the artifact on the shared CRN seeds (eval_artifacts_highprec.eval_artifact)
      -> best_learned_cost, paired difference vs the cached CDI cost vector.
3. Per instance: srp.run_over_seeds returns per_seed records + the standardized summary
   block (n_optimizer_seeds, learned/gate seed mean/std, savings_pct seed mean/std,
   frac_seeds_beating_gate, verdict_vs_same_protocol_gate).
4. TOP-LEVEL n_optimizer_seeds = MIN over the per-instance n_optimizer_seeds (documented
   choice; the generate-time gate reads exactly this key, and the min is the only value
   that cannot overstate any row). A cross_instance_mean_summary block additionally
   applies the standardized summary to per-seed costs AVERAGED across the 6 rows (a
   portfolio view; same srp keys).
5. Write the report. --smoke runs the identical pipeline at a tiny budget (3 CMA
   generations, fixed population 6, horizon 300, eval 2x1000, CRN 8 seeds x 4000,
   mp_num_processors 1) and writes ONLY to
   outputs/dual_sourcing_policy_search/smoke_seed_robust/seed_robust_report_smoke.json --
   a smoke run can NEVER touch the real artifact path (asserted).

CPU CAP / USAGE
---------------
  RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  python scripts/dual_sourcing/seed_robust_warmstart_soft_tree_vs_cdi_paired_crn.py \
      --budget full --seeds 9001 9002 9003 9004 9005 --mp_num_processors 2

  Smoke test (tiny budget, separate output path):
  python scripts/dual_sourcing/seed_robust_warmstart_soft_tree_vs_cdi_paired_crn.py \
      --smoke --seeds 9001 9002 9003 9004 9005 --mp_num_processors 1
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from invman.cpu_limits import configure_process_cpu_limits_from_argv  # noqa: E402

configure_process_cpu_limits_from_argv(sys.argv[1:], default=2)

import invman_rust  # noqa: F401,E402

from invman import optimizer_seed_robustness_policy as srp  # noqa: E402
from invman.experiment_runner import run_experiment  # noqa: E402
from invman.policy_build import build_policy  # noqa: E402

POLICY_SEARCH_DIR = PACKAGE_ROOT / "outputs" / "dual_sourcing_policy_search"
REAL_REPORT_PATH = POLICY_SEARCH_DIR / "seed_robust_report.json"
SMOKE_DIR = POLICY_SEARCH_DIR / "smoke_seed_robust"
SMOKE_REPORT_PATH = SMOKE_DIR / "seed_robust_report_smoke.json"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Reuse the EXISTING training / evaluation entry points verbatim (exemplar pattern):
#   train_warmstart.py     -> make_args / cdi_targets / targets_for_adapter / warmstart_x0
#   eval_artifacts_highprec -> cdi_costs_for / eval_artifact (+ .L.paired_stats)
_tw = _load_module("ds_train_warmstart", POLICY_SEARCH_DIR / "train_warmstart.py")
_ev = _load_module("ds_eval_artifacts_highprec", POLICY_SEARCH_DIR / "eval_artifacts_highprec.py")

ROWS = list(_tw.ROWS)  # the 6 Fig-9 rows: l_r in {2,3,4} x c_e in {105,110}

# THE chosen protocol (final_report.py's "ws_smallcap_axisconst" = the paper's lever):
ADAPTER = "capped_dual_index_delta_smallcap_targets"
DEPTH = 2
SPLIT = "axis_aligned"
LEAF = "constant"
TEMPERATURE = 0.25
DEFAULT_SIGMA_INIT = 0.5  # the warm-start runs were trained with sigma 0.5 (log_smallcap_sigma05)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="optimizer seeds (default: canonical 9001..9005; >=5 distinct enforced)")
    p.add_argument("--budget", choices=["screening", "full"], default="full")
    p.add_argument("--smoke", action="store_true",
                   help="tiny budget + separate smoke output path (never the real artifact)")
    p.add_argument("--rows", nargs="+", default=ROWS, choices=ROWS)
    p.add_argument("--mp_num_processors", type=int, default=2)
    p.add_argument("--sigma_init", type=float, default=DEFAULT_SIGMA_INIT)
    p.add_argument("--crn_eval_seeds", type=int, default=80,
                   help="number of shared CRN evaluation seeds (final_report protocol: 80)")
    p.add_argument("--crn_horizon", type=int, default=100000)
    p.add_argument("--crn_seed_start", type=int, default=3000)
    return p.parse_args()


def make_train_one_seed(row: str, parsed, run_root: Path, crn_seeds: list[int],
                        crn_horizon: int, inst: dict, gate_cost: float, cdi_costs):
    """Closure: train (warm-start at CDI) + paired-CRN eval for one (row, optimizer seed)."""

    def train_one_seed(seed: int) -> dict:
        t0 = time.time()
        args = _tw.make_args(row, depth=DEPTH, split=SPLIT, leaf=LEAF, adapter=ADAPTER,
                             temperature=TEMPERATURE, budget=parsed.budget, root=run_root,
                             seed=int(seed))
        args.mp_num_processors = int(parsed.mp_num_processors)
        args.sigma_init = float(parsed.sigma_init)
        if parsed.smoke:
            # Tiny smoke budget: a few CMA generations, small fixed population, short
            # horizons, single processor. Same pipeline, separate artifact tree.
            args.training_episodes = 3
            args.es_population = 6
            args.es_population_sampling = "fixed"
            args.es_population_candidates = None
            args.es_population_probabilities = None
            args.horizon = 300
            args.eval_horizon = 1000
            args.eval_seeds = 2
            args.mp_num_processors = 1
        # The final checkpoint must exist AND coincide with the deployed xbest:
        # es_mp saves every `save_every` episodes, so episodes must divide evenly.
        args.save_every = args.training_episodes if parsed.smoke else 100
        assert args.training_episodes % args.save_every == 0, \
            (args.training_episodes, args.save_every)
        # Seed-distinct experiment name: the checkpoint dir name is
        # {experiment_name}_{num_params}_{episode}; without the seed suffix every
        # optimizer seed would overwrite / reload the same artifacts.
        args.experiment_name = f"sr_{row}_{ADAPTER}_d{DEPTH}_{SPLIT[:4]}_{LEAF[:4]}_s{seed}"

        # Warm start CMA-ES at the encoded CDI control (train_warmstart protocol).
        model = build_policy(args)
        s_e, delta_r, cap_r = _tw.cdi_targets(row)
        tgt, adim = _tw.targets_for_adapter(ADAPTER, s_e, delta_r, cap_r)
        x0 = _tw.warmstart_x0(input_dim=model.input_dim, depth=DEPTH, leaf_type=LEAF,
                              action_dim=adim, min_values=list(model.min_values),
                              max_values=list(model.max_values), targets=tgt)
        assert len(x0) == model.num_params, (len(x0), model.num_params)
        args.cma_x0 = x0

        final_dir = (Path(args.trained_models_dir)
                     / f"{args.experiment_name}_{model.num_params}_{args.training_episodes}")
        result_path = Path(args.results_dir) / f"{args.experiment_name}.json"
        reused = result_path.exists() and (final_dir / "policy_artifact.json").exists()
        if reused:
            payload = json.loads(result_path.read_text())
        else:
            payload, _ = run_experiment(args)
        if not (final_dir / "policy_artifact.json").exists():
            raise FileNotFoundError(
                f"final checkpoint missing after training: {final_dir} "
                f"(save_every={args.save_every}, episodes={args.training_episodes})")
        art = json.loads((final_dir / "policy_artifact.json").read_text())

        learned_costs = _ev.eval_artifact(inst, art, crn_seeds, crn_horizon)
        learned = float(learned_costs.mean())
        paired_d, paired_sem = _ev.L.paired_stats(learned_costs, cdi_costs)
        rec = {
            "seed": int(seed),
            "gate_cost": gate_cost,
            "best_learned_cost": learned,
            "savings_pct_vs_gate": 100.0 * (gate_cost - learned) / gate_cost,
            "gap_pct_vs_cdi": 100.0 * (learned - gate_cost) / gate_cost,
            "paired_d_vs_cdi": paired_d,
            "paired_sem_vs_cdi": paired_sem,
            "suite_eval_cost": float(payload["evaluation"]["learned_policy"]["mean_cost"]),
            "model_dir": str(final_dir),
            "reused_cached_run": reused,
            "seconds": round(time.time() - t0, 1),
        }
        print(f"  [{row} seed {seed}] learned {learned:.4f} vs CDI {gate_cost:.4f}  "
              f"gap {rec['gap_pct_vs_cdi']:+.4f}%  paired_d {paired_d:+.4f}+/-{paired_sem:.4f}  "
              f"({'cached' if reused else 'trained'}, {rec['seconds']}s)")
        sys.stdout.flush()
        return rec

    return train_one_seed


def main():
    parsed = parse_args()
    if parsed.smoke:
        out_path = SMOKE_REPORT_PATH
        run_root = SMOKE_DIR / "runs"
        crn_n = min(parsed.crn_eval_seeds, 8)
        crn_horizon = min(parsed.crn_horizon, 4000)
    else:
        out_path = REAL_REPORT_PATH
        run_root = POLICY_SEARCH_DIR / "seed_robust_runs"
        crn_n = int(parsed.crn_eval_seeds)
        crn_horizon = int(parsed.crn_horizon)
    # A smoke run must NEVER write the real artifact path.
    assert parsed.smoke == ("smoke" in str(out_path)), (parsed.smoke, out_path)
    run_root.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crn_seeds = list(range(int(parsed.crn_seed_start), int(parsed.crn_seed_start) + crn_n))

    print(f"mode={'SMOKE' if parsed.smoke else 'REAL'}  budget={parsed.budget}  "
          f"rows={parsed.rows}  CRN={crn_n}x{crn_horizon}  out={out_path}")
    sys.stdout.flush()

    t_start = time.time()
    instances: dict[str, dict] = {}
    for row in parsed.rows:
        print(f"=== {row} ===")
        inst, cdi_params, cdi_costs = _ev.cdi_costs_for(row, crn_seeds, crn_horizon)
        gate_cost = float(cdi_costs.mean())
        print(f"  CDI(s_e={cdi_params['s_e']},dr={cdi_params['s_r'] - cdi_params['s_e']},"
              f"cap={cdi_params['cap_r']}) CRN mean {gate_cost:.4f}")
        sys.stdout.flush()
        train_one = make_train_one_seed(row, parsed, run_root, crn_seeds, crn_horizon,
                                        inst, gate_cost, cdi_costs)
        result = srp.run_over_seeds("dual_sourcing", train_one, seeds=parsed.seeds)
        result["cdi_params"] = {k: int(v) for k, v in cdi_params.items()}
        instances[row] = result
        print(f"  -> {row}: savings vs CDI {result['savings_pct_seed_mean']:+.4f}% "
              f"+/- {result['savings_pct_seed_std']:.4f}%  "
              f"beating {result['frac_seeds_beating_gate']}  "
              f"verdict {result['verdict_vs_same_protocol_gate']}")
        sys.stdout.flush()

    # Top-level n_optimizer_seeds = MIN over instances (documented in the module
    # docstring; the only value that cannot overstate any row's seed depth).
    n_min = min(r["n_optimizer_seeds"] for r in instances.values())
    seed_list = instances[parsed.rows[0]]["seeds"]

    # Cross-instance portfolio view: per optimizer seed, average learned/gate cost
    # across the rows, then apply the same standardized summary.
    cross_per_seed = []
    for i, sd in enumerate(seed_list):
        cross_per_seed.append({
            "seed": int(sd),
            "best_learned_cost": sum(instances[r]["per_seed"][i]["best_learned_cost"]
                                     for r in parsed.rows) / len(parsed.rows),
            "gate_cost": sum(instances[r]["per_seed"][i]["gate_cost"]
                             for r in parsed.rows) / len(parsed.rows),
        })
    cross_summary = srp.build_seed_robust_summary(cross_per_seed, problem_id="dual_sourcing",
                                                  savings_key=None)

    report = {
        "problem_id": "dual_sourcing",
        "n_optimizer_seeds": n_min,
        "n_optimizer_seeds_semantics": "min over per-instance n_optimizer_seeds",
        "seeds": seed_list,
        "smoke": bool(parsed.smoke),
        "budget": parsed.budget if not parsed.smoke else f"{parsed.budget}+SMOKE_OVERRIDES",
        "policy_protocol": {
            "spec": "warm-start-at-CDI soft tree (final_report 'ws_smallcap_axisconst')",
            "action_adapter": ADAPTER, "depth": DEPTH, "split_type": SPLIT,
            "leaf_type": LEAF, "temperature": TEMPERATURE,
            "sigma_init": float(parsed.sigma_init),
            "warm_start": "cma_x0 = encoded CDI (s_e, delta_r, cap_r) via build_warmstart.warmstart_x0",
            "trainer": "invman.experiment_runner.run_experiment (CMA-ES)",
        },
        "crn_protocol": {
            "n_eval_seeds": crn_n, "horizon": crn_horizon,
            "seed_start": int(parsed.crn_seed_start), "paired": True,
            "evaluator": "invman_rust.dual_sourcing_soft_tree_rollout (warm-up 0.2)",
        },
        "gate": ("capped_dual_index (CDI) encoded as a depth-1 constant tree, evaluated on "
                 "the SAME CRN seeds (paired); optimizer-seed-independent within an instance"),
        "cross_instance_mean_summary": cross_summary,
        "instances": instances,
        "generated_by": "scripts/dual_sourcing/seed_robust_warmstart_soft_tree_vs_cdi_paired_crn.py",
        "total_seconds": round(time.time() - t_start, 1),
    }
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 78)
    for row in parsed.rows:
        r = instances[row]
        print(f"{row:14s} learned {r['learned_seed_mean']:.4f}+/-{r['learned_seed_std']:.4f}  "
              f"CDI {r['gate_seed_mean']:.4f}  savings {r['savings_pct_seed_mean']:+.4f}%"
              f"+/-{r['savings_pct_seed_std']:.4f}%  {r['frac_seeds_beating_gate']}  "
              f"{r['verdict_vs_same_protocol_gate']}")
    print(f"n_optimizer_seeds (min over instances): {n_min}")
    print(f"cross-instance verdict: {cross_summary['verdict_vs_same_protocol_gate']}  "
          f"savings {cross_summary['savings_pct_seed_mean']:+.4f}%"
          f"+/-{cross_summary['savings_pct_seed_std']:.4f}%")
    print(f"WROTE_JSON: {out_path}")


if __name__ == "__main__":
    main()
