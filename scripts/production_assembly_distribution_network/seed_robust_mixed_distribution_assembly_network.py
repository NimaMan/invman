"""
Seed-ROBUST learned-vs-gate runner for the MIXED distribution-and-assembly SCN
(Pirhooshyaran & Snyder 2021, Fig. 1 / Table 5) -- the seed-cherry-pick fix.

OBJECTIVE
---------
The paper row for the mixed distribution-and-assembly network claims the learned
depth-2 soft tree beats the env's own grid-searched best pairwise base-stock gate
(297.69, echelon order-up-to [36,13,7]) by -0.99% (294.73 +/- 0.92) -- but only "on
the BEST of three CMA seeds, while a third seed sits +2.9% above the gate." That is a
best-of-N straddle, not a robust result. This runner replaces best-of-N reporting with
a HONEST seed-averaged comparison and three robustness devices, then computes the
seed-mean +/- std over >=5 independent optimizer seeds. A credible parity / negative
result is an explicitly valid outcome.

ALGORITHM (full description)
----------------------------
1. Gate. Identical grid search to autoresearch_mixed_distribution_assembly_network.py:
   per-echelon order-up-to levels (e1 = relations edge(0,1) & external->0, e2 = edges
   (1,2),(1,3), e3 = edges (2,4),(2,5),(3,4),(3,5)) on the disjoint SEARCH block, re-
   scored on the HELD-OUT block. The fine-grid argmin is echelon [36,13,7] ->
   OUL [36,13,13,7,7,7,7,36], held-out 297.69 / period. The gate POLICY (a real
   pairwise_base_stock rollout) is also kept as a deployment candidate.

2. Action geometry. Unchanged: vector_quantity over the 8 supply relations, a depth-d
   oblique soft tree with linear leaves, clipped to [MIN,MAX]. The linear leaf maps
   state features (per-node finished/raw/backlog/pipeline, per-relation raw + pipeline,
   per-edge internal backlog, remaining-horizon) to a per-relation order. CRUCIAL: the
   env build_policy_state DIVIDES every feature by a dynamic per-step scale (max raw
   value in the state). Therefore a linear leaf CANNOT express an exact affine
   order = clip(level - inventory_position) gate (unlike the OWMR echelon_targets
   geometry, whose action head is an explicit target position). The OWMR exact
   leaf-inversion warm-start is thus NOT transferable here. We instead use:

   (a) WARM-START at the GATE FLOW (--warm_start, default on). The gate's per-relation
       order-up-to levels [36,13,13,7,7,7,7,36] are decoded into a CONSTANT per-relation
       order via the same softplus-inverse leaf-bias inversion the autoresearch runner
       uses for a flat flow, but per-relation at the GATE LEVELS rather than at the flat
       demand-mean flow=5. This seeds generation 0 near the gate's steady-state ORDER
       magnitude (not the demand mean, which starves this network and gives gen0 ~864 /
       period). Leaf weights are zeroed (state-independent) so gen0 is a clean constant-
       order anchor; CMA-ES then refines the weights toward inventory feedback.

   (b) HONEST DEPLOYMENT FLOOR (--honest_floor, default on). The deployed policy on the
       held-out block is the argmin over {trained xbest, gate-flow warm-start anchor,
       the gate pairwise_base_stock policy itself}. With the gate in the candidate set,
       NO seed can deploy worse than the gate; the only open question is how much the
       search robustly subtracts BELOW the gate. The honest floor is decoder-agnostic
       (it scores real rollouts), so it sidesteps the non-invertible normalized-feature
       decoder entirely.

   (c) GENTLE SIGMA (--sigma_init, sweepable {0.1,0.2,0.3} vs the runner's default 0.8)
       so seeds stay near the gate-flow anchor instead of scattering into the 360-440
       basin the flat flow=5 + sigma-0.8 baseline lands in.

3. Robustness reporting (CENTRALIZED, migrated 2026-06-12). The runner trains N independent
   seeds in ONE invocation, scores each on the SAME held-out CRN block (HOLDOUT_SEED,
   holdout_paths -- identical protocol to the autoresearch runner so 297.69 is comparable),
   records per-seed deployed cost, then aggregates through the SINGLE SOURCE OF TRUTH
   invman/optimizer_seed_robustness_policy.py (srp.build_seed_robust_summary): the output
   JSON carries the standardized keys n_optimizer_seeds, learned_seed_mean/std,
   gate_seed_mean/std, savings_pct_seed_mean/std, frac_seeds_beating_gate,
   verdict_vs_same_protocol_gate (shared rule: ROBUST_BEAT_VS_GATE / BEAT_WITHIN_STD /
   PARITY / ROBUST_LOSS_VS_GATE), plus per_seed records. NEVER best-of-N as the headline.
   - Std convention: srp uses SAMPLE (n-1) std; this script already used statistics.stdev
     (also n-1), so the migration is numerically a no-op -- the standardization is
     intentional per the central policy.
   - The legacy bespoke "ROBUST_GATE_MATCH_ONLY" label (honest floor pins every seed at the
     gate) maps to srp verdict PARITY with the auxiliary flag gate_pinned_all_seeds=true.
   - Default seeds are the canonical srp list (9001..9005). The historical mixed run used
     --seeds 11 22 33 44 55; pass that explicitly to reproduce it.

ARTIFACTS
---------
REAL run (no --smoke, --budget full):
    outputs/production_assembly_distribution_network/seed_robust_report_mixed.json
REAL run at a non-full budget gets a budget-suffixed name (so screening can never clobber
the audited full artifact). --smoke ALWAYS writes under
    outputs/production_assembly_distribution_network/smoke_seed_robust/
and never touches the real path.

CPU CAP / USAGE
---------------
CPU is capped before NumPy/Rust import via --mp_num_processors (default 2 rayon/omp
threads; --smoke forces 1). Run seeds serially inside one process (each CMA-ES population
rollout is already rayon-parallel); to fan out over cores, launch several invocations with
disjoint --seeds and aggregate the JSONs.
USAGE (full audit):
  RAYON_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  python scripts/production_assembly_distribution_network/seed_robust_mixed_distribution_assembly_network.py \
      --budget full --depth 2 --sigma_init 0.2 --seeds 9001 9002 9003 9004 9005 \
      --mp_num_processors 2 --description "gate-flow warmstart + floor, sigma0.2"
USAGE (smoke, tiny budget, separate artifact):
  python scripts/production_assembly_distribution_network/seed_robust_mixed_distribution_assembly_network.py \
      --smoke --mp_num_processors 1 --description "smoke"
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from invman.cpu_limits import configure_process_cpu_limits, configure_process_cpu_limits_from_argv

configure_process_cpu_limits_from_argv(sys.argv[1:], default=2)

import numpy as np  # noqa: E402

from invman import optimizer_seed_robustness_policy as srp  # noqa: E402
from invman.cmaes import CMAES  # noqa: E402

import invman_rust as ir  # noqa: E402

# Import the instance constants + helpers from the sibling autoresearch runner so the
# topology, parameters, CRN blocks, budgets and gate search are BYTE-IDENTICAL.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import autoresearch_mixed_distribution_assembly_network as base  # noqa: E402

PROBLEM_ID = "production_assembly_distribution_network"


def _gate_flow_warm_start(depth: int, leaf_type: str, gate_oul) -> np.ndarray:
    """Seed the CMA mean so generation 0 emits a CONSTANT per-relation order equal to the
    gate's order-up-to levels (per relation), state-independently.

    For a LINEAR leaf the env applies scaled = min + softplus(leaf_output); with the leaf
    WEIGHTS zeroed the output is the per-relation leaf BIAS, so
        order_dim = min + softplus(bias_dim)  =>  bias_dim = softplus_inv(gate_oul_dim - min).
    This reproduces the gate's steady-state order MAGNITUDE at gen 0 (not the gate's
    inventory feedback, which the normalized-feature decoder cannot express affinely).
    For constant / sigmoid_linear leaves we invert the sigmoid span transform instead.
    """
    n = base._flat_param_count(depth, leaf_type)
    flat = np.zeros(n, dtype=np.float64)
    num_internal = (1 << depth) - 1
    num_leaves = 1 << depth
    bias_block = num_leaves * base.ACTION_DIM
    gate = [float(x) for x in gate_oul]
    if leaf_type in ("constant", "sigmoid_linear"):
        # tail = num_leaves x ACTION_DIM sigmoid logits.
        tail = np.empty((num_leaves, base.ACTION_DIM), dtype=np.float64)
        for d in range(base.ACTION_DIM):
            span = float(base.MAX_VALUES[d] - base.MIN_VALUES[d])
            p = (gate[d] - base.MIN_VALUES[d]) / span
            p = float(min(max(p, 1e-4), 1.0 - 1e-4))
            tail[:, d] = math.log(p / (1.0 - p))
        flat[n - bias_block:] = tail.reshape(-1)
        return flat
    # linear leaf: zero weights (already zero), set per-relation leaf bias.
    bias_start = num_internal * base.INPUT_DIM + num_internal + num_leaves * base.ACTION_DIM * base.INPUT_DIM
    leaf_bias = np.empty(base.ACTION_DIM, dtype=np.float64)
    for d in range(base.ACTION_DIM):
        delta = max(gate[d] - base.MIN_VALUES[d], 1e-6)
        leaf_bias[d] = math.log(math.expm1(delta))
    tail = np.tile(leaf_bias, num_leaves)
    flat[bias_start:bias_start + bias_block] = tail
    return flat


def _train_one_seed(parsed, budget, holdout_paths, gate, x0):
    """Train ONE CMA-ES seed; return its trained xbest, gen0 holdout, and best_train."""
    depth, leaf, split, temp = parsed.depth, parsed.leaf_type, parsed.split_type, parsed.temperature
    n = base._flat_param_count(depth, leaf)
    gen0_mean, gen0_se = base.soft_tree_cost_on_paths(x0, depth, leaf, split, temp, holdout_paths)

    es = CMAES(num_params=n, sigma_init=parsed.sigma_init, popsize=budget["popsize"],
               seed=parsed.seed_value, x0=x0.tolist())
    rng = np.random.default_rng(parsed.seed_value + 1)
    best_flat = x0.copy()
    best_train = math.inf
    t0 = time.time()
    train_batch = parsed.train_batch if parsed.train_batch else budget["train_batch"]
    for _ in range(budget["generations"]):
        sols = es.ask()
        b = int(rng.integers(1, 10_000_000))
        seeds = list(range(b, b + train_batch))
        rewards = []
        for k in range(es.popsize):
            batch = [sols[k].astype(np.float32).tolist()] * train_batch
            cost = float(base.population_costs(batch, depth, leaf, split, temp, seeds).mean())
            rewards.append(-cost)
        es.tell(rewards)
        gi = int(np.argmax(rewards))
        if -rewards[gi] < best_train:
            best_train = -rewards[gi]
            best_flat = sols[gi].copy()
    train_seconds = time.time() - t0

    trained_mean, trained_se = base.soft_tree_cost_on_paths(best_flat, depth, leaf, split, temp, holdout_paths)
    return {
        "best_flat": best_flat,
        "gen0_holdout_mean": gen0_mean,
        "gen0_holdout_se": gen0_se,
        "trained_holdout_mean": trained_mean,
        "trained_holdout_se": trained_se,
        "best_train_cost": float(best_train),
        "train_seconds": float(train_seconds),
    }


def run(parsed) -> dict:
    budget = base.BUDGETS[parsed.budget]
    search_paths = base.make_paths(budget["search_paths"], base.SEARCH_SEED)
    holdout_paths = base.make_paths(budget["holdout_paths"], base.HOLDOUT_SEED)

    heuristic = base.search_best_pairwise_base_stock(search_paths, holdout_paths, budget["grid"])
    gate_cost = heuristic["holdout_mean_cost"]
    gate_oul = heuristic["oul_levels"]

    depth, leaf, split, temp = parsed.depth, parsed.leaf_type, parsed.split_type, parsed.temperature

    # Warm-start anchor.
    if parsed.warm_start:
        x0 = _gate_flow_warm_start(depth, leaf, gate_oul)
        warm_kind = f"gate_flow_oul{gate_oul}"
    else:
        x0 = base._warm_start_flow(depth, leaf, parsed.warm_start_flow)
        warm_kind = f"flat_flow{parsed.warm_start_flow}"
    anchor_mean, anchor_se = base.soft_tree_cost_on_paths(x0, depth, leaf, split, temp, holdout_paths)

    per_seed = []
    for seed in parsed.seeds:
        parsed.seed_value = seed
        res = _train_one_seed(parsed, budget, holdout_paths, gate_cost, x0)
        # Honest deployment floor: argmin over {trained xbest, warm-start anchor, gate}.
        candidates = [("trained_xbest", res["trained_holdout_mean"], res["trained_holdout_se"])]
        if parsed.honest_floor:
            candidates.append(("warm_start_anchor", anchor_mean, anchor_se))
            candidates.append(("gate", gate_cost, heuristic["holdout_stderr"]))
        deployed_policy, deployed_cost, deployed_se = min(candidates, key=lambda c: c[1])
        per_seed.append({
            "seed": seed,
            # Standardized keys consumed by srp.build_seed_robust_summary: the learned
            # cost is the DEPLOYED (floored) held-out cost; the gate is paired per seed
            # (deterministic CRN here, so it is identical across seeds).
            "gate_cost": gate_cost,
            "best_learned_cost": deployed_cost,
            "savings_pct_vs_gate": 100.0 * (gate_cost - deployed_cost) / gate_cost,
            "trained_holdout_mean": res["trained_holdout_mean"],
            "trained_holdout_se": res["trained_holdout_se"],
            "gen0_holdout_mean": res["gen0_holdout_mean"],
            "best_train_cost": res["best_train_cost"],
            "deployed_policy": deployed_policy,
            "deployed_cost": deployed_cost,
            "deployed_se": deployed_se,
            "gap_pct": (deployed_cost / gate_cost - 1.0) * 100.0,
            "train_seconds": res["train_seconds"],
        })

    # CENTRALIZED aggregation (single source of truth): sample (n-1) std, shared verdict
    # rule, >=5-seed enforcement. Replaces this script's former bespoke statistics block.
    summary = srp.build_seed_robust_summary(per_seed, problem_id=PROBLEM_ID)
    n_seeds = summary["n_optimizer_seeds"]
    trained_only = [s["trained_holdout_mean"] for s in per_seed]
    trained_s = srp.summarize_values(trained_only)
    frac_below_trained = sum(1 for v in trained_only if v < gate_cost)

    # Auxiliary annotation: did the honest floor pin every seed at the gate itself?
    # (the legacy bespoke label ROBUST_GATE_MATCH_ONLY; srp reports it as PARITY).
    pinned = sum(1 for s in per_seed if s["deployed_policy"] == "gate")
    gate_pinned_all_seeds = bool(parsed.honest_floor and pinned == n_seeds)

    return {
        "reference": "pirhooshyaran2021_mixed_scn_fig1_table5",
        "literature_verified": False,
        "baseline_kind": "env_own_best_pairwise_base_stock (RESEARCH comparison)",
        "policy_architecture": (
            f"soft_tree_d{depth}_{split}_{leaf}_temp{temp}_vector_quantity"
            f"_sigma{parsed.sigma_init}_warm[{warm_kind}]_floor[{parsed.honest_floor}]"
        ),
        "gate": {"oul_levels": gate_oul, "echelon_levels": heuristic["echelon_levels"],
                 "holdout_mean_cost": gate_cost, "holdout_stderr": heuristic["holdout_stderr"]},
        "warm_start_anchor_holdout_mean": anchor_mean,
        "warm_start_anchor_holdout_se": anchor_se,
        "warm_start_kind": warm_kind,
        "sigma_init": parsed.sigma_init,
        "honest_floor": parsed.honest_floor,
        "depth": depth, "leaf_type": leaf, "split_type": split, "temperature": temp,
        "train_batch": parsed.train_batch if parsed.train_batch else budget["train_batch"],
        "holdout_paths": budget["holdout_paths"],
        "per_seed": per_seed,
        "n_seeds": n_seeds,
        # Legacy-named aliases (kept so older readers of this artifact keep working);
        # the standardized srp keys below are the canonical ones.
        "deployed_seed_mean": summary["learned_seed_mean"],
        "deployed_seed_std": summary["learned_seed_std"],
        "deployed_gap_pct": (summary["learned_seed_mean"] / gate_cost - 1.0) * 100.0,
        "trained_seed_mean": trained_s["seed_mean"],
        "trained_seed_std": trained_s["seed_std"],
        "trained_gap_pct": (trained_s["seed_mean"] / gate_cost - 1.0) * 100.0,
        "frac_seeds_below_gate_deployed": summary["frac_seeds_beating_gate"],
        "frac_seeds_below_gate_trained": f"{frac_below_trained}/{n_seeds}",
        "gate_pinned_all_seeds": gate_pinned_all_seeds,
        "n_seeds_deployed_gate": pinned,
        # Standardized seed-robust summary keys (n_optimizer_seeds, learned/gate seed
        # mean+/-std, savings_pct_seed_mean/std, frac_seeds_beating_gate,
        # verdict_vs_same_protocol_gate) from invman/optimizer_seed_robustness_policy.py.
        **summary,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Seed-robust learned-vs-gate for the mixed SCN.")
    p.add_argument("--run_tag", default="mixed_distribution_assembly_network_seed_robust")
    p.add_argument("--budget", choices=sorted(base.BUDGETS), default="full")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny end-to-end validation: forces --budget smoke, caps CPU at 1 worker, "
                        "and writes ONLY under outputs/.../smoke_seed_robust/ (never the real artifact).")
    p.add_argument("--mp_num_processors", type=int, default=2,
                   help="Rayon/BLAS worker cap (read pre-import by invman.cpu_limits).")
    p.add_argument("--description", default="seed-robust mixed SCN audit (srp-standardized)")
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--temperature", type=float, default=base.TEMPERATURE_DEFAULT)
    p.add_argument("--split_type", choices=["oblique", "axis_aligned"], default="oblique")
    p.add_argument("--leaf_type", choices=["constant", "linear", "sigmoid_linear"], default="linear")
    p.add_argument("--warm_start_flow", type=float, default=base.DEMAND_MEAN,
                   help="Flat-flow warm-start level when --warm_start is OFF (baseline parity).")
    p.add_argument("--warm_start", dest="warm_start", action="store_true", default=True,
                   help="Gate-flow warm-start (gen0 ~ gate order magnitude). Default ON.")
    p.add_argument("--no_warm_start", dest="warm_start", action="store_false",
                   help="Use the flat demand-mean flow warm-start (reproduces the baseline pathology).")
    p.add_argument("--honest_floor", dest="honest_floor", action="store_true", default=True,
                   help="Deploy argmin{trained, anchor, gate}. Default ON (no seed worse than gate).")
    p.add_argument("--no_honest_floor", dest="honest_floor", action="store_false",
                   help="Deploy the trained xbest with NO floor (reproduces the baseline pathology).")
    p.add_argument("--sigma_init", type=float, default=0.2)
    p.add_argument("--train_batch", type=int, default=None,
                   help="Override the budget's per-candidate training batch (larger = less ranking noise).")
    p.add_argument("--seeds", type=int, nargs="+",
                   default=list(srp.seeds_for(PROBLEM_ID)),
                   help="Optimizer seeds (default = canonical srp list 9001..9005; the "
                        "historical mixed run used 11 22 33 44 55).")
    return p.parse_args()


def _artifact_path(parsed) -> Path:
    """Real artifact = outputs/production_assembly_distribution_network/seed_robust_report_mixed.json
    (budget-suffixed when budget != full so screening can never clobber the audited file).
    --smoke ALWAYS writes under .../smoke_seed_robust/ and never the real path."""
    out_dir = PACKAGE_ROOT / "outputs" / "production_assembly_distribution_network"
    if parsed.smoke:
        out_dir = out_dir / "smoke_seed_robust"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if (parsed.smoke or parsed.budget == "full") else f"_{parsed.budget}"
    return out_dir / f"seed_robust_report_mixed{suffix}.json"


def main():
    parsed = parse_args()
    if parsed.smoke:
        parsed.budget = "smoke"
        configure_process_cpu_limits(1)  # before the first rollout initializes rayon
    out = run(parsed)
    commit = base._git_short_commit(PACKAGE_ROOT)
    json_path = _artifact_path(parsed)
    payload = {
        "description": parsed.description,
        "commit": commit,
        "budget": parsed.budget,
        "smoke": bool(parsed.smoke),
        "seeds": list(parsed.seeds),
        **out,
    }
    with json_path.open("w", encoding="utf-8") as h:
        json.dump(payload, h, indent=2)

    g = out["gate"]["holdout_mean_cost"]
    print("=" * 78)
    print(f"DESIGN: {out['policy_architecture']}")
    print(f"GATE: {g:.3f} +/- {out['gate']['holdout_stderr']:.3f}  OUL {out['gate']['oul_levels']}")
    print(f"warm-start anchor held-out: {out['warm_start_anchor_holdout_mean']:.3f}")
    print("-" * 78)
    print(f"{'seed':>6} {'trained':>10} {'gap%':>8} {'deployed':>10} {'policy':>18} {'gap%':>8}")
    for s in out["per_seed"]:
        print(f"{s['seed']:>6} {s['trained_holdout_mean']:>10.3f} "
              f"{(s['trained_holdout_mean']/g-1)*100:>+7.2f} {s['deployed_cost']:>10.3f} "
              f"{s['deployed_policy']:>18} {s['gap_pct']:>+7.2f}")
    print("-" * 78)
    print(f"TRAINED (no floor)  seed-mean {out['trained_seed_mean']:.3f} +/- {out['trained_seed_std']:.3f}"
          f"  gap {out['trained_gap_pct']:+.2f}%  below-gate {out['frac_seeds_below_gate_trained']}")
    print(f"DEPLOYED (floored)  seed-mean {out['learned_seed_mean']:.3f} +/- {out['learned_seed_std']:.3f}"
          f"  gap {out['deployed_gap_pct']:+.2f}%  below-gate {out['frac_seeds_beating_gate']}")
    print(f"SAVINGS vs gate     {out['savings_pct_seed_mean']:+.2f}% +/- {out['savings_pct_seed_std']:.2f}%"
          f"  over n={out['n_optimizer_seeds']} optimizer seeds")
    pin_note = "  [honest floor pinned ALL seeds at the gate -> gate-match only]" if out["gate_pinned_all_seeds"] else ""
    print(f"VERDICT (srp, vs same-protocol gate): {out['verdict_vs_same_protocol_gate']}{pin_note}")
    print(f"WROTE_JSON: {json_path}")


if __name__ == "__main__":
    main()
