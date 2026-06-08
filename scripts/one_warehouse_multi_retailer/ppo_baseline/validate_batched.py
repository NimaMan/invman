#!/usr/bin/env python
"""Validate the batched numpy env (batched_env.BatchedOWMR) against the scalar
instrument (instrument.rollout_decompose) on the gate echelon base-stock policy, over
the SAME holdout demand paths. Mean cost must agree to MC noise (independent emergency
RNG averaged over many paths). Also cross-check vs the Rust gate holdout mean.
"""
import sys, json
import numpy as np

PKG = "/home/nima/code/ml/invman"
OWMR = PKG + "/scripts/one_warehouse_multi_retailer"
for p in (PKG, OWMR, "/tmp/owmr_diag", "/tmp/owmr_ppo"):
    if p not in sys.path:
        sys.path.insert(0, p)

import common
import run_asymmetric_learned_vs_gate as R
from benchmark_learned_vs_heuristic import _sample_demand_paths, _heuristic_on_paths
import instrument as I
from batched_env import BatchedOWMR

NP = 1024
ref = common.get_reference("kaynov2024_instance_14")
ist = common.benchmark_initial_state(ref)

# gate levels from the soft-tree retrain artifact (proportional argmin)
st = json.load(open("/tmp/owmr_ppo/softtree_5seed_params.json")) if __import__("os").path.exists("/tmp/owmr_ppo/softtree_5seed_params.json") else None
if st:
    gW = st["gate_warehouse_level"]; gR = st["gate_retailer_levels"]
else:
    art = json.load(open("/tmp/owmr_diag/artifacts_full.json"))
    gW = art["gate_warehouse_level"]; gR = art["gate_retailer_levels"]
print(f"gate W={gW} R={gR}")

# holdout demand paths (CRN, same as protocol)
holdout = _sample_demand_paths(ref, NP, R.HOLDOUT_SEED_START)  # list of (T x K)
T = len(holdout[0]); K = len(holdout[0][0])
demands = np.asarray(holdout, dtype=np.int64)  # (NP, T, K)
print(f"demands {demands.shape}")

# ---- Rust gate holdout mean (reference truth) ----
rust = _heuristic_on_paths(ref, gW, gR, "proportional", holdout, R.ALLOC_SEED_HOLDOUT)
print(f"RUST gate mean {rust.mean():.2f} std {rust.std():.1f}")

# ---- scalar instrument gate mean ----
art = {"ref": ref, "ist": ist, "min_values": [0]*(K+1)}
rng = np.random.default_rng(12345)
sc = []
for path in holdout:
    rec = I.rollout_decompose(art, path, "proportional", rng,
                              action_fn=lambda s: (gW, list(gR)))
    sc.append(I.total_cost(rec))
sc = np.array(sc)
print(f"SCALAR-INSTR gate mean {sc.mean():.2f} std {sc.std():.1f}  diff vs rust {sc.mean()-rust.mean():+.2f} ({(sc.mean()-rust.mean())/rust.mean()*100:+.3f}%)")

# ---- batched env gate mean ----
env = BatchedOWMR(ref, ist, NP, emerg_seed=12345, allocation="proportional")
env.set_demands(demands)
env.reset()
total = np.zeros(NP)
for t in range(T):
    wh_order, ret_orders = env.echelon_orders(gW, np.asarray(gR))
    reward, cost = env.step(wh_order, ret_orders)
    total += cost
print(f"BATCHED gate mean {total.mean():.2f} std {total.std():.1f}  diff vs scalar {total.mean()-sc.mean():+.2f} ({(total.mean()-sc.mean())/sc.mean()*100:+.3f}%)  diff vs rust {(total.mean()-rust.mean())/rust.mean()*100:+.3f}%")

# paired check (same emergency draws would require same RNG order; here independent RNG,
# so compare distribution means). Use many seeds to tighten:
means = []
for sd in range(8):
    env = BatchedOWMR(ref, ist, NP, emerg_seed=1000+sd, allocation="proportional")
    env.set_demands(demands); env.reset()
    tot = np.zeros(NP)
    for t in range(T):
        wo, ro = env.echelon_orders(gW, np.asarray(gR)); _, c = env.step(wo, ro); tot += c
    means.append(tot.mean())
means = np.array(means)
print(f"BATCHED gate mean over 8 emerg-seeds {means.mean():.2f} +/- {means.std():.2f}")
