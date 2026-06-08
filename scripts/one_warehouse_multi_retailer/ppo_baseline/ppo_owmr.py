#!/usr/bin/env python
"""From-scratch PyTorch PPO for OWMR instance_14, trained & scored on the validated
batched instrument env (batched_env.BatchedOWMR). FAIR, COMPETENT baseline for the
head-to-head vs our CMA-ES soft-tree.

================================ ALGORITHM ===================================
Actor (Kaynov-style multi-discrete, heads LINEAR in K):
  - shared MLP trunk over the RAW network state (warehouse inv + pipeline + per-retailer
    inv + pipelines + positions + remaining horizon), standardized by a running normalizer.
  - K+1 = 11 independent categorical heads (warehouse order + one order head per retailer).
    Head j is a Categorical over {0..max_j} order quantities (max_j = soft-tree max_values).
    The joint action prob = product of per-head probs (factorized multi-discrete).
  - Value head: scalar baseline V(s).

Feasibility (matches Kaynov + our RandomSequential): the env resolves infeasible joint
allocations (sum of retailer orders > warehouse release capacity) with RANDOM-SEQUENTIAL
rationing, so any sampled joint action is feasible. The actor is NOT penalized for
over-ordering; the env clips via rationing. (This is exactly what Kaynov reports makes DRL
trainable on OWMR.)

Reward = -period_cost / REWARD_SCALE (per-step). GAE(lambda) advantages, clipped PPO
surrogate, value-function clipping, entropy bonus. Rollouts are fully vectorized over B
parallel demand paths (one PPO "episode batch" = B paths x T=100 steps).

WARM-START (behavior cloning): before PPO, the actor is BC-trained to imitate the tuned
gate echelon base-stock order vector on states visited by the gate policy, so PPO starts
from a strong, competitive policy (fair to PPO). The value head is then warmed by Monte-Carlo
returns of the gate policy.

Scoring: GREEDY (argmax) actor rolled out on the HOLDOUT CRN demand block (R.HOLDOUT_SEED_START,
>=1024 paths) under PROPORTIONAL rationing (the protocol's eval rule, same as the soft-tree
holdout score), on the SAME batched instrument. -> instrument-scored PPO cost X.
=============================================================================
"""
import sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(2)

PKG = "/home/nima/code/ml/invman"
OWMR = PKG + "/scripts/one_warehouse_multi_retailer"
for p in (PKG, OWMR, "/tmp/owmr_diag", "/tmp/owmr_ppo"):
    if p not in sys.path:
        sys.path.insert(0, p)

import common
import run_asymmetric_learned_vs_gate as R
from benchmark_learned_vs_heuristic import _sample_demand_paths
from batched_env import BatchedOWMR

DEVICE = "cpu"
REWARD_SCALE = 1000.0  # period costs ~ few hundred; scale rewards to O(1)


# ----------------------------- running normalizer ----------------------------
class RunningNorm:
    def __init__(self, dim):
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)
        self.count = 1e-4

    def update(self, x):  # x: (N, dim)
        bmean = x.mean(axis=0); bvar = x.var(axis=0); bn = x.shape[0]
        delta = bmean - self.mean
        tot = self.count + bn
        self.mean += delta * bn / tot
        m_a = self.var * self.count
        m_b = bvar * bn
        M2 = m_a + m_b + delta**2 * self.count * bn / tot
        self.var = M2 / tot
        self.count = tot

    def norm(self, x):
        return (x - self.mean) / np.sqrt(self.var + 1e-8)


# ------------------------------- actor-critic --------------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, head_sizes, hidden=128):
        super().__init__()
        self.head_sizes = head_sizes  # list of #categories per head (K+1 heads)
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        # heads LINEAR in K: one Linear(hidden -> size_j) per head
        self.heads = nn.ModuleList([nn.Linear(hidden, s) for s in head_sizes])
        self.value = nn.Linear(hidden, 1)
        # small init for the last layer of trunk & heads (stable start)
        for h in self.heads:
            nn.init.orthogonal_(h.weight, gain=0.01); nn.init.zeros_(h.bias)
        nn.init.orthogonal_(self.value.weight, gain=1.0); nn.init.zeros_(self.value.bias)

    def forward(self, obs):
        z = self.trunk(obs)
        logits = [head(z) for head in self.heads]
        v = self.value(z).squeeze(-1)
        return logits, v

    def act(self, obs, greedy=False):
        logits, v = self.forward(obs)
        actions, logps, ents = [], [], []
        for lg in logits:
            dist = torch.distributions.Categorical(logits=lg)
            a = lg.argmax(dim=-1) if greedy else dist.sample()
            actions.append(a)
            logps.append(dist.log_prob(a))
            ents.append(dist.entropy())
        actions = torch.stack(actions, dim=-1)        # (B, K+1)
        logp = torch.stack(logps, dim=-1).sum(-1)     # (B,)
        ent = torch.stack(ents, dim=-1).sum(-1)       # (B,)
        return actions, logp, ent, v

    def evaluate(self, obs, actions):
        logits, v = self.forward(obs)
        logps, ents = [], []
        for j, lg in enumerate(logits):
            dist = torch.distributions.Categorical(logits=lg)
            logps.append(dist.log_prob(actions[:, j]))
            ents.append(dist.entropy())
        logp = torch.stack(logps, dim=-1).sum(-1)
        ent = torch.stack(ents, dim=-1).sum(-1)
        return logp, ent, v


# ------------------------------- helpers -------------------------------------
def build_env(ref, ist, B, emerg_seed, allocation):
    env = BatchedOWMR(ref, ist, B, emerg_seed=emerg_seed, allocation=allocation)
    return env


def actions_to_orders(actions, max_values):
    """actions (B, K+1) integer categories -> (wh_order (B,), ret_orders (B,K)). Category
    index == order quantity (heads are over {0..max_j})."""
    a = actions.detach().cpu().numpy()
    wh_order = a[:, 0]
    ret_orders = a[:, 1:]
    return wh_order, ret_orders


# ------------------------------- BC warm-start -------------------------------
def collect_gate_states(ref, ist, gW, gR, n_paths, seed_start, emerg_seed, T):
    """Roll the gate policy through the batched env, recording (raw_obs, gate_order_vector)
    at each step for behavior cloning. Returns obs (N, F), targets (N, K+1) ints, and
    MC return-to-go (N,) for value warm-start."""
    paths = _sample_demand_paths(ref, n_paths, seed_start)
    demands = np.asarray(paths, dtype=np.int64)
    env = build_env(ref, ist, n_paths, emerg_seed, allocation="proportional")
    env.set_demands(demands); env.reset()
    obs_list, tgt_list, cost_list = [], [], []
    for t in range(T):
        raw = env.observe_raw()
        wh_order, ret_orders = env.echelon_orders(gW, np.asarray(gR))
        tgt = np.concatenate([wh_order[:, None], ret_orders], axis=1)
        obs_list.append(raw); tgt_list.append(tgt)
        _, cost = env.step(wh_order, ret_orders)
        cost_list.append(cost)
    obs = np.concatenate(obs_list, axis=0)
    tgt = np.concatenate(tgt_list, axis=0)
    # MC return-to-go per (path,t): sum of future -cost/scale, gamma=1 (undiscounted protocol)
    costs = np.stack(cost_list, axis=0)  # (T, B)
    rtg = np.zeros_like(costs)
    acc = np.zeros(costs.shape[1])
    for t in range(T - 1, -1, -1):
        acc = -costs[t] / REWARD_SCALE + acc
        rtg[t] = acc
    rtg = rtg.reshape(-1)
    return obs, tgt.astype(np.int64), rtg


def behavior_clone(ac, norm, obs, tgt, rtg, max_values, epochs, lr, batch, log):
    obs_n = norm.norm(obs).astype(np.float32)
    # clip targets to head ranges (gate orders can exceed max for warehouse rarely; clip)
    for j, mx in enumerate(max_values):
        tgt[:, j] = np.clip(tgt[:, j], 0, mx)
    obs_t = torch.tensor(obs_n)
    tgt_t = torch.tensor(tgt)
    rtg_t = torch.tensor(rtg.astype(np.float32))
    opt = torch.optim.Adam(ac.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    N = obs_t.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(N)
        tot = 0.0; vtot = 0.0; nb = 0
        for i in range(0, N, batch):
            idx = perm[i:i+batch]
            logits, v = ac.forward(obs_t[idx])
            loss = sum(ce(logits[j], tgt_t[idx, j]) for j in range(len(logits)))
            vloss = ((v - rtg_t[idx])**2).mean()
            opt.zero_grad(); (loss + 0.5 * vloss).backward(); opt.step()
            tot += loss.item(); vtot += vloss.item(); nb += 1
        if log and (ep % max(1, epochs // 5) == 0 or ep == epochs - 1):
            print(f"  [BC] ep {ep} ce={tot/nb:.4f} vloss={vtot/nb:.4f}", flush=True)


# ------------------------------- rollout -------------------------------------
def rollout(ac, norm, env, demands, T, max_values, greedy=False, update_norm=True,
            ration_rng=None):
    """Collect a vectorized rollout. Returns dict of tensors for PPO + total episode cost."""
    B = demands.shape[0]
    env.set_demands(demands); env.reset()
    obs_buf = np.zeros((T, B, ac.trunk[0].in_features), dtype=np.float32)
    act_buf = np.zeros((T, B, len(max_values)), dtype=np.int64)
    logp_buf = np.zeros((T, B), dtype=np.float32)
    rew_buf = np.zeros((T, B), dtype=np.float32)
    val_buf = np.zeros((T, B), dtype=np.float32)
    cost_total = np.zeros(B)
    for t in range(T):
        raw = env.observe_raw()
        if update_norm:
            norm.update(raw)
        obs_n = norm.norm(raw).astype(np.float32)
        obs_buf[t] = obs_n
        with torch.no_grad():
            ot = torch.tensor(obs_n)
            actions, logp, ent, v = ac.act(ot, greedy=greedy)
        wh_order, ret_orders = actions_to_orders(actions, max_values)
        _, cost = env.step(wh_order, ret_orders, ration_rng=ration_rng)
        act_buf[t] = actions.cpu().numpy()
        logp_buf[t] = logp.cpu().numpy()
        val_buf[t] = v.cpu().numpy()
        rew_buf[t] = -cost / REWARD_SCALE
        cost_total += cost
    return {
        "obs": obs_buf, "act": act_buf, "logp": logp_buf,
        "rew": rew_buf, "val": val_buf,
    }, cost_total


def compute_gae(rew, val, gamma, lam):
    T, B = rew.shape
    adv = np.zeros((T, B), dtype=np.float32)
    lastgae = np.zeros(B, dtype=np.float32)
    next_val = np.zeros(B, dtype=np.float32)  # terminal value = 0 (finite horizon)
    for t in range(T - 1, -1, -1):
        nonterminal = 1.0 if t < T - 1 else 0.0
        delta = rew[t] + gamma * next_val * nonterminal - val[t]
        lastgae = delta + gamma * lam * nonterminal * lastgae
        adv[t] = lastgae
        next_val = val[t]
    ret = adv + val
    return adv, ret


# ------------------------------- PPO train -----------------------------------
def train_ppo(args, log_prefix=""):
    ref = common.get_reference("kaynov2024_instance_14")
    ist = common.benchmark_initial_state(ref)
    K = len(ref["holding_cost_retailers"])
    T = int(ref["benchmark_periods"])
    max_values = [int(v) for v in args.max_values]
    head_sizes = [m + 1 for m in max_values]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    obs_dim = 1 + 2 + 1 + K + K * 2 + K + 1  # observe_raw layout for wh_L=2, ret_L=2, K=10
    ac = ActorCritic(obs_dim, head_sizes, hidden=args.hidden).to(DEVICE)
    norm = RunningNorm(obs_dim)

    # ---------------- warm-start: BC to the gate ----------------
    curve = []
    if args.bc_epochs > 0:
        gW, gR = args.gate_W, args.gate_R
        bc_obs, bc_tgt, bc_rtg = collect_gate_states(
            ref, ist, gW, gR, args.bc_paths, R.SEARCH_SEED_START + 50000,
            emerg_seed=args.seed * 7 + 11, T=T)
        norm.update(bc_obs)
        behavior_clone(ac, norm, bc_obs, bc_tgt, bc_rtg, max_values,
                       args.bc_epochs, args.bc_lr, args.bc_batch, log=args.verbose)

    # ---------------- holdout block for tracking (CRN, proportional) ----------------
    holdout_paths = _sample_demand_paths(ref, args.eval_paths, R.HOLDOUT_SEED_START)
    holdout_demands = np.asarray(holdout_paths, dtype=np.int64)
    eval_env = build_env(ref, ist, args.eval_paths, emerg_seed=20240601, allocation="proportional")

    def evaluate_greedy():
        _, cost = rollout(ac, norm, eval_env, holdout_demands, T, max_values,
                          greedy=True, update_norm=False)
        return float(cost.mean()), float(cost.std())

    bc_eval_mean, _ = evaluate_greedy()
    if args.verbose:
        print(f"{log_prefix}[after BC] greedy holdout cost {bc_eval_mean:.2f}", flush=True)
    curve.append({"iter": 0, "phase": "bc", "holdout_greedy_cost": bc_eval_mean})

    # ---------------- PPO loop ----------------
    opt = torch.optim.Adam(ac.parameters(), lr=args.lr)
    train_env = build_env(ref, ist, args.train_paths, emerg_seed=args.seed * 13 + 5,
                          allocation=args.train_alloc)
    ration_rng = np.random.default_rng(args.seed * 17 + 3)
    train_rng = np.random.RandomState(args.seed + 12345)

    best_eval = bc_eval_mean
    best_state = {k: v.detach().clone() for k, v in ac.state_dict().items()}
    for it in range(1, args.iters + 1):
        # fresh training demand paths each iteration (avoid overfitting a fixed block)
        seed_it = R.TRAIN_SEED_START + it * 1000 + args.seed
        train_paths = _sample_demand_paths(ref, args.train_paths, seed_it)
        train_demands = np.asarray(train_paths, dtype=np.int64)
        batch, train_cost = rollout(ac, norm, train_env, train_demands, T, max_values,
                                    greedy=False, update_norm=True, ration_rng=ration_rng)
        adv, ret = compute_gae(batch["rew"], batch["val"], args.gamma, args.lam)
        # flatten
        B = train_demands.shape[0]
        obs = torch.tensor(batch["obs"].reshape(T * B, -1))
        act = torch.tensor(batch["act"].reshape(T * B, -1))
        old_logp = torch.tensor(batch["logp"].reshape(T * B))
        adv_t = torch.tensor(adv.reshape(T * B))
        ret_t = torch.tensor(ret.reshape(T * B))
        val_old = torch.tensor(batch["val"].reshape(T * B))
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        N = T * B
        for epoch in range(args.ppo_epochs):
            perm = torch.randperm(N)
            for i in range(0, N, args.minibatch):
                idx = perm[i:i+args.minibatch]
                logp, ent, v = ac.evaluate(obs[idx], act[idx])
                ratio = torch.exp(logp - old_logp[idx])
                surr1 = ratio * adv_t[idx]
                surr2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv_t[idx]
                pg_loss = -torch.min(surr1, surr2).mean()
                # value clipping
                v_clip = val_old[idx] + torch.clamp(v - val_old[idx], -args.clip, args.clip)
                vf1 = (v - ret_t[idx])**2
                vf2 = (v_clip - ret_t[idx])**2
                v_loss = 0.5 * torch.max(vf1, vf2).mean()
                loss = pg_loss + args.vf_coef * v_loss - args.ent_coef * ent.mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), args.max_grad_norm)
                opt.step()

        if it % args.eval_every == 0 or it == args.iters:
            ev_mean, ev_std = evaluate_greedy()
            curve.append({"iter": it, "phase": "ppo", "train_cost": float(train_cost.mean()),
                          "holdout_greedy_cost": ev_mean})
            if ev_mean < best_eval:
                best_eval = ev_mean
                best_state = {k: v.detach().clone() for k, v in ac.state_dict().items()}
            if args.verbose:
                print(f"{log_prefix}[ppo it {it}] train {train_cost.mean():.1f} "
                      f"holdout-greedy {ev_mean:.1f} (best {best_eval:.1f})", flush=True)

    # restore best
    ac.load_state_dict(best_state)
    final_mean, final_std = evaluate_greedy()
    return ac, norm, curve, best_eval, final_mean, final_std, max_values


def default_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=120)
    p.add_argument("--train_paths", type=int, default=256)
    p.add_argument("--eval_paths", type=int, default=1024)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ppo_epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=2048)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--ent_coef", type=float, default=0.003)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--bc_epochs", type=int, default=30)
    p.add_argument("--bc_paths", type=int, default=256)
    p.add_argument("--bc_lr", type=float, default=1e-3)
    p.add_argument("--bc_batch", type=int, default=2048)
    p.add_argument("--train_alloc", default="random_sequential")
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--verbose", type=int, default=1)
    return p


if __name__ == "__main__":
    # quick smoke run (1 seed, short) when invoked directly
    parser = default_args()
    a = parser.parse_args()
    a.max_values = [255, 85, 75, 65, 55, 45, 30, 6, 18, 43, 54]
    a.gate_W = 440
    a.gate_R = [33, 30, 28, 26, 27, 30, 2, 10, 29, 39]
    t0 = time.time()
    ac, norm, curve, best, fmean, fstd, mv = train_ppo(a)
    print(f"DONE seed {a.seed}: best holdout-greedy {best:.2f} final {fmean:.2f} ({time.time()-t0:.0f}s)")
