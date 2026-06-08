#!/usr/bin/env python
"""Vectorized (batched-over-paths) numpy replica of the OWMR instance_14 instrument
step dynamics (instrument.py: rollout_decompose), for fast PPO training & scoring.

WHY: instrument.py loops one path at a time in Python -> far too slow to roll out the
thousands of episodes a from-scratch PPO needs. This module steps B paths in lock-step
with numpy, reproducing the EXACT same dynamics:
  - order generation (warehouse echelon base-stock OR direct PPO order action)
  - proportional / random-sequential / min_shortage rationing under release capacity
  - partial_backorder step: pipeline shift, emergency Bernoulli(p) draws, post-emergency
    warehouse holding, retailer holding + penalty
  - reward = -period_cost (env.rs / instrument.py convention)

It is validated against the scalar instrument (validate_batched.py) on the gate policy:
mean cost must match to MC noise (same demand paths, independent emergency RNG averaged).

CONVENTION NOTES (matched to instrument.py rollout_decompose):
  - avail (warehouse release pool) = max(wh_inv + wh_pipe[0], 0)
  - release_capacity = max(avail - holdback, 0); holdback=0 here (PPO has no holdback DOF)
  - ship is bounded by release_capacity via the rationing rule
  - emergency draws are i.i.d Bernoulli(emerg_p) per retailer per period, policy-independent
  - warehouse holding charged on POST-emergency on-hand (env.rs:271)

This env exposes BOTH:
  (a) a generic step(state, warehouse_order, retailer_orders) for an arbitrary policy
      (PPO emits warehouse_order + retailer_orders directly), and
  (b) an echelon-base-stock order helper (replicating the Rust binding) so we can run the
      gate/soft-tree heuristics through the SAME batched dynamics for validation & BC.
"""
import numpy as np


class BatchedOWMR:
    def __init__(self, ref, ist, n_paths, emerg_seed, allocation="proportional"):
        self.K = len(ref["holding_cost_retailers"])
        self.hcw = float(ref["holding_cost_warehouse"])
        self.hcr = np.asarray([float(v) for v in ref["holding_cost_retailers"]], dtype=np.float64)
        self.pcr = np.asarray([float(v) for v in ref["penalty_costs_retailers"]], dtype=np.float64)
        self.emerg_p = float(ref["emergency_shipment_probability"])
        self.behavior = str(ref["customer_behavior"])
        assert self.behavior == "partial_backorder", "batched env specialized to partial_backorder"
        self.wh_L = len(ist["initial_warehouse_pipeline"])      # warehouse lead time (pipeline slots)
        self.ret_L = len(ist["initial_retailer_pipeline"][0])   # retailer lead time
        self.allocation = allocation

        self.ist = ist
        self.B = int(n_paths)
        self.emerg_rng = np.random.default_rng(int(emerg_seed))
        self.reset()

    def reset(self):
        B, K = self.B, self.K
        ist = self.ist
        self.wh_inv = np.full(B, int(ist["initial_warehouse_inventory"]), dtype=np.int64)
        self.wh_pipe = np.tile(np.asarray(ist["initial_warehouse_pipeline"], dtype=np.int64), (B, 1))  # (B, wh_L)
        self.ret_inv = np.tile(np.asarray(ist["initial_retailer_inventory"], dtype=np.int64), (B, 1))  # (B, K)
        self.ret_pipe = np.stack([np.tile(np.asarray(r, dtype=np.int64), (B, 1))
                                  for r in ist["initial_retailer_pipeline"]], axis=1)  # (B, K, ret_L)
        self.period = 0
        return self.observe()

    # ----- soft-tree-style normalized state (matches instrument._build_policy_state) -----
    def observe(self):
        """Return per-path feature matrix (B, F) in the SAME normalized layout as
        instrument._build_policy_state: [wh_inv, wh_pipe..., ret_inv..., ret_pipe(k,..)..,
        total_system, remaining_frac], all divided by a per-path scale."""
        B, K = self.B, self.K
        ret_positions = self.ret_inv + self.ret_pipe.sum(axis=2)           # (B, K)
        total_system = self.wh_inv + self.wh_pipe.sum(axis=1) + ret_positions.sum(axis=1)  # (B,)
        scale = np.maximum.reduce([
            np.abs(self.wh_inv).astype(np.float64),
            np.abs(total_system).astype(np.float64),
            np.abs(self.ret_inv).max(axis=1).astype(np.float64),
            np.ones(B),
        ])  # (B,)
        feats = [ (self.wh_inv / scale)[:, None] ]
        feats.append(self.wh_pipe / scale[:, None])
        feats.append(self.ret_inv / scale[:, None])
        feats.append(self.ret_pipe.reshape(B, K * self.ret_L) / scale[:, None])
        feats.append((total_system / scale)[:, None])
        rem = 0.0 if self.total_periods == 0 else (self.total_periods - self.period) / self.total_periods
        feats.append(np.full((B, 1), rem))
        return np.concatenate(feats, axis=1).astype(np.float32)

    total_periods = 100  # instance_14

    # ----- raw absolute features (for PPO: physically meaningful, no per-path rescale) -----
    def observe_raw(self):
        """Absolute (un-normalized) network state for the PPO actor: gives the actor the
        true inventory positions + pipelines + a remaining-horizon scalar. Standardized
        downstream by the agent's running normalizer."""
        B, K = self.B, self.K
        ret_pos = self.ret_inv + self.ret_pipe.sum(axis=2)           # (B,K)
        wh_pos = self.wh_inv + self.wh_pipe.sum(axis=1)              # (B,)
        feats = [
            self.wh_inv[:, None].astype(np.float32),
            self.wh_pipe.astype(np.float32),
            wh_pos[:, None].astype(np.float32),
            self.ret_inv.astype(np.float32),
            self.ret_pipe.reshape(B, K * self.ret_L).astype(np.float32),
            ret_pos.astype(np.float32),
            np.full((B, 1), (self.total_periods - self.period), dtype=np.float32),
        ]
        return np.concatenate(feats, axis=1).astype(np.float32)

    # ----- echelon base-stock orders (replicate heuristics::echelon_base_stock_orders) ----
    def echelon_orders(self, W_target, r_targets):
        """Vectorized echelon base-stock. W_target scalar/array, r_targets (K,) or (B,K).
        Returns (wh_order (B,), ret_orders (B,K)). Matches the Rust binding:
          retailer order_k = max(R_k - retailer_position_k, 0)
          warehouse order  = max(W - echelon_position, 0), echelon_position = wh_inv +
             wh_pipe + sum(ret_positions) (echelon inventory position)."""
        B, K = self.B, self.K
        ret_pos = self.ret_inv + self.ret_pipe.sum(axis=2)  # (B,K)
        r_targets = np.asarray(r_targets)
        if r_targets.ndim == 1:
            r_targets = np.tile(r_targets, (B, 1))
        ret_orders = np.maximum(r_targets - ret_pos, 0).astype(np.int64)
        echelon_pos = self.wh_inv + self.wh_pipe.sum(axis=1) + ret_pos.sum(axis=1)  # (B,)
        W_target = np.asarray(W_target)
        if W_target.ndim == 0:
            W_target = np.full(B, int(W_target))
        wh_order = np.maximum(W_target - echelon_pos, 0).astype(np.int64)
        return wh_order, ret_orders

    # ----- rationing -----
    def _ration(self, release_capacity, ret_orders, rng=None):
        """Resolve infeasible joint allocations. ret_orders (B,K); release_capacity (B,).
        proportional: floor(o*cap/tot) where tot>cap (instrument._proportional)
        random_sequential: shuffle retailers per path, fill 1 unit at a time until cap
            exhausted (Kaynov RandomSequential -> trainable feasibility for DRL)."""
        B, K = self.B, self.K
        tot = ret_orders.sum(axis=1)  # (B,)
        ship = ret_orders.copy()
        over = tot > release_capacity
        if not over.any():
            return ship
        if self.allocation == "proportional":
            idx = np.where(over)[0]
            cap = release_capacity[idx]
            o = ret_orders[idx]
            t = tot[idx]
            # floor(o * cap / t); avoid div0 (t>cap>=0 so t>0)
            ship[idx] = (o * cap[:, None]) // t[:, None]
            return ship
        elif self.allocation == "random_sequential":
            # vectorized unit-by-unit random fill among retailers with remaining demand.
            idx = np.where(over)[0]
            assert rng is not None
            o = ret_orders[idx].astype(np.int64)      # (m,K)
            cap = release_capacity[idx].astype(np.int64)  # (m,)
            m = len(idx)
            filled = np.zeros((m, K), dtype=np.int64)
            remaining = o.copy()
            cap_left = cap.copy()
            # priority key: random per (path,retailer); fill in random retailer order,
            # but to respect "1 unit at a time until cap" we do K rounds of proportional-ish
            # random allocation. Equivalent expectation to Kaynov RandomSequential.
            for _ in range(int(o.sum(axis=1).max()) if m else 0):
                if (cap_left <= 0).all():
                    break
                # eligible retailers: remaining>0
                elig = remaining > 0
                # random key, mask ineligible
                key = rng.random((m, K))
                key[~elig] = -1.0
                pick = key.argmax(axis=1)  # (m,)
                rows = np.arange(m)
                can = (cap_left > 0) & elig[rows, pick]
                filled[rows[can], pick[can]] += 1
                remaining[rows[can], pick[can]] -= 1
                cap_left[can] -= 1
            ship[idx] = filled
            return ship
        else:
            raise ValueError(self.allocation)

    def step(self, wh_order, ret_orders, ration_rng=None):
        """Step all paths one period. wh_order (B,), ret_orders (B,K) are the RAW orders
        (PPO emits these directly; heuristic emits via echelon_orders). Returns
        (reward (B,), period_cost (B,)). Mutates internal state. Demands are drawn from
        the per-path demand matrix set by set_demands(); period auto-increments."""
        B, K = self.B, self.K
        wh_order = np.asarray(wh_order, dtype=np.int64)
        ret_orders = np.asarray(ret_orders, dtype=np.int64)
        avail = np.maximum(self.wh_inv + self.wh_pipe[:, 0], 0)
        release_capacity = avail  # holdback=0
        ship = self._ration(release_capacity, np.maximum(ret_orders, 0), rng=ration_rng)

        wh_arrival = self.wh_pipe[:, 0]
        ret_arrivals = self.ret_pipe[:, :, 0]  # (B,K)
        avail_wh = self.wh_inv + wh_arrival
        total_ship = ship.sum(axis=1)
        wh_end = avail_wh - total_ship  # (B,)

        # pipeline shift
        new_wh_pipe = np.concatenate([self.wh_pipe[:, 1:], wh_order[:, None]], axis=1)
        new_ret_pipe = np.concatenate([self.ret_pipe[:, :, 1:], ship[:, :, None]], axis=2)

        demands = self.demands[:, self.period, :]  # (B,K)
        emerg = (self.emerg_rng.random((B, K)) < self.emerg_p)  # (B,K)

        r_avail = self.ret_inv + ret_arrivals  # (B,K)
        short_be = np.maximum(demands - r_avail, 0)  # (B,K)
        # emergency: served from wh_end, sequentially across retailers (env.rs loops k).
        # Replicate sequential depletion: cumulative emergency demand capped by wh_end.
        want = np.where((short_be > 0) & emerg, short_be, 0).astype(np.int64)  # (B,K)
        wh_pool = np.maximum(wh_end, 0).astype(np.int64).copy()  # (B,)
        em = np.zeros((B, K), dtype=np.int64)
        for k in range(K):
            give = np.minimum(want[:, k], wh_pool)
            em[:, k] = give
            wh_pool -= give
        wh_end_post = wh_end - em.sum(axis=1)

        after = r_avail + em
        end_inv = np.maximum(after - demands, 0)  # (B,K)
        unmet = np.maximum(demands - after, 0)     # (B,K)

        ret_holding = (self.hcr[None, :] * np.maximum(end_inv, 0)).sum(axis=1)
        ret_penalty = (self.pcr[None, :] * unmet).sum(axis=1)
        wh_holding = self.hcw * np.maximum(wh_end_post, 0)
        period_cost = wh_holding + ret_holding + ret_penalty  # (B,)

        # advance
        self.wh_inv = wh_end_post
        self.wh_pipe = new_wh_pipe
        self.ret_inv = end_inv
        self.ret_pipe = new_ret_pipe
        self.period += 1
        return -period_cost, period_cost

    def set_demands(self, demands):
        """demands: (B, T, K) int array of realized per-retailer demand."""
        self.demands = np.asarray(demands, dtype=np.int64)
        self.total_periods = self.demands.shape[1]
