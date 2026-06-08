#!/usr/bin/env python
"""Vectorized (batched-over-paths) soft-tree constant-leaf oblique action evaluator,
so the CMA-ES soft-tree policy can be scored through the EXACT SAME batched env
(batched_env.BatchedOWMR) and the EXACT SAME emergency RNG as the PPO actor -> a truly
paired, instrument-bias-cancelling head-to-head.

Replicates instrument._soft_tree_action / _build_policy_state for the constant-leaf
oblique echelon_targets head, vectorized over B paths. Validated against the scalar
instrument learned_fn in validate_softtree_batched.
"""
import numpy as np


def _leaf_probs_batched(state, split_weights, split_bias, depth, temperature):
    """state (B, L); split_weights (n_internal*L,), split_bias (n_internal,).
    Returns leaf_probs (B, n_leaves). Oblique splits only."""
    B, L = state.shape
    n_internal = (1 << depth) - 1
    W = split_weights.reshape(n_internal, L)            # (n_internal, L)
    logits = state @ W.T + split_bias[None, :]          # (B, n_internal)
    gates = 1.0 / (1.0 + np.exp(-(logits / temperature)))  # (B, n_internal)
    gates = np.where(np.isnan(logits), 0.5, gates)
    level = np.ones((B, 1))
    for lv in range(depth):
        start = (1 << lv) - 1
        n_nodes = 1 << lv
        g = gates[:, start:start + n_nodes]             # (B, n_nodes)
        left = level * (1.0 - g)
        right = level * g
        # interleave: child order [left0,right0,left1,right1,...]
        nxt = np.empty((B, 2 * n_nodes))
        nxt[:, 0::2] = left
        nxt[:, 1::2] = right
        level = nxt
    return level  # (B, n_leaves)


class BatchedSoftTree:
    """Constant-leaf oblique soft tree (echelon_targets head). action_dim = K+1."""
    def __init__(self, flat, input_dim, depth, temperature, min_values, max_values):
        self.input_dim = int(input_dim)
        self.depth = int(depth)
        self.temperature = float(temperature)
        self.min_values = np.asarray(min_values, dtype=np.float64)
        self.max_values = np.asarray(max_values, dtype=np.float64)
        self.action_dim = len(min_values)
        n_internal = (1 << self.depth) - 1
        n_leaves = 1 << self.depth
        flat = np.asarray(flat, dtype=np.float64)
        we = n_internal * self.input_dim
        be = we + n_internal
        self.split_weights = flat[:we]
        self.split_bias = flat[we:be]
        # leaf outputs (n_leaves, action_dim)
        self.leaf_out = flat[be:be + n_leaves * self.action_dim].reshape(n_leaves, self.action_dim)

    def action(self, state):
        """state (B, input_dim) normalized features. Returns integer targets (B, action_dim):
        col 0 = W_target, cols 1: = retailer targets. (Round-half-away-from-zero + clip.)"""
        B = state.shape[0]
        lp = _leaf_probs_batched(state, self.split_weights, self.split_bias,
                                 self.depth, self.temperature)  # (B, n_leaves)
        span = (self.max_values - self.min_values)[None, :]     # (1, A)
        # scaled leaf actions: mn + sigmoid(leaf_out) * span  -> (n_leaves, A)
        scaled = self.min_values[None, :] + (1.0 / (1.0 + np.exp(-self.leaf_out))) * span
        av = lp @ scaled                                        # (B, A)
        r = np.where(av >= 0, np.floor(av + 0.5), np.ceil(av - 0.5))
        r = np.clip(r, self.min_values[None, :], self.max_values[None, :])
        return r.astype(np.int64)
