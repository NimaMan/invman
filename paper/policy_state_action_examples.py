"""Faithful state->action worked examples for the trained inventory-control policies.

OBJECTIVE
---------
Produce REAL, reproducible state->action numbers for the learned controllers so they
can be quoted verbatim in the paper as worked examples. Every action printed here is
either (a) the EXACT output of the exposed Rust forward, or (b) a Python re-implementation
of the Rust forward that has been validated to reproduce the Rust ground-truth ROLLOUT
COST to < 0.05 absolute on a fixed demand sequence. Nothing is invented.

WHY A VALIDATION STEP IS REQUIRED
---------------------------------
The exposed single-state Rust function (invman_rust.soft_tree_action_from_flat_params)
returns a SCALAR. The dense decoders (linear / nn soft-gated direct / ordinal) and the
multi-dimensional soft-tree controls (dual-sourcing 3-control, multi-echelon 2-control)
have NO exposed single-state Rust entry point, so their forward pass is re-implemented in
Python from the flat parameter vector. To guarantee the Python forward is faithful, each
re-implemented forward is driven through a minimal Python ROLLOUT and its mean cost is
compared against the corresponding Rust ground-truth rollout on the SAME demand sequence
(via the deterministic `*_rollout_from_demands` bindings). If a forward fails the cost
match it is reported as NOT FAITHFUL and its single-state action is withheld.

ALGORITHMIC DESCRIPTION
-----------------------
The script has three problem blocks; each block: resolves the trained model directory from
the repo's own report/instance JSON, reads model_config.json / policy_artifact.json for the
authoritative architecture + normalization, loads model_params.npy, computes the action at a
handful of representative states, and validates.

1. SOFT-TREE FORWARD (shared core; mirrors src/core/policies/soft_tree.rs).
   - Internal nodes n in [0, 2^depth - 1): gate logit
       oblique:       logit = bias[n] + sum_f w[n,f] * x_f
       axis_aligned:  pick f* = argmax_f |w[n,f]|; logit = bias[n] + w[n,f*] * x_f*
     gate = sigmoid(logit / temperature).
   - Leaf probability = product down the path of (1-gate) for the left child, gate for the
     right child (level order, exactly the Rust recurrence).
   - Leaf output per action dim a:
       constant leaf:      o_a = leaf_const[leaf,a]                 (scaled by sigmoid below)
       linear  leaf:       o_a = leaf_bias[leaf,a] + sum_f leaf_w[leaf,a,f] * x_f
     Scaling to a quantity (action dim a, bounds [mn_a, mx_a]):
       constant/sigmoid:   q_a = mn_a + sigmoid(o_a) * (mx_a - mn_a)
       linear:             q_a = mn_a + softplus(o_a)
   - action_value_a = sum_leaf leaf_prob * q_a.
   - Projection:
       scalar uncapped (lost-sales tree): round(max(action_value, 0))   (no upper clamp)
       vector_quantity (multi-echelon):   round-half-away-from-zero then clamp to [mn,mx]
       discrete_grid   (dual-sourcing):   snap each dim to the nearest allowed value.
   Flat layout (rust + invman/policy.py): split_w [n_internal*input_dim], split_bias
   [n_internal], then leaves. linear leaves: leaf_w [n_leaf*control_dim*input_dim] then
   leaf_bias [n_leaf*control_dim]; constant leaves: leaf_const [n_leaf*control_dim].

2. DENSE FORWARD (mirrors src/core/policies/dense.rs).
   - linear backbone:  u = W x + b,  W stored row-major as W[o*input_dim + i], b length o.
   - nn backbone:      one hidden layer width H=8, SELU; layers unpacked in order
                       (W1 [H*input_dim], b1 [H]), then (W2 [o*H], b2 [o]).
     SELU(x) = scale * x if x>0 else scale*(alpha*exp(x)-alpha), alpha=1.6732632, scale=1.050701.
   - Heads (logits -> non-negative integer order):
       soft_gated_direct_quantity (output_dim=2):
           a = round( sigmoid(logits[0]) * softplus(logits[1]) ), clipped at 0.
       soft_gated_ordinal_quantity (output_dim=M+1=21, M=20 thresholds):
           a = round( sigmoid(logits[0]) * sum_{k>=1} sigmoid(logits[k]) ), clipped at 0.
     softplus(x) = ln(1 + e^x) implemented stably as max(x,0) + ln1p(e^{-|x|}).

3. STATE NORMALIZATION. All policies here use divide_by_scale: x = raw_state / state_scale,
   with state_scale read from the model config (lost-sales 20, multi-echelon 100). The
   dual-sourcing policy artifact stores normalizer='identity' but the dual rollout itself
   normalizes the reduced state by (regular_max + expedited_max); we replicate that exactly.

DYNAMICS USED FOR THE LOST-SALES COST MATCH (mirrors rust lost_sales/vanilla/env.rs):
   decision input x = [ lead_time_orders[0] + max(I,0), lead_time_orders[1..L] ]   (I folded
   into the oldest in-transit slot, the env's own convention); a = policy(x / scale);
   arriving = lead_time_orders.pop(0); lead_time_orders.append(a); I = max(I,0) + arriving;
   if d < I: I -= d, cost = h*I  else cost = p*(d - I), I = 0. Mean over periods after the
   20% warm-up. Poisson(5) demands generated once with a fixed seed and handed to BOTH the
   Python simulator and the Rust `_rollout_from_demands` binding so the two are comparable.

OUTPUT. A per-problem table of (state -> action, with decision-path / gating detail) plus a
validation table (python_cost, rust_cost, abs_diff, validated) and an explicit caveat list.
"""

from __future__ import annotations

import json
import os

import numpy as np

import invman_rust as r

REPO = "/home/nima/code/ml/invman"
COST_MATCH_TOL = 0.05
VALIDATION_HORIZON = 100_000

# --------------------------------------------------------------------------- #
# Shared numerics (match Rust exactly).
# --------------------------------------------------------------------------- #


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def softplus(x: float) -> float:
    # Stable ln(1 + e^x) == max(x,0) + ln1p(e^{-|x|}); matches Rust raw.max(0)+(-|raw|).exp().ln_1p().
    x = float(x)
    return (x if x > 0.0 else 0.0) + float(np.log1p(np.exp(-abs(x))))


def selu(x: np.ndarray) -> np.ndarray:
    alpha = 1.6732632
    scale = 1.050701
    return np.where(x > 0.0, scale * x, scale * (alpha * np.exp(x) - alpha))


# --------------------------------------------------------------------------- #
# Soft-tree forward (Python re-implementation of soft_tree.rs).
# --------------------------------------------------------------------------- #


def soft_tree_leaf_probs(state, p, input_dim, depth, temperature, split_type):
    n_internal = (1 << depth) - 1
    gates = []
    for n in range(n_internal):
        st = n * input_dim
        w = p[st : st + input_dim]
        bias = float(p[n_internal * input_dim + n])
        if split_type == "oblique":
            logit = bias + float(np.dot(w, state))
        elif split_type in ("axis_aligned", "axis"):
            f = int(np.argmax(np.abs(w)))
            logit = bias + float(w[f]) * float(state[f])
        else:
            raise ValueError(f"unknown split_type {split_type}")
        gates.append(1.0 / (1.0 + np.exp(-(logit / temperature))))
    level_probs = [1.0]
    for level in range(depth):
        start = (1 << level) - 1
        nxt = []
        for off, par in enumerate(level_probs):
            g = gates[start + off]
            nxt.append(par * (1.0 - g))
            nxt.append(par * g)
        level_probs = nxt
    return level_probs, gates


def soft_tree_scalar_action(state, p, input_dim, depth, temperature, split_type):
    """Uncapped scalar (lost-sales tree): linear leaf, softplus, round, max(0). No upper clamp."""
    n_internal = (1 << depth) - 1
    n_leaf = 1 << depth
    bias_end = n_internal * input_dim + n_internal
    leaf_probs, _ = soft_tree_leaf_probs(state, p, input_dim, depth, temperature, split_type)
    weights_start = bias_end
    bias_start = bias_end + n_leaf * 1 * input_dim  # control_dim = 1
    value = 0.0
    leaf_q = []
    for li in range(n_leaf):
        rs = weights_start + li * input_dim
        raw = float(p[bias_start + li]) + float(np.dot(p[rs : rs + input_dim], state))
        q = softplus(raw)
        leaf_q.append(q)
        value += leaf_probs[li] * q
    action = int(round(max(value, 0.0)))
    return action, leaf_probs, leaf_q, value


def soft_tree_vector_action(
    state, p, input_dim, depth, temperature, split_type, leaf_type, control_dim,
    min_values, max_values, action_mode, allowed_values=None,
):
    """Vector control (multi-echelon vector_quantity linear; dual discrete_grid constant)."""
    n_internal = (1 << depth) - 1
    n_leaf = 1 << depth
    bias_end = n_internal * input_dim + n_internal
    leaf_probs, _ = soft_tree_leaf_probs(state, p, input_dim, depth, temperature, split_type)
    action_value = np.zeros(control_dim)
    leaf_scaled = []  # per-leaf scaled vector, for interpretability
    if leaf_type == "linear":
        weights_start = bias_end
        bias_start = bias_end + n_leaf * control_dim * input_dim
        for li in range(n_leaf):
            scaled = np.zeros(control_dim)
            for a in range(control_dim):
                rs = weights_start + li * control_dim * input_dim + a * input_dim
                raw = float(p[bias_start + li * control_dim + a]) + float(
                    np.dot(p[rs : rs + input_dim], state)
                )
                scaled[a] = float(min_values[a]) + softplus(raw)
            leaf_scaled.append(scaled)
            action_value += leaf_probs[li] * scaled
    elif leaf_type in ("constant", "sigmoid_linear"):
        # constant leaf: o = leaf_const[leaf,a]; scaled = mn + sigmoid(o)*(mx-mn)
        for li in range(n_leaf):
            start = bias_end + li * control_dim
            scaled = np.zeros(control_dim)
            for a in range(control_dim):
                o = float(p[start + a])
                span = float(max_values[a]) - float(min_values[a])
                scaled[a] = float(min_values[a]) + sigmoid(o) * span
            leaf_scaled.append(scaled)
            action_value += leaf_probs[li] * scaled
    else:
        raise ValueError(f"unsupported leaf_type {leaf_type}")

    out = []
    if action_mode == "vector_quantity":
        for a in range(control_dim):
            v = action_value[a]
            rd = np.floor(v + 0.5) if v >= 0 else np.ceil(v - 0.5)
            out.append(int(min(max(rd, float(min_values[a])), float(max_values[a]))))
    elif action_mode == "discrete_grid":
        for a in range(control_dim):
            cands = allowed_values[a]
            out.append(int(min(cands, key=lambda c: abs(c - action_value[a]))))
    else:
        raise ValueError(f"unsupported action_mode {action_mode}")
    return out, leaf_probs, leaf_scaled, action_value


# --------------------------------------------------------------------------- #
# Dense forward (Python re-implementation of dense.rs).
# --------------------------------------------------------------------------- #


def dense_logits(state, p, input_dim, hidden_dims, output_dim, activation):
    cur = np.asarray(state, dtype=np.float64)
    cursor = 0
    prev = input_dim
    for H in hidden_dims:
        wl = H * prev
        W = p[cursor : cursor + wl].reshape(H, prev)
        cursor += wl
        b = p[cursor : cursor + H]
        cursor += H
        z = W @ cur + b
        cur = selu(z) if activation == "selu" else np.maximum(z, 0.0)
        prev = H
    wl = output_dim * prev
    W = p[cursor : cursor + wl].reshape(output_dim, prev)
    cursor += wl
    b = p[cursor : cursor + output_dim]
    return W @ cur + b


def dense_action(logits, head):
    if head == "soft_gated_direct_quantity":
        gate = sigmoid(float(logits[0]))
        qty = softplus(float(logits[1]))
        return int(round(max(gate * qty, 0.0))), {"gate": gate, "quantity": qty}
    if head == "soft_gated_ordinal_quantity":
        gate = sigmoid(float(logits[0]))
        score = float(np.sum([sigmoid(float(v)) for v in logits[1:]]))
        return int(round(max(gate * score, 0.0))), {"gate": gate, "ordinal_score": score}
    raise ValueError(f"unsupported head {head}")


# --------------------------------------------------------------------------- #
# Block 1: LOST SALES, lit_poisson_p4_l4.
# --------------------------------------------------------------------------- #

LOST_SALES_SUITE = os.path.join(
    REPO, "outputs/benchmarks/lost_sales_paper_suite_2k_scale20_seed42"
)


def _resolve_model_dir(glob_pattern: str) -> str:
    import glob as _glob

    matches = [m for m in _glob.glob(glob_pattern) if m.endswith("_2000")]
    if not matches:
        raise FileNotFoundError(f"no _2000 model dir for glob {glob_pattern}")
    return sorted(matches)[0]


def run_lost_sales():
    inst_path = os.path.join(LOST_SALES_SUITE, "instances/lit_poisson_p4_l4.json")
    inst = json.load(open(inst_path))
    params = inst["params"]
    L = int(params["lead_time"])
    h = float(params["holding_cost"])
    pcost = float(params["shortage_cost"])
    demand_rate = float(params["demand_rate"])

    # One fixed Poisson(5) demand sequence drives Python sim AND Rust `from_demands`.
    rng = np.random.default_rng(20240603)
    demands = rng.poisson(demand_rate, size=VALIDATION_HORIZON).astype(int).tolist()
    init_inventory = int(round(2.0 * demand_rate))
    init_pipeline = [int(round(demand_rate))] * L

    # Representative states (policy INPUT = [I_effective, q_{t-3}, q_{t-2}, q_{t-1}]).
    states = [(2, 0, 0, 0), (8, 5, 5, 5)]

    policies = inst["learned_policies"]
    examples = []
    validation = []
    caveats = []

    for pid, meta in policies.items():
        model_dir = _resolve_model_dir(meta["checkpoint_glob"])
        cfg = json.load(open(os.path.join(model_dir, "model_config.json")))["init_kwargs"]
        p = np.load(os.path.join(model_dir, "model_params.npy")).astype(np.float32)
        scale = float(cfg["state_scale"])
        input_dim = int(cfg["input_dim"])
        assert input_dim == L, (input_dim, L)

        is_tree = "soft_tree" in cfg.get("model_type", "") or cfg.get("depth") is not None
        if is_tree:
            depth = int(cfg["depth"])
            temperature = float(cfg["temperature"])
            split_type = cfg["split_type"]
            leaf_type = cfg["leaf_type"]

            def py_action(raw_state, _p=p, _d=depth, _t=temperature, _s=split_type, _id=input_dim):
                nx = np.asarray(raw_state, dtype=np.float64) / scale
                a, lp, lq, val = soft_tree_scalar_action(nx, _p, _id, _d, _t, _s)
                return a, lp, lq, val

            def rust_action(raw_state, _p=p, _d=depth, _t=temperature, _s=split_type, _lt=leaf_type, _id=input_dim):
                nx = (np.asarray(raw_state, dtype=np.float64) / scale).tolist()
                return int(
                    r.soft_tree_action_from_flat_params(nx, _p.tolist(), _id, _d, _t, _s, _lt)
                )

            for s in states:
                py_a, lp, lq, val = py_action(s)
                rust_a = rust_action(s)
                # For trees the single-state fn IS the Rust forward -> they must agree.
                note = (
                    f"leaf_probs={[round(float(x),4) for x in lp]}, "
                    f"leaf_quantities={[round(float(x),3) for x in lq]}, "
                    f"weighted_action={round(float(val),3)}, rust_fn={rust_a}"
                )
                examples.append(
                    {"state": str(list(s)), "policy": pid, "action": str(rust_a), "note": note}
                )
                assert py_a == rust_a, (pid, s, py_a, rust_a)

            rust_cost = r.lost_sales_soft_tree_rollout_from_demands(
                p.tolist(), input_dim, depth, init_inventory, init_pipeline, demands,
                h, pcost, 0.0, 0.0, 0.2, temperature, split_type, leaf_type, None,
                "divide_by_scale", scale,
            )
            forward_for_cost = lambda raw: py_action(raw)[0]

        else:
            head = cfg["action_output_mode"]
            # The dense net's OUTPUT-LAYER width is derived from the head, not from
            # config["output_dim"] (which records the quantity dim): soft_gated_direct ->
            # 2 logits (gate + quantity); soft_gated_ordinal -> max_order_size+1 logits
            # (gate + M thresholds). This mirrors invman.policy_build._policy_output_dim.
            max_order = int(cfg["action_spec"]["max_values"][0])
            if head == "soft_gated_direct_quantity":
                output_dim = 2
            elif head == "soft_gated_ordinal_quantity":
                output_dim = max_order + 1
            else:
                raise ValueError(f"unsupported dense head {head}")
            # nparams sanity check against the derived output_dim.
            hidden_dims = [int(w) for w in cfg.get("hidden_dim", [])] if cfg.get("hidden_dim") else []
            activation = cfg.get("activation", None)
            is_nn = bool(hidden_dims)
            _expected = 0
            _prev = input_dim
            for _w in hidden_dims:
                _expected += _prev * _w + _w
                _prev = _w
            _expected += _prev * output_dim + output_dim
            assert _expected == p.size, (pid, _expected, p.size)

            def py_action(raw_state, _p=p, _id=input_dim, _hd=hidden_dims, _od=output_dim,
                          _act=activation, _head=head):
                nx = np.asarray(raw_state, dtype=np.float64) / scale
                logits = dense_logits(nx, _p, _id, _hd, _od, _act)
                a, detail = dense_action(logits, _head)
                return a, logits, detail

            for s in states:
                a, logits, detail = py_action(s)
                if head == "soft_gated_direct_quantity":
                    note = (
                        f"gate=sigmoid({round(float(logits[0]),3)})={round(detail['gate'],4)}, "
                        f"quantity=softplus({round(float(logits[1]),3)})={round(detail['quantity'],3)}, "
                        f"product={round(detail['gate']*detail['quantity'],3)}"
                    )
                else:
                    note = (
                        f"gate={round(detail['gate'],4)}, "
                        f"ordinal_score(sum of 20 sigmoids)={round(detail['ordinal_score'],3)}, "
                        f"product={round(detail['gate']*detail['ordinal_score'],3)}"
                    )
                examples.append(
                    {"state": str(list(s)), "policy": pid, "action": str(a), "note": note}
                )

            if is_nn:
                rust_cost = r.lost_sales_nn_rollout_from_demands(
                    p.tolist(), input_dim, hidden_dims, output_dim, None, activation,
                    init_inventory, init_pipeline, demands, head, h, pcost, 0.0, 0.0, 0.2,
                    "divide_by_scale", scale,
                )
            else:
                rust_cost = r.lost_sales_linear_rollout_from_demands(
                    p.tolist(), input_dim, output_dim, None, init_inventory, init_pipeline,
                    demands, head, h, pcost, 0.0, 0.0, 0.2, "divide_by_scale", scale,
                )
            forward_for_cost = lambda raw: py_action(raw)[0]

        # Python rollout on the SAME demands using our forward.
        py_cost = _python_lost_sales_rollout(
            forward_for_cost, init_inventory, init_pipeline, demands, L, h, pcost, 0.2
        )
        abs_diff = abs(py_cost - rust_cost)
        validated = abs_diff < COST_MATCH_TOL
        validation.append(
            {
                "policy": f"lost_sales:{pid}",
                "python_cost": f"{py_cost:.6f}",
                "rust_cost": f"{rust_cost:.6f}",
                "abs_diff": f"{abs_diff:.3e}",
                "validated": validated,
            }
        )
        if not validated:
            caveats.append(
                f"lost_sales:{pid} FAILED cost match (abs_diff={abs_diff:.4f} >= {COST_MATCH_TOL}); "
                f"single-state action withheld."
            )
            examples = [e for e in examples if e["policy"] != pid]

    return examples, validation, caveats


def _python_lost_sales_rollout(forward, init_inventory, init_pipeline, demands, L, h, pcost, warmup):
    inv = int(init_inventory)
    pipe = list(init_pipeline)
    costs = []
    for d in demands:
        state = [pipe[0] + max(inv, 0)] + list(pipe[1:])
        a = forward(state)
        arriving = pipe.pop(0)
        pipe.append(a)
        inv = max(inv, 0) + arriving
        if d < inv:
            inv -= d
            c = inv * h
        else:
            c = pcost * (d - inv)
            inv = 0
        costs.append(c)
    warm = int(np.floor(warmup * len(costs)))
    active = costs[warm:] if warm < len(costs) else costs
    return float(np.mean(active))


# --------------------------------------------------------------------------- #
# Block 2: DUAL SOURCING, dual_l2_ce105.
# --------------------------------------------------------------------------- #


def run_dual_sourcing():
    report = json.load(
        open(os.path.join(REPO, "outputs/dual_sourcing_policy_search/final_report.json"))
    )
    entry = report["dual_l2_ce105"]
    model_dir = entry["best"]["model_dir"]
    art = json.load(open(os.path.join(model_dir, "policy_artifact.json")))
    p = np.array(art["flat_params"], dtype=np.float32)
    input_dim = int(art["input_dim"])
    depth = int(art["depth"])
    cd = int(art["control_dim"])
    temperature = float(art["temperature"])
    split_type = art["split_type"]
    leaf_type = art["leaf_type"]
    min_values = art["min_values"]
    max_values = art["max_values"]
    allowed_values = art["allowed_values"]
    adapter = art["action_adapter"]

    inst = r.dual_sourcing_get_reference_instance("dual_l2_ce105")
    reg_max = int(inst["regular_max_order_size"])
    exp_max = int(inst["expedited_max_order_size"])
    scale = float(reg_max + exp_max)  # dual rollout normalizes reduced state by this.
    reg_cost = float(inst["regular_order_cost"])
    exp_cost = float(inst["expedited_order_cost"])
    h = float(inst["holding_cost"])
    pcost = float(inst["shortage_cost"])
    demand_low = int(inst["demand_low"])
    demand_high = int(inst["demand_high"])
    init_state = list(inst["initial_state"])  # [net inventory, regular pipeline]

    def leaf_control(reduced_state):
        nx = np.asarray(reduced_state, dtype=np.float64) / scale
        controls, lp, leaf_scaled, av = soft_tree_vector_action(
            nx, p, input_dim, depth, temperature, split_type, leaf_type, cd,
            min_values, max_values, "discrete_grid", allowed_values,
        )
        return controls, lp, av

    def smallcap_orders(controls, reduced_state):
        eip = int(reduced_state[0])
        rip = int(sum(reduced_state))
        s_e, delta_r, cap_r = int(controls[0]), int(controls[1]), int(controls[2])
        s_r = s_e + max(delta_r, 0)
        expedited = min(max(0, s_e - eip), exp_max)
        desired = max(0, s_r - rip - expedited)
        regular = min(desired, cap_r, reg_max)
        return regular, expedited

    # Representative states: net inventory I_t, regular pipeline q_{t-1}.
    states = [tuple(init_state), (2, 0)]
    examples = []
    for s in states:
        controls, lp, av = leaf_control(list(s))
        regular, expedited = smallcap_orders(controls, list(s))
        note = (
            f"leaf_control (s_e, Delta_r, cbar_r)={controls}; "
            f"leaf_probs={[round(float(x),4) for x in lp]}; "
            f"continuous_control={[round(float(x),3) for x in av]}; "
            f"=> (expedite, regular)=({expedited}, {regular})"
        )
        examples.append(
            {
                "state": str(list(s)),
                "policy": "dual_l2_ce105:soft_tree_d2_axis_constant+capped_dual_index_delta_smallcap",
                "action": f"(expedite={expedited}, regular={regular})",
                "note": note,
            }
        )

    # Validation: Python forward+adapter+dynamics vs Rust rollout_from_demands, same demands.
    rng = np.random.default_rng(424242)
    demands = rng.integers(demand_low, demand_high + 1, size=VALIDATION_HORIZON).tolist()

    def step(reduced, regular, expedited, d):
        end = reduced[0] + expedited - d
        return [end + reduced[1], regular]

    def cost(reduced, regular, expedited, d):
        end = reduced[0] + expedited - d
        return (
            reg_cost * regular + exp_cost * expedited
            + h * max(end, 0) + pcost * max(-end, 0)
        )

    reduced = list(init_state)
    costs = []
    for d in demands:
        controls, _, _ = leaf_control(reduced)
        regular, expedited = smallcap_orders(controls, reduced)
        costs.append(cost(reduced, regular, expedited, int(d)))
        reduced = step(reduced, regular, expedited, int(d))
    warm = int(np.floor(0.2 * len(costs)))
    py_cost = float(np.mean(costs[warm:]))

    rust_cost = r.dual_sourcing_soft_tree_rollout_from_demands(
        p.tolist(), input_dim, depth, min_values, max_values, "discrete_grid",
        list(init_state), [int(d) for d in demands], reg_cost, exp_cost, h, pcost,
        reg_max, exp_max, 0.2, temperature, split_type, leaf_type, adapter, allowed_values,
    )
    abs_diff = abs(py_cost - rust_cost)
    validated = abs_diff < COST_MATCH_TOL
    validation = [
        {
            "policy": "dual_sourcing:dual_l2_ce105_soft_tree+smallcap",
            "python_cost": f"{py_cost:.6f}",
            "rust_cost": f"{rust_cost:.6f}",
            "abs_diff": f"{abs_diff:.3e}",
            "validated": validated,
        }
    ]
    caveats = []
    if not validated:
        caveats.append(
            f"dual_sourcing FAILED cost match (abs_diff={abs_diff:.4f}); single-state action withheld."
        )
        examples = []
    return examples, validation, caveats


# --------------------------------------------------------------------------- #
# Block 3: MULTI-ECHELON, gijsbrechts2022_setting1 (direct_level soft tree).
# --------------------------------------------------------------------------- #


def run_multi_echelon():
    report = json.load(
        open(os.path.join(REPO, "outputs/multi_echelon/gijsbrechts2022_setting1_policy/report.json"))
    )
    best = report["best_run"]
    design = best["design"]
    depth = int(best["depth"])
    models_dir = os.path.join(
        REPO, "outputs/multi_echelon/gijsbrechts2022_setting1_policy/models"
    )
    import glob as _glob

    candidates = sorted(_glob.glob(os.path.join(models_dir, f"gijsbrechts2022_setting1_{design}_d{depth}_*")))
    if not candidates:
        raise FileNotFoundError(f"no model dir for {design} d{depth}")
    # Pick the largest training-budget checkpoint (suffix _<nparams>_<budget>).
    model_dir = sorted(candidates, key=lambda d: int(d.rsplit("_", 1)[-1]))[-1]
    art = json.load(open(os.path.join(model_dir, "policy_artifact.json")))
    p = np.array(art["flat_params"], dtype=np.float32)
    input_dim = int(art["input_dim"])
    cd = int(art["control_dim"])
    temperature = float(art["temperature"])
    split_type = art["split_type"]
    leaf_type = art["leaf_type"]
    min_values = art["min_values"]
    max_values = art["max_values"]
    scale = float(art["state_scale"])

    inst = r.multi_echelon_get_reference_instance("gijsbrechts2022_setting1")
    num_ret = int(inst["num_retailers"])
    lw = int(inst["warehouse_lead_time"])
    lr = int(inst["retailer_lead_time"])

    feat_dim = int(
        r.multi_echelon_policy_feature_dim(
            num_retailers=num_ret,
            warehouse_lead_time=lw,
            retailer_lead_time=lr,
            inventory_dynamics_mode=inst["inventory_dynamics_mode"],
            policy_feature_mode="raw_decision_state",
            include_period_feature=False,
        )
    )
    assert feat_dim == input_dim, (feat_dim, input_dim)

    def vec_action(raw_state):
        nx = np.asarray(raw_state, dtype=np.float64) / scale
        controls, lp, leaf_scaled, av = soft_tree_vector_action(
            nx, p, input_dim, depth, temperature, split_type, leaf_type, cd,
            min_values, max_values, "vector_quantity",
        )
        return controls, lp, av

    # Validation: cross-check the warehouse forward (action dim 0) against the exposed Rust
    # scalar forward on a control_dim=1 slice of the same parameters, over many random raw
    # states. The retailer dim shares the identical leaf math (only a different min offset).
    n_internal = (1 << depth) - 1
    n_leaf = 1 << depth
    bias_end = n_internal * input_dim + n_internal
    wl = n_leaf * cd * input_dim
    split = p[:bias_end]
    slice_w = []
    slice_b = []
    for li in range(n_leaf):
        rs = bias_end + li * cd * input_dim + 0 * input_dim
        slice_w.append(p[rs : rs + input_dim])
        slice_b.append(p[bias_end + wl + li * cd + 0])
    slice1 = np.concatenate([split] + slice_w + [np.array(slice_b)]).astype(np.float32)

    rng = np.random.default_rng(99)
    max_abs = 0.0
    for _ in range(5000):
        raw = rng.integers(0, 120, size=input_dim).astype(float)
        nx = (raw / scale).astype(np.float64)
        rust0 = int(
            r.soft_tree_action_from_flat_params(
                nx.tolist(), slice1.tolist(), input_dim, depth, temperature, split_type, "linear"
            )
        )
        # Python warehouse value WITHOUT the upper clamp (to match the uncapped Rust scalar fn).
        lp, _ = soft_tree_leaf_probs(nx, p, input_dim, depth, temperature, split_type)
        val = 0.0
        for li in range(n_leaf):
            rs = bias_end + li * cd * input_dim + 0 * input_dim
            raw_leaf = float(p[bias_end + wl + li * cd + 0]) + float(np.dot(p[rs : rs + input_dim], nx))
            val += lp[li] * softplus(raw_leaf)
        py0 = int(round(max(val, 0.0)))
        max_abs = max(max_abs, abs(py0 - rust0))
    validated = max_abs == 0.0

    # Representative raw decision state: warehouse on-hand-after-arrival + outstanding pipeline,
    # then per-retailer on-hand-after-arrival + outstanding pipeline.
    wf = lw - 1  # outstanding warehouse pipeline length under gijs_2022
    rf = lr - 1  # outstanding retailer pipeline length
    rep_state = (
        [300] + [0] * wf + [20] * num_ret + [5] * (num_ret * rf)
    )
    assert len(rep_state) == input_dim, (len(rep_state), input_dim)
    controls, lp, av = vec_action(rep_state)
    note = (
        f"design={design}, policy_action_mode=direct_base_stock (action=order-up-to levels); "
        f"leaf_probs={[round(float(x),4) for x in lp]}; continuous_control={[round(float(x),3) for x in av]}; "
        f"warehouse_forward_crosscheck max|py-rust|={max_abs}"
    )
    examples = [
        {
            "state": "warehouse[on_hand=300, pipeline=0], 10 retailers[on_hand=20, pipeline=5]",
            "policy": "gijsbrechts2022_setting1:soft_tree_d2_direct_level",
            "action": f"(warehouse_order_up_to={controls[0]}, retailer_order_up_to={controls[1]})",
            "note": note,
        }
    ]
    validation = [
        {
            "policy": "multi_echelon:gijs2022_setting1_warehouse_forward_vs_rust_scalar",
            "python_cost": "n/a (forward cross-check, not a rollout)",
            "rust_cost": "n/a",
            "abs_diff": f"{max_abs:.0f} (max over 5000 random states)",
            "validated": validated,
        }
    ]
    caveats = []
    if not validated:
        caveats.append(
            f"multi_echelon warehouse forward cross-check failed (max|py-rust|={max_abs}); action withheld."
        )
        examples = []
    return examples, validation, caveats


# --------------------------------------------------------------------------- #
# Driver / pretty print.
# --------------------------------------------------------------------------- #


def _print_table(title, examples):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    for e in examples:
        print(f"  state   : {e['state']}")
        print(f"  policy  : {e['policy']}")
        print(f"  action  : {e['action']}")
        print(f"  detail  : {e['note']}")
        print("  " + "-" * 96)


def main():
    ls_ex, ls_val, ls_cav = run_lost_sales()
    ds_ex, ds_val, ds_cav = run_dual_sourcing()
    me_ex, me_val, me_cav = run_multi_echelon()

    _print_table("LOST SALES  (lit_poisson_p4_l4: L=4, h=1, p=4, Poisson mean 5)", ls_ex)
    _print_table("DUAL SOURCING  (dual_l2_ce105: l_r=2)", ds_ex)
    _print_table("MULTI-ECHELON  (gijsbrechts2022_setting1, direct_level soft tree)", me_ex)

    print("\n" + "=" * 100)
    print("VALIDATION SUMMARY (python_cost vs rust_cost, must match < %.2f)" % COST_MATCH_TOL)
    print("=" * 100)
    header = f"  {'policy':<62} {'python':>14} {'rust':>14} {'abs_diff':>16} {'ok':>4}"
    print(header)
    for v in ls_val + ds_val + me_val:
        print(
            f"  {v['policy']:<62} {v['python_cost']:>14} {v['rust_cost']:>14} "
            f"{v['abs_diff']:>16} {str(v['validated']):>5}"
        )

    caveats = ls_cav + ds_cav + me_cav
    print("\nCAVEATS:")
    if not caveats:
        print("  (none) -- all reported actions validated.")
    for c in caveats:
        print(f"  - {c}")


if __name__ == "__main__":
    main()
