import numpy as np

from invman.policies.common import (
    as_float32_vector,
    normalize_action_spec,
    normalize_tree_action_adapter,
    normalize_tree_leaf_type,
    normalize_tree_split_type,
    round_nearest,
    sigmoid,
    softplus,
)
from invman.policies.es_module import ESModule
from invman.utils import save_init_args


class SoftTreePolicy(ESModule):
    @save_init_args
    def __init__(
        self,
        input_dim,
        max_order_size=None,
        action_spec=None,
        control_spec=None,
        depth=2,
        temperature=0.25,
        split_type="oblique",
        leaf_type="constant",
        action_adapter="identity",
        action_adapter_config=None,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.input_dim = int(input_dim)
        self.action_spec = normalize_action_spec(action_spec, default_max_order_size=max_order_size)
        self.control_spec = normalize_action_spec(
            self.action_spec if control_spec is None else control_spec,
            default_max_order_size=max_order_size,
        )
        self.action_dim = int(self.action_spec["action_dim"])
        self.control_dim = int(self.control_spec["action_dim"])
        self.action_mode = str(self.action_spec["action_mode"])
        self.control_mode = str(self.control_spec["action_mode"])
        self.min_values = [int(value) for value in self.control_spec["min_values"]]
        self.max_values = [int(value) for value in self.control_spec["max_values"]]
        self.max_order_size = int(self.action_spec["max_values"][0])
        self.depth = int(depth)
        self.temperature = float(temperature)
        self.split_type = normalize_tree_split_type(split_type)
        self.leaf_type = normalize_tree_leaf_type(leaf_type)
        self.action_adapter = normalize_tree_action_adapter(action_adapter)
        self.action_adapter_config = None if action_adapter_config is None else dict(action_adapter_config)
        self.num_internal_nodes = (2 ** self.depth) - 1
        self.num_leaves = 2 ** self.depth

        self.split_weights = np.empty((self.num_internal_nodes, self.input_dim), dtype=np.float32)
        self.split_bias = np.empty((self.num_internal_nodes,), dtype=np.float32)
        if self.leaf_type == "constant":
            self.leaf_logits = np.empty((self.num_leaves, self.control_dim), dtype=np.float32)
            self.leaf_weights = None
            self.leaf_bias = None
        elif self.leaf_type in {"linear", "sigmoid_linear"}:
            self.leaf_weights = np.empty((self.num_leaves, self.control_dim, self.input_dim), dtype=np.float32)
            self.leaf_bias = np.empty((self.num_leaves, self.control_dim), dtype=np.float32)
            self.leaf_logits = None
        else:
            raise NotImplementedError(f"Unknown tree leaf type: {self.leaf_type}")
        self.features = {}
        self.reset_parameters()

    def parameter_arrays(self):
        arrays = [self.split_weights, self.split_bias]
        if self.leaf_type == "constant":
            arrays.append(self.leaf_logits)
        elif self.leaf_type in {"linear", "sigmoid_linear"}:
            arrays.append(self.leaf_weights)
            arrays.append(self.leaf_bias)
        return arrays

    def reset_parameters(self):
        self.split_weights[...] = np.random.normal(0.0, 0.15, size=self.split_weights.shape).astype(np.float32)
        self.split_bias[...] = np.random.normal(0.0, 0.15, size=self.split_bias.shape).astype(np.float32)
        if self.leaf_type == "constant":
            self.leaf_logits[...] = np.random.normal(0.0, 0.15, size=self.leaf_logits.shape).astype(np.float32)
        elif self.leaf_type in {"linear", "sigmoid_linear"}:
            self.leaf_weights[...] = np.random.normal(0.0, 0.15, size=self.leaf_weights.shape).astype(np.float32)
            self.leaf_bias[...] = np.random.normal(0.0, 0.15, size=self.leaf_bias.shape).astype(np.float32)
        else:
            raise NotImplementedError(f"Unknown tree leaf type: {self.leaf_type}")

    def _leaf_probabilities(self, state):
        if self.split_type == "oblique":
            logits = (self.split_weights @ state + self.split_bias).astype(np.float32, copy=False)
            selected_feature_idx = None
            selected_feature_weight = None
        elif self.split_type == "axis_aligned":
            selector_idx = np.argmax(np.abs(self.split_weights), axis=-1)
            node_idx = np.arange(self.num_internal_nodes)
            selected_feature_idx = selector_idx.astype(np.int64, copy=False)
            selected_feature_weight = self.split_weights[node_idx, selector_idx].astype(np.float32, copy=True)
            selected_state = state[selector_idx]
            selected_weight = self.split_weights[node_idx, selector_idx]
            logits = (selected_state * selected_weight + self.split_bias).astype(np.float32, copy=False)
        else:
            raise NotImplementedError(f"Unknown tree split type: {self.split_type}")

        gates = sigmoid(logits / np.float32(self.temperature)).astype(np.float32, copy=False)

        level_probs = np.ones(1, dtype=np.float32)
        for depth in range(self.depth):
            next_level_probs = []
            start_idx = (2 ** depth) - 1
            for offset, parent_prob in enumerate(level_probs):
                gate = float(gates[start_idx + offset])
                next_level_probs.append(float(parent_prob) * (1.0 - gate))
                next_level_probs.append(float(parent_prob) * gate)
            level_probs = np.asarray(next_level_probs, dtype=np.float32)
        return gates, level_probs, selected_feature_idx, selected_feature_weight

    def _leaf_quantities(self, state):
        min_tensor = np.asarray(self.min_values, dtype=np.float32).reshape(1, self.control_dim)
        max_tensor = np.asarray(self.max_values, dtype=np.float32).reshape(1, self.control_dim)
        action_span = max_tensor - min_tensor
        if self.leaf_type == "constant":
            scaled = min_tensor + sigmoid(self.leaf_logits) * action_span
            return scaled.astype(np.float32, copy=False), None
        if self.leaf_type == "sigmoid_linear":
            raw_leaf_output = np.einsum("lai,i->la", self.leaf_weights, state, optimize=True) + self.leaf_bias
            scaled = min_tensor + sigmoid(raw_leaf_output) * action_span
            return scaled.astype(np.float32, copy=False), raw_leaf_output.astype(np.float32, copy=False)
        if self.leaf_type == "linear":
            raw_leaf_output = np.einsum("lai,i->la", self.leaf_weights, state, optimize=True) + self.leaf_bias
            scaled = min_tensor + softplus(raw_leaf_output)
            return scaled.astype(np.float32, copy=False), raw_leaf_output.astype(np.float32, copy=False)
        raise NotImplementedError(f"Unknown tree leaf type: {self.leaf_type}")

    def _project_controls(self, action_value):
        action_value = np.asarray(action_value, dtype=np.float32).reshape(self.control_dim)
        rounded = round_nearest(action_value).astype(np.int64)
        if self.control_mode == "scalar_quantity" and self.control_dim == 1 and self.leaf_type == "linear":
            scalar = int(max(int(rounded[0]), 0))
            return scalar, np.asarray(scalar, dtype=np.int64)
        clipped = np.clip(
            rounded,
            np.asarray(self.min_values, dtype=np.int64),
            np.asarray(self.max_values, dtype=np.int64),
        )

        if self.control_mode == "discrete_grid":
            projected_dims = []
            for dim_idx, allowed_values in enumerate(self.control_spec["allowed_values"]):
                allowed_tensor = np.asarray(allowed_values, dtype=np.float32)
                distances = np.abs(allowed_tensor - action_value[dim_idx])
                projected_dims.append(int(allowed_tensor[int(np.argmin(distances))]))
            if self.control_dim == 1:
                return projected_dims[0], np.asarray(projected_dims[0], dtype=np.int64)
            return tuple(projected_dims), np.asarray(projected_dims, dtype=np.int64)

        if self.control_dim == 1:
            scalar = int(clipped[0])
            return scalar, np.asarray(scalar, dtype=np.int64)
        projected = tuple(int(value) for value in clipped.tolist())
        return projected, np.asarray(projected, dtype=np.int64)

    def _finalize_action(self, projected_controls, state):
        from invman.problems.dual_sourcing.policies import apply_action_adapter

        controls = np.atleast_1d(projected_controls).astype(np.int64).tolist()
        normalized_state = np.asarray(state, dtype=np.float32)
        return apply_action_adapter(
            self.action_adapter,
            controls,
            normalized_state,
            self.action_spec,
            self.action_adapter_config,
        )

    def forward(self, state, return_features=False):
        state = as_float32_vector(state)
        split_probs, leaf_probs, selected_feature_idx, selected_feature_weight = self._leaf_probabilities(state)
        leaf_quantities, raw_leaf_output = self._leaf_quantities(state)
        action_value = np.sum(leaf_probs[:, None] * leaf_quantities, axis=0).astype(np.float32, copy=False)
        projected_controls, projected_array = self._project_controls(action_value)
        action = self._finalize_action(projected_array, state)

        if return_features:
            features = {
                "split_probs": split_probs.copy(),
                "leaf_probs": leaf_probs.copy(),
                "leaf_quantities": leaf_quantities[:, 0].copy() if self.control_dim == 1 else leaf_quantities.copy(),
                "action_value": float(action_value.item()) if self.control_dim == 1 else action_value.copy(),
                "projected_controls": projected_array.copy() if isinstance(projected_array, np.ndarray) else np.asarray(projected_array),
                "projected_action": np.asarray(action),
                "split_type": self.split_type,
                "leaf_type": self.leaf_type,
                "action_spec": dict(self.action_spec),
                "control_spec": dict(self.control_spec),
                "action_adapter": self.action_adapter,
            }
            if raw_leaf_output is not None:
                features["raw_leaf_output"] = (
                    raw_leaf_output[:, 0].copy() if self.control_dim == 1 else raw_leaf_output.copy()
                )
            if selected_feature_idx is not None:
                features["selected_feature_idx"] = selected_feature_idx.copy()
                features["selected_feature_weight"] = selected_feature_weight.copy()
            self.features = features
            return action, self.features
        self.features = {}
        return action
