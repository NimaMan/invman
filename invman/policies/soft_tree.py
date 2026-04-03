import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from invman.policies.common import (
    normalize_action_spec,
    normalize_tree_action_adapter,
    normalize_tree_leaf_type,
    normalize_tree_split_type,
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

        self.split_weights = nn.Parameter(torch.empty(self.num_internal_nodes, self.input_dim))
        self.split_bias = nn.Parameter(torch.empty(self.num_internal_nodes))
        if self.leaf_type == "constant":
            self.leaf_logits = nn.Parameter(torch.empty(self.num_leaves, self.control_dim))
            self.leaf_weights = None
            self.leaf_bias = None
        elif self.leaf_type in {"linear", "sigmoid_linear"}:
            self.leaf_weights = nn.Parameter(torch.empty(self.num_leaves, self.control_dim, self.input_dim))
            self.leaf_bias = nn.Parameter(torch.empty(self.num_leaves, self.control_dim))
            self.leaf_logits = None
        else:
            raise NotImplementedError(f"Unknown tree leaf type: {self.leaf_type}")
        self.features = {}
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.split_weights, mean=0.0, std=0.15)
        nn.init.normal_(self.split_bias, mean=0.0, std=0.15)
        if self.leaf_type == "constant":
            nn.init.normal_(self.leaf_logits, mean=0.0, std=0.15)
        elif self.leaf_type in {"linear", "sigmoid_linear"}:
            nn.init.normal_(self.leaf_weights, mean=0.0, std=0.15)
            nn.init.normal_(self.leaf_bias, mean=0.0, std=0.15)
        else:
            raise NotImplementedError(f"Unknown tree leaf type: {self.leaf_type}")

    def _leaf_probabilities(self, state):
        if self.split_type == "oblique":
            logits = F.linear(state.unsqueeze(0), self.split_weights, self.split_bias).squeeze(0)
            selected_feature_idx = None
            selected_feature_weight = None
        elif self.split_type == "axis_aligned":
            selector_idx = torch.argmax(torch.abs(self.split_weights), dim=-1)
            node_idx = torch.arange(self.num_internal_nodes, device=self.split_weights.device)
            selected_feature_idx = selector_idx.detach().cpu().numpy()
            selected_feature_weight = self.split_weights[node_idx, selector_idx].detach().cpu().numpy()
            selected_state = state[selector_idx]
            selected_weight = self.split_weights[node_idx, selector_idx]
            logits = (selected_state * selected_weight) + self.split_bias
        else:
            raise NotImplementedError(f"Unknown tree split type: {self.split_type}")

        gates = torch.sigmoid(logits / self.temperature)

        level_probs = state.new_ones(1)
        for depth in range(self.depth):
            next_level_probs = []
            start_idx = (2 ** depth) - 1
            for offset, parent_prob in enumerate(level_probs):
                gate = gates[start_idx + offset]
                next_level_probs.append(parent_prob * (1.0 - gate))
                next_level_probs.append(parent_prob * gate)
            level_probs = torch.stack(next_level_probs)
        return gates, level_probs, selected_feature_idx, selected_feature_weight

    def _control_scale(self, state):
        min_tensor = state.new_tensor(self.min_values, dtype=torch.float32).view(1, self.control_dim)
        max_tensor = state.new_tensor(self.max_values, dtype=torch.float32).view(1, self.control_dim)
        return min_tensor, max_tensor

    def _leaf_quantities(self, state):
        min_tensor = state.new_tensor(self.min_values, dtype=torch.float32).view(1, self.control_dim)
        max_tensor = state.new_tensor(self.max_values, dtype=torch.float32).view(1, self.control_dim)
        action_span = max_tensor - min_tensor
        if self.leaf_type == "constant":
            scaled = min_tensor + torch.sigmoid(self.leaf_logits) * action_span
            return scaled, None
        if self.leaf_type == "sigmoid_linear":
            raw_leaf_output = torch.einsum("lai,i->la", self.leaf_weights, state) + self.leaf_bias
            scaled = min_tensor + torch.sigmoid(raw_leaf_output) * action_span
            return scaled, raw_leaf_output
        if self.leaf_type == "linear":
            raw_leaf_output = torch.einsum("lai,i->la", self.leaf_weights, state) + self.leaf_bias
            scaled = min_tensor + F.softplus(raw_leaf_output)
            return scaled, raw_leaf_output
        raise NotImplementedError(f"Unknown tree leaf type: {self.leaf_type}")

    def _project_controls(self, action_value):
        rounded = torch.round(action_value).to(dtype=torch.int64)
        min_tensor = action_value.new_tensor(self.min_values, dtype=torch.int64)
        max_tensor = action_value.new_tensor(self.max_values, dtype=torch.int64)
        clipped = torch.minimum(torch.maximum(rounded, min_tensor), max_tensor)

        if self.control_mode == "discrete_grid":
            projected_dims = []
            for dim_idx, allowed_values in enumerate(self.control_spec["allowed_values"]):
                allowed_tensor = action_value.new_tensor(allowed_values, dtype=torch.float32)
                distances = torch.abs(allowed_tensor - action_value[dim_idx])
                projected_dims.append(int(allowed_tensor[torch.argmin(distances)].item()))
            if self.control_dim == 1:
                return projected_dims[0], np.asarray(projected_dims[0], dtype=np.int64)
            return tuple(projected_dims), np.asarray(projected_dims, dtype=np.int64)

        if self.control_dim == 1:
            return int(clipped[0].item()), np.asarray(int(clipped[0].item()), dtype=np.int64)
        projected = tuple(int(value) for value in clipped.detach().cpu().tolist())
        return projected, np.asarray(projected, dtype=np.int64)

    def _finalize_action(self, projected_controls, state):
        from invman.problems.dual_sourcing.policies import apply_action_adapter

        controls = np.atleast_1d(projected_controls).astype(np.int64).tolist()
        normalized_state = state.detach().cpu().numpy()
        return apply_action_adapter(
            self.action_adapter,
            controls,
            normalized_state,
            self.action_spec,
            self.action_adapter_config,
        )

    def forward(self, state, return_features=False):
        if state.dim() != 1:
            raise ValueError("SoftTreePolicy expects a single 1D state vector")

        state = state.to(dtype=torch.float32)
        split_probs, leaf_probs, selected_feature_idx, selected_feature_weight = self._leaf_probabilities(state)
        leaf_quantities, raw_leaf_output = self._leaf_quantities(state)
        action_value = torch.sum(leaf_probs.unsqueeze(-1) * leaf_quantities, dim=0)
        projected_controls, projected_array = self._project_controls(action_value)
        action = self._finalize_action(projected_array, state)

        if return_features:
            self.features["split_probs"] = split_probs.detach().cpu().numpy()
            self.features["leaf_probs"] = leaf_probs.detach().cpu().numpy()
            leaf_quantities_np = leaf_quantities.detach().cpu().numpy()
            self.features["leaf_quantities"] = (
                leaf_quantities_np[:, 0] if self.control_dim == 1 else leaf_quantities_np
            )
            action_value_np = action_value.detach().cpu().numpy()
            self.features["action_value"] = (
                float(action_value_np.item()) if self.control_dim == 1 else action_value_np
            )
            self.features["projected_controls"] = projected_array
            self.features["projected_action"] = np.asarray(action)
            self.features["split_type"] = self.split_type
            self.features["leaf_type"] = self.leaf_type
            self.features["action_spec"] = dict(self.action_spec)
            self.features["control_spec"] = dict(self.control_spec)
            self.features["action_adapter"] = self.action_adapter
            if raw_leaf_output is not None:
                raw_leaf_output_np = raw_leaf_output.detach().cpu().numpy()
                self.features["raw_leaf_output"] = (
                    raw_leaf_output_np[:, 0] if self.control_dim == 1 else raw_leaf_output_np
                )
            if selected_feature_idx is not None:
                self.features["selected_feature_idx"] = selected_feature_idx
                self.features["selected_feature_weight"] = selected_feature_weight
            return action, self.features
        return action
