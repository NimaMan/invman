import torch
import torch.nn as nn
import torch.nn.functional as F

from invman.policies.common import normalize_tree_leaf_type, normalize_tree_split_type
from invman.policies.es_module import ESModule
from invman.utils import save_init_args


class SoftTreePolicy(ESModule):
    @save_init_args
    def __init__(
        self,
        input_dim,
        max_order_size,
        depth=2,
        temperature=0.25,
        split_type="oblique",
        leaf_type="constant",
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.input_dim = int(input_dim)
        self.max_order_size = int(max_order_size)
        self.depth = int(depth)
        self.temperature = float(temperature)
        self.split_type = normalize_tree_split_type(split_type)
        self.leaf_type = normalize_tree_leaf_type(leaf_type)
        self.num_internal_nodes = (2 ** self.depth) - 1
        self.num_leaves = 2 ** self.depth

        self.split_weights = nn.Parameter(torch.empty(self.num_internal_nodes, self.input_dim))
        self.split_bias = nn.Parameter(torch.empty(self.num_internal_nodes))
        if self.leaf_type == "constant":
            self.leaf_logits = nn.Parameter(torch.empty(self.num_leaves))
            self.leaf_weights = None
            self.leaf_bias = None
        elif self.leaf_type == "linear":
            self.leaf_weights = nn.Parameter(torch.empty(self.num_leaves, self.input_dim))
            self.leaf_bias = nn.Parameter(torch.empty(self.num_leaves))
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
        elif self.leaf_type == "linear":
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

    def _leaf_quantities(self, state):
        if self.leaf_type == "constant":
            return torch.sigmoid(self.leaf_logits) * float(self.max_order_size), None
        if self.leaf_type == "linear":
            raw_leaf_output = F.linear(state.unsqueeze(0), self.leaf_weights, self.leaf_bias).squeeze(0)
            return torch.sigmoid(raw_leaf_output) * float(self.max_order_size), raw_leaf_output
        raise NotImplementedError(f"Unknown tree leaf type: {self.leaf_type}")

    def forward(self, state, return_features=False):
        if state.dim() != 1:
            raise ValueError("SoftTreePolicy expects a single 1D state vector")

        state = state.to(dtype=torch.float32)
        split_probs, leaf_probs, selected_feature_idx, selected_feature_weight = self._leaf_probabilities(state)
        leaf_quantities, raw_leaf_output = self._leaf_quantities(state)
        action_value = torch.sum(leaf_probs * leaf_quantities)
        action = torch.round(action_value).to(dtype=torch.int64)
        action = torch.clamp(action, min=0, max=int(self.max_order_size))

        if return_features:
            self.features["split_probs"] = split_probs.detach().cpu().numpy()
            self.features["leaf_probs"] = leaf_probs.detach().cpu().numpy()
            self.features["leaf_quantities"] = leaf_quantities.detach().cpu().numpy()
            self.features["action_value"] = float(action_value.detach().cpu().item())
            self.features["split_type"] = self.split_type
            self.features["leaf_type"] = self.leaf_type
            if raw_leaf_output is not None:
                self.features["raw_leaf_output"] = raw_leaf_output.detach().cpu().numpy()
            if selected_feature_idx is not None:
                self.features["selected_feature_idx"] = selected_feature_idx
                self.features["selected_feature_weight"] = selected_feature_weight
            return int(action.item()), self.features
        return int(action.item())
