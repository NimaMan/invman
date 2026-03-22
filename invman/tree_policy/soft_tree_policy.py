import torch
import torch.nn as nn
import torch.nn.functional as F

from invman.nn.es_module import ESModule
from invman.utils import save_init_args


class SoftTreePolicy(ESModule):
    @save_init_args
    def __init__(self, input_dim, max_order_size, depth=2, temperature=0.25):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.input_dim = int(input_dim)
        self.max_order_size = int(max_order_size)
        self.depth = int(depth)
        self.temperature = float(temperature)
        self.num_internal_nodes = (2 ** self.depth) - 1
        self.num_leaves = 2 ** self.depth

        self.split_weights = nn.Parameter(torch.empty(self.num_internal_nodes, self.input_dim))
        self.split_bias = nn.Parameter(torch.empty(self.num_internal_nodes))
        self.leaf_logits = nn.Parameter(torch.empty(self.num_leaves))
        self.features = {}
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.split_weights, mean=0.0, std=0.15)
        nn.init.normal_(self.split_bias, mean=0.0, std=0.15)
        nn.init.normal_(self.leaf_logits, mean=0.0, std=0.15)

    def _leaf_probabilities(self, state):
        logits = F.linear(state.unsqueeze(0), self.split_weights, self.split_bias).squeeze(0)
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
        return gates, level_probs

    def forward(self, state, return_features=False):
        if state.dim() != 1:
            raise ValueError("SoftTreePolicy expects a single 1D state vector")

        state = state.to(dtype=torch.float32)
        split_probs, leaf_probs = self._leaf_probabilities(state)
        leaf_quantities = torch.sigmoid(self.leaf_logits) * float(self.max_order_size)
        action_value = torch.sum(leaf_probs * leaf_quantities)
        action = torch.round(action_value).to(dtype=torch.int64)
        action = torch.clamp(action, min=0, max=int(self.max_order_size))

        if return_features:
            self.features["split_probs"] = split_probs.detach().cpu().numpy()
            self.features["leaf_probs"] = leaf_probs.detach().cpu().numpy()
            self.features["leaf_quantities"] = leaf_quantities.detach().cpu().numpy()
            self.features["action_value"] = float(action_value.detach().cpu().item())
            return int(action.item()), self.features
        return int(action.item())
