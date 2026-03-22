import torch
import torch.nn as nn

from invman.nn.es_module import ESModule
from invman.utils import save_init_args


def _normalize_action_output_mode(action_output_mode: str) -> str:
    aliases = {
        "categorical_quantity": "categorical_quantity",
        "direct_quantity": "direct_quantity",
        "gated_ordinal_quantity": "gated_ordinal_quantity",
        "discrete_logits": "categorical_quantity",
        "scalar_quantity": "direct_quantity",
    }
    normalized = aliases.get(action_output_mode)
    if normalized is None:
        valid = ", ".join(sorted(aliases))
        raise ValueError(f"Unknown action_output_mode '{action_output_mode}'. Expected one of: {valid}")
    return normalized


class LinearPolicyNet(ESModule):
    @save_init_args
    def __init__(
        self,
        input_dim,
        output_dim,
        output_activation=None,
        action_output_mode="discrete_logits",
        max_order_size=None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_activation = output_activation
        self.action_output_mode = _normalize_action_output_mode(action_output_mode)
        self.max_order_size = max_order_size
        out_features = output_dim if self.action_output_mode in {"categorical_quantity", "gated_ordinal_quantity"} else 1
        if self.action_output_mode == "direct_quantity" and max_order_size is None:
            raise ValueError("max_order_size is required when action_output_mode='direct_quantity'")
        if self.action_output_mode == "gated_ordinal_quantity" and max_order_size is None:
            raise ValueError("max_order_size is required when action_output_mode='gated_ordinal_quantity'")

        self.linear_app = nn.Linear(in_features=input_dim, out_features=out_features)
        self.features = {}

    def forward(self, state, return_features=False):
        raw_output = self.linear_app(state)
        if self.action_output_mode == "categorical_quantity":
            action = torch.argmax(raw_output, dim=-1)
        elif self.action_output_mode == "direct_quantity":
            scaled_quantity = torch.sigmoid(raw_output.squeeze(-1)) * float(self.max_order_size)
            action = torch.round(scaled_quantity).to(dtype=torch.int64)
        elif self.action_output_mode == "gated_ordinal_quantity":
            gate_logit = raw_output[..., 0]
            ordinal_logits = raw_output[..., 1:]
            gate_prob = torch.sigmoid(gate_logit)
            quantity_score = torch.sigmoid(ordinal_logits).sum(dim=-1)
            action = torch.round(gate_prob * quantity_score).to(dtype=torch.int64)
            action = torch.clamp(action, min=0, max=int(self.max_order_size))
        else:
            raise NotImplementedError(f"Unknown action_output_mode: {self.action_output_mode}")

        if return_features:
            self.features["linear"] = raw_output.detach().cpu().numpy()
            if self.action_output_mode == "gated_ordinal_quantity":
                self.features["gate_prob"] = gate_prob.detach().cpu().numpy()
                self.features["quantity_score"] = quantity_score.detach().cpu().numpy()
            return int(action.item()), self.features
        return int(action.item())
