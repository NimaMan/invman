import torch
import torch.nn as nn
import torch.nn.functional as F

from invman.nn.es_module import ESModule
from invman.utils import save_init_args


def get_activation_function(activation="gelu"):
    if activation == "selu":
        return F.selu
    if activation == "gelu":
        return F.gelu
    if activation == "relu":
        return F.relu
    raise NotImplementedError(f"Unsupported activation: {activation}")


def _normalize_action_output_mode(action_output_mode: str) -> str:
    aliases = {
        "categorical_quantity": "categorical_quantity",
        "direct_quantity": "direct_quantity",
        "gated_ordinal_quantity": "gated_ordinal_quantity",
        "two_stage_ordinal_quantity": "two_stage_ordinal_quantity",
        "conditional_ordinal_quantity": "two_stage_ordinal_quantity",
        "discrete_logits": "categorical_quantity",
        "scalar_quantity": "direct_quantity",
    }
    normalized = aliases.get(action_output_mode)
    if normalized is None:
        valid = ", ".join(sorted(aliases))
        raise ValueError(f"Unknown action_output_mode '{action_output_mode}'. Expected one of: {valid}")
    return normalized


class PolicyNet(ESModule):
    @save_init_args
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        activation="selu",
        action_output_mode="discrete_logits",
        max_order_size=None,
    ):
        super().__init__()
        if isinstance(hidden_dim, int):
            hidden_layers = [hidden_dim]
        else:
            hidden_layers = list(hidden_dim)
        if not hidden_layers:
            raise ValueError("hidden_dim must contain at least one hidden layer")

        self.input_dim = input_dim
        self.hidden_dim = hidden_layers
        self.output_dim = output_dim
        self.activation = get_activation_function(activation)
        self.action_output_mode = _normalize_action_output_mode(action_output_mode)
        self.max_order_size = max_order_size
        self.layers = nn.ModuleList()
        in_features = input_dim
        for width in hidden_layers:
            self.layers.append(nn.Linear(in_features=in_features, out_features=width))
            in_features = width
        out_features = (
            output_dim
            if self.action_output_mode in {"categorical_quantity", "gated_ordinal_quantity", "two_stage_ordinal_quantity"}
            else 1
        )
        if self.action_output_mode == "direct_quantity" and max_order_size is None:
            raise ValueError("max_order_size is required when action_output_mode='direct_quantity'")
        if self.action_output_mode in {"gated_ordinal_quantity", "two_stage_ordinal_quantity"} and max_order_size is None:
            raise ValueError(
                "max_order_size is required when action_output_mode is an ordinal quantity head"
            )

        self.output_layer = nn.Linear(in_features=in_features, out_features=out_features)
        self.features = {}

    def forward(self, state, return_features=False):
        h = state
        for layer_idx, layer in enumerate(self.layers):
            h = self.activation(layer(h))
            if return_features:
                self.features[layer_idx] = h.detach().cpu().numpy()

        raw_output = self.output_layer(h)
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
        elif self.action_output_mode == "two_stage_ordinal_quantity":
            gate_logit = raw_output[..., 0]
            ordinal_logits = raw_output[..., 1:]
            gate_prob = torch.sigmoid(gate_logit)
            quantity_score = torch.sigmoid(ordinal_logits).sum(dim=-1)
            order_flag = gate_prob >= 0.5
            positive_action = torch.round(quantity_score).to(dtype=torch.int64)
            positive_action = torch.clamp(positive_action, min=1, max=int(self.max_order_size))
            action = torch.where(order_flag, positive_action, torch.zeros_like(positive_action))
        else:
            raise NotImplementedError(f"Unknown action_output_mode: {self.action_output_mode}")

        if return_features:
            self.features["raw_output"] = raw_output.detach().cpu().numpy()
            if self.action_output_mode in {"gated_ordinal_quantity", "two_stage_ordinal_quantity"}:
                self.features["gate_prob"] = gate_prob.detach().cpu().numpy()
                self.features["quantity_score"] = quantity_score.detach().cpu().numpy()
                if self.action_output_mode == "two_stage_ordinal_quantity":
                    self.features["order_flag"] = order_flag.detach().cpu().numpy()
            return int(action.item()), self.features
        return int(action.item())
