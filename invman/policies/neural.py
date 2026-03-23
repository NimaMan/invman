import torch
import torch.nn as nn

from invman.policies.common import get_activation_function, normalize_action_spec, normalize_policy_head
from invman.policies.es_module import ESModule
from invman.policies.structured_actions import apply_structured_action_adapter, normalize_tree_action_adapter
from invman.utils import save_init_args


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
        action_spec=None,
        control_spec=None,
        action_adapter="identity",
        action_adapter_config=None,
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
        self.action_output_mode = normalize_policy_head(action_output_mode)
        self.max_order_size = max_order_size
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
        self.action_adapter = normalize_tree_action_adapter(action_adapter)
        self.action_adapter_config = None if action_adapter_config is None else dict(action_adapter_config)
        self.layers = nn.ModuleList()
        in_features = input_dim
        for width in hidden_layers:
            self.layers.append(nn.Linear(in_features=in_features, out_features=width))
            in_features = width
        out_features = (
            output_dim
            if self.action_output_mode in {"categorical_quantity", "gated_ordinal_quantity", "two_stage_ordinal_quantity"}
            else self.control_dim
        )
        if self.action_output_mode == "direct_quantity" and max_order_size is None:
            raise ValueError("max_order_size is required when action_output_mode='direct_quantity'")
        if self.action_output_mode in {"gated_ordinal_quantity", "two_stage_ordinal_quantity"} and max_order_size is None:
            raise ValueError(
                "max_order_size is required when action_output_mode is an ordinal quantity head"
            )
        if self.action_output_mode == "categorical_quantity" and (
            self.control_dim != 1 or self.control_mode != "scalar_quantity"
        ):
            raise ValueError("categorical_quantity requires a scalar_quantity control spec.")
        if self.action_output_mode in {"direct_quantity", "gated_ordinal_quantity", "two_stage_ordinal_quantity"} and (
            self.control_dim != 1 or self.control_mode != "scalar_quantity"
        ):
            raise ValueError(f"{self.action_output_mode} requires a scalar_quantity control spec.")
        if self.action_output_mode == "bounded_quantity" and self.control_dim < 1:
            raise ValueError("bounded_quantity requires at least one control dimension.")

        self.output_layer = nn.Linear(in_features=in_features, out_features=out_features)
        self.features = {}

    def _project_controls(self, control_value):
        rounded = torch.round(control_value).to(dtype=torch.int64)
        min_tensor = control_value.new_tensor(self.min_values, dtype=torch.int64)
        max_tensor = control_value.new_tensor(self.max_values, dtype=torch.int64)
        clipped = torch.minimum(torch.maximum(rounded, min_tensor), max_tensor)

        if self.control_mode == "discrete_grid":
            projected_dims = []
            for dim_idx, allowed_values in enumerate(self.control_spec["allowed_values"]):
                allowed_tensor = control_value.new_tensor(allowed_values, dtype=torch.float32)
                distances = torch.abs(allowed_tensor - control_value[dim_idx])
                projected_dims.append(int(allowed_tensor[torch.argmin(distances)].item()))
            if self.control_dim == 1:
                return projected_dims[0], projected_dims
            return tuple(projected_dims), projected_dims

        if self.control_dim == 1:
            scalar = int(clipped[0].item())
            return scalar, [scalar]
        projected = [int(value) for value in clipped.detach().cpu().tolist()]
        return tuple(projected), projected

    def _finalize_action(self, projected_controls, state):
        return apply_structured_action_adapter(
            self.action_adapter,
            projected_controls,
            state.detach().cpu().numpy(),
            self.action_spec,
            self.action_adapter_config,
        )

    def forward(self, state, return_features=False):
        h = state
        for layer_idx, layer in enumerate(self.layers):
            h = self.activation(layer(h))
            if return_features:
                self.features[layer_idx] = h.detach().cpu().numpy()

        raw_output = self.output_layer(h)
        if self.action_output_mode == "categorical_quantity":
            action = torch.argmax(raw_output, dim=-1)
            projected_controls = [int(action.item())]
        elif self.action_output_mode == "direct_quantity":
            scaled_quantity = torch.sigmoid(raw_output.squeeze(-1)) * float(self.max_order_size)
            action = torch.round(scaled_quantity).to(dtype=torch.int64)
            projected_controls = [int(action.item())]
        elif self.action_output_mode == "gated_ordinal_quantity":
            gate_logit = raw_output[..., 0]
            ordinal_logits = raw_output[..., 1:]
            gate_prob = torch.sigmoid(gate_logit)
            quantity_score = torch.sigmoid(ordinal_logits).sum(dim=-1)
            action = torch.round(gate_prob * quantity_score).to(dtype=torch.int64)
            action = torch.clamp(action, min=0, max=int(self.max_order_size))
            projected_controls = [int(action.item())]
        elif self.action_output_mode == "two_stage_ordinal_quantity":
            gate_logit = raw_output[..., 0]
            ordinal_logits = raw_output[..., 1:]
            gate_prob = torch.sigmoid(gate_logit)
            quantity_score = torch.sigmoid(ordinal_logits).sum(dim=-1)
            order_flag = gate_prob >= 0.5
            positive_action = torch.round(quantity_score).to(dtype=torch.int64)
            positive_action = torch.clamp(positive_action, min=1, max=int(self.max_order_size))
            action = torch.where(order_flag, positive_action, torch.zeros_like(positive_action))
            projected_controls = [int(action.item())]
        elif self.action_output_mode == "bounded_quantity":
            min_tensor = raw_output.new_tensor(self.min_values, dtype=torch.float32)
            max_tensor = raw_output.new_tensor(self.max_values, dtype=torch.float32)
            scaled_controls = min_tensor + torch.sigmoid(raw_output) * (max_tensor - min_tensor)
            _, projected_controls = self._project_controls(scaled_controls)
            action = self._finalize_action(projected_controls, state)
        else:
            raise NotImplementedError(f"Unknown action_output_mode: {self.action_output_mode}")

        if return_features:
            self.features["raw_output"] = raw_output.detach().cpu().numpy()
            if self.action_output_mode in {"gated_ordinal_quantity", "two_stage_ordinal_quantity"}:
                self.features["gate_prob"] = gate_prob.detach().cpu().numpy()
                self.features["quantity_score"] = quantity_score.detach().cpu().numpy()
                if self.action_output_mode == "two_stage_ordinal_quantity":
                    self.features["order_flag"] = order_flag.detach().cpu().numpy()
            if self.action_output_mode == "bounded_quantity":
                self.features["projected_controls"] = projected_controls
                self.features["action_adapter"] = self.action_adapter
                self.features["projected_action"] = action
                self.features["raw_output"] = raw_output.detach().cpu().numpy()
                return action, self.features
            return int(action.item()), self.features
        if self.action_output_mode == "bounded_quantity":
            return action
        return int(action.item())
