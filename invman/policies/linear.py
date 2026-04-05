import numpy as np

from invman.policies.common import (
    as_float32_vector,
    init_linear_layer,
    normalize_state_normalizer,
    normalize_state_vector,
    normalize_action_spec,
    normalize_policy_head,
    round_nearest,
    sigmoid,
    softplus,
)
from invman.policies.es_module import ESModule
from invman.utils import save_init_args


class LinearPolicyNet(ESModule):
    @save_init_args
    def __init__(
        self,
        input_dim,
        output_dim,
        output_activation=None,
        action_output_mode="discrete_logits",
        max_order_size=None,
        action_spec=None,
        control_spec=None,
        action_adapter="identity",
        action_adapter_config=None,
        state_normalizer="identity",
        state_scale=None,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.output_activation = output_activation
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
        self.action_adapter = str(action_adapter)
        self.action_adapter_config = None if action_adapter_config is None else dict(action_adapter_config)
        self.state_normalizer = normalize_state_normalizer(state_normalizer)
        self.state_scale = None if state_scale is None else float(state_scale)
        if self.action_output_mode in {
            "categorical_quantity",
            "soft_gated_ordinal_quantity",
            "hard_gated_ordinal_quantity",
        }:
            out_features = self.output_dim
        elif self.action_output_mode in {
            "gated_sigmoid_direct_quantity",
            "soft_gated_direct_quantity",
            "hard_gated_direct_quantity",
        }:
            out_features = 2
        else:
            out_features = self.control_dim
        self.policy_output_dim = int(out_features)
        if self.action_output_mode in {
            "capped_direct_quantity",
            "sigmoid_direct_quantity",
            "gated_sigmoid_direct_quantity",
            "soft_gated_direct_quantity",
            "soft_gated_ordinal_quantity",
            "hard_gated_ordinal_quantity",
            "hard_gated_direct_quantity",
        } and max_order_size is None:
            raise ValueError(
                "max_order_size is required when action_output_mode uses a decoder-side cap"
            )
        if self.action_output_mode == "categorical_quantity" and (
            self.control_dim != 1 or self.control_mode != "scalar_quantity"
        ):
            raise ValueError("categorical_quantity requires a scalar_quantity control spec.")
        if self.action_output_mode in {
            "direct_quantity",
            "capped_direct_quantity",
            "sigmoid_direct_quantity",
            "gated_sigmoid_direct_quantity",
            "soft_gated_direct_quantity",
            "soft_gated_ordinal_quantity",
            "hard_gated_ordinal_quantity",
            "hard_gated_direct_quantity",
        } and (
            self.control_dim != 1 or self.control_mode != "scalar_quantity"
        ):
            raise ValueError(f"{self.action_output_mode} requires a scalar_quantity control spec.")
        if self.action_output_mode == "bounded_quantity" and self.control_dim < 1:
            raise ValueError("bounded_quantity requires at least one control dimension.")

        self.linear_weight, self.linear_bias = init_linear_layer(self.input_dim, self.policy_output_dim)
        self.features = {}

    def parameter_arrays(self):
        return [self.linear_weight, self.linear_bias]

    def _project_controls(self, control_value):
        control_value = np.asarray(control_value, dtype=np.float32).reshape(self.control_dim)
        rounded = round_nearest(control_value).astype(np.int64)
        clipped = np.clip(
            rounded,
            np.asarray(self.min_values, dtype=np.int64),
            np.asarray(self.max_values, dtype=np.int64),
        )

        if self.control_mode == "discrete_grid":
            projected_dims = []
            for dim_idx, allowed_values in enumerate(self.control_spec["allowed_values"]):
                allowed_tensor = np.asarray(allowed_values, dtype=np.float32)
                distances = np.abs(allowed_tensor - control_value[dim_idx])
                projected_dims.append(int(allowed_tensor[int(np.argmin(distances))]))
            if self.control_dim == 1:
                return projected_dims[0], projected_dims
            return tuple(projected_dims), projected_dims

        if self.control_dim == 1:
            scalar = int(clipped[0])
            return scalar, [scalar]
        projected = [int(value) for value in clipped.tolist()]
        return tuple(projected), projected

    def _finalize_action(self, projected_controls, state):
        from invman.problems.dual_sourcing.policies import apply_action_adapter

        return apply_action_adapter(
            self.action_adapter,
            projected_controls,
            np.asarray(state, dtype=np.float32),
            self.action_spec,
            self.action_adapter_config,
        )

    def forward(self, state, return_features=False):
        raw_state = as_float32_vector(state)
        state = normalize_state_vector(
            raw_state,
            state_normalizer=self.state_normalizer,
            state_scale=self.state_scale,
        )
        raw_output = (self.linear_weight @ state + self.linear_bias).astype(np.float32, copy=False)
        features = {}

        if self.action_output_mode == "categorical_quantity":
            action = int(np.argmax(raw_output))
            projected_controls = [action]
        elif self.action_output_mode == "direct_quantity":
            quantity_value = float(softplus(raw_output[0]))
            action = int(max(round_nearest(np.asarray(quantity_value))[()], 0))
            projected_controls = [action]
        elif self.action_output_mode == "capped_direct_quantity":
            quantity_value = softplus(raw_output[0])
            _, projected_controls = self._project_controls(np.asarray([quantity_value], dtype=np.float32))
            action = self._finalize_action(projected_controls, raw_state)
        elif self.action_output_mode == "sigmoid_direct_quantity":
            quantity_value = float(sigmoid(raw_output[0]) * float(self.max_order_size))
            action = int(round_nearest(np.asarray(quantity_value))[()])
            projected_controls = [action]
        elif self.action_output_mode == "soft_gated_direct_quantity":
            gate_logit = float(raw_output[0])
            quantity_logit = float(raw_output[1])
            gate_prob = float(sigmoid(gate_logit))
            quantity_value = float(softplus(quantity_logit))
            action = int(np.clip(round_nearest(np.asarray(gate_prob * quantity_value))[()], 0, int(self.max_order_size)))
            projected_controls = [action]
        elif self.action_output_mode == "gated_sigmoid_direct_quantity":
            gate_logit = float(raw_output[0])
            quantity_logit = float(raw_output[1])
            gate_prob = float(sigmoid(gate_logit))
            quantity_value = float(sigmoid(quantity_logit) * float(self.max_order_size))
            action = int(np.clip(round_nearest(np.asarray(gate_prob * quantity_value))[()], 0, int(self.max_order_size)))
            projected_controls = [action]
        elif self.action_output_mode == "hard_gated_direct_quantity":
            gate_logit = float(raw_output[0])
            quantity_logit = float(raw_output[1])
            gate_prob = float(sigmoid(gate_logit))
            order_flag = gate_prob >= 0.5
            quantity_value = float(softplus(quantity_logit))
            positive_action = int(np.clip(round_nearest(np.asarray(quantity_value))[()], 1, int(self.max_order_size)))
            action = positive_action if order_flag else 0
            projected_controls = [action]
        elif self.action_output_mode == "soft_gated_ordinal_quantity":
            gate_logit = float(raw_output[0])
            ordinal_logits = raw_output[1:]
            gate_prob = float(sigmoid(gate_logit))
            quantity_score = float(np.sum(sigmoid(ordinal_logits)))
            action = int(np.clip(round_nearest(np.asarray(gate_prob * quantity_score))[()], 0, int(self.max_order_size)))
            projected_controls = [action]
        elif self.action_output_mode == "hard_gated_ordinal_quantity":
            gate_logit = float(raw_output[0])
            ordinal_logits = raw_output[1:]
            gate_prob = float(sigmoid(gate_logit))
            quantity_score = float(np.sum(sigmoid(ordinal_logits)))
            order_flag = gate_prob >= 0.5
            positive_action = int(np.clip(round_nearest(np.asarray(quantity_score))[()], 1, int(self.max_order_size)))
            action = positive_action if order_flag else 0
            projected_controls = [action]
        elif self.action_output_mode == "bounded_quantity":
            min_tensor = np.asarray(self.min_values, dtype=np.float32)
            max_tensor = np.asarray(self.max_values, dtype=np.float32)
            scaled_controls = min_tensor + sigmoid(raw_output) * (max_tensor - min_tensor)
            _, projected_controls = self._project_controls(scaled_controls)
            action = self._finalize_action(projected_controls, raw_state)
        else:
            raise NotImplementedError(f"Unknown action_output_mode: {self.action_output_mode}")

        if return_features:
            features["linear"] = raw_output.copy()
            if self.action_output_mode in {
                "gated_sigmoid_direct_quantity",
                "soft_gated_direct_quantity",
                "soft_gated_ordinal_quantity",
                "hard_gated_ordinal_quantity",
                "hard_gated_direct_quantity",
            }:
                features["gate_prob"] = np.asarray(gate_prob, dtype=np.float32)
                if self.action_output_mode in {
                    "soft_gated_direct_quantity",
                    "gated_sigmoid_direct_quantity",
                    "hard_gated_direct_quantity",
                }:
                    features["quantity_value"] = np.asarray(quantity_value, dtype=np.float32)
                else:
                    features["quantity_score"] = np.asarray(quantity_score, dtype=np.float32)
                if self.action_output_mode in {
                    "hard_gated_direct_quantity",
                    "hard_gated_ordinal_quantity",
                }:
                    features["order_flag"] = np.asarray(order_flag)
            if self.action_output_mode in {
                "bounded_quantity",
                "direct_quantity",
                "capped_direct_quantity",
            }:
                features["projected_controls"] = projected_controls
            features["normalized_state"] = state.copy()
            if self.action_output_mode == "direct_quantity":
                features["quantity_value"] = np.asarray(quantity_value, dtype=np.float32)
                features["projected_action"] = int(action)
            if self.action_output_mode == "capped_direct_quantity":
                features["quantity_value"] = np.asarray(quantity_value, dtype=np.float32)
                features["projected_action"] = action
            if self.action_output_mode == "bounded_quantity":
                features["action_adapter"] = self.action_adapter
                features["projected_action"] = action
            self.features = features
            if self.action_output_mode in {
                "bounded_quantity",
                "capped_direct_quantity",
            }:
                return action, self.features
            return int(action), self.features

        self.features = {}
        if self.action_output_mode in {
            "bounded_quantity",
            "capped_direct_quantity",
        }:
            return action
        return int(action)
