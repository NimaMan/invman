"""Self-contained learned-policy descriptor for CMA-ES training over a Rust rollout.

OBJECTIVE
---------
Python's only job in this codebase is to OPTIMIZE policy parameters with CMA-ES;
the policy forward pass and the environment rollout live in Rust (invman_rust,
reached over PyO3). This module defines the single object that crosses that
boundary: a `Policy` that fully describes a learned controller -- its
architecture, its ACTION BOUNDS, and its input normalization -- bundled together
with its flat parameter vector. "The bounds are part of the policy itself."

Everything here is data + bookkeeping. There is deliberately NO forward()/rollout
method: given a `Policy`, `invman.rollout_fitness` hands its fields to the Rust
rollout, which performs inference and returns a cost. CMA-ES never inspects the
weights' initial values -- it seeds its search from `num_params` alone (mean 0,
sigma_init) -- so this object only needs the parameter COUNT to be correct, plus
the architecture/bound fields the Rust call reads.

POLICY vs PROBLEM
-----------------
A `Policy` carries only policy-defining fields:
  - backbone: "soft_tree" | "linear" | "nn"
  - flat_params: the trained weights (np.float32, length == num_params)
  - input_dim: state-vector length
  - soft_tree: depth, temperature, split_type, leaf_type
  - dense (linear/nn): output_dim, action_output_mode (policy head),
                       and for nn: hidden_dim, activation_name
  - action bounds: control_dim, control_mode, min_values, max_values,
                   allowed_values, max_order_size, action_adapter
  - input transform: state_normalizer, state_scale
Problem/env fields (demand, costs, lead time, horizon, seed) are NOT stored here;
they are read from `args` at fitness-evaluation time.

PARAMETER LAYOUT (num_params)
-----------------------------
The flat vector is the concatenation of the per-array blocks below, in order.
Rust unpacks the same convention. With n_in = 2**depth - 1 internal nodes and
n_leaf = 2**depth leaves, control_dim = c, input_dim = i, output_dim = o:
  soft_tree, constant leaf: n_in*i + n_in + n_leaf*c
  soft_tree, linear/sigmoid_linear leaf: n_in*i + n_in + n_leaf*c*i + n_leaf*c
  linear: i*o + o
  nn (hidden widths h_1..h_k): sum over layers of (prev*width + width),
      starting prev=i through the hidden widths, then prev*o + o
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass, field

import numpy as np

ARTIFACT_VERSION = 1


@dataclass
class Policy:
    backbone: str
    input_dim: int

    # Action bounds (part of the policy itself).
    control_dim: int = 1
    control_mode: str = "scalar_quantity"
    min_values: tuple[int, ...] = (0,)
    max_values: tuple[int, ...] = (0,)
    allowed_values: list[list[int]] | None = None
    max_order_size: int | None = None
    action_adapter: str = "identity"

    # Input normalization.
    state_normalizer: str = "identity"
    state_scale: float | None = None

    # Soft-tree architecture.
    depth: int | None = None
    temperature: float | None = None
    split_type: str | None = None
    leaf_type: str | None = None

    # Dense (linear / nn) architecture.
    output_dim: int | None = None
    action_output_mode: str | None = None
    hidden_dim: tuple[int, ...] = ()
    activation_name: str | None = None

    # Trained weights; defaults to zeros (CMA-ES seeds from num_params, not these).
    flat_params: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        if self.backbone not in {"soft_tree", "linear", "nn"}:
            raise ValueError(f"Unknown policy backbone: {self.backbone}")
        self.input_dim = int(self.input_dim)
        self.control_dim = int(self.control_dim)
        self.min_values = tuple(int(v) for v in self.min_values)
        self.max_values = tuple(int(v) for v in self.max_values)
        self.hidden_dim = tuple(int(w) for w in self.hidden_dim)
        if self.flat_params is None:
            self.flat_params = np.zeros(self.num_params, dtype=np.float32)
        else:
            self.set_model_params(self.flat_params)

    # --- CMA-ES interface (es_mp.train relies on exactly these three) ---------

    @property
    def num_params(self) -> int:
        if self.backbone == "soft_tree":
            n_internal = (2 ** int(self.depth)) - 1
            n_leaf = 2 ** int(self.depth)
            count = n_internal * self.input_dim + n_internal
            if self.leaf_type == "constant":
                count += n_leaf * self.control_dim
            else:  # linear / sigmoid_linear
                count += n_leaf * self.control_dim * self.input_dim + n_leaf * self.control_dim
            return int(count)
        if self.backbone == "linear":
            return int(self.input_dim * int(self.output_dim) + int(self.output_dim))
        # nn
        count = 0
        prev = self.input_dim
        for width in self.hidden_dim:
            count += prev * width + width
            prev = width
        count += prev * int(self.output_dim) + int(self.output_dim)
        return int(count)

    def get_model_flat_params(self) -> np.ndarray:
        return np.asarray(self.flat_params, dtype=np.float32)

    def set_model_params(self, flat_params) -> "Policy":
        flat = np.asarray(flat_params, dtype=np.float32).reshape(-1)
        if flat.size != self.num_params:
            raise ValueError(
                f"flat_params length {flat.size} != policy num_params {self.num_params}"
            )
        self.flat_params = flat
        return self

    # --- self-contained artifact (what Rust loads to run/backtest the policy) --

    def to_artifact(self) -> dict:
        """Language-neutral dict bundling architecture + bounds + weights."""
        fields = asdict(self)
        fields.pop("flat_params", None)
        return {
            "artifact_version": ARTIFACT_VERSION,
            "num_params": self.num_params,
            "flat_params": self.get_model_flat_params().tolist(),
            **fields,
        }

    def save(self, save_directory, override=False) -> None:
        if os.path.exists(save_directory):
            if not os.path.isdir(save_directory):
                raise NotADirectoryError(f"Save target is an existing file: {save_directory}")
            if not override:
                raise FileExistsError(f"Save directory already exists: {save_directory}")
            shutil.rmtree(save_directory)
        os.makedirs(save_directory)
        np.save(os.path.join(save_directory, "model_params.npy"),
                self.get_model_flat_params(), allow_pickle=False)
        with open(os.path.join(save_directory, "policy_artifact.json"), "w", encoding="utf-8") as fh:
            json.dump(self.to_artifact(), fh, indent=2, sort_keys=True)

    @classmethod
    def load(cls, save_directory) -> "Policy":
        with open(os.path.join(save_directory, "policy_artifact.json"), "r", encoding="utf-8") as fh:
            artifact = json.load(fh)
        artifact.pop("artifact_version", None)
        artifact.pop("num_params", None)
        flat = artifact.pop("flat_params", None)
        params_path = os.path.join(save_directory, "model_params.npy")
        if os.path.exists(params_path):
            flat = np.load(params_path, allow_pickle=False)
        policy = cls(**artifact)
        if flat is not None:
            policy.set_model_params(flat)
        return policy
