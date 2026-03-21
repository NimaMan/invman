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


class PolicyNet(ESModule):
    @save_init_args
    def __init__(self, input_dim, hidden_dim, output_dim, activation="selu"):
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
        self.layers = nn.ModuleList()
        in_features = input_dim
        for width in hidden_layers:
            self.layers.append(nn.Linear(in_features=in_features, out_features=width))
            in_features = width
        self.output_layer = nn.Linear(in_features=in_features, out_features=output_dim)
        self.features = {}

    def forward(self, state, return_features=False):
        h = state
        for layer_idx, layer in enumerate(self.layers):
            h = self.activation(layer(h))
            if return_features:
                self.features[layer_idx] = h.detach().cpu().numpy()
        logits = self.output_layer(h)
        action = torch.argmax(logits, dim=-1)
        if return_features:
            self.features["logits"] = logits.detach().cpu().numpy()
            return int(action.item()), self.features
        return int(action.item())
