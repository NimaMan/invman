import torch
import torch.nn as nn

from invman.nn.es_module import ESModule
from invman.utils import save_init_args


class LinearPolicyNet(ESModule):
    @save_init_args
    def __init__(self, input_dim, output_dim, output_activation=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_activation = output_activation
        self.linear_app = nn.Linear(in_features=input_dim, out_features=output_dim)
        self.features = {}

    def forward(self, state, return_features=False):
        logits = self.linear_app(state)
        action = torch.argmax(logits, dim=-1)
        if return_features:
            self.features["linear"] = logits.detach().cpu().numpy()
            return int(action.item()), self.features
        return int(action.item())
