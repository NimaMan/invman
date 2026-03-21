import copy
import importlib
import json
import os
import shutil
import sys

import numpy as np
import torch
from torch.nn import Module


class ESModule(Module):

    def save(self, save_directory, override=False):
        """Save initial parameters and pytorch model parameters to a given directory.

        Parameters
        ----------
        save_directory : string
            Directory to which to save the model parameters and rufous config.

        override : bool
            If true, removes and recreates save_directory if it already exists.
        """
        if os.path.exists(save_directory):
            if not os.path.isdir(save_directory):
                raise NotADirectoryError(
                    'The directory to which to save the model is an existing file: "{}"'.format(save_directory)
                )
            if not override:
                raise FileExistsError(
                    'The directory to which to save the model, already exists: "{}"'.format(save_directory)
                )
            else:
                shutil.rmtree(save_directory)

        os.makedirs(save_directory)
        torch.save(self.state_dict(), os.path.join(save_directory, "model_params.torch"))
        config = {
            "init_args": self._init_args,
            "init_kwargs": self._init_kwargs,
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "model_type": type(self).__module__ + "." + type(self).__qualname__,
        }
        with open(os.path.join(save_directory, "model_config.json"), "w") as f:
            json.dump(config, f)

    @staticmethod
    def load(directory, device="cpu", strict=True):
        """Load model from a given directory.
        Parameters
        ----------
        directory: str
            Base dirrectory to which the model was saved.
        device: str
            String representation of the pytorch device to which this model should be mapped.
            This allows (among other things) to load models trained on a GPU to be loaded on a CPU only system.
            Defaults to `cpu` for compatibility.
        strict: boolean
            Flag used for `torch.nn.module.load_state_dict` which can force strict agreement between the model's
            registered modules and all keys in the stored file. Use False, if the models has been modified
            by adding more weights to it, e.g., by using it in combination with a pyro-ppl model where some of
            the pytorch variables where registered with the rufous model.
        """
        with open(os.path.join(directory, "model_config.json"), "r") as f:
            config = json.load(f)

        model_type = config["model_type"]
        module_str, class_str = model_type.rsplit(".", 1)
        module = importlib.import_module(module_str)
        model_class = getattr(module, class_str)
        model = model_class(*config["init_args"], **config["init_kwargs"])

        model.load_state_dict(
            torch.load(
                os.path.join(directory, "model_params.torch"),
                map_location=torch.device(device),
            ),
            strict=strict,
        )
        # set to evaluation mode on load, as the default in pytorch is training mode
        model.eval()
        return model

    def get_model_shapes(self):
        model_shapes = []
        for param in self.parameters():
            p = param.data.cpu().numpy()
            model_shapes.append(p.shape)
            param.requires_grad = False
        return model_shapes

    @property
    def model_shapes(self, ):
        return self.get_model_shapes()

    def set_model_params(self, flat_params):
        model_shapes = self.model_shapes
        idx = 0

        for i, param in enumerate(self.parameters()):
            delta = np.product(model_shapes[i])
            block = flat_params[idx: idx + delta]
            block = np.reshape(block, model_shapes[i])
            idx += delta
            block_data = torch.from_numpy(block).float()
            param.data = block_data

        return self

    def load_model(self, filename):
        with open(filename) as f:
            data = json.load(f)
        print("loading file %s" % (filename))
        self.data = data
        model_params = np.array(data[0])  # assuming other stuff is in data
        self.set_model_params(model_params)

    def count_model_params(self):
        orig_model = copy.deepcopy(self)
        orig_params = []
        model_shapes = []
        for param in orig_model.parameters():
            p = param.data.cpu().numpy()
            model_shapes.append(p.shape)
            orig_params.append(p.flatten())
        orig_params_flat = np.concatenate(orig_params)

        return len(orig_params_flat)

    @property
    def num_params(self):
        return self.count_model_params()
    
    def get_model_params(
        self,
    ):
        # Get the model params in the original order and shape
        orig_model = copy.deepcopy(self)
        orig_params = []
        model_shapes = []
        for param in orig_model.parameters():
            p = param.data.cpu().numpy()
            model_shapes.append(p.shape)
            orig_params.append(p)

        return orig_params, model_shapes        

    def get_model_flat_params(
        self,
    ):
        orig_model = copy.deepcopy(self)
        orig_params = []
        for param in orig_model.parameters():
            p = param.data.cpu().numpy()
            orig_params.append(p.flatten())
        orig_params_flat = np.concatenate(orig_params)
        return orig_params_flat

    def reset_parameters(self):
        def weight_reset(m):
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv1d):
                m.reset_parameters()

        self.apply(weight_reset)