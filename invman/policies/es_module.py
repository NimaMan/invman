import importlib
import json
import os
import shutil
import sys

import numpy as np


class ESModule:
    def __init__(self):
        self.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def train(self, mode=True):
        self.training = bool(mode)
        return self

    def eval(self):
        return self.train(False)

    def parameter_arrays(self):
        raise NotImplementedError("Subclasses must return parameter arrays in flat-parameter order")

    def save(self, save_directory, override=False):
        if os.path.exists(save_directory):
            if not os.path.isdir(save_directory):
                raise NotADirectoryError(
                    'The directory to which to save the model is an existing file: "{}"'.format(save_directory)
                )
            if not override:
                raise FileExistsError(
                    'The directory to which to save the model, already exists: "{}"'.format(save_directory)
                )
            shutil.rmtree(save_directory)

        os.makedirs(save_directory)
        np.save(os.path.join(save_directory, "model_params.npy"), self.get_model_flat_params(), allow_pickle=False)
        config = {
            "init_args": self._init_args,
            "init_kwargs": self._init_kwargs,
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "model_type": type(self).__module__ + "." + type(self).__qualname__,
        }
        with open(os.path.join(save_directory, "model_config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f)

    @staticmethod
    def load(directory, device="cpu", strict=True):
        del device
        del strict
        with open(os.path.join(directory, "model_config.json"), "r", encoding="utf-8") as f:
            config = json.load(f)

        model_type = config["model_type"]
        module_str, class_str = model_type.rsplit(".", 1)
        module = importlib.import_module(module_str)
        model_class = getattr(module, class_str)
        model = model_class(*config["init_args"], **config["init_kwargs"])
        params_path = os.path.join(directory, "model_params.npy")
        if not os.path.exists(params_path):
            raise FileNotFoundError(f"Missing NumPy parameter file: {params_path}")
        model.set_model_params(np.load(params_path, allow_pickle=False))
        model.eval()
        return model

    def get_model_shapes(self):
        return [tuple(param.shape) for param in self.parameter_arrays()]

    @property
    def model_shapes(self):
        return self.get_model_shapes()

    def set_model_params(self, flat_params):
        flat_params = np.asarray(flat_params, dtype=np.float32).reshape(-1)
        idx = 0
        for param in self.parameter_arrays():
            delta = int(param.size)
            block = flat_params[idx: idx + delta]
            if block.size != delta:
                raise ValueError("flat_params length does not match model parameter count")
            param[...] = np.reshape(block, param.shape).astype(np.float32, copy=False)
            idx += delta
        if idx != flat_params.size:
            raise ValueError("flat_params length does not match model parameter count")
        return self

    def load_model(self, filename):
        with open(filename, encoding="utf-8") as f:
            data = json.load(f)
        print("loading file %s" % (filename))
        self.data = data
        model_params = np.array(data[0], dtype=np.float32)
        self.set_model_params(model_params)

    def count_model_params(self):
        return int(sum(param.size for param in self.parameter_arrays()))

    @property
    def num_params(self):
        return self.count_model_params()

    def get_model_params(self):
        params = [param.copy() for param in self.parameter_arrays()]
        return params, [tuple(param.shape) for param in params]

    def get_model_flat_params(self):
        flat_params = [param.reshape(-1).astype(np.float32, copy=False) for param in self.parameter_arrays()]
        if not flat_params:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(flat_params).astype(np.float32, copy=False)

    def reset_parameters(self):
        return None
