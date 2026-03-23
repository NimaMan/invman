import gzip
import os
import pickle
import random
from pathlib import Path

import numpy as np


def env_time_limit(iteration, min_steps, max_steps, num_cma_iterations, scaling_exponent=1):
    assert scaling_exponent > 0, "For the polynomial scaling, you need to provide a positive scaling-exponent"
    limit_step_fn = lambda iteration: max_steps + (min_steps - max_steps) * np.power(
        np.clip((1 - iteration/(num_cma_iterations-1)), 0, 1), scaling_exponent)

    return limit_step_fn(iteration)


class Seeder:
    def __init__(self, init_seed=0):
        self.rng = np.random.RandomState(init_seed)
        self.limit = np.int32(2**31-1)

    def next_seed(self, batch_size=1):
        seed = int(self.rng.randint(self.limit))
        result = [seed] * batch_size
        return result

    def next_batch_seeds(self, batch_size):
        result = self.rng.randint(self.limit, size=batch_size).tolist()
        return result


def set_global_seeds(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is optional for some paths
        torch = None
    if torch is not None:
        torch.manual_seed(seed)

def save_init_args(init_method):
    def wrapper(self, *args, **kwargs):
        self._init_args = args
        self._init_kwargs = kwargs
        init_method(self, *args, **kwargs)
    return wrapper


def write_log(log, model, args):
    log_file_name = f"log_{args.experiment_name}.txt"
    os.makedirs(args.log_dir, exist_ok=True)
    log_file_dir = os.path.join(args.log_dir, log_file_name)
    with open(log_file_dir, "a") as f:
        f.write(log)
        f.write("\n")


def save_model_solutions(model, solutions, episode, args, save_solutions=False, es_parameter_type='avg'):
    model_dir_name = f"{args.experiment_name}_{model.num_params}_{episode}"
    model_root = Path(args.trained_models_dir)
    model_root.mkdir(parents=True, exist_ok=True)
    model_save_dir = model_root / model_dir_name
    model.save(str(model_save_dir), override=True)
    if save_solutions:
        solutions_name = f"{args.experiment_name}_{model.num_params}_{episode}.pkl.gz"
        solutions_save_dir = model_save_dir / solutions_name
        with gzip.open(solutions_save_dir, "wb") as f:
            pickle.dump(solutions, f, protocol=pickle.HIGHEST_PROTOCOL)
