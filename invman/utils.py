import gzip
import json
import os
import pickle
import random
import signal
import time
import traceback
from datetime import datetime, timezone
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


class RunTerminationRequested(KeyboardInterrupt):
    pass


def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def experiment_status_path(args):
    return Path(args.results_dir) / f"status_{args.experiment_name}.json"


class RunStatusTracker:
    def __init__(self, status_path, metadata=None):
        self.status_path = Path(status_path)
        self.metadata = {} if metadata is None else dict(metadata)
        self.stage = "initializing"
        self.started_at = None
        self.completed = False
        self.completion_details = {}
        self._previous_handlers = {}

    def _payload(self, status, reason=None, exception_type=None, traceback_text=None, extra=None):
        now = time.time()
        payload = {
            "status": status,
            "stage": self.stage,
            "reason": reason,
            "exception_type": exception_type,
            "pid": os.getpid(),
            "started_at": None if self.started_at is None else datetime.fromtimestamp(self.started_at, tz=timezone.utc).isoformat(),
            "updated_at": _utc_now_iso(),
            "elapsed_seconds": None if self.started_at is None else round(now - self.started_at, 6),
            "metadata": self.metadata,
        }
        if traceback_text is not None:
            payload["traceback"] = traceback_text
        if extra:
            payload["details"] = dict(extra)
        return payload

    def _write(self, status, reason=None, exception_type=None, traceback_text=None, extra=None):
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._payload(
            status=status,
            reason=reason,
            exception_type=exception_type,
            traceback_text=traceback_text,
            extra=extra,
        )
        self.status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def update(self, stage, **details):
        self.stage = str(stage)
        self._write("running", reason=f"entered stage '{self.stage}'", extra=details or None)

    def mark_completed(self, **details):
        self.completed = True
        self.completion_details = dict(details)
        self._write("completed", reason="run completed successfully", extra=self.completion_details or None)

    def _signal_handler(self, signum, _frame):
        signame = signal.Signals(signum).name
        self._write("interrupted", reason=f"received {signame}")
        raise RunTerminationRequested(f"received {signame}")

    def __enter__(self):
        self.started_at = time.time()
        for sig in (signal.SIGINT, signal.SIGTERM):
            self._previous_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, self._signal_handler)
        self._write("running", reason="run started")
        return self

    def __exit__(self, exc_type, exc, tb):
        for sig, handler in self._previous_handlers.items():
            signal.signal(sig, handler)

        if self.completed:
            return False

        if exc is None:
            self._write("completed", reason="run completed without explicit completion marker")
            return False

        if isinstance(exc, (KeyboardInterrupt, RunTerminationRequested)):
            self._write(
                "interrupted",
                reason=str(exc) or exc.__class__.__name__,
                exception_type=exc.__class__.__name__,
            )
            return False

        if isinstance(exc, SystemExit):
            code = getattr(exc, "code", None)
            if code in (None, 0):
                self._write("completed", reason="run exited via SystemExit(0)")
            else:
                self._write("failed", reason=f"SystemExit({code})", exception_type="SystemExit")
            return False

        self._write(
            "failed",
            reason=f"{exc.__class__.__name__}: {exc}",
            exception_type=exc.__class__.__name__,
            traceback_text="".join(traceback.format_exception(exc_type, exc, tb)),
        )
        return False

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
