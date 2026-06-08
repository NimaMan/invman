import multiprocessing as mp
import time
from copy import copy

import numpy as np

from invman.cmaes import CMAES
from invman.cpu_limits import normalize_args_cpu_limits
from invman.utils import Seeder, env_time_limit, save_model_solutions, write_log


def _coerce_int_list(raw_value):
    if raw_value is None:
        return None
    if isinstance(raw_value, str):
        parts = [part.strip() for part in raw_value.split(",") if part.strip()]
    else:
        parts = list(raw_value)
    return [int(part) for part in parts]


def _coerce_float_list(raw_value):
    if raw_value is None:
        return None
    if isinstance(raw_value, str):
        parts = [part.strip() for part in raw_value.split(",") if part.strip()]
    else:
        parts = list(raw_value)
    return [float(part) for part in parts]


class PopulationScheduler:
    def __init__(self, args):
        self.base_population = int(getattr(args, "es_population", 0))
        if self.base_population <= 0:
            raise ValueError(f"ES population must be positive; got {self.base_population}")

        raw_mode = getattr(args, "es_population_sampling", "fixed")
        self.mode = "fixed" if raw_mode is None else str(raw_mode)
        self.candidates = _coerce_int_list(getattr(args, "es_population_candidates", None))
        raw_probabilities = _coerce_float_list(getattr(args, "es_population_probabilities", None))
        if self.candidates is not None and self.mode == "fixed":
            self.mode = "categorical"

        if self.mode not in {"fixed", "categorical"}:
            raise ValueError(
                f"Unsupported ES population sampling mode '{self.mode}'. Expected 'fixed' or 'categorical'."
            )

        self.probabilities = None
        if self.mode == "categorical":
            if not self.candidates:
                raise ValueError("Categorical ES population sampling requires at least one candidate population size.")
            if any(value <= 0 for value in self.candidates):
                raise ValueError(f"Population candidates must all be positive; got {self.candidates}")
            if raw_probabilities is None:
                self.probabilities = [1.0 / len(self.candidates)] * len(self.candidates)
            else:
                if len(raw_probabilities) != len(self.candidates):
                    raise ValueError(
                        "ES population probabilities must match the candidate list length "
                        f"({len(raw_probabilities)} != {len(self.candidates)})"
                    )
                if any(value < 0 for value in raw_probabilities):
                    raise ValueError(f"ES population probabilities must be nonnegative; got {raw_probabilities}")
                total_weight = float(sum(raw_probabilities))
                if total_weight <= 0.0:
                    raise ValueError("ES population probabilities must sum to a positive value.")
                self.probabilities = [float(value) / total_weight for value in raw_probabilities]
            self.rng = np.random.RandomState(int(getattr(args, "seed", 0)))
        else:
            self.candidates = None
            self.rng = None

    def sample(self) -> int:
        if self.mode == "fixed":
            return self.base_population
        sampled_index = int(self.rng.choice(len(self.candidates), p=self.probabilities))
        return int(self.candidates[sampled_index])

    def protocol_label(self) -> str:
        if self.mode == "fixed":
            return f"fixed(pop={self.base_population})"
        formatted_probs = ", ".join(
            f"{population}:{probability:.3f}"
            for population, probability in zip(self.candidates, self.probabilities)
        )
        return f"categorical(base={self.base_population}; {formatted_probs})"

    def summarize(self, observed_populations):
        observed = [int(value) for value in observed_populations]
        protocol = {
            "base_population": int(self.base_population),
            "sampling_mode": self.mode,
            "candidates": None if self.candidates is None else [int(value) for value in self.candidates],
            "probabilities": None if self.probabilities is None else [float(value) for value in self.probabilities],
        }
        if not observed:
            return protocol
        unique_values, counts = np.unique(np.asarray(observed, dtype=np.int64), return_counts=True)
        protocol.update(
            {
                "observed_counts": {str(int(value)): int(count) for value, count in zip(unique_values, counts)},
                "observed_mean": float(np.mean(observed)),
                "observed_min": int(np.min(observed)),
                "observed_max": int(np.max(observed)),
            }
        )
        return protocol


def get_es_optimizer(model, args):
    if args.training_method == "ppo":
        raise NotImplementedError(
            "training_method='ppo' is a neural actor-critic trainer and does not use the "
            "es_mp ask/tell loop. Invoke the reusable Rust PPO trainer directly via "
            "invman.ppo_trainer.train_ppo(problem, ...) instead."
        )
    if args.training_method != "cma":
        raise NotImplementedError(f"Unsupported optimizer '{args.training_method}'. Only CMA-ES is supported.")
    return CMAES(
        model.num_params,
        sigma_init=args.sigma_init,
        popsize=args.es_population,
        seed=getattr(args, "seed", None),
        x0=getattr(args, "cma_x0", None),
    )


def train(
    model,
    get_model_fitness,
    args,
    get_population_fitness=None,
    same_seed=False,
    limit_env_time=False,
    min_steps=100,
    max_steps=5000,
    return_optimizer=False,
):
    # ADDITIVE/REVERSIBLE (training-path audit 2026-06-06): ``return_optimizer``
    # defaults to False so the historical return signature ``(model, fitness_hist)``
    # and the deployed endpoint (CMA-ES ``xbest`` via ``es.best_param()``) are
    # UNCHANGED for every existing caller. When True, the live ``es`` optimizer is
    # also returned so a caller can read the distribution-mean endpoint
    # ``es.current_param()`` (CMA-ES ``xfavorite`` = ``es.result[5]``) WITHOUT this
    # function silently flipping the global default deployment. This is the only
    # clean way to extract both endpoints from the SAME run.
    def _log_terminal(message):
        print(message)
        history.append(message)
        write_log(message, model, args)

    episodes = args.training_episodes
    mp_num_processors = normalize_args_cpu_limits(args)
    history = []
    fitness_hist = []
    population_hist = []

    es = get_es_optimizer(model, args)
    population_scheduler = PopulationScheduler(args)
    seeder = Seeder()
    print(f"Starting {args.training_method} with {model.num_params} parameters")
    protocol_line = f"ES population protocol: {population_scheduler.protocol_label()}"
    print(protocol_line)
    history.append(protocol_line)
    write_log(protocol_line, model, args)
    start = time.time()
    ctx = mp.get_context("spawn")
    pool = None
    base_horizon = args.horizon
    save_every = max(1, getattr(args, "save_every", getattr(args, "model_save_step", 100)))
    save_solutions = getattr(args, "save_solutions", False)

    try:
        for episode in range(1, episodes + 1):
            worker_args = args
            rollout_horizon = base_horizon
            if limit_env_time:
                rollout_horizon = int(
                    env_time_limit(
                        episode - 1,
                        min_steps=min_steps,
                        max_steps=max_steps,
                        num_cma_iterations=episodes,
                    )
                )
                worker_args = copy(args)
                worker_args.horizon = rollout_horizon

            current_population = population_scheduler.sample()
            population_hist.append(current_population)
            solutions = es.ask(popsize=current_population)
            if same_seed:
                seeds = seeder.next_seed(current_population)
            else:
                seeds = seeder.next_batch_seeds(current_population)

            pop_fitness = None
            if get_population_fitness is not None:
                pop_fitness = get_population_fitness(model, worker_args, solutions, seeds)

            if pop_fitness is None:
                if pool is None:
                    pool = ctx.Pool(processes=mp_num_processors)
                results = [
                    pool.apply_async(
                        get_model_fitness,
                        args=(model, worker_args, solution, seeds[indiv_id], indiv_id),
                    )
                    for indiv_id, solution in enumerate(solutions)
                ]
                pop_fitness = [result.get() for result in results]

            pop_fitness = sorted(pop_fitness, key=lambda item: item[1])
            es_fitness = np.array([fitness for fitness, _ in pop_fitness], dtype=np.float64)
            es.tell(es_fitness)

            fitness_hist.append(es_fitness)
            log_line = (
                f"e{episode} reward -> best: {np.max(es_fitness):.3f} mean: {np.mean(es_fitness):.3f}, "
                f"std: {np.std(es_fitness):.3f}, horizon: {rollout_horizon}, popsize: {current_population}"
            )
            print(log_line)
            history.append(log_line)
            write_log(log_line, model, args)

            if episode % save_every == 0:
                curr_solution = es.best_param()
                model = model.set_model_params(curr_solution)
                save_model_solutions(
                    model,
                    solutions,
                    episode,
                    args,
                    save_solutions=save_solutions,
                )
    except KeyboardInterrupt as exc:
        elapsed_seconds = time.time() - start
        _log_terminal(
            f"the optimization was interrupted after {elapsed_seconds:.2f}s: "
            f"{exc.__class__.__name__}{f' ({exc})' if str(exc) else ''}"
        )
        raise
    except Exception as exc:
        elapsed_seconds = time.time() - start
        _log_terminal(
            f"the optimization failed after {elapsed_seconds:.2f}s: "
            f"{exc.__class__.__name__}: {exc}"
        )
        raise
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    elapsed_seconds = time.time() - start
    _log_terminal(f"the optimization ended in {elapsed_seconds:.2f}s")
    model = model.set_model_params(es.best_param())
    setattr(
        model,
        "training_run_metadata",
        {"es_population_protocol": population_scheduler.summarize(population_hist)},
    )
    if return_optimizer:
        return model, fitness_hist, es
    return model, fitness_hist
