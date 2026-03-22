import multiprocessing as mp
import time
from copy import copy

import numpy as np

from invman.es import CMAES, OpenES, SimpleGA, XNES
from invman.utils import Seeder, env_time_limit, save_model_solutions, write_log


def get_es_optimizer(model, args):
    if args.training_method == "cma":
        es = CMAES(
            model.num_params,
            sigma_init=args.sigma_init,
            popsize=args.es_population,
        )
    elif args.training_method == "ga":
        es = SimpleGA(
            model.num_params,
            sigma_init=args.sigma_init,
            popsize=args.es_population,
            sigma_decay=args.sigma_decay,
        )
    elif args.training_method == "es":
        es = OpenES(
            model.num_params,
            sigma_init=args.sigma_init,
            popsize=args.es_population,
        )
    elif args.training_method == "xnes":
        es = XNES(
            model.num_params,
            sigma_init=args.sigma_init,
            popsize=args.es_population,
            sigma_decay=args.sigma_decay,
        )
    else:
        raise NotImplementedError(f"Unknown ES optimizer: {args.training_method}")

    return es


def train(
    model,
    get_model_fitness,
    args,
    get_population_fitness=None,
    same_seed=False,
    limit_env_time=False,
    min_steps=100,
    max_steps=5000,
):
    episodes = args.training_episodes
    history = []
    fitness_hist = []

    es = get_es_optimizer(model, args)
    seeder = Seeder()
    print(f"Starting {args.training_method} with {model.num_params} parameters")
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
                        episode,
                        min_steps=min_steps,
                        max_steps=max_steps,
                        num_cma_iterations=episodes,
                    )
                )
                worker_args = copy(args)
                worker_args.horizon = rollout_horizon

            solutions = es.ask()
            if same_seed:
                seeds = seeder.next_seed(args.es_population)
            else:
                seeds = seeder.next_batch_seeds(args.es_population)

            pop_fitness = None
            if get_population_fitness is not None:
                pop_fitness = get_population_fitness(model, worker_args, solutions, seeds)

            if pop_fitness is None:
                if pool is None:
                    pool = ctx.Pool(processes=args.mp_num_processors)
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
                f"std: {np.std(es_fitness):.3f}, horizon: {rollout_horizon}"
            )
            print(log_line)
            history.append(log_line)
            write_log(log_line, model, args)

            if episode % save_every == 0:
                curr_solution = es.current_param()
                model = model.set_model_params(curr_solution)
                save_model_solutions(
                    model,
                    solutions,
                    episode,
                    args,
                    save_solutions=save_solutions,
                )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    elapsed_minutes = (time.time() - start) / 60.0
    print(f"the optimization ended in {elapsed_minutes}")
    model = model.set_model_params(es.current_param())
    return model, fitness_hist
