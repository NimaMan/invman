import numpy as np
from types import SimpleNamespace

from invman.cmaes import CMAES
from invman.config import get_config
from invman.es_mp import PopulationScheduler, train
from invman.utils import Seeder, set_global_seeds


def test_seeder_uses_private_rng():
    np.random.seed(123)
    before = np.random.randint(0, 1000)

    seeder = Seeder(init_seed=7)
    seeds_a = seeder.next_batch_seeds(5)
    seeds_b = seeder.next_seed(3)
    after = np.random.randint(0, 1000)

    np.random.seed(123)
    replay_before = np.random.randint(0, 1000)
    replay_after = np.random.randint(0, 1000)

    assert before == replay_before
    assert after == replay_after
    assert len(seeds_a) == 5
    assert len(seeds_b) == 3
    assert len(set(seeds_b)) == 1


def test_cmaes_seed_makes_sampling_reproducible():
    set_global_seeds(11)
    es_a = CMAES(num_params=5, sigma_init=1.0, popsize=4, seed=11)
    sols_a = es_a.ask()

    set_global_seeds(999)
    es_b = CMAES(num_params=5, sigma_init=1.0, popsize=4, seed=11)
    sols_b = es_b.ask()

    assert np.allclose(sols_a, sols_b)


def test_get_config_parses_sampled_es_population_args():
    args = get_config(
        [
            "--problem",
            "dual_sourcing",
            "--policy_name",
            "soft_tree_d2_t0p25_oblique_linear_leaf_adapter-capped_dual_index_targets",
            "--es_population",
            "128",
            "--es_population_sampling",
            "categorical",
            "--es_population_candidates",
            "64,96,128",
            "--es_population_probabilities",
            "0.2,0.3,0.5",
        ]
    )

    assert args.es_population == 128
    assert args.es_population_sampling == "categorical"
    assert args.es_population_candidates == [64, 96, 128]
    assert args.es_population_probabilities == [0.2, 0.3, 0.5]


def test_population_scheduler_seed_makes_sampling_reproducible():
    args = SimpleNamespace(
        es_population=128,
        es_population_sampling="categorical",
        es_population_candidates=[64, 96, 128],
        es_population_probabilities=[0.2, 0.3, 0.5],
        seed=11,
    )
    scheduler_a = PopulationScheduler(args)
    scheduler_b = PopulationScheduler(args)

    sequence_a = [scheduler_a.sample() for _ in range(20)]
    sequence_b = [scheduler_b.sample() for _ in range(20)]

    assert sequence_a == sequence_b


class _DummyModel:
    def __init__(self, params=None):
        self.num_params = 3
        self.params = np.zeros(self.num_params, dtype=np.float64) if params is None else np.asarray(params, dtype=np.float64)

    def set_model_params(self, params):
        return _DummyModel(params)

    def save(self, *_args, **_kwargs):
        return None


def _dummy_population_fitness(_model, _args, solutions, _seeds):
    return [(-float(np.sum(np.square(solution))), indiv_idx) for indiv_idx, solution in enumerate(solutions)]


def test_train_records_sampled_population_protocol(tmp_path):
    args = SimpleNamespace(
        training_method="cma",
        training_episodes=12,
        mp_num_processors=1,
        sigma_init=1.0,
        es_population=128,
        es_population_sampling="categorical",
        es_population_candidates=[64, 96, 128],
        es_population_probabilities=[0.2, 0.3, 0.5],
        horizon=10,
        save_every=1000,
        save_solutions=False,
        seed=17,
        log_dir=str(tmp_path / "logs"),
        trained_models_dir=str(tmp_path / "models"),
        experiment_name="sampled_population_protocol_smoke",
    )

    trained_model, _ = train(
        model=_DummyModel(),
        get_model_fitness=None,
        get_population_fitness=_dummy_population_fitness,
        args=args,
    )

    protocol = trained_model.training_run_metadata["es_population_protocol"]
    assert protocol["base_population"] == 128
    assert protocol["sampling_mode"] == "categorical"
    assert protocol["candidates"] == [64, 96, 128]
    assert protocol["probabilities"] == [0.2, 0.3, 0.5]
    assert sum(protocol["observed_counts"].values()) == args.training_episodes
    assert protocol["observed_min"] in {64, 96, 128}
    assert protocol["observed_max"] in {64, 96, 128}
