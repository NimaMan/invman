import numpy as np

from invman.es import CMAES
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
