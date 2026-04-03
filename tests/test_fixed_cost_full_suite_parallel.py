from types import SimpleNamespace

from scripts.lost_sales_fixed_order_cost.benchmark_full_suite import (
    _effective_instance_jobs,
    _shared_child_command_args,
)


def _build_parsed(**overrides):
    payload = {
        "grid_name": "literature_subset_poisson_mu5",
        "run_tag": "fixed_cost_parallel_test",
        "seed": 123,
        "mp_num_processors": 16,
        "instance_jobs": 4,
        "eval_horizon": 1000,
        "eval_seeds": 2,
        "same_seed": False,
        "search_horizon": 10000,
        "reuse_existing": True,
        "reuse_existing_instance_summary": True,
        "only": ["linear_categorical_quantity", "nn_soft_gated_ordinal_quantity"],
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


def test_effective_instance_jobs_respects_rollout_worker_budget():
    parsed = _build_parsed(mp_num_processors=3, instance_jobs=8)
    assert _effective_instance_jobs(parsed, num_instances=16) == 3


def test_child_command_forces_single_instance_execution_and_summary_skip():
    parsed = _build_parsed()
    command = _shared_child_command_args(parsed, mp_num_processors=4)

    assert "--instance_jobs" in command
    assert command[command.index("--instance_jobs") + 1] == "1"
    assert "--skip_suite_summary" in command
    assert "--only" in command
    assert "linear_categorical_quantity" in command
    assert "nn_soft_gated_ordinal_quantity" in command
