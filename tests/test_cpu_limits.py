from types import SimpleNamespace

from invman.cpu_limits import (
    CPU_THREAD_ENV_VARS,
    bounded_worker_count,
    configure_process_cpu_limits,
    normalize_args_cpu_limits,
)


def test_bounded_worker_count_has_four_core_hard_cap():
    assert bounded_worker_count(16) == 4
    assert bounded_worker_count(4) == 4
    assert bounded_worker_count(2) == 2
    assert bounded_worker_count(0) == 1


def test_configure_process_cpu_limits_caps_high_thread_env_values():
    env = {
        "RAYON_NUM_THREADS": "32",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "not-an-int",
    }

    limit = configure_process_cpu_limits(8, environ=env)

    assert limit == 4
    assert env["RAYON_NUM_THREADS"] == "4"
    assert env["OMP_NUM_THREADS"] == "1"
    assert env["OPENBLAS_NUM_THREADS"] == "4"
    for name in CPU_THREAD_ENV_VARS:
        assert 1 <= int(env[name]) <= 4


def test_normalize_args_cpu_limits_mutates_requested_worker_count(monkeypatch):
    monkeypatch.setenv("RAYON_NUM_THREADS", "99")
    args = SimpleNamespace(mp_num_processors=16)

    limit = normalize_args_cpu_limits(args)

    assert limit == 4
    assert args.mp_num_processors == 4


def test_lost_sales_instance_jobs_use_capped_rollout_budget():
    from scripts.lost_sales.benchmark_full_suite import _effective_instance_jobs

    parsed = SimpleNamespace(mp_num_processors=16, instance_jobs=8)

    assert _effective_instance_jobs(parsed, num_instances=32) == 4


def test_lost_sales_child_subprocess_receives_capped_cpu_environment(monkeypatch):
    from scripts.lost_sales import benchmark_full_suite

    captured = {}

    def fake_run(command, *, check, cwd, env):
        captured.update(command=command, check=check, cwd=cwd, env=env)

    monkeypatch.setenv("RAYON_NUM_THREADS", "32")
    monkeypatch.setenv("OMP_NUM_THREADS", "32")
    monkeypatch.setattr(benchmark_full_suite.subprocess, "run", fake_run)
    parsed = SimpleNamespace(
        grid_name="xin2020_extended_lost_sales",
        run_tag="cpu_limit_test",
        seed=42,
        eval_horizon=1000,
        eval_seeds=2,
        training_episodes=None,
        training_horizon=None,
        state_scale=None,
        same_seed=False,
        reuse_existing=True,
        reuse_existing_instance_summary=False,
        only=["linear_categorical_quantity_q20"],
    )

    benchmark_full_suite._run_instance_subprocess(
        parsed,
        "lit_mmpp2_neg_p19_l6",
        mp_num_processors=2,
    )

    command = captured["command"]
    assert command[command.index("--mp_num_processors") + 1] == "2"
    assert command[-2:] == ["--references", "lit_mmpp2_neg_p19_l6"]
    assert captured["check"] is True
    assert captured["env"]["RAYON_NUM_THREADS"] == "2"
    assert captured["env"]["OMP_NUM_THREADS"] == "2"
