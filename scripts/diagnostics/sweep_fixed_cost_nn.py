import json
import subprocess
import sys
from pathlib import Path


EXPERIMENTS = [
    {
        "experiment_name": "diag_fixed_cost_nn_h8x8_sig5_pop10_1k",
        "policy_type": "nn",
        "hidden_dim": [8, 8],
        "sigma_init": 5.0,
        "es_population": 10,
        "training_episodes": 1000,
    },
    {
        "experiment_name": "diag_fixed_cost_nn_h16x16_sig5_pop10_1k",
        "policy_type": "nn",
        "hidden_dim": [16, 16],
        "sigma_init": 5.0,
        "es_population": 10,
        "training_episodes": 1000,
    },
    {
        "experiment_name": "diag_fixed_cost_nn_h16x16x16_sig5_pop10_1k",
        "policy_type": "nn",
        "hidden_dim": [16, 16, 16],
        "sigma_init": 5.0,
        "es_population": 10,
        "training_episodes": 1000,
    },
    {
        "experiment_name": "diag_fixed_cost_nn_h16x16x16_sig1_pop10_1k",
        "policy_type": "nn",
        "hidden_dim": [16, 16, 16],
        "sigma_init": 1.0,
        "es_population": 10,
        "training_episodes": 1000,
    },
]


def build_command(experiment):
    cmd = [
        sys.executable,
        "scripts/run_experiment.py",
        "--problem",
        "lost_sales_fixed_order_cost",
        "--policy_type",
        experiment["policy_type"],
        "--training_episodes",
        str(experiment["training_episodes"]),
        "--es_population",
        str(experiment["es_population"]),
        "--sigma_init",
        str(experiment["sigma_init"]),
        "--mp_num_processors",
        "4",
        "--lead_time",
        "4",
        "--shortage_cost",
        "4",
        "--fixed_order_cost",
        "5",
        "--demand_dist_name",
        "Poisson",
        "--demand_rate",
        "5",
        "--max_order_size",
        "50",
        "--horizon",
        "2000",
        "--eval_horizon",
        "20000",
        "--eval_seeds",
        "3",
        "--track_demand",
        "--same_seed",
        "--experiment_name",
        experiment["experiment_name"],
    ]
    if experiment["policy_type"] == "nn":
        cmd.extend(["--hidden_dim", *[str(width) for width in experiment["hidden_dim"]]])
    return cmd


def load_result(results_dir: Path, experiment_name: str):
    result_path = results_dir / f"{experiment_name}.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    evaluation = payload["evaluation"]
    learned = evaluation["learned_policy"]["mean_cost"]
    heuristics = evaluation["heuristics"]
    best_heuristic_name, best_heuristic = min(
        heuristics.items(),
        key=lambda item: item[1]["mean_cost"],
    )
    return {
        "experiment_name": experiment_name,
        "learned_policy_mean_cost": learned,
        "best_heuristic_name": best_heuristic_name,
        "best_heuristic_mean_cost": best_heuristic["mean_cost"],
        "gap_to_best_heuristic_pct": 100.0 * (learned - best_heuristic["mean_cost"]) / best_heuristic["mean_cost"],
    }


def main():
    package_root = Path(__file__).resolve().parents[2]
    workspace_root = package_root
    results_dir = package_root / "outputs" / "results"
    log_dir = package_root / "outputs" / "logs" / "fixed_cost_sweep"
    summary_path = results_dir / "diag_fixed_cost_nn_sweep_summary.json"
    log_dir.mkdir(parents=True, exist_ok=True)
    summaries = []

    for experiment in EXPERIMENTS:
        result_path = results_dir / f"{experiment['experiment_name']}.json"
        if not result_path.exists():
            cmd = build_command(experiment)
            log_path = log_dir / f"{experiment['experiment_name']}.log"
            print(f"running: {' '.join(cmd)}")
            print(f"log: {log_path}")
            with log_path.open("w", encoding="utf-8") as log_file:
                subprocess.run(
                    cmd,
                    cwd=workspace_root,
                    check=True,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
        else:
            print(f"reusing existing result: {result_path}")
        summaries.append(load_result(results_dir, experiment["experiment_name"]))

    summaries.sort(key=lambda item: item["learned_policy_mean_cost"])
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))
    print(f"saved summary to {summary_path}")


if __name__ == "__main__":
    main()
