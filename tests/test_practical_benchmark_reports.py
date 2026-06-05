import json

from invman.benchmarks.practical import load_dataset, write_report


def test_practical_report_helper_loads_dataset_and_writes_json_markdown(tmp_path):
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            {
                "name": "toy_trace",
                "source_kind": "repo_curated",
                "source_note": "Small checked-in trace for report rendering.",
                "practical_goal": "Exercise the practical report helper.",
            }
        ),
        encoding="utf-8",
    )

    dataset = load_dataset(dataset_path)
    output_json = tmp_path / "reports" / "latest_report.json"
    output_markdown = tmp_path / "reports" / "README.md"
    payload = write_report(
        {
            "family": "toy_family",
            "dataset": dataset,
            "calibration_protocol": "Use a deterministic toy policy.",
            "dataset_diagnostics": {"periods": 2, "mean_demand": 3.5},
            "metric_order": ["mean_period_cost"],
            "metric_labels": {"mean_period_cost": "Mean Period Cost"},
            "policy_rows": [
                {
                    "policy": "toy_policy",
                    "split": "eval",
                    "params": [1],
                    "metrics": {"mean_period_cost": 4.25},
                    "notes": "unit-test row",
                }
            ],
        },
        output_json=output_json,
        output_markdown=output_markdown,
    )

    archived = json.loads(output_json.read_text(encoding="utf-8"))
    markdown = output_markdown.read_text(encoding="utf-8")

    assert archived == payload
    assert archived["dataset"]["name"] == "toy_trace"
    assert "# toy_family Practical Benchmark" in markdown
    assert "| `toy_policy` | `eval` | `[1]` | `4.2500` | unit-test row |" in markdown
