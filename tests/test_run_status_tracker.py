import json

import pytest

from invman.utils import RunStatusTracker, RunTerminationRequested, experiment_status_path


def test_experiment_status_path_uses_results_dir_and_experiment_name(tmp_path):
    class Args:
        results_dir = str(tmp_path)
        experiment_name = "demo_run"

    assert experiment_status_path(Args()) == tmp_path / "status_demo_run.json"


def test_run_status_tracker_marks_completed(tmp_path):
    status_path = tmp_path / "status.json"

    with RunStatusTracker(status_path, metadata={"experiment_name": "demo"}) as tracker:
        tracker.update("training")
        tracker.mark_completed(results_path="results/demo.json")

    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["stage"] == "training"
    assert payload["details"]["results_path"] == "results/demo.json"


def test_run_status_tracker_marks_failure(tmp_path):
    status_path = tmp_path / "status.json"

    with pytest.raises(RuntimeError):
        with RunStatusTracker(status_path, metadata={"experiment_name": "demo"}) as tracker:
            tracker.update("training")
            raise RuntimeError("boom")

    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["stage"] == "training"
    assert payload["exception_type"] == "RuntimeError"
    assert "boom" in payload["reason"]


def test_run_status_tracker_marks_interrupted(tmp_path):
    status_path = tmp_path / "status.json"

    with pytest.raises(RunTerminationRequested):
        with RunStatusTracker(status_path, metadata={"experiment_name": "demo"}) as tracker:
            tracker.update("training")
            raise RunTerminationRequested("received SIGTERM")

    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["status"] == "interrupted"
    assert payload["stage"] == "training"
    assert payload["exception_type"] == "RunTerminationRequested"
    assert payload["reason"] == "received SIGTERM"
