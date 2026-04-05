from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_dataset(path: str | Path) -> dict[str, Any]:
    dataset_path = Path(path)
    return json.loads(dataset_path.read_text(encoding="utf-8"))


def ensure_parent(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def dumps_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def render_practical_benchmark_markdown(payload: dict[str, Any]) -> str:
    dataset = payload["dataset"]
    metric_order = payload["metric_order"]
    metric_labels = payload.get("metric_labels", {})

    lines = [
        f"# {payload['family']} Practical Benchmark",
        "",
        f"- dataset: `{dataset['name']}`",
        f"- source_kind: `{dataset['source_kind']}`",
        f"- source_note: {dataset['source_note']}",
        f"- practical_goal: {dataset['practical_goal']}",
    ]
    if payload.get("calibration_protocol"):
        lines.append(f"- calibration: {payload['calibration_protocol']}")
    if payload.get("dataset_diagnostics"):
        for key, value in payload["dataset_diagnostics"].items():
            if isinstance(value, float):
                lines.append(f"- {key}: `{value:.4f}`")
            else:
                lines.append(f"- {key}: `{value}`")

    lines.extend(
        [
            "",
            "| Policy | Split | Params | "
            + " | ".join(metric_labels.get(metric, metric) for metric in metric_order)
            + " | Notes |",
            "| --- | --- | --- | "
            + " | ".join("---:" for _ in metric_order)
            + " | --- |",
        ]
    )

    for row in payload["policy_rows"]:
        metrics = row["metrics"]
        rendered_metrics = []
        for metric in metric_order:
            value = metrics.get(metric)
            if value is None:
                rendered_metrics.append("`-`")
            elif isinstance(value, float):
                rendered_metrics.append(f"`{value:.4f}`")
            else:
                rendered_metrics.append(f"`{value}`")
        lines.append(
            f"| `{row['policy']}` | `{row.get('split', 'eval')}` | `{row['params']}` | "
            + " | ".join(rendered_metrics)
            + f" | {row.get('notes', '')} |"
        )

    return "\n".join(lines)


def write_report(
    payload: dict[str, Any],
    *,
    output_json: str | Path | None = None,
    output_markdown: str | Path | None = None,
) -> dict[str, Any]:
    markdown = render_practical_benchmark_markdown(payload)
    payload = dict(payload)
    payload["markdown"] = markdown

    if output_json is not None:
        path = ensure_parent(output_json)
        path.write_text(dumps_json(payload), encoding="utf-8")
    if output_markdown is not None:
        path = ensure_parent(output_markdown)
        path.write_text(markdown + "\n", encoding="utf-8")
    return payload
