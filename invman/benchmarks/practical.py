from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_dataset(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dumps_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=str)


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _format_code(value: Any) -> str:
    return f"`{_format_value(value)}`"


def _render_policy_table(payload: dict[str, Any]) -> list[str]:
    metric_order = list(payload.get("metric_order", []))
    metric_labels = dict(payload.get("metric_labels", {}))
    rows = list(payload.get("policy_rows", []))
    if not metric_order or not rows:
        return []

    headers = ["Policy", "Split", "Params"]
    headers.extend(metric_labels.get(metric, metric) for metric in metric_order)
    headers.append("Notes")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| --- | --- | --- | " + " | ".join("---:" for _ in metric_order) + " | --- |",
    ]
    for row in rows:
        metrics = dict(row.get("metrics", {}))
        rendered_metrics = [_format_code(metrics.get(metric, "")) for metric in metric_order]
        lines.append(
            f"| {_format_code(row.get('policy', ''))} | "
            f"{_format_code(row.get('split', ''))} | "
            f"{_format_code(row.get('params', ''))} | "
            + " | ".join(rendered_metrics)
            + f" | {row.get('notes', '')} |"
        )
    return lines


def render_markdown(payload: dict[str, Any]) -> str:
    family = str(payload.get("family", "benchmark"))
    dataset = dict(payload.get("dataset", {}))
    diagnostics = dict(payload.get("dataset_diagnostics", {}))

    lines = [f"# {family} Practical Benchmark", ""]
    for key, label, code_format in (
        ("name", "dataset", True),
        ("source_kind", "source_kind", True),
        ("source_note", "source_note", False),
        ("practical_goal", "practical_goal", False),
    ):
        if key in dataset:
            value = _format_code(dataset[key]) if code_format else str(dataset[key])
            lines.append(f"- {label}: {value}")
    if payload.get("calibration_protocol"):
        lines.append(f"- calibration: {payload['calibration_protocol']}")
    for key, value in diagnostics.items():
        lines.append(f"- {key}: {_format_code(value)}")

    table = _render_policy_table(payload)
    if table:
        lines.extend(["", *table])

    return "\n".join(lines)


def write_report(
    payload: dict[str, Any],
    *,
    output_json: str | Path,
    output_markdown: str | Path,
) -> dict[str, Any]:
    payload = dict(payload)
    markdown = render_markdown(payload)
    payload["markdown"] = markdown

    output_json_path = Path(output_json)
    output_markdown_path = Path(output_markdown)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_markdown_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(dumps_json(payload) + "\n", encoding="utf-8")
    output_markdown_path.write_text(markdown + "\n", encoding="utf-8")
    return payload
