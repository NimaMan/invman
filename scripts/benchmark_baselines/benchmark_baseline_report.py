"""Shared experiment harness for the executable-baseline reports.

================================================================================
ALGORITHMIC DESCRIPTION
================================================================================
Objective
---------
Give every per-family `run_*_baselines.py` script ONE implementation of "load the
reference instances of a problem, show the published baselines, optionally re-run
them on the live env, and (optionally) score a candidate policy — then emit a
comparison table." The scripts under this folder are thin: they pick the problem
+ the columns; this module does the work via the executable layer
(`invman.benchmarks.runners`).

Why each piece exists (maps to the objective)
---------------------------------------------
* `collect_rows(problem, simulate, protocol, instances)` — for each reference
  instance: read params + published baselines (free), pick the canonical
  reference cost, and — when `simulate` — RE-RUN the shipped baseline on the live
  env (`ReferenceInstance.run_baselines`). Returns plain dicts so the rows are
  JSON-serializable and the rendering is decoupled from the runner.
* `render_markdown` / `write_outputs` — a stable markdown table + a JSON sidecar
  so a consumer can both read the comparison and diff it in CI. The published
  number and the recomputed number sit in adjacent columns so the reproduction
  gap is visible at a glance (the repo's honesty discipline).
* `evaluate_zero_policy` — a zero-effort "the env is runnable end-to-end" probe:
  it scores an all-zeros soft-tree on each instance through the SAME seam the
  CMA-ES optimizer uses (`ReferenceInstance.evaluate`). It is NOT a trained
  result — it only proves the evaluate path works; a real comparison plugs a
  trained weight vector into the same call.
* `build_arg_parser` / `run` — the common CLI: `--simulate` (re-run baselines),
  `--full` (literature-faithful protocol vs the fast smoke one), `--instances`,
  `--evaluate-zeros`, `--out`.

This module never trains. Training a soft-tree to actually BEAT a baseline is the
job of the per-family CMA-ES scripts (`scripts/<family>/...`); here we establish
the reference numbers and the comparison harness a user drops their result into.

Usage (from a per-family script):
    from benchmark_baseline_report import run
    run("lost_sales", default_columns=[...])
Dependencies: `invman.benchmarks.runners` (-> `invman_rust`), numpy, stdlib.
================================================================================
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from invman.benchmarks import runners


# ---------------------------------------------------------------------------
# Row collection
# ---------------------------------------------------------------------------


def _protocol_for(runner, full: bool):
    return runner.published_protocol if full else runner.smoke_protocol


def collect_rows(
    problem: str,
    *,
    simulate: bool = False,
    full: bool = False,
    instances: Optional[list[str]] = None,
    evaluate_zeros: bool = False,
) -> list[dict]:
    """One JSON-able row per reference instance of `problem`.

    Each row carries the env source, the published baselines (cost or gap), the
    canonical reference cost, and — when `simulate` — the recomputed baselines.
    """
    runner = runners.get_runner(problem)
    names = instances if instances is not None else runner.list_instances()
    protocol = _protocol_for(runner, full)

    rows: list[dict] = []
    for name in names:
        inst = runner.load_instance(name)
        published = {
            b.name: {
                "cost": b.mean_cost,
                "is_published": b.is_published,
                "is_optimal": b.is_optimal,
                "is_reference": b.is_reference,
                "params": b.params,
                "note": b.note,
            }
            for b in inst.published_baselines
        }
        ref = inst.reference_baseline
        row = {
            "problem": problem,
            "instance": name,
            "subfamily": inst.subfamily,
            "source": inst.source,
            "reference_baseline": None if ref is None else ref.name,
            "reference_cost": None if ref is None else ref.mean_cost,
            "published": published,
        }
        if simulate:
            recomputed = inst.run_baselines(protocol)
            row["recomputed"] = {
                k: {"cost": v.mean_cost, "params": v.params, "source": v.source, "note": v.note}
                for k, v in recomputed.items()
            }
            row["protocol"] = {
                "seeds": list(protocol.seeds),
                "horizon": protocol.horizon,
                "warm_up_periods_ratio": protocol.warm_up_periods_ratio,
                "replications": protocol.replications,
            }
        if evaluate_zeros:
            n = inst.policy_param_count()
            row["zero_policy_cost"] = inst.evaluate([0.0] * n, protocol=protocol)
            row["policy_param_count"] = n
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_markdown(problem: str, rows: list[dict], *, simulate: bool) -> str:
    lines: list[str] = [f"# Baseline report — `{problem}`", ""]
    runner = runners.get_runner(problem)
    lines.append(
        f"Generated from the executable baseline layer "
        f"(`invman.benchmarks.runners`). {len(rows)} reference instance(s)."
    )
    lines.append("")

    # Union of published-baseline names across rows, in stable order.
    pub_names: list[str] = []
    for row in rows:
        for name in row["published"]:
            if name not in pub_names:
                pub_names.append(name)

    headers = ["Instance", "Subfamily", "Reference", "Ref cost"]
    headers += [f"pub:{n}" for n in pub_names]
    if simulate:
        headers.append("recomputed (name=cost)")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")

    for row in rows:
        cells = [
            f"`{row['instance']}`",
            row["subfamily"],
            row["reference_baseline"] or "—",
            _fmt(row["reference_cost"]),
        ]
        for name in pub_names:
            entry = row["published"].get(name)
            if entry is None:
                cells.append("—")
            elif entry["cost"] is not None:
                cells.append(_fmt(entry["cost"]))
            elif entry["params"] and "published_gap_pct" in entry["params"]:
                cells.append(f"gap {entry['params']['published_gap_pct']}%")
            elif entry["params"] and "published_savings_pct" in entry["params"]:
                cells.append(f"save {entry['params']['published_savings_pct']}%")
            else:
                cells.append("(pub)")
        if simulate:
            rec = row.get("recomputed", {})
            cells.append(
                ", ".join(
                    f"{k}={_fmt(v['cost'])}" for k, v in rec.items() if v["cost"] is not None
                )
                or "—"
            )
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    lines.append(
        "_pub:* = published literature number (cost, or `gap`/`save` % where the "
        "paper reports a relative figure). `recomputed` = re-run on the live env "
        "via the runner. To compare your policy: "
        "`inst = catalog.get(problem).load_instance(name); "
        "inst.compare(inst.evaluate(my_params))`._"
    )
    lines.append("")
    return "\n".join(lines)


def write_outputs(problem: str, rows: list[dict], out_dir: Path, *, simulate: bool) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{problem}_baselines.md"
    json_path = out_dir / f"{problem}_baselines.json"
    md_path.write_text(render_markdown(problem, rows, simulate=simulate) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(rows, indent=2, default=str) + "\n", encoding="utf-8")
    return {"markdown": md_path, "json": json_path}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser(problem: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Executable baseline report for the {problem!r} family."
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="re-run the shipped baselines on the live env (else only read published numbers)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="use the literature-faithful protocol (slower) instead of the fast smoke one",
    )
    parser.add_argument(
        "--instances",
        nargs="*",
        default=None,
        help="restrict to these reference-instance names (default: all)",
    )
    parser.add_argument(
        "--evaluate-zeros",
        action="store_true",
        help="also score an all-zeros soft-tree on each instance (proves the evaluate seam runs)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="directory to write <problem>_baselines.{md,json}; default: print to stdout only",
    )
    return parser


def run(problem: str) -> list[dict]:
    """Parse args, collect rows, print the table, optionally write outputs."""
    args = build_arg_parser(problem).parse_args()
    rows = collect_rows(
        problem,
        simulate=args.simulate,
        full=args.full,
        instances=args.instances,
        evaluate_zeros=args.evaluate_zeros,
    )
    table = render_markdown(problem, rows, simulate=args.simulate)
    print(table)
    if args.evaluate_zeros:
        print("\nZero-policy (untrained) sanity costs:")
        for row in rows:
            print(f"  {row['instance']}: {row.get('zero_policy_cost')}")
    if args.out:
        paths = write_outputs(problem, rows, Path(args.out), simulate=args.simulate)
        print(f"\nWrote {paths['markdown']} and {paths['json']}")
    return rows
