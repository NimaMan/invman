import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


DEFAULT_RUN_TAG = "dual_sourcing_factor_screen_v1"
DEFAULT_INPUT = (
    PACKAGE_ROOT / "outputs" / "autoresearch" / DEFAULT_RUN_TAG / "factor_screen_summary.json"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "factor_screen_results.md"

FOLLOWUP_SCREENING_SUMMARIES = [
    PACKAGE_ROOT
    / "outputs"
    / "autoresearch"
    / "dual_l2_ce110_axis_family_probe"
    / "screening_summary.json",
    PACKAGE_ROOT
    / "outputs"
    / "autoresearch"
    / "dual_l3_axis_linear_cappeddelta_probe"
    / "screening_summary.json",
    PACKAGE_ROOT
    / "outputs"
    / "autoresearch"
    / "dual_hard_axis_linear_smallcap_probe"
    / "screening_summary.json",
]
FOLLOWUP_RAW_RESULTS = [
    (
        "dual_l2_ce105",
        "tree_axis_linear_smallcap_delta",
        PACKAGE_ROOT
        / "outputs"
        / "autoresearch"
        / "dual_l2_axis_linear_probe"
        / "results"
        / "dual_l2_ce105_soft_tree_d2_t0p25_axis_aligned_linear_leaf_adapter-capped_dual_index_delta_smallcap_targets.json",
    ),
]
FOLLOWUP_POLICY_ALIASES = {
    "soft_tree_d2_t0p25_axis_aligned_linear_leaf_adapter-capped_dual_index_delta_targets": "tree_axis_linear_capped_delta",
    "soft_tree_d2_t0p25_axis_aligned_linear_leaf_adapter-capped_dual_index_delta_smallcap_targets": "tree_axis_linear_smallcap_delta",
}


PAIR_COMPARISONS = [
    (
        "tree_capped_dual_index",
        "tree_capped_delta",
        "Factorizing regular targets (`s_r = s_e + delta_r`) versus unfactorized capped dual-index targets",
    ),
    (
        "tree_capped_delta",
        "tree_smallcap_delta",
        "Adding a small discrete regular-cap grid on top of the capped-delta tree",
    ),
    (
        "tree_smallcap_delta",
        "tree_axis_constant_smallcap_delta",
        "Replacing the oblique linear tree with an axis-aligned constant-leaf tree on the same small-cap control family",
    ),
    (
        "linear_smallcap_delta",
        "nn_smallcap_delta",
        "Replacing a linear dense small-cap policy with a wider neural backbone on the same control family",
    ),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Render a markdown summary for the dual-sourcing factor screen.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _pair_delta(rows: list[dict], left_id: str, right_id: str):
    by_reference = {}
    for row in rows:
        by_reference.setdefault(row["reference"], {})[row["policy_id"]] = row

    deltas = []
    for reference, items in sorted(by_reference.items()):
        if left_id not in items or right_id not in items:
            continue
        left_gap = float(items[left_id]["gap_pct_vs_best_heuristic"])
        right_gap = float(items[right_id]["gap_pct_vs_best_heuristic"])
        deltas.append(
            {
                "reference": reference,
                "left_gap_pct": left_gap,
                "right_gap_pct": right_gap,
                "improvement_pct": left_gap - right_gap,
            }
        )
    mean_improvement = (
        sum(item["improvement_pct"] for item in deltas) / len(deltas) if deltas else None
    )
    return deltas, mean_improvement


def _followup_gap_from_results(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    learned_cost = float(payload["evaluation"]["learned_policy"]["mean_cost"])
    best_heuristic_cost = min(
        float(summary["mean_cost"])
        for summary in payload["evaluation"]["heuristics"].values()
        if isinstance(summary, dict) and "mean_cost" in summary
    )
    return 100.0 * (learned_cost / best_heuristic_cost - 1.0)


def _load_followup_axis_linear_rows():
    rows = []
    for reference, policy_id, path in FOLLOWUP_RAW_RESULTS:
        if not path.exists():
            continue
        rows.append(
            {
                "reference": reference,
                "policy_id": policy_id,
                "gap_pct_vs_best_heuristic": _followup_gap_from_results(path),
            }
        )

    for path in FOLLOWUP_SCREENING_SUMMARIES:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload:
            policy_id = FOLLOWUP_POLICY_ALIASES.get(row["policy_name"])
            if policy_id is None:
                continue
            rows.append(
                {
                    "reference": row["reference"],
                    "policy_id": policy_id,
                    "gap_pct_vs_best_heuristic": float(row["gap_pct_vs_best_heuristic"]),
                }
            )
    return rows


def _render(summary: dict):
    rows = summary["rows"]
    policy_summary = summary["aggregate"]["policy_summary"]
    row_summary = summary["aggregate"]["row_summary"]

    lines = [
        "# Dual-Sourcing Factor Screen",
        "",
        f"Run tag: `{summary['run_tag']}`",
        f"Budget: `{summary['budget']}`",
        "",
        "This note uses the six Gijs Figure 9 benchmark rows as a controlled policy-design testbed.",
        "The goal is not to defend one fixed policy class, but to identify which design choices drive learned-policy performance.",
        "",
        "## Aggregate Ranking",
        "",
        "| policy | mean gap vs best heuristic (%) | wins vs best heuristic | control family | structure |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for policy_id, item in sorted(
        policy_summary.items(),
        key=lambda kv: kv[1]["mean_gap_pct_vs_best_heuristic"],
    ):
        lines.append(
            f"| `{policy_id}` | `{item['mean_gap_pct_vs_best_heuristic']:.4f}` | "
            f"`{item['wins_vs_best_heuristic']}/{item['num_instances']}` | "
            f"`{item['control_family']}` | `{item['backbone']}` |"
        )

    lines.extend(
        [
            "",
            "## Best Policy By Benchmark Row",
            "",
            "| reference | best policy | gap vs best heuristic (%) |",
            "| --- | --- | ---: |",
        ]
    )
    for item in sorted(row_summary, key=lambda row: row["reference"]):
        lines.append(
            f"| `{item['reference']}` | `{item['best_policy_id']}` | `{item['best_gap_pct_vs_best_heuristic']:.4f}` |"
        )

    lines.extend(["", "## Factor Effects", ""])
    for left_id, right_id, label in PAIR_COMPARISONS:
        deltas, mean_improvement = _pair_delta(rows, left_id, right_id)
        if not deltas:
            continue
        lines.append(f"### {label}")
        lines.append("")
        lines.append(
            f"Average improvement in gap when moving from `{left_id}` to `{right_id}`: "
            f"`{mean_improvement:.4f}` percentage points."
        )
        lines.append("")
        lines.append("| reference | left gap (%) | right gap (%) | improvement (pp) |")
        lines.append("| --- | ---: | ---: | ---: |")
        for item in deltas:
            lines.append(
                f"| `{item['reference']}` | `{item['left_gap_pct']:.4f}` | "
                f"`{item['right_gap_pct']:.4f}` | `{item['improvement_pct']:.4f}` |"
            )
        lines.append("")

    hardest = max(row_summary, key=lambda item: item["best_gap_pct_vs_best_heuristic"])
    easiest = min(row_summary, key=lambda item: item["best_gap_pct_vs_best_heuristic"])
    best_policy_id, best_policy = min(
        policy_summary.items(),
        key=lambda kv: kv[1]["mean_gap_pct_vs_best_heuristic"],
    )
    followup_rows = _load_followup_axis_linear_rows()
    followup_best_by_reference = {}
    for row in followup_rows:
        current = followup_best_by_reference.get(row["reference"])
        if current is None or row["gap_pct_vs_best_heuristic"] < current["gap_pct_vs_best_heuristic"]:
            followup_best_by_reference[row["reference"]] = row
    lines.extend(
        [
            "## Current Reading",
            "",
            f"- Hardest row under the current search surface: `{hardest['reference']}` with best learned gap `{hardest['best_gap_pct_vs_best_heuristic']:.4f}%`.",
            f"- Easiest row under the current search surface: `{easiest['reference']}` with best learned gap `{easiest['best_gap_pct_vs_best_heuristic']:.4f}%`.",
            "- The dominant factor is control geometry, not just parameter count.",
            "- Factorized dual-index controls help more than staying in unfactorized target coordinates.",
            "- A small discrete cap on regular orders is not enough by itself; it helps most when paired with a tighter policy geometry.",
            "- On the hard rows, tighter tree geometry can beat a more flexible oblique tree on the same control family.",
            "- The right next step is to keep the good control family and search more policy classes on top of it, not to retreat to raw direct-order outputs.",
            "",
            "## Promotion Candidates",
            "",
            f"- Promote `{best_policy_id}` as the default six-row search family. Its mean gap is `{best_policy['mean_gap_pct_vs_best_heuristic']:.4f}%` and it is the best policy on five of the six rows.",
            "- Keep `tree_capped_delta` alive as the main alternate family. It wins `dual_l2_ce110` and is the best non-axis-constant option on the easy rows.",
            "- Deprioritize `tree_capped_dual_index`: it is consistently worse than the factorized variants.",
            "- Deprioritize `linear_smallcap_delta` as a main family: the control family is reasonable, but the backbone is too weak to compete consistently.",
            "",
        ]
    )
    if followup_best_by_reference:
        lines.extend(
            [
                "## Follow-Up Axis-Linear Probes",
                "",
                "The factor screen suggested trying axis-aligned linear leaves on the factorized delta control families.",
                "Those probes show that linear leaves are useful, but not as a universal replacement.",
                "",
                "| reference | best factor-screen family | factor-screen gap (%) | best axis-linear follow-up | follow-up gap (%) | delta (pp) |",
                "| --- | --- | ---: | --- | ---: | ---: |",
            ]
        )
        for baseline in sorted(row_summary, key=lambda row: row["reference"]):
            followup = followup_best_by_reference.get(baseline["reference"])
            if followup is None:
                continue
            delta = (
                baseline["best_gap_pct_vs_best_heuristic"]
                - followup["gap_pct_vs_best_heuristic"]
            )
            lines.append(
                f"| `{baseline['reference']}` | `{baseline['best_policy_id']}` | "
                f"`{baseline['best_gap_pct_vs_best_heuristic']:.4f}` | `{followup['policy_id']}` | "
                f"`{followup['gap_pct_vs_best_heuristic']:.4f}` | `{delta:.4f}` |"
            )

        improved_refs = sorted(
            reference
            for reference, followup in followup_best_by_reference.items()
            if followup["gap_pct_vs_best_heuristic"]
            < next(
                row["best_gap_pct_vs_best_heuristic"]
                for row in row_summary
                if row["reference"] == reference
            )
        )
        regressed_refs = sorted(
            reference
            for reference, followup in followup_best_by_reference.items()
            if followup["gap_pct_vs_best_heuristic"]
            >= next(
                row["best_gap_pct_vs_best_heuristic"]
                for row in row_summary
                if row["reference"] == reference
            )
        )
        if improved_refs or regressed_refs:
            lines.extend(["", "### Updated Direction", ""])
        if improved_refs:
            joined = ", ".join(f"`{reference}`" for reference in improved_refs)
            lines.append(
                f"- Axis-linear follow-ups improve on the factor-screen best family for {joined}."
            )
        if regressed_refs:
            joined = ", ".join(f"`{reference}`" for reference in regressed_refs)
            lines.append(
                f"- Axis-linear follow-ups are worse than the factor-screen best family for {joined}."
            )
        lines.extend(
            [
                "- The common ingredient across the winning rows is still the factorized capped-delta control surface.",
                "- The split is in policy geometry: `l_r = 2` rows benefit from axis-aligned linear leaves, while `l_r in {3,4}` still prefer the tighter `tree_axis_constant_smallcap_delta` family.",
                "- The strongest next design is a lead-time-conditioned portfolio or mixture on top of the same factorized control basis, rather than one universal backbone.",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main():
    args = parse_args()
    summary = json.loads(args.input.read_text(encoding="utf-8"))
    rendered = _render(summary)
    args.output.write_text(rendered, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
