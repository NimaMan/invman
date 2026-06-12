"""Fast, pure-data tests for the baseline catalog API (no training, no Rust).

Asserts the manifest loads, every entry has a valid difficulty + rationale, the
catalog lists all 14 problems, `get()` works for every problem and fails loudly
on an unknown one, and `render_card()` round-trips to non-empty markdown.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from invman.benchmarks import catalog

MANIFEST_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "benchmarks"
    / "BENCHMARK_MANIFEST.json"
)

VALID_DIFFICULTIES = {"easy", "medium", "hard"}
VALID_TIERS = {"strict", "reference", "faithful", "mixed"}
EXPECTED_PROBLEM_COUNT = 14

# The honest, ledger-reconciled headline verification tier per family
# (docs/benchmarks/VERIFICATION_LEDGER.md + FUNDAMENTAL_QUESTIONS.md are the
# authority; these are read VERBATIM from the manifest, never string-derived).
EXPECTED_TIER = {
    "lost_sales": "strict",
    "dual_sourcing": "strict",
    "multi_echelon": "mixed",
    "one_warehouse_multi_retailer": "faithful",
    "perishable_inventory": "strict",
    "ameliorating_inventory": "reference",
    "joint_replenishment": "reference",
    "joint_pricing_inventory": "faithful",
    "nonstationary_lot_sizing": "reference",
    "procurement_removal_inventory": "faithful",
    "random_yield_inventory": "faithful",
    "spare_parts_inventory": "reference",
    "vendor_managed_inventory": "faithful",
    "decentralized_inventory_control": "reference",
}


def test_manifest_is_valid_json_list_of_14() -> None:
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) == EXPECTED_PROBLEM_COUNT


def test_every_entry_has_valid_difficulty_and_nonempty_rationale() -> None:
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for entry in data:
        problem = entry.get("problem")
        difficulty = entry.get("difficulty")
        rationale = entry.get("difficulty_rationale", "")
        assert difficulty in VALID_DIFFICULTIES, (problem, difficulty)
        assert isinstance(rationale, str) and rationale.strip(), problem


def test_list_problems_returns_14_in_manifest_order() -> None:
    problems = catalog.list_problems()
    assert len(problems) == EXPECTED_PROBLEM_COUNT
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert problems == [entry["problem"] for entry in data]


def test_list_problems_difficulty_filter_partitions_all() -> None:
    buckets = {
        diff: set(catalog.list_problems(difficulty=diff))
        for diff in VALID_DIFFICULTIES
    }
    union = set().union(*buckets.values())
    assert union == set(catalog.list_problems())
    # Buckets are disjoint (each problem has exactly one difficulty).
    total = sum(len(b) for b in buckets.values())
    assert total == EXPECTED_PROBLEM_COUNT


def test_list_problems_verified_filter_partitions_all() -> None:
    buckets = {
        tier: set(catalog.list_problems(verified=tier)) for tier in VALID_TIERS
    }
    union = set().union(*buckets.values())
    assert union == set(catalog.list_problems())
    total = sum(len(b) for b in buckets.values())
    assert total == EXPECTED_PROBLEM_COUNT


def test_every_entry_has_explicit_manifest_verification_tier() -> None:
    """The tier MUST be an explicit, valid manifest field — never derived."""
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for entry in data:
        problem = entry.get("problem")
        tier = entry.get("verification_tier")
        assert tier in VALID_TIERS, (problem, tier)


def test_verification_tier_matches_ledger_reconciled_values() -> None:
    """Lock the honest, ledger-reconciled headline tier for every family.

    Guards the five honesty corrections specifically so they cannot silently
    regress: OWMR=faithful (approx-only, not strict), joint_replenishment=
    reference (reproduces an ACTION not a cost), spare_parts=reference (the
    strict bit is the ADJACENT Kranenburg analytical module), multi_echelon=
    mixed (umbrella), ameliorating=reference (companion LP bound, env faithful).
    """
    for problem, expected in EXPECTED_TIER.items():
        assert catalog.get(problem).verification.tier == expected, problem
    # The five corrected families, asserted by name so the intent is explicit.
    assert catalog.get("one_warehouse_multi_retailer").verification.tier == "faithful"
    assert catalog.get("joint_replenishment").verification.tier == "reference"
    assert catalog.get("spare_parts_inventory").verification.tier == "reference"
    assert catalog.get("multi_echelon").verification.tier == "mixed"
    assert catalog.get("ameliorating_inventory").verification.tier == "reference"


def test_strict_filter_excludes_approx_action_and_adjacent_module_families() -> None:
    """verified='strict' must NOT leak the over-labelled families."""
    strict = set(catalog.list_problems(verified="strict"))
    assert strict == {"lost_sales", "dual_sourcing", "perishable_inventory"}
    for leaked in (
        "one_warehouse_multi_retailer",
        "joint_replenishment",
        "spare_parts_inventory",
        "multi_echelon",
        "ameliorating_inventory",
    ):
        assert leaked not in strict, leaked


def test_multi_echelon_has_per_subfamily_verification_tier_map() -> None:
    card = catalog.get("multi_echelon")
    sub_map = card.verification.tier_by_subfamily
    assert sub_map, "mixed umbrella entry should map subfamily tiers"
    assert sub_map.get("serial") == "strict"
    assert sub_map.get("assembly") == "faithful"
    assert sub_map.get("production_assembly_distribution_network") == "faithful"
    assert sub_map.get("divergent_special_delivery") == "strict"
    assert sub_map.get("general_backorder_fixed_cost") == "strict"
    for tier in sub_map.values():
        assert tier in VALID_TIERS


def test_split_families_carry_a_tier_note() -> None:
    """Split / adjacent-module families must record the split in a note."""
    for problem in (
        "spare_parts_inventory",
        "ameliorating_inventory",
        "joint_replenishment",
        "decentralized_inventory_control",
    ):
        assert catalog.get(problem).verification.tier_note.strip(), problem


def test_list_problems_rejects_bad_filters() -> None:
    with pytest.raises(ValueError):
        catalog.list_problems(difficulty="trivial")
    with pytest.raises(ValueError):
        catalog.list_problems(verified="gold")


@pytest.mark.parametrize("problem", catalog.list_problems())
def test_get_returns_card_for_every_problem(problem: str) -> None:
    card = catalog.get(problem)
    assert card.problem == problem
    assert card.difficulty in VALID_DIFFICULTIES
    assert card.difficulty_rationale.strip()
    assert card.verification.tier in VALID_TIERS
    # A card must expose the consumer-facing surface.
    assert card.instances, problem
    assert isinstance(card.baselines.heuristics, list)
    assert isinstance(card.reproduce_commands, list)


def test_get_accepts_full_problem_string_and_is_case_insensitive() -> None:
    assert catalog.get("multi_echelon").problem == "multi_echelon"
    assert catalog.get("  Lost_Sales  ").problem == "lost_sales"


def test_get_raises_on_unknown_problem() -> None:
    with pytest.raises(KeyError):
        catalog.get("not_a_real_problem")
    with pytest.raises(KeyError):
        catalog.get("")


@pytest.mark.parametrize("problem", catalog.list_problems())
def test_render_card_produces_nonempty_markdown(problem: str) -> None:
    markdown = catalog.render_card(problem)
    assert isinstance(markdown, str)
    assert markdown.strip()
    # Round-trip: the card must name the problem, its difficulty, and a reproduce block.
    assert f"`{problem}`" in markdown
    assert "Difficulty:" in markdown
    assert "How to reproduce & compare" in markdown


def test_multi_echelon_has_per_subfamily_difficulty_map() -> None:
    card = catalog.get("multi_echelon")
    assert card.difficulty_by_subfamily, "umbrella entry should map subfamilies"
    assert card.difficulty_by_subfamily.get("serial") == "medium"
    for diff in card.difficulty_by_subfamily.values():
        assert diff in VALID_DIFFICULTIES


def test_render_all_cards_writes_15_files(tmp_path: Path) -> None:
    written = catalog.render_all_cards(tmp_path)
    # 14 problem cards + the index.
    assert len(written) == EXPECTED_PROBLEM_COUNT + 1
    assert (tmp_path / "README.md").exists()
    for problem in catalog.list_problems():
        path = tmp_path / f"{problem}.md"
        assert path.exists()
        assert path.read_text(encoding="utf-8").strip()
