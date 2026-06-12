"""Tests for the executable baseline layer (`invman.benchmarks.runners`).

Covers the uniform surface the build plan's workstream (a) promises: the runner
registry, `catalog.get(problem).load_instance(name)`, the published baselines,
the canonical-reference selection, `compare()` math, and — running the live env
on small protocols — that `run_baselines` reproduces the published numbers and
`evaluate` round-trips through the CMA-ES seam.

The env-running tests use tiny horizons so the suite stays fast; they are skipped
cleanly if the compiled `invman_rust` extension is unavailable.
"""

from __future__ import annotations

import pytest

invman_rust = pytest.importorskip("invman_rust")

from invman.benchmarks import catalog, runners
from invman.benchmarks.runners.base import EvalProtocol

SMALL = EvalProtocol(seeds=(1234,), horizon=2000, warm_up_periods_ratio=0.2)
SMALL_ME = EvalProtocol(seeds=(123,), horizon=1500, warm_up_periods_ratio=0.0, replications=2)


# --- registry ---------------------------------------------------------------


def test_available_runners_are_the_three_built_families() -> None:
    assert set(runners.available_runners()) == {"lost_sales", "dual_sourcing", "multi_echelon"}


def test_get_runner_unknown_problem_raises() -> None:
    with pytest.raises(KeyError):
        runners.get_runner("not_a_problem")


def test_runner_instances_are_cached() -> None:
    assert runners.get_runner("lost_sales") is runners.get_runner("lost_sales")


# --- catalog bridge ---------------------------------------------------------


def test_catalog_card_exposes_runner_for_built_families() -> None:
    for problem in ("lost_sales", "dual_sourcing", "multi_echelon"):
        card = catalog.get(problem)
        assert card.has_runner
        assert card.runner().problem == problem


def test_catalog_card_without_runner_reports_false() -> None:
    card = catalog.get("perishable_inventory")
    assert card.has_runner is False
    with pytest.raises(KeyError):
        card.runner()


def test_list_instances_matches_rust() -> None:
    assert len(catalog.get("lost_sales").list_instances()) == len(
        invman_rust.lost_sales_reference_instance_names()
    ) + len(invman_rust.lost_sales_fixed_order_cost_list_reference_instances())
    assert len(catalog.get("dual_sourcing").list_instances()) == 6
    assert len(catalog.get("multi_echelon").list_instances()) == 5


# --- load + metadata --------------------------------------------------------


def test_lost_sales_vanilla_published_baselines() -> None:
    inst = catalog.get("lost_sales").load_instance("lit_poisson_p4_l4")
    assert inst.subfamily == "vanilla"
    assert inst.published_costs["optimal"] == pytest.approx(4.73, abs=1e-6)
    # The exact optimum is the canonical reference.
    assert inst.reference_baseline.name == "optimal"
    assert inst.reference_cost == pytest.approx(4.73, abs=1e-6)


def test_lost_sales_fixed_routing_and_published() -> None:
    inst = catalog.get("lost_sales").load_instance("bijvank2015_table1_l2_p14_k5")
    assert inst.subfamily == "fixed_order_cost"
    assert inst.published_costs["optimal_dp"] == pytest.approx(11.46, abs=1e-6)
    assert inst.reference_baseline.name == "optimal_dp"


def test_dual_sourcing_published_gaps_no_absolute_cost() -> None:
    inst = catalog.get("dual_sourcing").load_instance("dual_l2_ce105")
    # The paper reports gaps, not absolute costs -> no published cost, no reference.
    assert inst.published_costs == {}
    assert inst.reference_baseline is None
    gaps = {b.name: b.params["published_gap_pct"] for b in inst.published_baselines}
    assert gaps["capped_dual_index"] == pytest.approx(0.0, abs=1e-6)


def test_multi_echelon_reference_is_constant_base_stock() -> None:
    inst = catalog.get("multi_echelon").load_instance("van_roy1997_case_study1")
    # constant base-stock is the declared canonical comparator even though the
    # best-NDP cost (1179) is lower than it (1302).
    assert inst.reference_baseline.name == "constant_base_stock_published"
    assert inst.reference_cost == pytest.approx(1302.0, abs=1e-6)


def test_unknown_instance_raises() -> None:
    with pytest.raises(KeyError):
        runners.load_instance("lost_sales", "does_not_exist")


# --- compare ----------------------------------------------------------------


def test_compare_reports_signed_gap_and_beats() -> None:
    inst = catalog.get("lost_sales").load_instance("lit_poisson_p4_l4")
    worse = inst.compare(5.0)  # above optimal 4.73 -> does not beat
    assert worse["beats"] is False
    assert worse["gap_abs"] == pytest.approx(0.27, abs=1e-6)
    better = inst.compare(4.70)  # below optimal -> beats (and gap is negative)
    assert better["beats"] is True
    assert better["gap_pct"] < 0.0


def test_compare_against_named_baseline() -> None:
    inst = catalog.get("lost_sales").load_instance("lit_poisson_p4_l4")
    out = inst.compare(4.82, against="myopic2")
    assert out["reference"] == "myopic2"
    assert out["gap_abs"] == pytest.approx(0.0, abs=1e-6)


# --- run the live env (small protocols) -------------------------------------


def test_fixed_cost_run_baselines_reproduces_published() -> None:
    inst = catalog.get("lost_sales").load_instance("bijvank2015_table1_l2_p14_k5")
    recomputed = inst.run_baselines()  # exact average-cost VI; deterministic
    assert recomputed["optimal_dp"].mean_cost == pytest.approx(11.46, abs=0.05)
    assert recomputed["s_s"].mean_cost == pytest.approx(11.62, abs=0.05)


def test_vanilla_run_baselines_close_to_published() -> None:
    inst = catalog.get("lost_sales").load_instance("lit_poisson_p4_l4")
    recomputed = inst.run_baselines(SMALL)
    assert recomputed["myopic2"].mean_cost == pytest.approx(4.82, rel=0.05)


def test_dual_sourcing_capped_dual_index_is_cheapest() -> None:
    inst = catalog.get("dual_sourcing").load_instance("dual_l2_ce105")
    recomputed = inst.run_baselines(SMALL)
    costs = {k: v.mean_cost for k, v in recomputed.items() if v.available}
    assert min(costs, key=costs.get) == "capped_dual_index"
    assert recomputed["capped_dual_index"].is_optimal is True


def test_multi_echelon_run_baselines_reproduces_simple_problem() -> None:
    inst = catalog.get("multi_echelon").load_instance("van_roy1997_simple_problem")
    recomputed = inst.run_baselines(SMALL_ME)
    cbs = recomputed["constant_base_stock"]
    assert cbs.available and cbs.mean_cost > 0.0
    # published constant base-stock is 51.7; env-sim reproduces it loosely.
    assert cbs.mean_cost == pytest.approx(51.7, rel=0.20)


# --- evaluate seam ----------------------------------------------------------


def test_policy_param_count_and_evaluate_round_trip() -> None:
    inst = catalog.get("lost_sales").load_instance("lit_poisson_p4_l4")
    n = inst.policy_param_count()
    assert n > 0
    cost = inst.evaluate([0.0] * n, protocol=SMALL)
    assert cost > 0.0 and cost == cost  # finite, not NaN


def test_evaluate_wrong_param_count_raises() -> None:
    inst = catalog.get("lost_sales").load_instance("lit_poisson_p4_l4")
    with pytest.raises(ValueError):
        inst.evaluate([0.0, 0.0, 0.0], protocol=SMALL)  # wrong length
