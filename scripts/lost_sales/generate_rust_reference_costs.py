"""
Generate src/problems/lost_sales/reference_costs.rs: the lost-sales
benchmark grid with per-instance heuristic reference costs.

Sources (literature preferred, repo-computed for the gaps):
  - Literature optimal / M1 / M2 / SVBS / CappedBS: the (now-deleted) Python
    problem_info.py, recovered from git to /tmp/old_problem_info.py.
  - Computed M1 / M2 / SVBS for the literature gaps (Geometric M1/SVBS, Poisson
    L10 M1/SVBS, all MMPP2): outputs/computed_lost_sales_heuristics_*.jsonl,
    produced by the Rust heuristics evaluator (IID + MMPP2 stationary-marginal).

Each instance records a `source`: "literature", "literature+computed", or
"computed".

Usage:
  python3 scripts/lost_sales/generate_rust_reference_costs.py \
      > src/problems/lost_sales/reference_costs.rs
"""

from __future__ import annotations

import glob
import importlib.util
import json
import sys

LIT_PATH = "/tmp/old_problem_info.py"
COMPUTED_GLOB = "outputs/computed_lost_sales_heuristics_*.jsonl"

DEMAND_CASES = {
    "poisson": dict(token="poisson", lit="Poisson", kind="Poisson",
                    lo=0.0, hi=0.0, p00=0.0, p11=0.0),
    "geometric": dict(token="geometric", lit="Geometric", kind="Geometric",
                      lo=0.0, hi=0.0, p00=0.0, p11=0.0),
    "mmpp2_pos": dict(token="mmpp2_pos", lit=None, kind="MarkovModulatedPoisson2",
                      lo=3.0, hi=7.0, p00=0.9, p11=0.9),
    "mmpp2_neg": dict(token="mmpp2_neg", lit=None, kind="MarkovModulatedPoisson2",
                      lo=3.0, hi=7.0, p00=0.1, p11=0.1),
}
SHORTAGES = [4, 19]
LEADS = [4, 6, 8, 10]


def _valid(v):
    return v is not None and not (isinstance(v, float) and v != v)  # not None, not NaN


def _load_literature():
    spec = importlib.util.spec_from_file_location("old_pi", LIT_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.problem_info


def _load_computed():
    path = sorted(glob.glob(COMPUTED_GLOB))[-1]
    comp = {}
    for line in open(path):
        r = json.loads(line)
        comp.setdefault(r["instance"], {})[r["heuristic"]] = r["value"]
    return comp, path


def _opt(v):
    return "None" if not _valid(v) else f"Some({float(v)})"


def _merge(lit_cell, comp_cell):
    """Return (costs_dict, source) merging literature (preferred) with computed."""
    lit_cell = lit_cell or {}
    comp_cell = comp_cell or {}
    used_lit = used_comp = False

    def pick(lit_key, comp_key):
        nonlocal used_lit, used_comp
        lv = lit_cell.get(lit_key)
        if _valid(lv):
            used_lit = True
            return float(lv)
        cv = comp_cell.get(comp_key)
        if _valid(cv):
            used_comp = True
            return float(cv)
        return None

    costs = {
        "optimal": (float(lit_cell["optimal"]) if _valid(lit_cell.get("optimal")) else None),
        "myopic1": pick("M1", "myopic1"),
        "myopic2": pick("M2", "myopic2"),
        "svbs": pick("SVBS", "svbs"),
        "capped_base_stock": (float(lit_cell["CappedBS"]) if _valid(lit_cell.get("CappedBS")) else None),
    }
    if _valid(lit_cell.get("optimal")) or _valid(lit_cell.get("CappedBS")):
        used_lit = True
    source = ("literature+computed" if used_lit and used_comp
              else "literature" if used_lit else "computed")
    return costs, source


def _entry(name, case, p, L, costs, source):
    return f"""    LostSalesReferenceInstance {{
        name: "{name}",
        demand: LostSalesDemandConfig {{
            kind: LostSalesDemandKind::{case['kind']},
            demand_rate: 5.0,
            demand_lambda_low: {case['lo']},
            demand_lambda_high: {case['hi']},
            demand_p00: {case['p00']},
            demand_p11: {case['p11']},
        }},
        lead_time: {L},
        holding_cost: 1.0,
        shortage_cost: {float(p)},
        costs: HeuristicReferenceCosts {{
            optimal: {_opt(costs['optimal'])},
            myopic1: {_opt(costs['myopic1'])},
            myopic2: {_opt(costs['myopic2'])},
            svbs: {_opt(costs['svbs'])},
            capped_base_stock: {_opt(costs['capped_base_stock'])},
        }},
        source: "{source}",
    }},"""


def main():
    lit = _load_literature()
    comp, comp_path = _load_computed()

    entries = []
    # canonical vanilla instance (all literature)
    entries.append(_entry(
        "vanilla_l4_p4_poisson5", DEMAND_CASES["poisson"], 4, 4,
        {"optimal": 4.73, "myopic1": 5.06, "myopic2": 4.82, "svbs": 5.83, "capped_base_stock": 4.80},
        "literature"))
    for case_key, case in DEMAND_CASES.items():
        for p in SHORTAGES:
            for L in LEADS:
                name = f"lit_{case['token']}_p{p}_l{L}"
                lit_cell = lit.get(f"{case['lit']}_demand_shortage_cost_{p}", {}).get(L) if case["lit"] else None
                comp_cell = comp.get(name)
                costs, source = _merge(lit_cell, comp_cell)
                entries.append(_entry(name, case, p, L, costs, source))

    body = "\n".join(entries)
    print(f"""//! Reference heuristic costs for the lost-sales benchmark grid.
//!
//! Per-instance optimal / Myopic-1 / Myopic-2 / SVBS / capped-base-stock costs.
//! Literature values (Xin 2020 extension of Zipkin 2008) are used where the
//! literature reports them; the remaining cells (Geometric M1/SVBS, Poisson
//! L=10 M1/SVBS, and all Markov-modulated rows) are repo-computed by the
//! `heuristics` evaluator in this crate (IID demand directly; MMPP2 via the
//! stationary-marginal demand law). Each instance records which in `source`.
//!
//! Generated by scripts/lost_sales/generate_rust_reference_costs.py from the
//! recovered literature table and {comp_path}.

use crate::problems::lost_sales::demand::{{LostSalesDemandConfig, LostSalesDemandKind}};

/// Reference heuristic mean costs for one lost-sales instance. `None` means the
/// value is neither in the literature nor computed for this instance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeuristicReferenceCosts {{
    pub optimal: Option<f64>,
    pub myopic1: Option<f64>,
    pub myopic2: Option<f64>,
    pub svbs: Option<f64>,
    pub capped_base_stock: Option<f64>,
}}

/// A benchmark-grid instance with its demand process and reference costs.
#[derive(Clone, Copy)]
pub struct LostSalesReferenceInstance {{
    pub name: &'static str,
    pub demand: LostSalesDemandConfig,
    pub lead_time: usize,
    pub holding_cost: f64,
    pub shortage_cost: f64,
    pub costs: HeuristicReferenceCosts,
    /// "literature" | "literature+computed" | "computed".
    pub source: &'static str,
}}

/// The full benchmark grid: canonical vanilla instance plus
/// {{Poisson, Geometric, MMPP2+, MMPP2-}} x p in {{4,19}} x L in {{4,6,8,10}}.
pub const REFERENCE_INSTANCES: &[LostSalesReferenceInstance] = &[
{body}
];

/// Look up a reference instance by name.
pub fn reference_instance(name: &str) -> Option<&'static LostSalesReferenceInstance> {{
    REFERENCE_INSTANCES.iter().find(|inst| inst.name == name)
}}

/// All reference-instance names.
pub fn reference_instance_names() -> Vec<&'static str> {{
    REFERENCE_INSTANCES.iter().map(|inst| inst.name).collect()
}}

#[cfg(test)]
mod tests {{
    use super::*;

    #[test]
    fn vanilla_matches_literature() {{
        let v = reference_instance("vanilla_l4_p4_poisson5").unwrap();
        assert_eq!(v.costs.myopic2, Some(4.82));
        assert_eq!(v.costs.svbs, Some(5.83));
    }}

    #[test]
    fn every_grid_instance_has_the_three_heuristics() {{
        // Myopic-1, Myopic-2, SVBS are populated (literature or computed) for
        // every instance on the active L in {{4,6,8,10}} grid.
        for inst in REFERENCE_INSTANCES {{
            assert!(inst.costs.myopic1.is_some(), "{{}} missing myopic1", inst.name);
            assert!(inst.costs.myopic2.is_some(), "{{}} missing myopic2", inst.name);
            assert!(inst.costs.svbs.is_some(), "{{}} missing svbs", inst.name);
        }}
    }}
}}
""")
    print(f"// {len(entries)} instances emitted", file=sys.stderr)


if __name__ == "__main__":
    main()
