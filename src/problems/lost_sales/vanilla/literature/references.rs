//! Literature anchor for the vanilla (no fixed-order-cost) lost-sales family.
//!
//! Algorithmic / provenance description
//! ------------------------------------
//! The canonical vanilla instance `vanilla_l4_p4_poisson5` (discrete-time periodic-review
//! lost-sales, Poisson demand mean lambda=5, lead time L=4, holding h=1, shortage p=4) is the
//! benchmark whose per-policy average costs are reproduced by this crate's env + heuristic
//! rollout (see `../heuristics/mod.rs::vanilla_heuristic_mean_costs_match_literature_numbers`,
//! which simulates the env and asserts the simulated mean costs match the numbers below).
//!
//! Source of those numbers (pinned): Zipkin (2008), "Old and New Methods for Lost-Sales
//! Inventory Systems," Operations Research 56(5):1256-1263, **Table 3(a)** (Poisson, penalty
//! p=4), lead-time column 4 (p.1261). The Poisson mean lambda=5 is the value Zipkin uses for
//! these experiments and is restated by Gijsbrechts, Boute, Van Mieghem & Zhang (2022),
//! Management Science 68(3):1885-1903, p.11 ("demand is Poisson distributed with lambda = 5").
//!
//! Mapping of carried policy names to Zipkin's Table 3 rows:
//!   - `optimal`  -> Zipkin "Optimal"                  = 4.73  (published_optimal_cost)
//!   - `myopic1`  -> Zipkin "Myopic"  (= myopic-1)     = 5.06  (+7.0% over optimal)
//!   - `myopic2`  -> Zipkin "Myopic-2"                 = 4.82  (+1.9%)
//!   - `svbs`     -> Zipkin "Standard vector base-stock" (Eq. 5, Morton 1969/1971) = 5.83 (+23.3%)
//!   - `better_vector_base_stock` -> Zipkin "Better vector base-stock" = 4.80 (+1.5%)
//!
//! Note on `capped_base_stock` in `reference_costs.rs`: the carried value 4.80 is Zipkin's
//! "Better vector base-stock" row. The capped base-stock policy of Xin (2021), "Understanding
//! the Performance of Capped Base-Stock Policies in Lost-Sales Inventory Models," Operations
//! Research 69(1):61-70 (DOI 10.1287/opre.2020.2019), Table 1, reports a comparable ~4.8 cost on
//! this same Zipkin instance; it is corroborating, but the load-bearing pin for 4.80 is Zipkin's
//! own better-vector-base-stock row.

/// A published benchmark source (one paper / one table) carried for provenance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PublishedBenchmarkReference {
    pub source: &'static str,
    pub url: &'static str,
    pub benchmark_policies: &'static [&'static str],
    pub reported_numbers_available: bool,
    pub notes: &'static str,
}

/// One published per-policy average-cost cell from the source table.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PublishedHeuristicRow {
    /// Policy identifier as used elsewhere in the crate (matches `reference_costs.rs`).
    pub policy_name: &'static str,
    /// Zipkin Table 3 row label the value is transcribed from.
    pub published_row_label: &'static str,
    pub mean_cost: f64,
}

/// A literature reference instance: the published numbers + the flag asserting that this crate's
/// env + heuristics reproduce them (proven by an in-crate test).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VanillaLostSalesReferenceInstance {
    pub name: &'static str,
    pub source: &'static str,
    pub url: &'static str,
    pub literature_verified: bool,
    pub lead_time: usize,
    pub demand_distribution: &'static str,
    pub demand_mean: f64,
    pub holding_cost: f64,
    pub shortage_cost: f64,
    pub published_optimal_cost: Option<f64>,
    pub published_heuristic_rows: &'static [PublishedHeuristicRow],
    pub benchmark_policies: &'static [&'static str],
    pub notes: &'static str,
}

pub const ZIPKIN_2008_REFERENCE: PublishedBenchmarkReference = PublishedBenchmarkReference {
    source: "Zipkin (2008), \"Old and New Methods for Lost-Sales Inventory Systems\", Operations Research 56(5):1256-1263, Table 3(a) (Poisson, p=4), lead-time column 4 (p.1261)",
    url: "https://doi.org/10.1287/opre.1070.0471",
    benchmark_policies: &["optimal", "myopic1", "myopic2", "svbs", "better_vector_base_stock"],
    reported_numbers_available: true,
    notes: "Poisson mean lambda=5 (restated by Gijsbrechts et al. 2022, Management Science 68(3), p.11). 'myopic1' is Zipkin's 'Myopic'; 'svbs' is Zipkin's 'Standard vector base-stock' (Eq. 5, Morton 1969/1971). The carried capped_base_stock=4.80 equals Zipkin's 'Better vector base-stock' row; Xin (2021), Operations Research 69(1):61-70 (DOI 10.1287/opre.2020.2019), Table 1, reports a comparable ~4.8 capped-base-stock cost on this instance (corroborating).",
};

pub const ZIPKIN_2008_TABLE3A_L4_HEURISTICS: &[PublishedHeuristicRow] = &[
    PublishedHeuristicRow {
        policy_name: "myopic1",
        published_row_label: "Myopic",
        mean_cost: 5.06,
    },
    PublishedHeuristicRow {
        policy_name: "myopic2",
        published_row_label: "Myopic-2",
        mean_cost: 4.82,
    },
    PublishedHeuristicRow {
        policy_name: "svbs",
        published_row_label: "Standard vector base-stock",
        mean_cost: 5.83,
    },
    PublishedHeuristicRow {
        policy_name: "better_vector_base_stock",
        published_row_label: "Better vector base-stock",
        mean_cost: 4.80,
    },
];

pub const ZIPKIN_2008_TABLE3A_L4_REFERENCE: VanillaLostSalesReferenceInstance =
    VanillaLostSalesReferenceInstance {
        name: "vanilla_l4_p4_poisson5",
        source: ZIPKIN_2008_REFERENCE.source,
        url: ZIPKIN_2008_REFERENCE.url,
        literature_verified: true,
        lead_time: 4,
        demand_distribution: "poisson",
        demand_mean: 5.0,
        holding_cost: 1.0,
        shortage_cost: 4.0,
        published_optimal_cost: Some(4.73),
        published_heuristic_rows: ZIPKIN_2008_TABLE3A_L4_HEURISTICS,
        benchmark_policies: ZIPKIN_2008_REFERENCE.benchmark_policies,
        notes: "Published Zipkin Table 3(a) L=4 validation row. A live env+heuristic rollout reproduces the Myopic-1 (5.06), Myopic-2 (4.82) and SVBS (5.83) average costs to within ~0.015 (see ../heuristics/mod.rs::vanilla_heuristic_mean_costs_match_literature_numbers, tolerance 0.12). The 4.73 optimal is the DP value, not produced by a heuristic rollout.",
    };

/// Repo-canonical primary instance (Standard Module Contract).
pub const PRIMARY_REFERENCE_INSTANCE: VanillaLostSalesReferenceInstance =
    ZIPKIN_2008_TABLE3A_L4_REFERENCE;

/// The instance whose published numbers the in-crate verification test reproduces.
pub const VERIFICATION_PROBLEM_INSTANCE: VanillaLostSalesReferenceInstance =
    ZIPKIN_2008_TABLE3A_L4_REFERENCE;

pub const REFERENCE_INSTANCES: &[VanillaLostSalesReferenceInstance] =
    &[ZIPKIN_2008_TABLE3A_L4_REFERENCE];

pub fn list_reference_instances() -> Vec<&'static str> {
    REFERENCE_INSTANCES
        .iter()
        .map(|instance| instance.name)
        .collect()
}

pub fn get_reference_instance(name: &str) -> Option<&'static VanillaLostSalesReferenceInstance> {
    REFERENCE_INSTANCES
        .iter()
        .find(|instance| instance.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problems::lost_sales::vanilla::reference_costs;

    /// The canonical instance is asserted literature-verified, and the pinned published numbers
    /// are kept in lock-step with the carried `reference_costs.rs` cells so the two cannot drift.
    #[test]
    fn pinned_zipkin_table3a_numbers_match_carried_reference_costs() {
        let inst = PRIMARY_REFERENCE_INSTANCE;
        assert!(inst.literature_verified, "canonical instance must be literature-verified");

        let carried = reference_costs::reference_instance("vanilla_l4_p4_poisson5")
            .expect("canonical reference_costs instance must exist");

        assert_eq!(carried.costs.optimal, inst.published_optimal_cost);

        for row in inst.published_heuristic_rows {
            let carried_cost = match row.policy_name {
                "myopic1" => carried.costs.myopic1,
                "myopic2" => carried.costs.myopic2,
                "svbs" => carried.costs.svbs,
                "better_vector_base_stock" => carried.costs.capped_base_stock,
                other => panic!("unexpected pinned policy {other}"),
            };
            assert_eq!(
                carried_cost,
                Some(row.mean_cost),
                "pinned {} ({}) != carried reference_costs cell",
                row.policy_name,
                row.published_row_label
            );
        }
    }
}
