#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PublishedBenchmarkReference {
    pub source: &'static str,
    pub url: &'static str,
    pub benchmark_policies: &'static [&'static str],
    pub reported_numbers_available: bool,
    pub notes: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PublishedHeuristicRow {
    pub policy_name: &'static str,
    pub params: &'static [usize],
    pub mean_cost: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FixedCostLostSalesReferenceInstance {
    pub name: &'static str,
    pub source: &'static str,
    pub url: &'static str,
    pub literature_verified: bool,
    pub review_periods: usize,
    pub lead_time: usize,
    pub demand_distribution: &'static str,
    pub demand_mean_per_review_period: f64,
    pub holding_cost: f64,
    pub shortage_cost: f64,
    pub fixed_order_cost: f64,
    pub published_optimal_cost: Option<f64>,
    pub published_heuristic_rows: &'static [PublishedHeuristicRow],
    pub benchmark_policies: &'static [&'static str],
    pub notes: &'static str,
}

pub const BIJVANK_2015_REFERENCE: PublishedBenchmarkReference = PublishedBenchmarkReference {
    source: "Bijvank, Bhulai, and Huh (2015), EJOR 241(2):381-390",
    url: "https://www.math.vu.nl/~sbhulai/publications/ejor2015b.pdf",
    benchmark_policies: &["optimal_dp", "s_s", "s_nq", "modified_s_s_q"],
    reported_numbers_available: true,
    notes: "Table 1 gives a published validation instance for R=1, L=2, h=1, p=14, K=5 with Poisson demand mean 5, including the optimal average cost and the best (s,S), (s,nQ), and modified (s,S,q) rows.",
};

pub const BIJVANK_2015_TABLE1_HEURISTICS: &[PublishedHeuristicRow] = &[
    PublishedHeuristicRow {
        policy_name: "s_s",
        params: &[17, 23],
        mean_cost: 11.62,
    },
    PublishedHeuristicRow {
        policy_name: "s_nq",
        params: &[17, 7],
        mean_cost: 11.56,
    },
    PublishedHeuristicRow {
        policy_name: "modified_s_s_q",
        params: &[17, 23, 7],
        mean_cost: 11.50,
    },
];

pub const BIJVANK_2015_TABLE1_REFERENCE: FixedCostLostSalesReferenceInstance =
    FixedCostLostSalesReferenceInstance {
        name: "bijvank2015_table1_l2_p14_k5",
        source: BIJVANK_2015_REFERENCE.source,
        url: BIJVANK_2015_REFERENCE.url,
        literature_verified: true,
        review_periods: 1,
        lead_time: 2,
        demand_distribution: "poisson",
        demand_mean_per_review_period: 5.0,
        holding_cost: 1.0,
        shortage_cost: 14.0,
        fixed_order_cost: 5.0,
        published_optimal_cost: Some(11.46),
        published_heuristic_rows: BIJVANK_2015_TABLE1_HEURISTICS,
        benchmark_policies: BIJVANK_2015_REFERENCE.benchmark_policies,
        notes: "Published Table 1 validation row. The current Rust average-cost value-iteration solver and exact heuristic evaluators reproduce the published costs tightly for this instance.",
    };

pub const REFERENCE_INSTANCES: &[FixedCostLostSalesReferenceInstance] =
    &[BIJVANK_2015_TABLE1_REFERENCE];

pub fn list_reference_instances() -> Vec<&'static str> {
    REFERENCE_INSTANCES.iter().map(|instance| instance.name).collect()
}

pub fn get_reference_instance(
    name: &str,
) -> Option<&'static FixedCostLostSalesReferenceInstance> {
    REFERENCE_INSTANCES.iter().find(|instance| instance.name == name)
}
