use statrs::distribution::{Continuous, ContinuousCDF, Normal as StatNormal};

use crate::problems::multi_echelon::production_assembly_distribution_network::literature::{
    SingleNodeBenchmarkRow, PIRHOOSHYARAN_2021_REFERENCE, SINGLE_NODE_BENCHMARK_ROWS,
};

#[derive(Clone, Debug, PartialEq)]
pub struct SingleNodeBenchmarkResult {
    pub case_idx: usize,
    pub published_analytical_oul: f64,
    pub reproduced_analytical_oul: f64,
    pub published_analytical_average_cost: f64,
    pub reproduced_analytical_average_cost: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NetworkInventoryLiteratureBenchmarkSummary {
    pub source: &'static str,
    pub url: &'static str,
    pub single_node_results: Vec<SingleNodeBenchmarkResult>,
}

fn single_node_analytical_solution(row: &SingleNodeBenchmarkRow) -> SingleNodeBenchmarkResult {
    let normal = StatNormal::new(row.demand_mean, row.demand_stddev)
        .expect("single-node benchmark row must define a valid normal distribution");
    let critical_ratio = row.shortage_cost / (row.holding_cost + row.shortage_cost);
    let reproduced_analytical_oul = normal.inverse_cdf(critical_ratio);
    let z = (reproduced_analytical_oul - row.demand_mean) / row.demand_stddev;
    let standard_normal = StatNormal::new(0.0, 1.0).expect("standard normal must build");
    let pdf = standard_normal.pdf(z);
    let cdf = standard_normal.cdf(z);
    let expected_overage = row.demand_stddev * (pdf + z * cdf);
    let expected_underage = row.demand_stddev * (pdf - z * (1.0 - cdf));
    let reproduced_analytical_average_cost =
        row.holding_cost * expected_overage + row.shortage_cost * expected_underage;

    SingleNodeBenchmarkResult {
        case_idx: row.case_idx,
        published_analytical_oul: row.published_analytical_oul,
        reproduced_analytical_oul,
        published_analytical_average_cost: row.published_analytical_average_cost,
        reproduced_analytical_average_cost,
    }
}

pub fn literature_benchmark_summary(
    _serial_replications: usize,
    _seed: u64,
) -> NetworkInventoryLiteratureBenchmarkSummary {
    let single_node_results = SINGLE_NODE_BENCHMARK_ROWS
        .iter()
        .map(single_node_analytical_solution)
        .collect::<Vec<_>>();

    NetworkInventoryLiteratureBenchmarkSummary {
        source: PIRHOOSHYARAN_2021_REFERENCE.source,
        url: PIRHOOSHYARAN_2021_REFERENCE.url,
        single_node_results,
    }
}
