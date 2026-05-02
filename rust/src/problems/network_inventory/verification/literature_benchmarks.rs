use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal as RandNormal};
use statrs::distribution::{Continuous, ContinuousCDF, Normal as StatNormal};

use crate::problems::network_inventory::literature::{
    SerialBenchmarkRow, SingleNodeBenchmarkRow, PIRHOOSHYARAN_2021_REFERENCE,
    SERIAL_BENCHMARK_ROWS, SINGLE_NODE_BENCHMARK_ROWS,
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
pub struct SerialBenchmarkResult {
    pub case_idx: usize,
    pub published_analytical_ouls: Vec<f64>,
    pub published_average_cost: f64,
    pub reproduced_average_cost: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NetworkInventoryLiteratureBenchmarkSummary {
    pub source: &'static str,
    pub url: &'static str,
    pub single_node_results: Vec<SingleNodeBenchmarkResult>,
    pub serial_results: Vec<SerialBenchmarkResult>,
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

// Paper-facing serial benchmark audit.
//
// The paper's serial recurrence mixes explicit backorders with negative finished-goods inventory
// in a way that is not fully algebraically consistent. This simulator follows the paper's stated
// event order and local inventory-position definition while keeping the update logic executable:
//
// - continuous normal demand, truncated at zero
// - fixed OULs in the published upstream-to-downstream edge order
// - finished inventory initialized to local lead-time demand
// - raw inventories and pipelines initialized to zero
// - raw-only inventory positions
// - current demand is subtracted from finished inventory as written in the paper
// - backlog is carried explicitly per node/customer
// - the carried customer backlog is also used as the order-position backlog term
// - under shortage we use the paper's printed backlog update `BO_t = BO_{t-1} - S_t`, which is
//   not fully coherent as an inventory model but is the closest paper-facing executable surrogate
//
// The result is an audit surrogate, not a claim that the executable Rust environment is already
// literature-verified on the serial rows.
fn simulate_serial_row(
    row: &SerialBenchmarkRow,
    replications: usize,
    seed: u64,
) -> SerialBenchmarkResult {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = RandNormal::new(row.demand_mean, row.demand_stddev)
        .expect("serial benchmark row must define a valid normal distribution");
    let mut average_cost_sum = 0.0;

    for _ in 0..replications {
        let num_nodes = row.num_echelons;
        let mut finished_inventory = row
            .lead_times
            .iter()
            .map(|lead_time| row.demand_mean * *lead_time as f64)
            .collect::<Vec<_>>();
        let mut raw_inventory = vec![0.0; num_nodes];
        let mut backlog = vec![0.0; num_nodes];
        let mut pipelines = row
            .lead_times
            .iter()
            .map(|lead_time| vec![0.0; *lead_time])
            .collect::<Vec<_>>();

        let mut episode_cost = 0.0;
        for _period in 0..row.horizon_periods {
            let realized_external_demand = normal.sample(&mut rng).max(0.0);
            let mut orders = vec![0.0; num_nodes];
            let mut downstream_demand = realized_external_demand;

            for node_idx in (0..num_nodes).rev() {
                let pipeline_inbound = pipelines[node_idx].iter().sum::<f64>();
                let inventory_position =
                    raw_inventory[node_idx] - downstream_demand + pipeline_inbound + backlog[node_idx];
                orders[node_idx] =
                    (row.published_analytical_ouls[node_idx] - inventory_position).max(0.0);
                downstream_demand = orders[node_idx];
            }

            for node_idx in 0..num_nodes {
                let arrival = if row.lead_times[node_idx] == 0 {
                    0.0
                } else {
                    pipelines[node_idx].remove(0)
                };
                raw_inventory[node_idx] += arrival;
                if node_idx == 0 && row.lead_times[0] > 0 {
                    pipelines[0].push(orders[0]);
                }

                finished_inventory[node_idx] += raw_inventory[node_idx];
                raw_inventory[node_idx] = 0.0;

                let current_demand = if node_idx + 1 < num_nodes {
                    orders[node_idx + 1]
                } else {
                    realized_external_demand
                };
                let total_due = backlog[node_idx] + current_demand;
                let shipped = finished_inventory[node_idx].max(0.0).min(total_due);
                backlog[node_idx] = (backlog[node_idx] - shipped).max(0.0);

                if node_idx + 1 < num_nodes && row.lead_times[node_idx + 1] > 0 {
                    pipelines[node_idx + 1].push(shipped);
                }

                finished_inventory[node_idx] -= current_demand;
            }

            let mut period_cost = 0.0;
            for node_idx in 0..num_nodes {
                let outbound_in_transit = if node_idx + 1 < num_nodes {
                    pipelines[node_idx + 1].iter().sum::<f64>()
                } else {
                    0.0
                };
                period_cost += row.holding_costs[node_idx]
                    * (raw_inventory[node_idx] + finished_inventory[node_idx].max(0.0) + outbound_in_transit);
            }
            period_cost += row.shortage_costs[num_nodes - 1] * backlog[num_nodes - 1];
            episode_cost += period_cost;
        }

        average_cost_sum += episode_cost / row.horizon_periods as f64;
    }

    SerialBenchmarkResult {
        case_idx: row.case_idx,
        published_analytical_ouls: row.published_analytical_ouls.to_vec(),
        published_average_cost: row.published_average_cost,
        reproduced_average_cost: average_cost_sum / replications as f64,
    }
}

pub fn literature_benchmark_summary(
    serial_replications: usize,
    seed: u64,
) -> NetworkInventoryLiteratureBenchmarkSummary {
    let single_node_results = SINGLE_NODE_BENCHMARK_ROWS
        .iter()
        .map(single_node_analytical_solution)
        .collect::<Vec<_>>();
    let serial_results = SERIAL_BENCHMARK_ROWS
        .iter()
        .map(|row| simulate_serial_row(row, serial_replications, seed + row.case_idx as u64))
        .collect::<Vec<_>>();

    NetworkInventoryLiteratureBenchmarkSummary {
        source: PIRHOOSHYARAN_2021_REFERENCE.source,
        url: PIRHOOSHYARAN_2021_REFERENCE.url,
        single_node_results,
        serial_results,
    }
}
