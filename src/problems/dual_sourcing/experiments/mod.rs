#![allow(dead_code)]

use crate::problems::dual_sourcing::literature::{
    get_figure_9_gap_reference, get_reference_instance, GIJSBRECHTS_2022_REFERENCE,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DualSourcingExperimentGrid {
    pub name: &'static str,
    pub description: &'static str,
    pub source: &'static str,
    pub url: &'static str,
    pub reference_instance_names: &'static [&'static str],
    pub regular_lead_times: &'static [usize],
    pub expedited_order_costs: &'static [f64],
    pub regular_order_cost: f64,
    pub holding_cost: f64,
    pub shortage_cost: f64,
    pub demand_low: usize,
    pub demand_high: usize,
    pub regular_max_order_size: usize,
    pub expedited_max_order_size: usize,
    pub horizon: usize,
    pub eval_horizon: usize,
    pub eval_seeds: usize,
    pub search_seed: u64,
    pub inventory_lower: i64,
    pub inventory_upper: i64,
    pub solver_tolerance: f64,
    pub max_iterations: usize,
    pub warm_up_periods_ratio: f64,
    pub state_features: &'static str,
    pub notes: &'static str,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DualSourcingExperimentInstance {
    pub name: String,
    pub description: String,
    pub reference_instance_name: &'static str,
    pub source: &'static str,
    pub url: &'static str,
    pub literature_verified: bool,
    pub literature_verification_metric: &'static str,
    pub regular_lead_time: usize,
    pub expedited_lead_time: usize,
    pub regular_order_cost: f64,
    pub expedited_order_cost: f64,
    pub holding_cost: f64,
    pub shortage_cost: f64,
    pub regular_max_order_size: usize,
    pub expedited_max_order_size: usize,
    pub demand_low: usize,
    pub demand_high: usize,
    pub horizon: usize,
    pub eval_horizon: usize,
    pub eval_seeds: usize,
    pub seed: u64,
    pub inventory_lower: i64,
    pub inventory_upper: i64,
    pub solver_tolerance: f64,
    pub max_iterations: usize,
    pub warm_up_periods_ratio: f64,
    pub state_features: &'static str,
    pub benchmark_family: &'static str,
    pub benchmark_policies: &'static [&'static str],
    pub notes: String,
}

pub const GIJSBRECHTS_2022_FIGURE9_FAMILY_NAME: &str = "gijsbrechts2022_figure9_family";

const FIGURE9_REFERENCE_INSTANCE_NAMES: &[&str] = &[
    "dual_l2_ce105",
    "dual_l2_ce110",
    "dual_l3_ce105",
    "dual_l3_ce110",
    "dual_l4_ce105",
    "dual_l4_ce110",
];

const FIGURE9_REGULAR_LEAD_TIMES: &[usize] = &[2, 3, 4];
const FIGURE9_EXPEDITED_ORDER_COSTS: &[f64] = &[105.0, 110.0];

pub const GIJSBRECHTS_2022_FIGURE9_FAMILY: DualSourcingExperimentGrid =
    DualSourcingExperimentGrid {
        name: GIJSBRECHTS_2022_FIGURE9_FAMILY_NAME,
        description: "Six small-scale dual-sourcing benchmark rows from Gijsbrechts et al. (2022), Section 6.2 / Figure 9.",
        source: GIJSBRECHTS_2022_REFERENCE.source,
        url: GIJSBRECHTS_2022_REFERENCE.url,
        reference_instance_names: FIGURE9_REFERENCE_INSTANCE_NAMES,
        regular_lead_times: FIGURE9_REGULAR_LEAD_TIMES,
        expedited_order_costs: FIGURE9_EXPEDITED_ORDER_COSTS,
        regular_order_cost: 100.0,
        holding_cost: 5.0,
        shortage_cost: 495.0,
        demand_low: 0,
        demand_high: 4,
        regular_max_order_size: 12,
        expedited_max_order_size: 12,
        horizon: 6000,
        eval_horizon: 20_000,
        eval_seeds: 3,
        search_seed: 123,
        inventory_lower: -12,
        inventory_upper: 24,
        solver_tolerance: 1e-8,
        max_iterations: 250,
        warm_up_periods_ratio: 0.2,
        state_features: "pipeline",
        notes: "This grid matches the six published Figure 9 problem rows. The literature verification target is the published relative optimality gaps, not unpublished absolute costs.",
    };

pub const EXPERIMENT_GRIDS: &[DualSourcingExperimentGrid] = &[GIJSBRECHTS_2022_FIGURE9_FAMILY];

pub fn list_experiment_grids() -> &'static [DualSourcingExperimentGrid] {
    EXPERIMENT_GRIDS
}

pub fn get_experiment_grid(name: &str) -> Option<&'static DualSourcingExperimentGrid> {
    EXPERIMENT_GRIDS.iter().find(|grid| grid.name == name)
}

pub fn expand_experiment_grid(name: &str) -> Result<Vec<DualSourcingExperimentInstance>, String> {
    let grid =
        get_experiment_grid(name).ok_or_else(|| format!("unknown dual-sourcing grid '{name}'"))?;
    let mut instances = Vec::with_capacity(grid.reference_instance_names.len());
    for &reference_name in grid.reference_instance_names.iter() {
        let reference = get_reference_instance(reference_name)
            .ok_or_else(|| format!("unknown dual-sourcing reference '{reference_name}'"))?;
        let notes = match get_figure_9_gap_reference(reference.name) {
            Some(gaps) => format!(
                "{} Figure 9 publishes rounded optimality-gap labels of capped dual-index {:.2}%, dual-index {:.2}%, single-index {:.2}%, tailored base-surge {:.2}%, and A3C {:.2}%. {}",
                reference.notes,
                gaps.capped_dual_index_gap_pct,
                gaps.dual_index_gap_pct,
                gaps.single_index_gap_pct,
                gaps.tailored_base_surge_gap_pct,
                gaps.a3c_gap_pct,
                grid.notes,
            ),
            None => format!("{} {}", reference.notes, grid.notes),
        };
        instances.push(DualSourcingExperimentInstance {
            name: reference.name.to_string(),
            description: format!(
                "Gijsbrechts Figure 9 dual-sourcing benchmark row with l_r={}, c_e={:.0}, c_r=100, h=5, b=495, and demand U{{0,1,2,3,4}}.",
                reference.regular_lead_time, reference.expedited_order_cost
            ),
            reference_instance_name: reference.name,
            source: reference.source,
            url: reference.url,
            literature_verified: true,
            literature_verification_metric: "published_relative_optimality_gap_pct",
            regular_lead_time: reference.regular_lead_time,
            expedited_lead_time: reference.expedited_lead_time,
            regular_order_cost: reference.regular_order_cost,
            expedited_order_cost: reference.expedited_order_cost,
            holding_cost: reference.holding_cost,
            shortage_cost: reference.shortage_cost,
            regular_max_order_size: reference.regular_max_order_size,
            expedited_max_order_size: reference.expedited_max_order_size,
            demand_low: reference.demand_low,
            demand_high: reference.demand_high,
            horizon: grid.horizon,
            eval_horizon: grid.eval_horizon,
            eval_seeds: grid.eval_seeds,
            seed: grid.search_seed,
            inventory_lower: grid.inventory_lower,
            inventory_upper: grid.inventory_upper,
            solver_tolerance: grid.solver_tolerance,
            max_iterations: grid.max_iterations,
            warm_up_periods_ratio: grid.warm_up_periods_ratio,
            state_features: grid.state_features,
            benchmark_family: "Gijsbrechts2022Figure9DualSourcing",
            benchmark_policies: GIJSBRECHTS_2022_REFERENCE.benchmark_policies,
            notes,
        });
    }
    Ok(instances)
}
