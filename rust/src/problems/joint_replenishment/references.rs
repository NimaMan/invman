use crate::problems::joint_replenishment::demand::DemandRange;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PublishedBenchmarkReference {
    pub source: &'static str,
    pub url: &'static str,
    pub benchmark_policies: &'static [&'static str],
    pub notes: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointReplenishmentReferenceInstance {
    pub name: &'static str,
    pub source: &'static str,
    pub url: &'static str,
    pub num_items: usize,
    pub truck_capacity: usize,
    pub major_order_cost: f64,
    pub minor_order_costs: &'static [f64],
    pub holding_costs: &'static [f64],
    pub shortage_costs: &'static [f64],
    pub demand_ranges: &'static [DemandRange],
    pub notes: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorkedTransitionReference {
    pub source: &'static str,
    pub url: &'static str,
    pub initial_inventory_levels: &'static [i32],
    pub action: &'static [usize],
    pub realized_demands: &'static [usize],
    pub truck_capacity: usize,
    pub major_order_cost: f64,
    pub minor_order_costs: &'static [f64],
    pub holding_costs: &'static [f64],
    pub shortage_costs: &'static [f64],
    pub expected_next_inventory_levels: &'static [i32],
    pub expected_trucks_used: usize,
    pub expected_order_cost: f64,
    pub expected_holding_cost: f64,
    pub expected_shortage_cost: f64,
    pub expected_period_cost: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExactVerificationReference {
    pub source: &'static str,
    pub url: &'static str,
    pub periods: usize,
    pub discount_factor: f64,
    pub truck_capacity: usize,
    pub max_order_quantities: &'static [usize],
    pub initial_inventory_levels: &'static [i32],
    pub major_order_cost: f64,
    pub minor_order_costs: &'static [f64],
    pub holding_costs: &'static [f64],
    pub shortage_costs: &'static [f64],
    pub demand_ranges: &'static [DemandRange],
    pub moq_item_targets: &'static [usize],
    pub moq_review_period: usize,
    pub moq_rounding_threshold: f64,
    pub dynout_item_targets: &'static [usize],
    pub expected_optimal_discounted_cost: f64,
    pub expected_optimal_first_action: &'static [usize],
    pub expected_moq_discounted_cost: f64,
    pub expected_moq_first_action: &'static [usize],
    pub expected_dynout_discounted_cost: f64,
    pub expected_dynout_first_action: &'static [usize],
    pub notes: &'static str,
}

pub const VANVUCHELEN_2020_REFERENCE: PublishedBenchmarkReference = PublishedBenchmarkReference {
    source: "Vanvuchelen et al. (2020), Computers in Industry 122, 103300",
    url: "https://www.sciencedirect.com/science/article/pii/S0166361519308218",
    benchmark_policies: &["ppo", "(Q,S|T)_moq", "dyn-out"],
    notes: "The paper studies the stochastic joint replenishment problem with full-truckload coupling and compares PPO against the MOQ and DYN-OUT heuristics. Search-result snippets expose the small-scale setting family with n = 2, V = 6, K = 75, k_i = 10, l_i = 0, d_1 ~ U[0,5], and d_2 ~ U[0,3].",
};

pub const SMALL_SCALE_DEMAND_RANGES: &[DemandRange] = &[
    DemandRange { low: 0, high: 5 },
    DemandRange { low: 0, high: 3 },
];

pub const SMALL_SCALE_MINOR_COSTS: &[f64] = &[10.0, 10.0];

pub const SMALL_SCALE_SETTINGS: [JointReplenishmentReferenceInstance; 4] = [
    JointReplenishmentReferenceInstance {
        name: "vanvuchelen2020_small_scale_setting_1",
        source: VANVUCHELEN_2020_REFERENCE.source,
        url: VANVUCHELEN_2020_REFERENCE.url,
        num_items: 2,
        truck_capacity: 6,
        major_order_cost: 75.0,
        minor_order_costs: SMALL_SCALE_MINOR_COSTS,
        holding_costs: &[1.0, 1.0],
        shortage_costs: &[19.0, 19.0],
        demand_ranges: SMALL_SCALE_DEMAND_RANGES,
        notes: "Small-scale setting 1 from the paper preview snippets.",
    },
    JointReplenishmentReferenceInstance {
        name: "vanvuchelen2020_small_scale_setting_2",
        source: VANVUCHELEN_2020_REFERENCE.source,
        url: VANVUCHELEN_2020_REFERENCE.url,
        num_items: 2,
        truck_capacity: 6,
        major_order_cost: 75.0,
        minor_order_costs: SMALL_SCALE_MINOR_COSTS,
        holding_costs: &[1.0, 5.0],
        shortage_costs: &[19.0, 95.0],
        demand_ranges: SMALL_SCALE_DEMAND_RANGES,
        notes: "Small-scale setting 2 from the paper preview snippets.",
    },
    JointReplenishmentReferenceInstance {
        name: "vanvuchelen2020_small_scale_setting_3",
        source: VANVUCHELEN_2020_REFERENCE.source,
        url: VANVUCHELEN_2020_REFERENCE.url,
        num_items: 2,
        truck_capacity: 6,
        major_order_cost: 75.0,
        minor_order_costs: SMALL_SCALE_MINOR_COSTS,
        holding_costs: &[5.0, 1.0],
        shortage_costs: &[95.0, 19.0],
        demand_ranges: SMALL_SCALE_DEMAND_RANGES,
        notes: "Small-scale setting 3 from the paper preview snippets.",
    },
    JointReplenishmentReferenceInstance {
        name: "vanvuchelen2020_small_scale_setting_4",
        source: VANVUCHELEN_2020_REFERENCE.source,
        url: VANVUCHELEN_2020_REFERENCE.url,
        num_items: 2,
        truck_capacity: 6,
        major_order_cost: 75.0,
        minor_order_costs: SMALL_SCALE_MINOR_COSTS,
        holding_costs: &[5.0, 5.0],
        shortage_costs: &[95.0, 95.0],
        demand_ranges: SMALL_SCALE_DEMAND_RANGES,
        notes: "Small-scale setting 4 from the paper preview snippets.",
    },
];

pub const PRIMARY_REFERENCE_INSTANCE: JointReplenishmentReferenceInstance =
    SMALL_SCALE_SETTINGS[0];

pub const WORKED_TRANSITION_REFERENCE: WorkedTransitionReference = WorkedTransitionReference {
    source: VANVUCHELEN_2020_REFERENCE.source,
    url: VANVUCHELEN_2020_REFERENCE.url,
    initial_inventory_levels: &[1, -2],
    action: &[4, 1],
    realized_demands: &[3, 0],
    truck_capacity: 6,
    major_order_cost: 75.0,
    minor_order_costs: SMALL_SCALE_MINOR_COSTS,
    holding_costs: &[1.0, 1.0],
    shortage_costs: &[19.0, 19.0],
    expected_next_inventory_levels: &[2, -1],
    expected_trucks_used: 1,
    expected_order_cost: 95.0,
    expected_holding_cost: 2.0,
    expected_shortage_cost: 19.0,
    expected_period_cost: 116.0,
};

pub const VERIFICATION_PROBLEM_INSTANCE: ExactVerificationReference = ExactVerificationReference {
    source: VANVUCHELEN_2020_REFERENCE.source,
    url: VANVUCHELEN_2020_REFERENCE.url,
    periods: 4,
    discount_factor: 0.99,
    truck_capacity: 6,
    max_order_quantities: &[12, 12],
    initial_inventory_levels: &[2, 0],
    major_order_cost: 75.0,
    minor_order_costs: SMALL_SCALE_MINOR_COSTS,
    holding_costs: &[1.0, 1.0],
    shortage_costs: &[19.0, 19.0],
    demand_ranges: SMALL_SCALE_DEMAND_RANGES,
    moq_item_targets: &[8, 5],
    moq_review_period: 1,
    moq_rounding_threshold: 2.0,
    dynout_item_targets: &[8, 5],
    expected_optimal_discounted_cost: 266.3863465996094,
    expected_optimal_first_action: &[6, 6],
    expected_moq_discounted_cost: 386.10114499218747,
    expected_moq_first_action: &[7, 5],
    expected_dynout_discounted_cost: 383.9596796015626,
    expected_dynout_first_action: &[6, 6],
    notes: "Finite-horizon repo-native exact verifier built on the paper's small-scale setting 1 model family. The expected costs and first actions are frozen from the exact DP and heuristic evaluations once the implementation is solved.",
};
