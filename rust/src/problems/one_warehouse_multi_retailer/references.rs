use crate::problems::one_warehouse_multi_retailer::allocation::AllocationPolicy;
use crate::problems::one_warehouse_multi_retailer::demand::{DemandDistributionKind, DemandModel};
use crate::problems::one_warehouse_multi_retailer::env::CustomerBehaviorModel;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PublishedBenchmarkReference {
    pub source: &'static str,
    pub url: &'static str,
    pub benchmark_policies: &'static [&'static str],
    pub notes: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PublishedPolicyBenchmark {
    pub source: &'static str,
    pub url: &'static str,
    pub policy_name: &'static str,
    pub allocation_policy: Option<AllocationPolicy>,
    pub mean_cost: f64,
    pub standard_error: f64,
    pub relative_gap_percent: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OneWarehouseMultiRetailerReferenceInstance {
    pub name: &'static str,
    pub source: &'static str,
    pub url: &'static str,
    pub customer_behavior: CustomerBehaviorModel,
    pub warehouse_lead_time: usize,
    pub retailer_lead_times: &'static [usize],
    pub demand_models: &'static [DemandModel],
    pub holding_cost_warehouse: f64,
    pub holding_cost_retailers: &'static [f64],
    pub penalty_costs_retailers: &'static [f64],
    pub emergency_shipment_probability: f64,
    pub published_min_shortage_benchmark: Option<PublishedPolicyBenchmark>,
    pub published_proportional_benchmark: Option<PublishedPolicyBenchmark>,
    pub published_ppo_benchmark: Option<PublishedPolicyBenchmark>,
    pub notes: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorkedTransitionReference {
    pub source: &'static str,
    pub url: &'static str,
    pub customer_behavior: CustomerBehaviorModel,
    pub initial_warehouse_inventory: i32,
    pub initial_warehouse_pipeline: &'static [usize],
    pub initial_retailer_inventory: &'static [i32],
    pub initial_retailer_pipeline: &'static [&'static [usize]],
    pub warehouse_order: usize,
    pub retailer_orders: &'static [usize],
    pub retailer_shipments: &'static [usize],
    pub realized_demands: &'static [usize],
    pub expected_next_warehouse_inventory: i32,
    pub expected_next_warehouse_pipeline: &'static [usize],
    pub expected_next_retailer_inventory: &'static [i32],
    pub expected_next_retailer_pipeline: &'static [&'static [usize]],
    pub expected_holding_cost: f64,
    pub expected_shortage_cost: f64,
    pub expected_period_cost: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExactVerificationReference {
    pub source: &'static str,
    pub url: &'static str,
    pub customer_behavior: CustomerBehaviorModel,
    pub periods: usize,
    pub discount_factor: f64,
    pub warehouse_lead_time: usize,
    pub retailer_lead_times: &'static [usize],
    pub initial_warehouse_inventory: i32,
    pub initial_warehouse_pipeline: &'static [usize],
    pub initial_retailer_inventory: &'static [i32],
    pub initial_retailer_pipeline: &'static [&'static [usize]],
    pub holding_cost_warehouse: f64,
    pub holding_cost_retailers: &'static [f64],
    pub penalty_costs_retailers: &'static [f64],
    pub emergency_shipment_probability: f64,
    pub optimal_allocation_policy: AllocationPolicy,
    pub heuristic_warehouse_base_stock_level: usize,
    pub heuristic_retailer_base_stock_levels: &'static [usize],
    pub demand_supports: &'static [&'static [u32]],
    pub demand_probabilities: &'static [&'static [f64]],
    pub max_action_levels: &'static [usize],
    pub expected_optimal_discounted_cost: f64,
    pub expected_optimal_first_action: &'static [usize],
    pub expected_proportional_discounted_cost: f64,
    pub expected_proportional_first_action: &'static [usize],
    pub expected_proportional_shipments: &'static [usize],
    pub expected_min_shortage_discounted_cost: f64,
    pub expected_min_shortage_first_action: &'static [usize],
    pub expected_min_shortage_shipments: &'static [usize],
    pub notes: &'static str,
}

pub const KAYNOV_2024_REFERENCE: PublishedBenchmarkReference = PublishedBenchmarkReference {
    source: "Kaynov et al. (2024), International Journal of Production Economics 267, 109088",
    url: "https://doi.org/10.1016/j.ijpe.2023.109088",
    benchmark_policies: &[
        "echelon_base_stock_min_shortage",
        "echelon_base_stock_proportional",
        "ppo",
    ],
    notes: "The paper formulates OWMR as a periodic-review MDP with warehouse ordering, retailer ordering, and downstream allocation. Benchmarks are echelon base-stock policies paired with proportional allocation and min-shortage allocation.",
};

macro_rules! benchmark_row {
    ($policy:expr, $allocation:expr, $mean:expr, $se:expr, $gap:expr) => {
        Some(PublishedPolicyBenchmark {
            source: KAYNOV_2024_REFERENCE.source,
            url: KAYNOV_2024_REFERENCE.url,
            policy_name: $policy,
            allocation_policy: $allocation,
            mean_cost: $mean,
            standard_error: $se,
            relative_gap_percent: $gap,
        })
    };
}

pub const POISSON_3: DemandModel = DemandModel {
    kind: DemandDistributionKind::Poisson,
    param1: 3.0,
    param2: 0.0,
};
pub const POISSON_05: DemandModel = DemandModel {
    kind: DemandDistributionKind::Poisson,
    param1: 0.5,
    param2: 0.0,
};
pub const NORMAL_3_1: DemandModel = DemandModel {
    kind: DemandDistributionKind::RoundedNormal,
    param1: 3.0,
    param2: 1.0,
};
pub const NORMAL_1_5: DemandModel = DemandModel {
    kind: DemandDistributionKind::RoundedNormal,
    param1: 1.0,
    param2: 5.0,
};
pub const NORMAL_5_1: DemandModel = DemandModel {
    kind: DemandDistributionKind::RoundedNormal,
    param1: 5.0,
    param2: 1.0,
};
pub const UNIFORM_0_6: DemandModel = DemandModel {
    kind: DemandDistributionKind::DiscreteUniform,
    param1: 0.0,
    param2: 6.0,
};
pub const NORMAL_5_14: DemandModel = DemandModel {
    kind: DemandDistributionKind::RoundedNormal,
    param1: 5.0,
    param2: 14.0,
};
pub const NORMAL_0_20: DemandModel = DemandModel {
    kind: DemandDistributionKind::RoundedNormal,
    param1: 0.0,
    param2: 20.0,
};
pub const NORMAL_2_16: DemandModel = DemandModel {
    kind: DemandDistributionKind::RoundedNormal,
    param1: 2.0,
    param2: 16.0,
};
pub const NORMAL_4_12: DemandModel = DemandModel {
    kind: DemandDistributionKind::RoundedNormal,
    param1: 4.0,
    param2: 12.0,
};
pub const NORMAL_6_8: DemandModel = DemandModel {
    kind: DemandDistributionKind::RoundedNormal,
    param1: 6.0,
    param2: 8.0,
};
pub const NORMAL_8_4: DemandModel = DemandModel {
    kind: DemandDistributionKind::RoundedNormal,
    param1: 8.0,
    param2: 4.0,
};
pub const NORMAL_10_0: DemandModel = DemandModel {
    kind: DemandDistributionKind::RoundedNormal,
    param1: 10.0,
    param2: 0.0,
};
pub const POISSON_9: DemandModel = DemandModel {
    kind: DemandDistributionKind::Poisson,
    param1: 9.0,
    param2: 0.0,
};
pub const POISSON_12: DemandModel = DemandModel {
    kind: DemandDistributionKind::Poisson,
    param1: 12.0,
    param2: 0.0,
};

macro_rules! owmr_standard_instance {
    (
        $name:expr,
        $customer_behavior:expr,
        $warehouse_lead_time:expr,
        $retailer_lead_times:expr,
        $demand_models:expr,
        $min_mean:expr,
        $min_se:expr,
        $min_gap:expr,
        $prop_mean:expr,
        $prop_se:expr,
        $ppo_mean:expr,
        $ppo_se:expr,
        $ppo_gap:expr
    ) => {
        OneWarehouseMultiRetailerReferenceInstance {
            name: $name,
            source: KAYNOV_2024_REFERENCE.source,
            url: KAYNOV_2024_REFERENCE.url,
            customer_behavior: $customer_behavior,
            warehouse_lead_time: $warehouse_lead_time,
            retailer_lead_times: $retailer_lead_times,
            demand_models: $demand_models,
            holding_cost_warehouse: 0.5,
            holding_cost_retailers: &[1.0, 1.0, 1.0],
            penalty_costs_retailers: &[9.0, 9.0, 9.0],
            emergency_shipment_probability: 0.0,
            published_min_shortage_benchmark: benchmark_row!(
                "echelon_base_stock",
                Some(AllocationPolicy::MinShortage),
                $min_mean,
                $min_se,
                $min_gap
            ),
            published_proportional_benchmark: benchmark_row!(
                "echelon_base_stock",
                Some(AllocationPolicy::Proportional),
                $prop_mean,
                $prop_se,
                0.0
            ),
            published_ppo_benchmark: benchmark_row!("ppo", None, $ppo_mean, $ppo_se, $ppo_gap),
            notes: "Table 1 / Table A.3 instance from Kaynov et al. (2024).",
        }
    };
}

pub const TABLE_A3_INSTANCES: [OneWarehouseMultiRetailerReferenceInstance; 14] = [
    owmr_standard_instance!(
        "kaynov2024_instance_1",
        CustomerBehaviorModel::Backorder,
        2,
        &[1, 1, 1],
        &[POISSON_3, POISSON_3, POISSON_3],
        -1609.47,
        1.67,
        -2.78,
        -1655.51,
        1.66,
        -1637.2,
        5.23,
        -1.11
    ),
    owmr_standard_instance!(
        "kaynov2024_instance_2",
        CustomerBehaviorModel::Backorder,
        2,
        &[1, 1, 1],
        &[UNIFORM_0_6, NORMAL_3_1, POISSON_3],
        -1426.84,
        1.53,
        3.1,
        -1383.88,
        1.36,
        -1417.46,
        4.26,
        2.43
    ),
    owmr_standard_instance!(
        "kaynov2024_instance_3",
        CustomerBehaviorModel::Backorder,
        2,
        &[1, 1, 1],
        &[NORMAL_1_5, NORMAL_5_1, POISSON_05],
        -1800.5,
        2.08,
        1.38,
        -1776.04,
        2.1,
        -1731.67,
        7.16,
        -2.5
    ),
    owmr_standard_instance!(
        "kaynov2024_instance_4",
        CustomerBehaviorModel::Backorder,
        2,
        &[1, 2, 3],
        &[POISSON_3, POISSON_3, POISSON_3],
        -1857.3,
        1.76,
        -0.09,
        -1858.93,
        1.89,
        -1908.95,
        6.73,
        2.69
    ),
    owmr_standard_instance!(
        "kaynov2024_instance_5",
        CustomerBehaviorModel::Backorder,
        5,
        &[3, 3, 3],
        &[NORMAL_1_5, NORMAL_5_1, POISSON_05],
        -2306.89,
        3.47,
        2.67,
        -2246.84,
        3.18,
        -2331.07,
        13.45,
        3.75
    ),
    owmr_standard_instance!(
        "kaynov2024_instance_6",
        CustomerBehaviorModel::LostSales,
        1,
        &[1, 1, 1],
        &[POISSON_3, POISSON_3, POISSON_3],
        -1366.51,
        0.91,
        -0.54,
        -1373.91,
        0.92,
        -1347.34,
        2.89,
        -1.93
    ),
    owmr_standard_instance!(
        "kaynov2024_instance_7",
        CustomerBehaviorModel::LostSales,
        2,
        &[1, 1, 1],
        &[POISSON_3, POISSON_3, POISSON_3],
        -1408.08,
        0.95,
        0.13,
        -1406.27,
        0.99,
        -1405.08,
        3.11,
        -0.09
    ),
    owmr_standard_instance!(
        "kaynov2024_instance_8",
        CustomerBehaviorModel::LostSales,
        5,
        &[1, 1, 1],
        &[POISSON_3, POISSON_3, POISSON_3],
        -1516.67,
        1.14,
        0.57,
        -1508.12,
        1.11,
        -1495.49,
        3.76,
        -0.84
    ),
    owmr_standard_instance!(
        "kaynov2024_instance_9",
        CustomerBehaviorModel::LostSales,
        2,
        &[1, 2, 3],
        &[POISSON_3, POISSON_3, POISSON_3],
        -1561.56,
        1.17,
        1.67,
        -1535.96,
        1.14,
        -1511.68,
        3.65,
        -1.58
    ),
    owmr_standard_instance!(
        "kaynov2024_instance_10",
        CustomerBehaviorModel::LostSales,
        5,
        &[3, 3, 3],
        &[NORMAL_1_5, NORMAL_5_1, POISSON_05],
        -1741.6,
        1.85,
        0.29,
        -1736.55,
        1.69,
        -1674.54,
        5.45,
        -3.57
    ),
    OneWarehouseMultiRetailerReferenceInstance {
        name: "kaynov2024_instance_11",
        source: KAYNOV_2024_REFERENCE.source,
        url: KAYNOV_2024_REFERENCE.url,
        customer_behavior: CustomerBehaviorModel::PartialBackorder,
        warehouse_lead_time: 2,
        retailer_lead_times: &[1, 1, 1],
        demand_models: &[POISSON_3, POISSON_3, POISSON_3],
        holding_cost_warehouse: 0.5,
        holding_cost_retailers: &[1.0, 1.0, 1.0],
        penalty_costs_retailers: &[9.0, 9.0, 9.0],
        emergency_shipment_probability: 0.8,
        published_min_shortage_benchmark: benchmark_row!(
            "echelon_base_stock",
            Some(AllocationPolicy::MinShortage),
            -1109.96,
            1.02,
            -0.16
        ),
        published_proportional_benchmark: benchmark_row!(
            "echelon_base_stock",
            Some(AllocationPolicy::Proportional),
            -1111.76,
            1.02,
            0.0
        ),
        published_ppo_benchmark: benchmark_row!("ppo", None, -971.86, 3.13, -12.58),
        notes: "Partial-backorder instance from Table 2 / Table A.3.",
    },
    OneWarehouseMultiRetailerReferenceInstance {
        name: "kaynov2024_instance_12",
        source: KAYNOV_2024_REFERENCE.source,
        url: KAYNOV_2024_REFERENCE.url,
        customer_behavior: CustomerBehaviorModel::PartialBackorder,
        warehouse_lead_time: 2,
        retailer_lead_times: &[1, 1, 1],
        demand_models: &[NORMAL_1_5, NORMAL_5_1, POISSON_05],
        holding_cost_warehouse: 0.5,
        holding_cost_retailers: &[1.0, 1.0, 1.0],
        penalty_costs_retailers: &[9.0, 9.0, 9.0],
        emergency_shipment_probability: 0.8,
        published_min_shortage_benchmark: benchmark_row!(
            "echelon_base_stock",
            Some(AllocationPolicy::MinShortage),
            -1406.43,
            1.45,
            0.29
        ),
        published_proportional_benchmark: benchmark_row!(
            "echelon_base_stock",
            Some(AllocationPolicy::Proportional),
            -1402.38,
            1.44,
            0.0
        ),
        published_ppo_benchmark: benchmark_row!("ppo", None, -1118.92, 4.51, -20.21),
        notes: "Partial-backorder instance from Table 2 / Table A.3.",
    },
    OneWarehouseMultiRetailerReferenceInstance {
        name: "kaynov2024_instance_13",
        source: KAYNOV_2024_REFERENCE.source,
        url: KAYNOV_2024_REFERENCE.url,
        customer_behavior: CustomerBehaviorModel::PartialBackorder,
        warehouse_lead_time: 2,
        retailer_lead_times: &[2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        demand_models: &[
            NORMAL_5_14,
            NORMAL_5_14,
            NORMAL_5_14,
            NORMAL_5_14,
            NORMAL_5_14,
            NORMAL_5_14,
            NORMAL_5_14,
            NORMAL_5_14,
            NORMAL_5_14,
            NORMAL_5_14,
        ],
        holding_cost_warehouse: 3.0,
        holding_cost_retailers: &[3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
        penalty_costs_retailers: &[60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0],
        emergency_shipment_probability: 0.8,
        published_min_shortage_benchmark: benchmark_row!(
            "echelon_base_stock",
            Some(AllocationPolicy::MinShortage),
            -99882.51,
            86.58,
            -1.81
        ),
        published_proportional_benchmark: benchmark_row!(
            "echelon_base_stock",
            Some(AllocationPolicy::Proportional),
            -101727.47,
            87.32,
            0.0
        ),
        published_ppo_benchmark: benchmark_row!("ppo", None, -79727.39, 215.39, -21.63),
        notes: "Large 10-retailer partial-backorder instance aligned with prior OWMR work.",
    },
    OneWarehouseMultiRetailerReferenceInstance {
        name: "kaynov2024_instance_14",
        source: KAYNOV_2024_REFERENCE.source,
        url: KAYNOV_2024_REFERENCE.url,
        customer_behavior: CustomerBehaviorModel::PartialBackorder,
        warehouse_lead_time: 2,
        retailer_lead_times: &[2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        demand_models: &[
            NORMAL_0_20,
            NORMAL_2_16,
            NORMAL_4_12,
            NORMAL_6_8,
            NORMAL_8_4,
            NORMAL_10_0,
            POISSON_05,
            POISSON_3,
            POISSON_9,
            POISSON_12,
        ],
        holding_cost_warehouse: 3.0,
        holding_cost_retailers: &[3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
        penalty_costs_retailers: &[60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0],
        emergency_shipment_probability: 0.8,
        published_min_shortage_benchmark: benchmark_row!(
            "echelon_base_stock",
            Some(AllocationPolicy::MinShortage),
            -52787.41,
            49.29,
            -1.07
        ),
        published_proportional_benchmark: benchmark_row!(
            "echelon_base_stock",
            Some(AllocationPolicy::Proportional),
            -53358.86,
            45.4,
            0.0
        ),
        published_ppo_benchmark: benchmark_row!("ppo", None, -42835.02, 124.09, -19.72),
        notes: "Asymmetric 10-retailer partial-backorder instance from Table 2 / Table A.3.",
    },
];

pub const PRIMARY_REFERENCE_INSTANCE: OneWarehouseMultiRetailerReferenceInstance =
    TABLE_A3_INSTANCES[6];

pub const WORKED_TRANSITION_REFERENCE: WorkedTransitionReference = WorkedTransitionReference {
    source: KAYNOV_2024_REFERENCE.source,
    url: KAYNOV_2024_REFERENCE.url,
    customer_behavior: CustomerBehaviorModel::LostSales,
    initial_warehouse_inventory: 2,
    initial_warehouse_pipeline: &[3, 1],
    initial_retailer_inventory: &[1, 2],
    initial_retailer_pipeline: &[&[1], &[0]],
    warehouse_order: 4,
    retailer_orders: &[4, 3],
    retailer_shipments: &[2, 2],
    realized_demands: &[2, 1],
    expected_next_warehouse_inventory: 1,
    expected_next_warehouse_pipeline: &[1, 4],
    expected_next_retailer_inventory: &[0, 1],
    expected_next_retailer_pipeline: &[&[2], &[2]],
    expected_holding_cost: 1.5,
    expected_shortage_cost: 0.0,
    expected_period_cost: 1.5,
};

pub const VERIFICATION_RETAILER_SUPPORT: &[u32] = &[0, 1];
pub const VERIFICATION_RETAILER_PROBABILITIES: &[f64] = &[0.5, 0.5];

pub const VERIFICATION_PROBLEM_INSTANCE: ExactVerificationReference = ExactVerificationReference {
    source: KAYNOV_2024_REFERENCE.source,
    url: KAYNOV_2024_REFERENCE.url,
    customer_behavior: CustomerBehaviorModel::LostSales,
    periods: 2,
    discount_factor: 0.99,
    warehouse_lead_time: 1,
    retailer_lead_times: &[1, 1],
    initial_warehouse_inventory: 2,
    initial_warehouse_pipeline: &[1],
    initial_retailer_inventory: &[1, 0],
    initial_retailer_pipeline: &[&[1], &[0]],
    holding_cost_warehouse: 0.5,
    holding_cost_retailers: &[1.0, 1.0],
    penalty_costs_retailers: &[9.0, 9.0],
    emergency_shipment_probability: 0.0,
    optimal_allocation_policy: AllocationPolicy::Proportional,
    heuristic_warehouse_base_stock_level: 6,
    heuristic_retailer_base_stock_levels: &[3, 3],
    demand_supports: &[VERIFICATION_RETAILER_SUPPORT, VERIFICATION_RETAILER_SUPPORT],
    demand_probabilities: &[
        VERIFICATION_RETAILER_PROBABILITIES,
        VERIFICATION_RETAILER_PROBABILITIES,
    ],
    max_action_levels: &[3, 3, 3],
    expected_optimal_discounted_cost: 8.485000000000001,
    expected_optimal_first_action: &[0, 0, 1],
    expected_proportional_discounted_cost: 9.2225,
    expected_proportional_first_action: &[1, 1, 3],
    expected_proportional_shipments: &[0, 2],
    expected_min_shortage_discounted_cost: 9.465,
    expected_min_shortage_first_action: &[1, 1, 3],
    expected_min_shortage_shipments: &[1, 2],
    notes: "Repo-native exact verifier shaped after the lost-sales OWMR setting with positive lead times. Two retailers, binary demand support, and a two-period horizon keep the finite-horizon dynamic program cheap enough for routine testing.",
};
