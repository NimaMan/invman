#![allow(dead_code)]

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PublishedBenchmarkReference {
    pub source: &'static str,
    pub url: &'static str,
    pub benchmark_policies: &'static [&'static str],
    pub reported_numbers_available: bool,
    pub numbers_anchor_repo_assertions: bool,
    pub notes: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LiteratureBenchmarkPolicyResult {
    pub policy_name: &'static str,
    pub base_stock_levels: &'static [usize],
    pub reported_cost_value: f64,
    pub reported_cost_half_width: f64,
    pub reported_readiness_percent: f64,
    pub reported_readiness_half_width: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LiteratureBenchmarkScenario {
    pub name: &'static str,
    pub source: &'static str,
    pub url: &'static str,
    pub literature_verified: bool,
    pub verification_source: &'static str,
    pub model_family: &'static str,
    pub am_location: &'static str,
    pub echelons: usize,
    pub simulation_horizon_days: usize,
    pub table_replications: usize,
    pub demand_rate_per_hour: f64,
    pub review_intervals_hours: &'static [f64],
    pub transport_lead_times_hours: &'static [f64],
    pub am_lead_time_hours: Option<f64>,
    pub regular_sourcing_cost: f64,
    pub am_sourcing_cost: Option<f64>,
    pub holding_costs_as_reported: &'static [f64],
    pub downtime_cost_as_reported: f64,
    pub published_policy_results: &'static [LiteratureBenchmarkPolicyResult],
    pub notes: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KranenburgLateralTransshipmentReferenceInstance {
    pub name: &'static str,
    pub source: &'static str,
    pub url: &'static str,
    pub literature_verified: bool,
    pub verification_source: &'static str,
    pub table: &'static str,
    pub varied_parameter: &'static str,
    pub varied_value_label: &'static str,
    pub demand_rate_per_local_warehouse: f64,
    pub num_local_warehouses: usize,
    pub holding_cost: f64,
    pub emergency_cost: f64,
    pub lateral_transshipment_cost: f64,
    pub joint_warehouse_cost: f64,
    pub waiting_time_target: f64,
    pub emergency_time: f64,
    pub lateral_transshipment_time: f64,
    pub joint_warehouse_time: f64,
    pub regular_replenishment_time: f64,
    pub published_situation1_optimal_r: f64,
    pub published_situation1_cost: f64,
    pub published_situation3_optimal_r: f64,
    pub published_situation3_cost: f64,
    pub published_cost_ratio_situation1_over_situation3: f64,
    pub notes: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SparePartsReferenceInstance {
    pub name: &'static str,
    pub source: &'static str,
    pub url: &'static str,
    pub periods: usize,
    pub installed_base: usize,
    pub procurement_lead_time: usize,
    pub repair_lead_time: usize,
    pub initial_on_hand_inventory: usize,
    pub initial_backlog: usize,
    pub initial_procurement_pipeline: &'static [usize],
    pub initial_repair_pipeline: &'static [usize],
    pub failure_probability: f64,
    pub holding_cost: f64,
    pub downtime_cost: f64,
    pub procurement_cost: f64,
    pub benchmark_base_stock_level: usize,
    pub benchmark_lead_time_mean_cover_safety_buffer: f64,
    pub notes: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorkedTransitionReference {
    pub source: &'static str,
    pub url: &'static str,
    pub installed_base: usize,
    pub initial_on_hand_inventory: usize,
    pub initial_backlog: usize,
    pub initial_procurement_pipeline: &'static [usize],
    pub initial_repair_pipeline: &'static [usize],
    pub action: usize,
    pub realized_failures: usize,
    pub holding_cost: f64,
    pub downtime_cost: f64,
    pub procurement_cost: f64,
    pub expected_procurement_arrival: usize,
    pub expected_repair_return: usize,
    pub expected_post_failure_on_hand_inventory: usize,
    pub expected_post_failure_backlog: usize,
    pub expected_next_on_hand_inventory: usize,
    pub expected_next_backlog: usize,
    pub expected_next_procurement_pipeline: &'static [usize],
    pub expected_next_repair_pipeline: &'static [usize],
    pub expected_period_cost: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExactVerificationReference {
    pub source: &'static str,
    pub url: &'static str,
    pub periods: usize,
    pub discount_factor: f64,
    pub installed_base: usize,
    pub procurement_lead_time: usize,
    pub repair_lead_time: usize,
    pub initial_on_hand_inventory: usize,
    pub initial_backlog: usize,
    pub initial_procurement_pipeline: &'static [usize],
    pub initial_repair_pipeline: &'static [usize],
    pub failure_probability: f64,
    pub holding_cost: f64,
    pub downtime_cost: f64,
    pub procurement_cost: f64,
    pub max_order_quantity: usize,
    pub base_stock_level: usize,
    pub lead_time_mean_cover_safety_buffer: f64,
    pub notes: &'static str,
}

pub const SPARE_PARTS_REVIEW_REFERENCE: PublishedBenchmarkReference = PublishedBenchmarkReference {
    source: "Spare Parts Inventory Management: A Literature Review (2021), Sustainability 13(5), 2460",
    url: "https://www.mdpi.com/2071-1050/13/5/2460",
    benchmark_policies: &["base_stock", "one_for_one_replenishment", "multi-echelon transshipment"],
    reported_numbers_available: false,
    numbers_anchor_repo_assertions: false,
    notes: "The review highlights that spare-parts systems are often single-echelon, repairable, and continuous-review. It explicitly notes that one-for-one replenishment or base-stock control is widely used for repairable spare parts.",
};

pub const ZHOU_2024_REFERENCE: PublishedBenchmarkReference = PublishedBenchmarkReference {
    source: "Zhou et al. (2024), Applied Mathematical Modelling 125(B), 827-844",
    url: "https://doi.org/10.1016/j.apm.2023.10.039",
    benchmark_policies: &["marl", "multi-echelon spare-parts baselines"],
    reported_numbers_available: false,
    numbers_anchor_repo_assertions: false,
    notes: "This paper motivates spare_parts_inventory as a distinct RL family through a multi-echelon spare-parts setting with recent multi-agent DRL results.",
};

pub const VAN_DER_HAAR_2025_REFERENCE: PublishedBenchmarkReference = PublishedBenchmarkReference {
    source: "van der Haar et al. (2025), Industrializing Deep Reinforcement Learning for Operational Spare Parts Inventory Management",
    url: "https://ssrn.com/abstract=4999374",
    benchmark_policies: &["drl", "distance-based transshipment and expediting heuristics"],
    reported_numbers_available: false,
    numbers_anchor_repo_assertions: false,
    notes: "The paper shows that large-scale operational spare-parts management relies on proactive transshipments and expediting, which motivates keeping spare_parts_inventory separate from general general_network.",
};

pub const KRANENBURG_2006_REFERENCE: PublishedBenchmarkReference = PublishedBenchmarkReference {
    source: "Kranenburg (2006), Chapter 5, Tables 5.1-5.3",
    url: "https://pure.tue.nl/ws/files/2461454/200612097.pdf",
    benchmark_policies: &[
        "situation1_separate_stock_points",
        "situation2_joint_warehouse",
        "situation3_lateral_transshipment",
    ],
    reported_numbers_available: true,
    numbers_anchor_repo_assertions: true,
    notes: "Chapter 5 gives an exact analytical comparison between separate stock points, a joint warehouse, and lateral transshipment for expensive spare parts with low demand rates. Tables 5.1 and 5.2 publish both the benchmark parameters and the optimal randomized-stock and cost numbers, which makes this an executable literature-verification family.",
};

pub const VAN_OERS_2024_REFERENCE: PublishedBenchmarkReference = PublishedBenchmarkReference {
    source: "van Oers et al. (2024), IFAC PapersOnLine 58(19), 1006-1011",
    url: "https://doi.org/10.1016/j.ifacol.2024.09.144",
    benchmark_policies: &["enumeration", "newsvendor", "echelon_separation"],
    reported_numbers_available: true,
    numbers_anchor_repo_assertions: true,
    notes: "Table 1 is open-access and reports a fully specified N=2 serial spare-parts benchmark with three public scenarios: no AM, upstream AM, and downstream AM. The repo stores the table values exactly as reported. Inference: because the no-AM enumeration row is exactly 100.0 and the paper text describes downstream AM as a 28% cost reduction, the printed cost figures are likely normalized table values rather than literal dollar totals; the repo therefore preserves the table entries without forcing an absolute-dollar interpretation.",
};

pub const VAN_OERS_2024_REVIEW_INTERVALS_HOURS: &[f64] = &[48.0, 96.0];
pub const VAN_OERS_2024_TRANSPORT_LEAD_TIMES_HOURS: &[f64] = &[24.0, 24.0];
pub const VAN_OERS_2024_HOLDING_COSTS_AS_REPORTED: &[f64] = &[7200.0, 2400.0];

pub const VAN_OERS_2024_NO_AM_ENUMERATION_LEVELS: &[usize] = &[8, 4];
pub const VAN_OERS_2024_NO_AM_NEWSVENDOR_LEVELS: &[usize] = &[5, 7];
pub const VAN_OERS_2024_NO_AM_ECHELON_SEPARATION_LEVELS: &[usize] = &[6, 6];
pub const VAN_OERS_2024_UPSTREAM_AM_ENUMERATION_LEVELS: &[usize] = &[11, 0];
pub const VAN_OERS_2024_UPSTREAM_AM_NEWSVENDOR_LEVELS: &[usize] = &[5, 0];
pub const VAN_OERS_2024_UPSTREAM_AM_ECHELON_SEPARATION_LEVELS: &[usize] = &[6, 13];
pub const VAN_OERS_2024_DOWNSTREAM_AM_ENUMERATION_LEVELS: &[usize] = &[5, 0];
pub const VAN_OERS_2024_DOWNSTREAM_AM_NEWSVENDOR_LEVELS: &[usize] = &[0, 8];
pub const VAN_OERS_2024_DOWNSTREAM_AM_ECHELON_SEPARATION_LEVELS: &[usize] = &[4, 1];

pub const VAN_OERS_2024_NO_AM_POLICY_RESULTS: &[LiteratureBenchmarkPolicyResult] = &[
    LiteratureBenchmarkPolicyResult {
        policy_name: "enumeration",
        base_stock_levels: VAN_OERS_2024_NO_AM_ENUMERATION_LEVELS,
        reported_cost_value: 100.0,
        reported_cost_half_width: 1.14,
        reported_readiness_percent: 99.57,
        reported_readiness_half_width: 0.027,
    },
    LiteratureBenchmarkPolicyResult {
        policy_name: "newsvendor",
        base_stock_levels: VAN_OERS_2024_NO_AM_NEWSVENDOR_LEVELS,
        reported_cost_value: 117.0,
        reported_cost_half_width: 1.65,
        reported_readiness_percent: 99.08,
        reported_readiness_half_width: 0.037,
    },
    LiteratureBenchmarkPolicyResult {
        policy_name: "echelon_separation",
        base_stock_levels: VAN_OERS_2024_NO_AM_ECHELON_SEPARATION_LEVELS,
        reported_cost_value: 105.9,
        reported_cost_half_width: 1.44,
        reported_readiness_percent: 99.36,
        reported_readiness_half_width: 0.033,
    },
];

pub const VAN_OERS_2024_UPSTREAM_AM_POLICY_RESULTS: &[LiteratureBenchmarkPolicyResult] = &[
    LiteratureBenchmarkPolicyResult {
        policy_name: "enumeration",
        base_stock_levels: VAN_OERS_2024_UPSTREAM_AM_ENUMERATION_LEVELS,
        reported_cost_value: 108.1,
        reported_cost_half_width: 1.13,
        reported_readiness_percent: 99.61,
        reported_readiness_half_width: 0.020,
    },
    LiteratureBenchmarkPolicyResult {
        policy_name: "newsvendor",
        base_stock_levels: VAN_OERS_2024_UPSTREAM_AM_NEWSVENDOR_LEVELS,
        reported_cost_value: 171.3,
        reported_cost_half_width: 2.21,
        reported_readiness_percent: 97.84,
        reported_readiness_half_width: 0.040,
    },
    LiteratureBenchmarkPolicyResult {
        policy_name: "echelon_separation",
        base_stock_levels: VAN_OERS_2024_UPSTREAM_AM_ECHELON_SEPARATION_LEVELS,
        reported_cost_value: 142.3,
        reported_cost_half_width: 2.11,
        reported_readiness_percent: 98.81,
        reported_readiness_half_width: 0.067,
    },
];

pub const VAN_OERS_2024_DOWNSTREAM_AM_POLICY_RESULTS: &[LiteratureBenchmarkPolicyResult] = &[
    LiteratureBenchmarkPolicyResult {
        policy_name: "enumeration",
        base_stock_levels: VAN_OERS_2024_DOWNSTREAM_AM_ENUMERATION_LEVELS,
        reported_cost_value: 71.98,
        reported_cost_half_width: 0.53,
        reported_readiness_percent: 99.77,
        reported_readiness_half_width: 0.003,
    },
    LiteratureBenchmarkPolicyResult {
        policy_name: "newsvendor",
        base_stock_levels: VAN_OERS_2024_DOWNSTREAM_AM_NEWSVENDOR_LEVELS,
        reported_cost_value: 138.00,
        reported_cost_half_width: 0.67,
        reported_readiness_percent: 99.56,
        reported_readiness_half_width: 0.003,
    },
    LiteratureBenchmarkPolicyResult {
        policy_name: "echelon_separation",
        base_stock_levels: VAN_OERS_2024_DOWNSTREAM_AM_ECHELON_SEPARATION_LEVELS,
        reported_cost_value: 72.01,
        reported_cost_half_width: 0.55,
        reported_readiness_percent: 99.77,
        reported_readiness_half_width: 0.003,
    },
];

pub const VAN_OERS_2024_NO_AM_SCENARIO: LiteratureBenchmarkScenario = LiteratureBenchmarkScenario {
    name: "van_oers2024_table1_no_am",
    source: VAN_OERS_2024_REFERENCE.source,
    url: VAN_OERS_2024_REFERENCE.url,
    literature_verified: true,
    verification_source: "published_benchmark_table_from_literature",
    model_family: "two_echelon_periodic_review_serial_spare_parts",
    am_location: "none",
    echelons: 2,
    simulation_horizon_days: 1000,
    table_replications: 100,
    demand_rate_per_hour: 0.04,
    review_intervals_hours: VAN_OERS_2024_REVIEW_INTERVALS_HOURS,
    transport_lead_times_hours: VAN_OERS_2024_TRANSPORT_LEAD_TIMES_HOURS,
    am_lead_time_hours: None,
    regular_sourcing_cost: 25.2,
    am_sourcing_cost: None,
    holding_costs_as_reported: VAN_OERS_2024_HOLDING_COSTS_AS_REPORTED,
    downtime_cost_as_reported: 3.75,
    published_policy_results: VAN_OERS_2024_NO_AM_POLICY_RESULTS,
    notes: "Table 1 no-AM scenario. The paper states Poisson demand with rate 0.04 events per hour, R2 = 4 days, R1 = 2 days, and l1 = l2 = 1 day. The table rows are copied exactly as printed.",
};

pub const VAN_OERS_2024_UPSTREAM_AM_SCENARIO: LiteratureBenchmarkScenario =
    LiteratureBenchmarkScenario {
        name: "van_oers2024_table1_upstream_am",
        source: VAN_OERS_2024_REFERENCE.source,
        url: VAN_OERS_2024_REFERENCE.url,
        literature_verified: true,
        verification_source: "published_benchmark_table_from_literature",
        model_family: "two_echelon_periodic_review_serial_spare_parts_with_upstream_am",
        am_location: "upstream",
        echelons: 2,
        simulation_horizon_days: 1000,
        table_replications: 100,
        demand_rate_per_hour: 0.04,
        review_intervals_hours: VAN_OERS_2024_REVIEW_INTERVALS_HOURS,
        transport_lead_times_hours: VAN_OERS_2024_TRANSPORT_LEAD_TIMES_HOURS,
        am_lead_time_hours: Some(6.42),
        regular_sourcing_cost: 25.2,
        am_sourcing_cost: Some(84.0),
        holding_costs_as_reported: VAN_OERS_2024_HOLDING_COSTS_AS_REPORTED,
        downtime_cost_as_reported: 3.75,
        published_policy_results: VAN_OERS_2024_UPSTREAM_AM_POLICY_RESULTS,
        notes: "Table 1 upstream-AM scenario. The paper reports a single AM lead-time input l_AM = 6.42 hours and compares enumeration, newsvendor, and echelon-separation base-stock choices.",
    };

pub const VAN_OERS_2024_DOWNSTREAM_AM_SCENARIO: LiteratureBenchmarkScenario =
    LiteratureBenchmarkScenario {
        name: "van_oers2024_table1_downstream_am",
        source: VAN_OERS_2024_REFERENCE.source,
        url: VAN_OERS_2024_REFERENCE.url,
        literature_verified: true,
        verification_source: "published_benchmark_table_from_literature",
        model_family: "two_echelon_periodic_review_serial_spare_parts_with_downstream_am",
        am_location: "downstream",
        echelons: 2,
        simulation_horizon_days: 1000,
        table_replications: 100,
        demand_rate_per_hour: 0.04,
        review_intervals_hours: VAN_OERS_2024_REVIEW_INTERVALS_HOURS,
        transport_lead_times_hours: VAN_OERS_2024_TRANSPORT_LEAD_TIMES_HOURS,
        am_lead_time_hours: Some(6.42),
        regular_sourcing_cost: 25.2,
        am_sourcing_cost: Some(84.0),
        holding_costs_as_reported: VAN_OERS_2024_HOLDING_COSTS_AS_REPORTED,
        downtime_cost_as_reported: 3.75,
        published_policy_results: VAN_OERS_2024_DOWNSTREAM_AM_POLICY_RESULTS,
        notes: "Table 1 downstream-AM scenario. The paper reports that this is the strongest AM placement in the example system, with enumeration (5, 0) and echelon separation (4, 1) nearly tied in the published table.",
    };

pub const VAN_OERS_2024_TABLE_1_SCENARIOS: &[LiteratureBenchmarkScenario] = &[
    VAN_OERS_2024_NO_AM_SCENARIO,
    VAN_OERS_2024_UPSTREAM_AM_SCENARIO,
    VAN_OERS_2024_DOWNSTREAM_AM_SCENARIO,
];

pub const KRANENBURG_2006_TABLE_5_2_BASE_CASE: KranenburgLateralTransshipmentReferenceInstance =
    KranenburgLateralTransshipmentReferenceInstance {
        name: "kranenburg2006_table5_2_base_case",
        source: KRANENBURG_2006_REFERENCE.source,
        url: KRANENBURG_2006_REFERENCE.url,
        literature_verified: true,
        verification_source: "published_exact_table_reproduced_from_literature",
        table: "5.2",
        varied_parameter: "base_case",
        varied_value_label: "base_case",
        demand_rate_per_local_warehouse: 0.001,
        num_local_warehouses: 10,
        holding_cost: 10.0,
        emergency_cost: 1000.0,
        lateral_transshipment_cost: 500.0,
        joint_warehouse_cost: 450.0,
        waiting_time_target: 0.2,
        emergency_time: 2.0,
        lateral_transshipment_time: 0.5,
        joint_warehouse_time: 0.45,
        regular_replenishment_time: 10.0,
        published_situation1_optimal_r: 9.09,
        published_situation1_cost: 91.90,
        published_situation3_optimal_r: 6.10,
        published_situation3_cost: 63.00,
        published_cost_ratio_situation1_over_situation3: 1.46,
        notes: "Base case from Tables 5.1 and 5.2. Kranenburg models a single-item spare-parts system with symmetric local warehouses, Poisson demand, emergency replenishment from a central warehouse, and optional lateral transshipment. Situation 3 assumes each local warehouse carries at most one item, so the total randomized stock satisfies 0 <= R <= |J|.",
    };

macro_rules! kranenburg_table52_row {
    (
        $name:expr,
        $varied_parameter:expr,
        $varied_value_label:expr,
        { $($field:ident : $value:expr),* $(,)? },
        $s1_r:expr,
        $s1_cost:expr,
        $s3_r:expr,
        $s3_cost:expr,
        $ratio:expr
    ) => {
        KranenburgLateralTransshipmentReferenceInstance {
            name: $name,
            varied_parameter: $varied_parameter,
            varied_value_label: $varied_value_label,
            $(
                $field: $value,
            )*
            published_situation1_optimal_r: $s1_r,
            published_situation1_cost: $s1_cost,
            published_situation3_optimal_r: $s3_r,
            published_situation3_cost: $s3_cost,
            published_cost_ratio_situation1_over_situation3: $ratio,
            notes: "Published Table 5.2 row from Kranenburg (2006).",
            ..KRANENBURG_2006_TABLE_5_2_BASE_CASE
        }
    };
}

pub const KRANENBURG_2006_TABLE_5_2_ROWS: &[KranenburgLateralTransshipmentReferenceInstance] = &[
    KRANENBURG_2006_TABLE_5_2_BASE_CASE,
    kranenburg_table52_row!(
        "kranenburg2006_table5_2_m_0p00001",
        "m",
        "0.00001",
        { demand_rate_per_local_warehouse: 0.00001 },
        9.00,
        90.02,
        6.00,
        60.03,
        1.50
    ),
    kranenburg_table52_row!(
        "kranenburg2006_table5_2_m_0p0001",
        "m",
        "0.0001",
        { demand_rate_per_local_warehouse: 0.0001 },
        9.01,
        90.19,
        6.01,
        60.30,
        1.50
    ),
    kranenburg_table52_row!(
        "kranenburg2006_table5_2_m_0p001",
        "m",
        "0.001",
        { demand_rate_per_local_warehouse: 0.001 },
        9.09,
        91.90,
        6.10,
        63.00,
        1.46
    ),
    kranenburg_table52_row!(
        "kranenburg2006_table5_2_m_0p01",
        "m",
        "0.01",
        { demand_rate_per_local_warehouse: 0.01 },
        9.90,
        109.00,
        7.00,
        90.01,
        1.21
    ),
    kranenburg_table52_row!(
        "kranenburg2006_table5_2_clat_over_cem_0p25",
        "Clat/Cem",
        "0.25",
        { lateral_transshipment_cost: 250.0 },
        9.09,
        91.90,
        6.10,
        62.00,
        1.48
    ),
    kranenburg_table52_row!(
        "kranenburg2006_table5_2_clat_over_cem_0p5",
        "Clat/Cem",
        "0.5",
        { lateral_transshipment_cost: 500.0 },
        9.09,
        91.90,
        6.10,
        63.00,
        1.46
    ),
    kranenburg_table52_row!(
        "kranenburg2006_table5_2_ch_over_cem_0p001",
        "Ch/Cem",
        "0.001",
        { holding_cost: 1.0 },
        9.09,
        10.09,
        6.10,
        8.10,
        1.25
    ),
    kranenburg_table52_row!(
        "kranenburg2006_table5_2_ch_over_cem_0p01",
        "Ch/Cem",
        "0.01",
        { holding_cost: 10.0 },
        9.09,
        91.90,
        6.10,
        63.00,
        1.46
    ),
    kranenburg_table52_row!(
        "kranenburg2006_table5_2_ch_over_cem_0p1",
        "Ch/Cem",
        "0.1",
        { holding_cost: 100.0 },
        9.09,
        910.00,
        6.10,
        612.00,
        1.49
    ),
    kranenburg_table52_row!(
        "kranenburg2006_table5_2_ch_over_cem_1",
        "Ch/Cem",
        "1",
        { holding_cost: 1000.0 },
        9.09,
        9091.00,
        6.10,
        6102.00,
        1.49
    ),
    kranenburg_table52_row!(
        "kranenburg2006_table5_2_wt_over_tem_0p05",
        "WTobj/tem",
        "0.05",
        { waiting_time_target: 0.10 },
        9.60,
        96.45,
        8.10,
        82.00,
        1.18
    ),
    kranenburg_table52_row!(
        "kranenburg2006_table5_2_wt_over_tem_0p10",
        "WTobj/tem",
        "0.10",
        { waiting_time_target: 0.20 },
        9.09,
        91.90,
        6.10,
        63.00,
        1.46
    ),
    kranenburg_table52_row!(
        "kranenburg2006_table5_2_wt_over_tem_0p15",
        "WTobj/tem",
        "0.15",
        { waiting_time_target: 0.30 },
        8.59,
        87.35,
        4.10,
        44.00,
        1.99
    ),
    kranenburg_table52_row!(
        "kranenburg2006_table5_2_wt_over_tem_0p20",
        "WTobj/tem",
        "0.20",
        { waiting_time_target: 0.40 },
        8.08,
        82.80,
        2.21,
        26.04,
        3.18
    ),
    kranenburg_table52_row!(
        "kranenburg2006_table5_2_wt_over_tem_0p25",
        "WTobj/tem",
        "0.25",
        { waiting_time_target: 0.50 },
        7.58,
        78.25,
        1.51,
        19.60,
        3.99
    ),
    kranenburg_table52_row!(
        "kranenburg2006_table5_2_wt_over_tem_0p30",
        "WTobj/tem",
        "0.30",
        { waiting_time_target: 0.60 },
        7.07,
        73.70,
        0.99,
        14.97,
        4.92
    ),
    kranenburg_table52_row!(
        "kranenburg2006_table5_2_tlat_over_tem_0p25",
        "tlat/tem",
        "0.25",
        { lateral_transshipment_time: 0.50 },
        9.09,
        91.90,
        6.10,
        63.00,
        1.46
    ),
    kranenburg_table52_row!(
        "kranenburg2006_table5_2_tlat_over_tem_0p50",
        "tlat/tem",
        "0.5",
        { lateral_transshipment_time: 1.00 },
        9.09,
        91.90,
        8.10,
        82.00,
        1.12
    ),
    kranenburg_table52_row!(
        "kranenburg2006_table5_2_treg_over_tem_2",
        "treg/tem",
        "2",
        { regular_replenishment_time: 4.0 },
        9.04,
        91.36,
        6.04,
        62.40,
        1.46
    ),
    kranenburg_table52_row!(
        "kranenburg2006_table5_2_treg_over_tem_3",
        "treg/tem",
        "3",
        { regular_replenishment_time: 6.0 },
        9.05,
        91.54,
        6.06,
        62.60,
        1.46
    ),
    kranenburg_table52_row!(
        "kranenburg2006_table5_2_treg_over_tem_5",
        "treg/tem",
        "5",
        { regular_replenishment_time: 10.0 },
        9.09,
        91.90,
        6.10,
        63.00,
        1.46
    ),
    kranenburg_table52_row!(
        "kranenburg2006_table5_2_treg_over_tem_10",
        "treg/tem",
        "10",
        { regular_replenishment_time: 20.0 },
        9.18,
        92.80,
        6.20,
        64.00,
        1.45
    ),
    kranenburg_table52_row!(
        "kranenburg2006_table5_2_j_equal_m_2",
        "|J| (m equal)",
        "2",
        { num_local_warehouses: 2 },
        1.82,
        18.38,
        1.30,
        13.39,
        1.37
    ),
    kranenburg_table52_row!(
        "kranenburg2006_table5_2_j_equal_m_3",
        "|J| (m equal)",
        "3",
        { num_local_warehouses: 3 },
        2.73,
        27.57,
        1.87,
        19.27,
        1.43
    ),
    kranenburg_table52_row!(
        "kranenburg2006_table5_2_j_equal_m_5",
        "|J| (m equal)",
        "5",
        { num_local_warehouses: 5 },
        4.55,
        45.95,
        3.05,
        31.50,
        1.46
    ),
    kranenburg_table52_row!(
        "kranenburg2006_table5_2_j_equal_m_10",
        "|J| (m equal)",
        "10",
        { num_local_warehouses: 10 },
        9.09,
        91.90,
        6.10,
        63.00,
        1.46
    ),
    kranenburg_table52_row!(
        "kranenburg2006_table5_2_j_equal_m_20",
        "|J| (m equal)",
        "20",
        { num_local_warehouses: 20 },
        18.18,
        183.80,
        12.20,
        126.00,
        1.46
    ),
    kranenburg_table52_row!(
        "kranenburg2006_table5_2_j_equal_m_50",
        "|J| (m equal)",
        "50",
        { num_local_warehouses: 50 },
        45.45,
        459.50,
        30.50,
        315.00,
        1.46
    ),
    kranenburg_table52_row!(
        "kranenburg2006_table5_2_j_equal_mtotal_2",
        "|J| (M equal)",
        "2",
        { num_local_warehouses: 2, demand_rate_per_local_warehouse: 0.005 },
        1.89,
        19.90,
        1.55,
        17.11,
        1.16
    ),
    kranenburg_table52_row!(
        "kranenburg2006_table5_2_j_equal_mtotal_3",
        "|J| (M equal)",
        "3",
        { num_local_warehouses: 3, demand_rate_per_local_warehouse: 0.01 / 3.0 },
        2.79,
        28.90,
        1.97,
        21.59,
        1.34
    ),
    kranenburg_table52_row!(
        "kranenburg2006_table5_2_j_equal_mtotal_5",
        "|J| (M equal)",
        "5",
        { num_local_warehouses: 5, demand_rate_per_local_warehouse: 0.002 },
        4.59,
        46.90,
        3.10,
        33.02,
        1.42
    ),
    kranenburg_table52_row!(
        "kranenburg2006_table5_2_j_equal_mtotal_10",
        "|J| (M equal)",
        "10",
        { num_local_warehouses: 10, demand_rate_per_local_warehouse: 0.001 },
        9.09,
        91.90,
        6.10,
        63.00,
        1.46
    ),
    kranenburg_table52_row!(
        "kranenburg2006_table5_2_j_equal_mtotal_20",
        "|J| (M equal)",
        "20",
        { num_local_warehouses: 20, demand_rate_per_local_warehouse: 0.0005 },
        18.09,
        181.90,
        12.10,
        123.00,
        1.48
    ),
    kranenburg_table52_row!(
        "kranenburg2006_table5_2_j_equal_mtotal_50",
        "|J| (M equal)",
        "50",
        { num_local_warehouses: 50, demand_rate_per_local_warehouse: 0.0002 },
        45.09,
        451.90,
        30.10,
        303.00,
        1.49
    ),
];

pub const PRIMARY_REFERENCE_INSTANCE: SparePartsReferenceInstance = SparePartsReferenceInstance {
    name: "single_echelon_repairable_operational_spares",
    source: SPARE_PARTS_REVIEW_REFERENCE.source,
    url: SPARE_PARTS_REVIEW_REFERENCE.url,
    periods: 17,
    installed_base: 12,
    procurement_lead_time: 3,
    repair_lead_time: 2,
    initial_on_hand_inventory: 2,
    initial_backlog: 0,
    initial_procurement_pipeline: &[0, 0, 0],
    initial_repair_pipeline: &[1, 0],
    failure_probability: 0.08,
    holding_cost: 0.25,
    downtime_cost: 20.0,
    procurement_cost: 3.0,
    benchmark_base_stock_level: 5,
    benchmark_lead_time_mean_cover_safety_buffer: 1.0,
    notes: "Canonical repo interpretation of spare parts as a single-echelon repairable service-parts problem with deterministic repair returns and explicit procurement to grow the rotable pool. The executable primary benchmark uses a 17-period finite horizon.",
};

pub const WORKED_TRANSITION_REFERENCE: WorkedTransitionReference = WorkedTransitionReference {
    source: PRIMARY_REFERENCE_INSTANCE.source,
    url: PRIMARY_REFERENCE_INSTANCE.url,
    installed_base: 3,
    initial_on_hand_inventory: 1,
    initial_backlog: 1,
    initial_procurement_pipeline: &[1, 0],
    initial_repair_pipeline: &[0, 2],
    action: 2,
    realized_failures: 1,
    holding_cost: 0.5,
    downtime_cost: 6.0,
    procurement_cost: 2.0,
    expected_procurement_arrival: 1,
    expected_repair_return: 0,
    expected_post_failure_on_hand_inventory: 0,
    expected_post_failure_backlog: 1,
    expected_next_on_hand_inventory: 0,
    expected_next_backlog: 0,
    expected_next_procurement_pipeline: &[0, 2],
    expected_next_repair_pipeline: &[2, 1],
    expected_period_cost: 10.0,
};

pub const VERIFICATION_PROBLEM_INSTANCE: ExactVerificationReference = ExactVerificationReference {
    source: SPARE_PARTS_REVIEW_REFERENCE.source,
    url: SPARE_PARTS_REVIEW_REFERENCE.url,
    periods: 4,
    discount_factor: 0.99,
    installed_base: 3,
    procurement_lead_time: 2,
    repair_lead_time: 2,
    initial_on_hand_inventory: 1,
    initial_backlog: 0,
    initial_procurement_pipeline: &[0, 0],
    initial_repair_pipeline: &[0, 0],
    failure_probability: 0.4,
    holding_cost: 0.5,
    downtime_cost: 6.0,
    procurement_cost: 2.0,
    max_order_quantity: 4,
    base_stock_level: 3,
    lead_time_mean_cover_safety_buffer: 1.0,
    notes: "Repo-native exact verifier on a reduced repairable spare-parts instance. The state is small enough for routine finite-horizon DP while preserving installed-base failures, repair returns, and procurement decisions.",
};
