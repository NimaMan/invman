#![allow(dead_code)]

use crate::problems::perishable_inventory::env::IssuingPolicy;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PublishedBenchmarkReference {
    pub source: &'static str,
    pub url: &'static str,
    pub benchmark_policies: &'static [&'static str],
    pub notes: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PublishedScenarioAReturns {
    pub source: &'static str,
    pub url: &'static str,
    pub value_iteration_mean_return: i32,
    pub value_iteration_return_std: i32,
    pub best_base_stock_mean_return: i32,
    pub best_base_stock_return_std: i32,
    pub optimality_gap_pct: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PublishedFigure3Verification {
    pub source: &'static str,
    pub url: &'static str,
    pub published_base_stock_level: usize,
    pub published_optimal_policy: &'static [[usize; 9]; 9],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PerishableReferenceInstance {
    pub name: &'static str,
    pub demand_mean: f64,
    pub demand_cov: f64,
    pub shelf_life: usize,
    pub lead_time: usize,
    pub shortage_cost: f64,
    pub holding_cost: f64,
    pub waste_cost: f64,
    pub procurement_cost: f64,
    pub max_order_size: usize,
    pub issuing_policy: IssuingPolicy,
    pub horizon: usize,
    pub eval_horizon: usize,
    pub warm_up_periods_ratio: f64,
    pub published_scenario_a_returns: Option<PublishedScenarioAReturns>,
    pub published_figure3_verification: Option<PublishedFigure3Verification>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VerificationProblemInstance {
    pub name: &'static str,
    pub reference_instance_name: &'static str,
    pub published_base_stock_level: usize,
    pub published_value_iteration_mean_return: i32,
    pub published_optimal_policy: &'static [[usize; 9]; 9],
}

pub const BENCHMARK_POLICIES: &[&str] = &["base_stock", "bsp_low_ew", "dqn", "shaped_dqn"];

pub const DE_MOOR_2022_REFERENCE: PublishedBenchmarkReference = PublishedBenchmarkReference {
    source: "De Moor, Gijsbrechts, Boute (2022), \"Reward shaping to improve the performance of deep reinforcement learning in perishable inventory management\", European Journal of Operational Research, 301(2), 535-545",
    url: "https://doi.org/10.1016/j.ejor.2021.10.045",
    benchmark_policies: BENCHMARK_POLICIES,
    notes: "De Moor et al. (2022) fully specified the optimal and best base-stock policies for the m=2 experiments 1 (LIFO) and 2 (FIFO); these are the published-policy tables and base-stock levels (5 LIFO, 7 FIFO) re-derived by the exact MDP. The exact figure number in the published EJOR article was not independently confirmed by the librarian audit (paywalled full text).",
};

pub const FARRINGTON_2025_REFERENCE: PublishedBenchmarkReference = PublishedBenchmarkReference {
    source: "Farrington, Wong, Li, Utley (2025), \"Going faster to see further: graphics processing unit-accelerated value iteration and simulation for perishable inventory control using JAX\", Annals of Operations Research, 349(3), 1609-1638, Table 3",
    url: "https://doi.org/10.1007/s10479-025-06551-6",
    benchmark_policies: &["value_iteration", "base_stock"],
    notes: "Table 3 reports value-iteration and best base-stock mean returns and standard deviations for all 32 Scenario A settings from De Moor et al. (2022). PMC open-access copy: https://pmc.ncbi.nlm.nih.gov/articles/PMC12350524/ ; arXiv preprint: https://arxiv.org/abs/2303.10672 .",
};

pub const DE_MOOR_M2_EXP1_FIGURE3_POLICY: [[usize; 9]; 9] = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [2, 2, 2, 2, 2, 2, 2, 2, 2],
    [3, 3, 3, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3, 3, 3, 3],
];

pub const DE_MOOR_M2_EXP2_FIGURE3_POLICY: [[usize; 9]; 9] = [
    [1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0, 0],
    [2, 1, 1, 0, 0, 0, 0, 0, 0],
    [2, 2, 1, 1, 0, 0, 0, 0, 0],
    [3, 2, 2, 1, 1, 1, 1, 0, 0],
    [3, 3, 2, 2, 1, 1, 1, 1, 1],
    [4, 3, 3, 3, 2, 2, 2, 2, 2],
    [4, 4, 3, 3, 3, 3, 3, 3, 3],
    [4, 4, 4, 4, 4, 4, 4, 4, 4],
];

macro_rules! scenario_a_instance {
    (
        $name:expr,
        $m:expr,
        $exp:expr,
        $lead:expr,
        $waste:expr,
        $policy:expr,
        $vi_mean:expr,
        $vi_std:expr,
        $base_mean:expr,
        $base_std:expr,
        $gap:expr,
        $fig:expr
    ) => {
        PerishableReferenceInstance {
            name: $name,
            demand_mean: 4.0,
            demand_cov: 0.50,
            shelf_life: $m,
            lead_time: $lead,
            shortage_cost: 5.0,
            holding_cost: 1.0,
            waste_cost: $waste,
            procurement_cost: 3.0,
            max_order_size: 10,
            issuing_policy: $policy,
            horizon: 465,
            eval_horizon: 365,
            warm_up_periods_ratio: 100.0 / 465.0,
            published_scenario_a_returns: Some(PublishedScenarioAReturns {
                source: FARRINGTON_2025_REFERENCE.source,
                url: FARRINGTON_2025_REFERENCE.url,
                value_iteration_mean_return: $vi_mean,
                value_iteration_return_std: $vi_std,
                best_base_stock_mean_return: $base_mean,
                best_base_stock_return_std: $base_std,
                optimality_gap_pct: $gap,
            }),
            published_figure3_verification: $fig,
        }
    };
}

pub const SCENARIO_A_REFERENCE_INSTANCES: [PerishableReferenceInstance; 32] = [
    scenario_a_instance!(
        "de_moor2022_m2_exp1_l1_cp7_lifo",
        2,
        1,
        1,
        7.0,
        IssuingPolicy::Lifo,
        -1553,
        61,
        -1565,
        62,
        0.80,
        Some(PublishedFigure3Verification {
            source: DE_MOOR_2022_REFERENCE.source,
            url: DE_MOOR_2022_REFERENCE.url,
            published_base_stock_level: 5,
            published_optimal_policy: &DE_MOOR_M2_EXP1_FIGURE3_POLICY,
        })
    ),
    scenario_a_instance!(
        "de_moor2022_m2_exp2_l1_cp7_fifo",
        2,
        2,
        1,
        7.0,
        IssuingPolicy::Fifo,
        -1457,
        59,
        -1474,
        56,
        1.20,
        Some(PublishedFigure3Verification {
            source: DE_MOOR_2022_REFERENCE.source,
            url: DE_MOOR_2022_REFERENCE.url,
            published_base_stock_level: 7,
            published_optimal_policy: &DE_MOOR_M2_EXP2_FIGURE3_POLICY,
        })
    ),
    scenario_a_instance!(
        "de_moor2022_m2_exp3_l1_cp10_lifo",
        2,
        3,
        1,
        10.0,
        IssuingPolicy::Lifo,
        -1571,
        61,
        -1581,
        62,
        0.64,
        None
    ),
    scenario_a_instance!(
        "de_moor2022_m2_exp4_l1_cp10_fifo",
        2,
        4,
        1,
        10.0,
        IssuingPolicy::Fifo,
        -1463,
        60,
        -1485,
        61,
        1.46,
        None
    ),
    scenario_a_instance!(
        "de_moor2022_m2_exp5_l2_cp7_lifo",
        2,
        5,
        2,
        7.0,
        IssuingPolicy::Lifo,
        -1551,
        62,
        -1590,
        64,
        2.49,
        None
    ),
    scenario_a_instance!(
        "de_moor2022_m2_exp6_l2_cp7_fifo",
        2,
        6,
        2,
        7.0,
        IssuingPolicy::Fifo,
        -1461,
        58,
        -1495,
        60,
        2.31,
        None
    ),
    scenario_a_instance!(
        "de_moor2022_m2_exp7_l2_cp10_lifo",
        2,
        7,
        2,
        10.0,
        IssuingPolicy::Lifo,
        -1569,
        61,
        -1606,
        64,
        2.35,
        None
    ),
    scenario_a_instance!(
        "de_moor2022_m2_exp8_l2_cp10_fifo",
        2,
        8,
        2,
        10.0,
        IssuingPolicy::Fifo,
        -1469,
        59,
        -1504,
        60,
        2.41,
        None
    ),
    scenario_a_instance!(
        "de_moor2022_m3_exp1_l1_cp7_lifo",
        3,
        1,
        1,
        7.0,
        IssuingPolicy::Lifo,
        -1490,
        58,
        -1500,
        59,
        0.71,
        None
    ),
    scenario_a_instance!(
        "de_moor2022_m3_exp2_l1_cp7_fifo",
        3,
        2,
        1,
        7.0,
        IssuingPolicy::Fifo,
        -1424,
        56,
        -1435,
        52,
        0.74,
        None
    ),
    scenario_a_instance!(
        "de_moor2022_m3_exp3_l1_cp10_lifo",
        3,
        3,
        1,
        10.0,
        IssuingPolicy::Lifo,
        -1498,
        61,
        -1512,
        58,
        0.90,
        None
    ),
    scenario_a_instance!(
        "de_moor2022_m3_exp4_l1_cp10_fifo",
        3,
        4,
        1,
        10.0,
        IssuingPolicy::Fifo,
        -1425,
        55,
        -1436,
        52,
        0.82,
        None
    ),
    scenario_a_instance!(
        "de_moor2022_m3_exp5_l2_cp7_lifo",
        3,
        5,
        2,
        7.0,
        IssuingPolicy::Lifo,
        -1513,
        61,
        -1533,
        61,
        1.32,
        None
    ),
    scenario_a_instance!(
        "de_moor2022_m3_exp6_l2_cp7_fifo",
        3,
        6,
        2,
        7.0,
        IssuingPolicy::Fifo,
        -1435,
        56,
        -1456,
        58,
        1.42,
        None
    ),
    scenario_a_instance!(
        "de_moor2022_m3_exp7_l2_cp10_lifo",
        3,
        7,
        2,
        10.0,
        IssuingPolicy::Lifo,
        -1526,
        60,
        -1544,
        61,
        1.16,
        None
    ),
    scenario_a_instance!(
        "de_moor2022_m3_exp8_l2_cp10_fifo",
        3,
        8,
        2,
        10.0,
        IssuingPolicy::Fifo,
        -1437,
        56,
        -1457,
        58,
        1.42,
        None
    ),
    scenario_a_instance!(
        "de_moor2022_m4_exp1_l1_cp7_lifo",
        4,
        1,
        1,
        7.0,
        IssuingPolicy::Lifo,
        -1459,
        56,
        -1476,
        54,
        1.15,
        None
    ),
    scenario_a_instance!(
        "de_moor2022_m4_exp2_l1_cp7_fifo",
        4,
        2,
        1,
        7.0,
        IssuingPolicy::Fifo,
        -1422,
        56,
        -1430,
        52,
        0.54,
        None
    ),
    scenario_a_instance!(
        "de_moor2022_m4_exp3_l1_cp10_lifo",
        4,
        3,
        1,
        10.0,
        IssuingPolicy::Lifo,
        -1465,
        56,
        -1481,
        60,
        1.08,
        None
    ),
    scenario_a_instance!(
        "de_moor2022_m4_exp4_l1_cp10_fifo",
        4,
        4,
        1,
        10.0,
        IssuingPolicy::Fifo,
        -1422,
        56,
        -1430,
        52,
        0.54,
        None
    ),
    scenario_a_instance!(
        "de_moor2022_m4_exp5_l2_cp7_lifo",
        4,
        5,
        2,
        7.0,
        IssuingPolicy::Lifo,
        -1480,
        59,
        -1496,
        59,
        1.07,
        None
    ),
    scenario_a_instance!(
        "de_moor2022_m4_exp6_l2_cp7_fifo",
        4,
        6,
        2,
        7.0,
        IssuingPolicy::Fifo,
        -1432,
        55,
        -1453,
        58,
        1.44,
        None
    ),
    scenario_a_instance!(
        "de_moor2022_m4_exp7_l2_cp10_lifo",
        4,
        7,
        2,
        10.0,
        IssuingPolicy::Lifo,
        -1489,
        59,
        -1505,
        58,
        1.07,
        None
    ),
    scenario_a_instance!(
        "de_moor2022_m4_exp8_l2_cp10_fifo",
        4,
        8,
        2,
        10.0,
        IssuingPolicy::Fifo,
        -1432,
        55,
        -1453,
        58,
        1.44,
        None
    ),
    scenario_a_instance!(
        "de_moor2022_m5_exp1_l1_cp7_lifo",
        5,
        1,
        1,
        7.0,
        IssuingPolicy::Lifo,
        -1443,
        55,
        -1454,
        55,
        0.73,
        None
    ),
    scenario_a_instance!(
        "de_moor2022_m5_exp2_l1_cp7_fifo",
        5,
        2,
        1,
        7.0,
        IssuingPolicy::Fifo,
        -1422,
        56,
        -1430,
        52,
        0.54,
        None
    ),
    scenario_a_instance!(
        "de_moor2022_m5_exp3_l1_cp10_lifo",
        5,
        3,
        1,
        10.0,
        IssuingPolicy::Lifo,
        -1446,
        56,
        -1460,
        55,
        0.94,
        None
    ),
    scenario_a_instance!(
        "de_moor2022_m5_exp4_l1_cp10_fifo",
        5,
        4,
        1,
        10.0,
        IssuingPolicy::Fifo,
        -1422,
        56,
        -1430,
        52,
        0.54,
        None
    ),
    scenario_a_instance!(
        "de_moor2022_m5_exp5_l2_cp7_lifo",
        5,
        5,
        2,
        7.0,
        IssuingPolicy::Lifo,
        -1463,
        58,
        -1480,
        60,
        1.22,
        None
    ),
    scenario_a_instance!(
        "de_moor2022_m5_exp6_l2_cp7_fifo",
        5,
        6,
        2,
        7.0,
        IssuingPolicy::Fifo,
        -1432,
        55,
        -1453,
        58,
        1.44,
        None
    ),
    scenario_a_instance!(
        "de_moor2022_m5_exp7_l2_cp10_lifo",
        5,
        7,
        2,
        10.0,
        IssuingPolicy::Lifo,
        -1467,
        58,
        -1484,
        59,
        1.15,
        None
    ),
    scenario_a_instance!(
        "de_moor2022_m5_exp8_l2_cp10_fifo",
        5,
        8,
        2,
        10.0,
        IssuingPolicy::Fifo,
        -1432,
        55,
        -1453,
        58,
        1.44,
        None
    ),
];

pub const PRIMARY_REFERENCE_INSTANCE_NAME: &str = "de_moor2022_m2_exp2_l1_cp7_fifo";

pub const VERIFICATION_EXP1: VerificationProblemInstance = VerificationProblemInstance {
    name: "de_moor2022_figure3_m2_exp1_lifo",
    reference_instance_name: "de_moor2022_m2_exp1_l1_cp7_lifo",
    published_base_stock_level: 5,
    published_value_iteration_mean_return: -1553,
    published_optimal_policy: &DE_MOOR_M2_EXP1_FIGURE3_POLICY,
};

pub const VERIFICATION_EXP2: VerificationProblemInstance = VerificationProblemInstance {
    name: "de_moor2022_figure3_m2_exp2_fifo",
    reference_instance_name: "de_moor2022_m2_exp2_l1_cp7_fifo",
    published_base_stock_level: 7,
    published_value_iteration_mean_return: -1457,
    published_optimal_policy: &DE_MOOR_M2_EXP2_FIGURE3_POLICY,
};

pub const VERIFICATION_PROBLEM_INSTANCES: [VerificationProblemInstance; 2] =
    [VERIFICATION_EXP1, VERIFICATION_EXP2];

pub const VERIFICATION_PROBLEM_INSTANCE: VerificationProblemInstance = VERIFICATION_EXP2;

pub fn get_primary_reference_instance() -> PerishableReferenceInstance {
    get_reference_instance(PRIMARY_REFERENCE_INSTANCE_NAME)
        .expect("primary reference instance must exist")
}

pub fn get_reference_instance(name: &str) -> Option<PerishableReferenceInstance> {
    SCENARIO_A_REFERENCE_INSTANCES
        .iter()
        .copied()
        .find(|instance| instance.name == name)
}

pub fn list_reference_instances() -> &'static [PerishableReferenceInstance] {
    &SCENARIO_A_REFERENCE_INSTANCES
}

pub fn build_lifetime_sweep_instances(lifetimes: &[usize]) -> Vec<PerishableReferenceInstance> {
    SCENARIO_A_REFERENCE_INSTANCES
        .iter()
        .copied()
        .filter(|instance| lifetimes.contains(&instance.shelf_life))
        .collect()
}
