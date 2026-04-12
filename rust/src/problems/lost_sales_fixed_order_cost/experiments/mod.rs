use crate::problems::lost_sales::demand::{
    LostSalesDemandConfig, LostSalesDemandKind, DEFAULT_MMPP2_LAMBDA_HIGH,
    DEFAULT_MMPP2_LAMBDA_LOW, DEFAULT_MMPP2_NEGATIVE_P00, DEFAULT_MMPP2_NEGATIVE_P11,
    DEFAULT_MMPP2_POSITIVE_P00, DEFAULT_MMPP2_POSITIVE_P11,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FixedCostExperimentDemandCase {
    pub key: &'static str,
    pub display_name: &'static str,
    pub name_token: &'static str,
    pub demand_distribution: &'static str,
    pub demand_rate: f64,
    pub demand_lambda_low: Option<f64>,
    pub demand_lambda_high: Option<f64>,
    pub demand_p00: Option<f64>,
    pub demand_p11: Option<f64>,
    pub notes: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FixedCostExperimentGrid {
    pub name: &'static str,
    pub description: &'static str,
    pub demand_cases: &'static [FixedCostExperimentDemandCase],
    pub shortage_costs: &'static [f64],
    pub fixed_order_costs: &'static [f64],
    pub lead_times: &'static [usize],
    pub holding_cost: f64,
    pub mean_demand: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FixedCostExperimentInstance {
    pub name: String,
    pub description: String,
    pub demand_case_key: &'static str,
    pub demand_case_display_name: &'static str,
    pub demand_distribution: &'static str,
    pub demand_rate: f64,
    pub demand_lambda_low: Option<f64>,
    pub demand_lambda_high: Option<f64>,
    pub demand_p00: Option<f64>,
    pub demand_p11: Option<f64>,
    pub lead_time: usize,
    pub shortage_cost: f64,
    pub fixed_order_cost: f64,
    pub holding_cost: f64,
    pub procurement_cost: f64,
    pub max_order_size: usize,
    pub horizon: usize,
    pub eval_horizon: usize,
    pub warm_up_periods_ratio: f64,
    pub seed: u64,
    pub state_normalizer: &'static str,
    pub state_scale: f64,
    pub search_horizon: usize,
    pub search_seed: u64,
    pub position_upper_bound: usize,
    pub top_k_s_s_pairs: usize,
    pub q_window: usize,
    pub evaluation_eval_horizon: usize,
    pub evaluation_eval_seeds: usize,
    pub benchmark_family: &'static str,
    pub parent_problem_family: &'static str,
    pub notes: String,
}

pub const FULL_GRID_NAME: &str = "lost_sales_style_full_grid_mu5";
pub const DEFAULT_MAX_ORDER_SIZE: usize = 50;
pub const DEFAULT_HORIZON: usize = 3000;
pub const DEFAULT_EVAL_HORIZON: usize = 50_000;
pub const DEFAULT_WARM_UP_PERIODS_RATIO: f64 = 0.2;
pub const DEFAULT_SEED: u64 = 42;
pub const DEFAULT_STATE_NORMALIZER: &str = "quantity_scale";
pub const DEFAULT_STATE_SCALE: f64 = 50.0;
pub const DEFAULT_SEARCH_HORIZON: usize = 3000;
pub const DEFAULT_SEARCH_SEED: u64 = 42;
pub const DEFAULT_TOP_K_S_S_PAIRS: usize = 12;
pub const DEFAULT_Q_WINDOW: usize = 8;
pub const DEFAULT_EVALUATION_SEEDS: usize = 3;

pub const FIXED_COST_DEMAND_CASES: &[FixedCostExperimentDemandCase] = &[
    FixedCostExperimentDemandCase {
        key: "poisson",
        display_name: "Poisson",
        name_token: "pois",
        demand_distribution: "Poisson",
        demand_rate: 5.0,
        demand_lambda_low: None,
        demand_lambda_high: None,
        demand_p00: None,
        demand_p11: None,
        notes: "Poisson demand with mean 5.",
    },
    FixedCostExperimentDemandCase {
        key: "geometric",
        display_name: "Geometric",
        name_token: "geom",
        demand_distribution: "Geometric",
        demand_rate: 5.0,
        demand_lambda_low: None,
        demand_lambda_high: None,
        demand_p00: None,
        demand_p11: None,
        notes: "Geometric demand with mean 5.",
    },
    FixedCostExperimentDemandCase {
        key: "mmpp2_positive",
        display_name: "MMPP2 positive",
        name_token: "mmpp2_pos",
        demand_distribution: "MarkovModulatedPoisson2",
        demand_rate: 5.0,
        demand_lambda_low: Some(DEFAULT_MMPP2_LAMBDA_LOW),
        demand_lambda_high: Some(DEFAULT_MMPP2_LAMBDA_HIGH),
        demand_p00: Some(DEFAULT_MMPP2_POSITIVE_P00),
        demand_p11: Some(DEFAULT_MMPP2_POSITIVE_P11),
        notes:
            "Two-state Markov-modulated Poisson demand with positive lag-1 autocorrelation and stationary mean 5.",
    },
    FixedCostExperimentDemandCase {
        key: "mmpp2_negative",
        display_name: "MMPP2 negative",
        name_token: "mmpp2_neg",
        demand_distribution: "MarkovModulatedPoisson2",
        demand_rate: 5.0,
        demand_lambda_low: Some(DEFAULT_MMPP2_LAMBDA_LOW),
        demand_lambda_high: Some(DEFAULT_MMPP2_LAMBDA_HIGH),
        demand_p00: Some(DEFAULT_MMPP2_NEGATIVE_P00),
        demand_p11: Some(DEFAULT_MMPP2_NEGATIVE_P11),
        notes:
            "Two-state Markov-modulated Poisson demand with negative lag-1 autocorrelation and stationary mean 5.",
    },
];

pub const FULL_GRID_SHORTAGE_COSTS: &[f64] = &[4.0, 19.0];
pub const FULL_GRID_FIXED_ORDER_COSTS: &[f64] = &[5.0, 25.0];
pub const FULL_GRID_LEAD_TIMES: &[usize] = &[2, 4, 6, 8, 10];

pub const LOST_SALES_STYLE_FULL_GRID_MU5: FixedCostExperimentGrid = FixedCostExperimentGrid {
    name: FULL_GRID_NAME,
    description: "Fixed-cost lost-sales paper grid aligned with the vanilla lost-sales benchmark shape: lead times {2,4,6,8,10}, shortage costs {4,19}, fixed costs {5,25}, and demand families {Poisson, Geometric, MMPP2 positive, MMPP2 negative}, all with mean demand 5.",
    demand_cases: FIXED_COST_DEMAND_CASES,
    shortage_costs: FULL_GRID_SHORTAGE_COSTS,
    fixed_order_costs: FULL_GRID_FIXED_ORDER_COSTS,
    lead_times: FULL_GRID_LEAD_TIMES,
    holding_cost: 1.0,
    mean_demand: 5.0,
};

pub const EXPERIMENT_GRIDS: &[FixedCostExperimentGrid] = &[LOST_SALES_STYLE_FULL_GRID_MU5];

fn build_demand_config(case: &FixedCostExperimentDemandCase) -> LostSalesDemandConfig {
    let kind = match case.demand_distribution {
        "Poisson" => LostSalesDemandKind::Poisson,
        "Geometric" => LostSalesDemandKind::Geometric,
        "MarkovModulatedPoisson2" => LostSalesDemandKind::MarkovModulatedPoisson2,
        other => panic!("unsupported fixed-cost experiment demand distribution '{other}'"),
    };
    LostSalesDemandConfig {
        kind,
        demand_rate: case.demand_rate,
        demand_lambda_low: case.demand_lambda_low.unwrap_or(0.0),
        demand_lambda_high: case.demand_lambda_high.unwrap_or(0.0),
        demand_p00: case.demand_p00.unwrap_or(0.0),
        demand_p11: case.demand_p11.unwrap_or(0.0),
    }
}

fn cumulative_variance(config: &LostSalesDemandConfig, periods: usize) -> Result<f64, String> {
    let mut variance = 0.0;
    for lag in 0..periods {
        let weight = if lag == 0 {
            periods as f64
        } else {
            2.0 * (periods - lag) as f64
        };
        variance += weight * config.lag_k_autocovariance(lag)?;
    }
    Ok(variance)
}

fn default_position_upper_bound(
    case: &FixedCostExperimentDemandCase,
    lead_time: usize,
) -> Result<usize, String> {
    let config = build_demand_config(case);
    let protection_mean = (lead_time + 1) as f64 * case.demand_rate;
    let protection_std = cumulative_variance(&config, lead_time + 1)?.sqrt();
    let upper = (protection_mean + 4.0 * protection_std).ceil() as usize;
    Ok(upper.clamp(1, DEFAULT_MAX_ORDER_SIZE))
}

fn format_number(value: f64) -> String {
    if (value.fract()).abs() < 1e-12 {
        format!("{}", value as i64)
    } else {
        format!("{}", value).replace('.', "p")
    }
}

fn build_instance_name(
    case: &FixedCostExperimentDemandCase,
    lead_time: usize,
    shortage_cost: f64,
    fixed_order_cost: f64,
) -> String {
    format!(
        "lit_{}_mu{}_l{}_p{}_k{}",
        case.name_token,
        format_number(case.demand_rate),
        lead_time,
        format_number(shortage_cost),
        format_number(fixed_order_cost)
    )
}

pub fn list_experiment_grids() -> &'static [FixedCostExperimentGrid] {
    EXPERIMENT_GRIDS
}

pub fn get_experiment_grid(name: &str) -> Option<&'static FixedCostExperimentGrid> {
    EXPERIMENT_GRIDS.iter().find(|grid| grid.name == name)
}

pub fn expand_experiment_grid(name: &str) -> Result<Vec<FixedCostExperimentInstance>, String> {
    let grid = get_experiment_grid(name)
        .ok_or_else(|| format!("unknown fixed-cost experiment grid '{name}'"))?;
    let mut instances = Vec::new();
    for case in grid.demand_cases.iter() {
        for &lead_time in grid.lead_times.iter() {
            for &shortage_cost in grid.shortage_costs.iter() {
                for &fixed_order_cost in grid.fixed_order_costs.iter() {
                    let position_upper_bound = default_position_upper_bound(case, lead_time)?;
                    instances.push(FixedCostExperimentInstance {
                        name: build_instance_name(case, lead_time, shortage_cost, fixed_order_cost),
                        description: format!(
                            "Fixed-cost lost-sales full-grid instance with {} demand, mean demand {}, L={}, p={}, and K={}.",
                            case.display_name,
                            format_number(case.demand_rate),
                            lead_time,
                            format_number(shortage_cost),
                            format_number(fixed_order_cost)
                        ),
                        demand_case_key: case.key,
                        demand_case_display_name: case.display_name,
                        demand_distribution: case.demand_distribution,
                        demand_rate: case.demand_rate,
                        demand_lambda_low: case.demand_lambda_low,
                        demand_lambda_high: case.demand_lambda_high,
                        demand_p00: case.demand_p00,
                        demand_p11: case.demand_p11,
                        lead_time,
                        shortage_cost,
                        fixed_order_cost,
                        holding_cost: grid.holding_cost,
                        procurement_cost: 0.0,
                        max_order_size: DEFAULT_MAX_ORDER_SIZE,
                        horizon: DEFAULT_HORIZON,
                        eval_horizon: DEFAULT_EVAL_HORIZON,
                        warm_up_periods_ratio: DEFAULT_WARM_UP_PERIODS_RATIO,
                        seed: DEFAULT_SEED,
                        state_normalizer: DEFAULT_STATE_NORMALIZER,
                        state_scale: DEFAULT_STATE_SCALE,
                        search_horizon: DEFAULT_SEARCH_HORIZON,
                        search_seed: DEFAULT_SEARCH_SEED,
                        position_upper_bound,
                        top_k_s_s_pairs: DEFAULT_TOP_K_S_S_PAIRS,
                        q_window: DEFAULT_Q_WINDOW,
                        evaluation_eval_horizon: DEFAULT_EVAL_HORIZON,
                        evaluation_eval_seeds: DEFAULT_EVALUATION_SEEDS,
                        benchmark_family: "repo_fixed_cost_lost_sales_full_grid",
                        parent_problem_family: "Bijvank2015ParametricPolicies",
                        notes: format!(
                            "Fixed-cost extension of the lost-sales paper grid with the extra setup-cost axis K in {{5,25}}; Poisson and Geometric slices align with the literature-style single-item benchmark surface, while the two MMPP2 slices are repo extensions used to match the four-demand benchmark shape of the vanilla lost-sales suite. {}",
                            case.notes
                        ),
                    });
                }
            }
        }
    }
    Ok(instances)
}
