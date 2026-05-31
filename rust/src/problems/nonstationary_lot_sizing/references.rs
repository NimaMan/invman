#![allow(dead_code)]

use std::f64::consts::PI;

use crate::problems::nonstationary_lot_sizing::demand::DemandDistributionKind;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PublishedBenchmarkReference {
    pub source: &'static str,
    pub url: &'static str,
    pub benchmark_policies: &'static [&'static str],
    pub notes: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ForecastDefinition {
    pub id: usize,
    pub name: &'static str,
    pub description: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PublishedPolicyBenchmark {
    pub source: &'static str,
    pub url: &'static str,
    pub demand_kind: DemandDistributionKind,
    pub demand_cv: f64,
    pub mean_cost: f64,
    pub cost_std: f64,
    pub shortage_rate: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorkedTransitionReference {
    pub source: &'static str,
    pub url: &'static str,
    pub forecast_horizon: usize,
    pub lead_time: usize,
    pub holding_cost: f64,
    pub shortage_cost: f64,
    pub fixed_order_cost: f64,
    pub procurement_cost: f64,
    pub lost_sales: bool,
    pub initial_forecast_window: &'static [f64; 4],
    pub initial_net_inventory: f64,
    pub initial_pipeline: &'static [f64; 1],
    pub action: f64,
    pub realized_demand: f64,
    pub next_forecast_mean: f64,
    pub expected_next_forecast_window: &'static [f64; 4],
    pub expected_next_net_inventory: f64,
    pub expected_next_pipeline: &'static [f64; 1],
    pub expected_reward: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NonstationaryLotSizingReferenceInstance {
    pub name: &'static str,
    pub source: &'static str,
    pub url: &'static str,
    pub forecast_id: usize,
    pub periods: usize,
    pub forecast_horizon: usize,
    pub lead_time: usize,
    pub holding_cost: f64,
    pub shortage_cost: f64,
    pub fixed_order_cost: f64,
    pub procurement_cost: f64,
    pub lost_sales: bool,
    pub initial_net_inventory: f64,
    pub demand_kind: DemandDistributionKind,
    pub demand_cv: f64,
    pub published_simple_benchmark: Option<PublishedPolicyBenchmark>,
    pub published_rolling_dp_benchmark: Option<PublishedPolicyBenchmark>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VerificationProblemInstance {
    pub name: &'static str,
    pub reference_instance_name: &'static str,
    pub simulation_replications: usize,
    pub mean_cost_tolerance: f64,
    pub shortage_rate_tolerance: f64,
}

pub const DEHAYBE_2024_REFERENCE: PublishedBenchmarkReference = PublishedBenchmarkReference {
    source: "Dehaybe et al. (2024), Tables 3-7 and Section 4.2",
    url: "https://doi.org/10.1016/j.ejor.2023.10.007",
    benchmark_policies: &["rolling_dp_s_s", "simple_s_s", "ppo"],
    notes: "The paper (PPO is its DRL agent) defines the rolling-forecast single-item lot-sizing setting with fixed ordering cost, lead time, and backorder/lost-sales variants. Citation metadata (EJOR 314(2):433-445, 2024) confirmed via IDEAS/RePEc in the 2026 audit. The worked-transition numbers used by this crate are reproduced via the author's public testbed code (HenriDeh/DRL_MMULS), NOT independently confirmed against a printed Section 4.2 / Table 3-4 in the article during that audit (PDF inaccessible).",
};

pub const DRL_MMULS_SINGLE_ITEM_REFERENCE: PublishedBenchmarkReference =
    PublishedBenchmarkReference {
        source: "HenriDeh/DRL_MMULS single-item branch",
        url: "https://github.com/HenriDeh/DRL_MMULS/tree/single-item",
        benchmark_policies: &["rolling_dp_s_s", "simple_s_s", "ppo"],
        notes: "The single-item branch ships the forecast library and per-instance benchmark CSVs for the paper's fixed-forecast experiments. In that branch, the lost-sales rolling-DP benchmark is evaluated with Poisson demand while the simple baseline is evaluated with CVNormal demand.",
    };

pub const FORECAST_DEFINITIONS: [ForecastDefinition; 8] = [
    ForecastDefinition {
        id: 1,
        name: "constant_0_5",
        description: "Constant demand forecast at 5 units.",
    },
    ForecastDefinition {
        id: 2,
        name: "constant_1_0",
        description: "Constant demand forecast at 10 units.",
    },
    ForecastDefinition {
        id: 3,
        name: "constant_1_5",
        description: "Constant demand forecast at 15 units.",
    },
    ForecastDefinition {
        id: 4,
        name: "seasonal_1",
        description: "Sinusoidal demand with 104-period seasonality.",
    },
    ForecastDefinition {
        id: 5,
        name: "seasonal_2",
        description: "Sinusoidal demand with 52-period seasonality.",
    },
    ForecastDefinition {
        id: 6,
        name: "seasonal_4",
        description: "Sinusoidal demand with 26-period seasonality.",
    },
    ForecastDefinition {
        id: 7,
        name: "growth",
        description: "Linear growth from 5 to 15 units over 136 periods.",
    },
    ForecastDefinition {
        id: 8,
        name: "decline",
        description: "Linear decline from 15 to 5 units over 136 periods.",
    },
];

pub const WORKED_EXAMPLE_REFERENCE: WorkedTransitionReference = WorkedTransitionReference {
    source: DEHAYBE_2024_REFERENCE.source,
    url: DEHAYBE_2024_REFERENCE.url,
    forecast_horizon: 4,
    lead_time: 1,
    holding_cost: 1.0,
    shortage_cost: 10.0,
    fixed_order_cost: 100.0,
    procurement_cost: 0.0,
    lost_sales: false,
    initial_forecast_window: &[7.0, 13.0, 10.0, 15.0],
    initial_net_inventory: -2.0,
    initial_pipeline: &[5.0],
    action: 19.0,
    realized_demand: 6.0,
    next_forecast_mean: 5.0,
    expected_next_forecast_window: &[13.0, 10.0, 15.0, 5.0],
    expected_next_net_inventory: -3.0,
    expected_next_pipeline: &[19.0],
    expected_reward: -130.0,
};

macro_rules! lost_sales_reference_instance {
    (
        $name:expr,
        $forecast_id:expr,
        $simple_cost:expr,
        $simple_std:expr,
        $simple_shortage:expr,
        $dp_cost:expr,
        $dp_std:expr,
        $dp_shortage:expr
    ) => {
        NonstationaryLotSizingReferenceInstance {
            name: $name,
            source: DEHAYBE_2024_REFERENCE.source,
            url: DEHAYBE_2024_REFERENCE.url,
            forecast_id: $forecast_id,
            periods: 104,
            forecast_horizon: 32,
            lead_time: 2,
            holding_cost: 1.0,
            shortage_cost: 5.0,
            fixed_order_cost: 10.0,
            procurement_cost: 0.0,
            lost_sales: true,
            initial_net_inventory: 20.0,
            demand_kind: DemandDistributionKind::CvNormal,
            demand_cv: 0.2,
            published_simple_benchmark: Some(PublishedPolicyBenchmark {
                source: DRL_MMULS_SINGLE_ITEM_REFERENCE.source,
                url: "https://raw.githubusercontent.com/HenriDeh/DRL_MMULS/single-item/data/single-item/scarf_testbed_simple_lostsales.csv",
                demand_kind: DemandDistributionKind::CvNormal,
                demand_cv: 0.2,
                mean_cost: $simple_cost,
                cost_std: $simple_std,
                shortage_rate: $simple_shortage,
            }),
            published_rolling_dp_benchmark: Some(PublishedPolicyBenchmark {
                source: DRL_MMULS_SINGLE_ITEM_REFERENCE.source,
                url: "https://raw.githubusercontent.com/HenriDeh/DRL_MMULS/single-item/data/single-item/scarf_testbed_DP_lostsales.csv",
                demand_kind: DemandDistributionKind::Poisson,
                demand_cv: 0.0,
                mean_cost: $dp_cost,
                cost_std: $dp_std,
                shortage_rate: $dp_shortage,
            }),
        }
    };
}

pub const LOST_SALES_FORECAST_BENCHMARKS: [NonstationaryLotSizingReferenceInstance; 8] = [
    lost_sales_reference_instance!(
        "dehaybe2024_lostsales_lt2_b5_k10_constant_5",
        1,
        1252.4885126630645,
        24.997247864746488,
        0.002257224822374979,
        1215.264,
        51.88591766994637,
        0.08371429560108733
    ),
    lost_sales_reference_instance!(
        "dehaybe2024_lostsales_lt2_b5_k10_constant_10",
        2,
        1832.9142436489014,
        61.86262354870222,
        0.0029443487165113735,
        1711.741,
        79.3574793483798,
        0.04793465748308879
    ),
    lost_sales_reference_instance!(
        "dehaybe2024_lostsales_lt2_b5_k10_constant_15",
        3,
        2369.6265719327503,
        83.31123474706921,
        0.010798230024562525,
        2072.164,
        86.43122966533255,
        0.03265778250574352
    ),
    lost_sales_reference_instance!(
        "dehaybe2024_lostsales_lt2_b5_k10_seasonal_1",
        4,
        1824.9849305221624,
        54.79894632381554,
        0.005102263384820955,
        1675.81,
        72.5810023484238,
        0.04499945105003535
    ),
    lost_sales_reference_instance!(
        "dehaybe2024_lostsales_lt2_b5_k10_seasonal_2",
        5,
        1869.9015804632895,
        53.58261747099499,
        0.00556035793112148,
        1680.512,
        73.24216183504055,
        0.04560985552056054
    ),
    lost_sales_reference_instance!(
        "dehaybe2024_lostsales_lt2_b5_k10_seasonal_4",
        6,
        1858.1096981637254,
        55.17347892586996,
        0.0068329782121353015,
        1687.426,
        72.36991037667468,
        0.045789060677398144
    ),
    lost_sales_reference_instance!(
        "dehaybe2024_lostsales_lt2_b5_k10_growth",
        7,
        1754.7650626733312,
        54.80707006265809,
        0.0016976563165351682,
        1603.741,
        69.61177859183543,
        0.05073870776319464
    ),
    lost_sales_reference_instance!(
        "dehaybe2024_lostsales_lt2_b5_k10_decline",
        8,
        1964.4606533055787,
        68.82477147038543,
        0.011555343257297896,
        1840.866,
        81.30478775885534,
        0.05170177110825886
    ),
];

pub const PRIMARY_REFERENCE_INSTANCE_NAME: &str = "dehaybe2024_lostsales_lt2_b5_k10_constant_10";

pub const ROLLING_DP_DISCOUNT_FACTOR: f64 = 0.99;
pub const ROLLING_DP_STATIONARY_TAIL_PERIODS: usize = 32;

pub const VERIFICATION_PROBLEM_INSTANCE: VerificationProblemInstance =
    VerificationProblemInstance {
        name: "constant_10_rolling_dp_reference",
        reference_instance_name: PRIMARY_REFERENCE_INSTANCE_NAME,
        simulation_replications: 25_000,
        mean_cost_tolerance: 35.0,
        shortage_rate_tolerance: 0.01,
    };

pub fn list_forecast_definitions() -> &'static [ForecastDefinition] {
    &FORECAST_DEFINITIONS
}

pub fn list_reference_instances() -> &'static [NonstationaryLotSizingReferenceInstance] {
    &LOST_SALES_FORECAST_BENCHMARKS
}

pub fn get_reference_instance(
    name: &str,
) -> Option<&'static NonstationaryLotSizingReferenceInstance> {
    LOST_SALES_FORECAST_BENCHMARKS
        .iter()
        .find(|instance| instance.name == name)
}

pub fn get_primary_reference_instance() -> &'static NonstationaryLotSizingReferenceInstance {
    get_reference_instance(PRIMARY_REFERENCE_INSTANCE_NAME)
        .expect("primary nonstationary lot sizing reference must exist")
}

pub fn build_forecast_path(forecast_id: usize, length: usize) -> Option<Vec<f64>> {
    if length == 0 {
        return Some(Vec::new());
    }
    let mut values = Vec::with_capacity(length);
    for period in 0..length {
        let t = period as f64 + 1.0;
        let value = match forecast_id {
            1 => 5.0,
            2 => 10.0,
            3 => 15.0,
            4 => 10.0 + 5.0 * (2.0 * PI * t / 104.0).sin(),
            5 => 10.0 + 5.0 * (2.0 * PI * 2.0 * t / 104.0).sin(),
            6 => 10.0 + 5.0 * (2.0 * PI * 4.0 * t / 104.0).sin(),
            7 => 5.0 + 10.0 * period as f64 / (length.saturating_sub(1).max(1) as f64),
            8 => 15.0 - 10.0 * period as f64 / (length.saturating_sub(1).max(1) as f64),
            _ => return None,
        };
        values.push(value);
    }
    Some(values)
}
