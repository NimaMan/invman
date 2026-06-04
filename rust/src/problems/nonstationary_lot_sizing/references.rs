#![allow(dead_code)]

// =============================================================================
// nonstationary_lot_sizing / references.rs
//
// PURPOSE
//   Source of truth for the literature instances carried for the single-item
//   non-stationary stochastic lot-sizing problem with rolling forecasts
//   (Dehaybe, Catanzaro, Chevalier 2024, EJOR 314(2):433-445,
//    DOI 10.1016/j.ejor.2023.10.007).
//
// WHAT THE CARRIED NUMBERS ACTUALLY ARE  (read before trusting any flag)
//   The per-instance `mean_cost`, `cost_std`, and `shortage_rate` values in
//   LOST_SALES_FORECAST_BENCHMARKS are NOT printed in the peer-reviewed EJOR
//   article. They are copied row-for-row from the author's PUBLIC COMPANION
//   CODE testbed CSVs:
//       data/single-item/scarf_testbed_DP_lostsales.csv      (rolling-DP rows)
//       data/single-item/scarf_testbed_simple_lostsales.csv  (simple (s,S) rows)
//   on branch `single-item` of https://github.com/HenriDeh/DRL_MMULS .
//   Those CSVs are produced by `scripts/single-item/experiments/DP_solve
//   lostsales.jl`, whose instance grid is
//       Iterators.product([2,4,8], [5,10], [10,20,30], [true])    (LT, b, K, lostsales)
//   i.e. lead times {2,4,8}, shortage {5,10}, setup {10,20,30}, CV=0.2, H=32.
//   This grid is DIFFERENT from the article's reported experiment grid in
//   `experiment_parameters_lostsales.jl`
//       leadtimes [8,4,1,0], shortages_ls [50,75,100], setups [0,80,1280],
//       CVs [0.1,0.3], horizons [16,8,4]   (defaults bold in the article's
//       parameter table).
//   So the rows we reproduce are author-testbed (reference-implementation)
//   outputs, NOT a per-instance value printed in any article table or figure.
//
// LITERATURE-VERIFICATION STATUS  (repo rule: literature-verified == an in-crate
//   test re-runs the env/solver and reproduces a number PRINTED IN A PAPER within
//   a stated tolerance; a reference-implementation / author-CSV match does NOT
//   count, nor does a self-consistent mechanics check).
//   - Every reference instance carries `literature_verified: false`.
//   - `verification_source` records WHAT each row is actually checked against
//     ("henrideh_drl_mmuls_public_testbed_csv_reference_impl_not_paper_table").
//   - The Section 4.2 worked transition is carried as an INTERNAL mechanics /
//     self-consistency check of `step_state`; the EJOR full text was not
//     accessible (paywalled; OA submitted version on the UCLouvain DIAL
//     repository was unreachable), so we make NO claim that the period
//     cost 130 / reward -130 is a value printed in the article. The flag stays
//     false and `verification_source` says so.
//
// ALGORITHM (what the executable family computes for these references)
//   1. build_forecast_path: deterministic rolling forecast path (constant /
//      sinusoidal-seasonal / linear growth / decline) used as the mean demand
//      signal feeding the env.
//   2. simple (s,S): closed-form newsvendor-style levels from the lead-time
//      demand distribution (heuristics::simple_ss).
//   3. rolling (s,S): per-period Scarf-style finite-horizon DP over the rolling
//      window, re-solved each period (heuristics::rolling_dp), discount 0.99,
//      stationary tail of 32 periods.
//   4. simulate_policy / simulate_periodic_s_s_policy: Monte-Carlo rollout of
//      the chosen policy through `step_state`, returning mean cost + shortage
//      rate, which the verifier compares to the author-testbed CSV row.
// =============================================================================

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
    /// `false`: the period cost / reward below is an INTERNAL mechanics check of
    /// `step_state`, not a value confirmed to be printed in the article.
    pub literature_verified: bool,
    /// What the worked transition is actually checked against.
    pub verification_source: &'static str,
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
    /// `false`: the benchmark rows reproduce the author's PUBLIC TESTBED CSVs
    /// (reference implementation), not a value printed in the EJOR article.
    pub literature_verified: bool,
    /// What each benchmark row is actually checked against.
    pub verification_source: &'static str,
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
    /// `false`: the verifier reproduces the author's public testbed CSV row, not
    /// a paper-printed number.
    pub literature_verified: bool,
    pub verification_source: &'static str,
    pub simulation_replications: usize,
    pub mean_cost_tolerance: f64,
    pub shortage_rate_tolerance: f64,
}

pub const DEHAYBE_2024_REFERENCE: PublishedBenchmarkReference = PublishedBenchmarkReference {
    source: "Dehaybe, Catanzaro & Chevalier (2024), EJOR 314(2):433-445",
    url: "https://doi.org/10.1016/j.ejor.2023.10.007",
    benchmark_policies: &["rolling_dp_s_s", "simple_s_s", "ppo"],
    notes: "Paper title: 'Deep Reinforcement Learning for inventory optimization with non-stationary uncertain demand'. It defines the single-item rolling-forecast lot-sizing MDP (holding cost, shortage cost, fixed setup cost, lead time, backorders OR lost sales) and reports DRL-vs-DP comparisons. HONEST STATUS: the EJOR full text was not accessible to this repo (paywalled; the OA submitted version on the UCLouvain DIAL repository was unreachable). We therefore do NOT carry any number confirmed to be printed in an article table or figure. The Section-4 worked transition is reproduced as an internal mechanics check only, and the per-instance benchmark rows are reproduced from the author's public companion-code testbed CSVs (see DRL_MMULS_SINGLE_ITEM_REFERENCE), not from an article table. literature_verified = false on every instance.",
};

pub const DRL_MMULS_SINGLE_ITEM_REFERENCE: PublishedBenchmarkReference =
    PublishedBenchmarkReference {
        source: "HenriDeh/DRL_MMULS single-item branch",
        url: "https://github.com/HenriDeh/DRL_MMULS/tree/single-item",
        benchmark_policies: &["rolling_dp_s_s", "simple_s_s", "ppo"],
        notes: "Author's public companion code for the EJOR article. The carried per-instance benchmark rows are copied EXACTLY from this repo's testbed CSVs (data/single-item/scarf_testbed_DP_lostsales.csv and scarf_testbed_simple_lostsales.csv), produced by scripts/single-item/experiments/'DP_solve lostsales.jl'. That script's grid is Iterators.product([2,4,8],[5,10],[10,20,30],[true]) with CV=0.2, horizon=32 -- the rows we reproduce are its (LT=2,b=5,K=10) family. In that branch the lost-sales rolling-DP benchmark is evaluated with Poisson demand and the simple (s,S) baseline with CVNormal demand. This is a REFERENCE-IMPLEMENTATION match, NOT a number printed in the peer-reviewed article; the article's reported experiment grid (experiment_parameters_lostsales.jl: leadtimes [8,4,1,0], shortages_ls [50,75,100], setups [0,80,1280], CVs [0.1,0.3], horizons [16,8,4]) differs from this testbed grid.",
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
    literature_verified: false,
    verification_source: "internal_step_state_mechanics_self_consistency_not_a_paper_printed_number",
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
            literature_verified: false,
            verification_source:
                "henrideh_drl_mmuls_public_testbed_csv_reference_impl_not_paper_table",
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
        literature_verified: false,
        verification_source:
            "henrideh_drl_mmuls_public_testbed_csv_reference_impl_not_paper_table",
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
