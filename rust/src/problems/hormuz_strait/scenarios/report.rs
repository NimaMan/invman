#[derive(Clone, Debug, PartialEq)]
pub struct HormuzScenarioAssumption {
    pub name: String,
    pub value: f64,
    pub units: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HormuzDailyPriceSummary {
    pub day_index: usize,
    pub mean_brent_price_usd_per_bbl: f64,
    pub p10_brent_price_usd_per_bbl: f64,
    pub p50_brent_price_usd_per_bbl: f64,
    pub p90_brent_price_usd_per_bbl: f64,
    pub closure_fraction: f64,
    pub blocked_flow_million_bpd: f64,
    pub rerouted_flow_million_bpd: f64,
    pub reserve_release_million_bpd: f64,
    pub floating_storage_release_million_bpd: f64,
    pub non_hormuz_supply_response_million_bpd: f64,
    pub inventory_buffer_draw_million_bpd: f64,
    pub effective_tightness_million_bpd: f64,
    pub target_price_usd_per_bbl: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HormuzScenarioSimulationSummary {
    pub scenario_id: String,
    pub label: String,
    pub description: String,
    pub day_30_mean_brent_price_usd_per_bbl: f64,
    pub day_30_p10_brent_price_usd_per_bbl: f64,
    pub day_30_p50_brent_price_usd_per_bbl: f64,
    pub day_30_p90_brent_price_usd_per_bbl: f64,
    pub monthly_average_mean_brent_price_usd_per_bbl: f64,
    pub peak_mean_brent_price_usd_per_bbl: f64,
    pub peak_mean_price_day: usize,
    pub mean_effective_tightness_million_bpd: f64,
    pub max_effective_tightness_million_bpd: f64,
    pub assumptions: Vec<HormuzScenarioAssumption>,
    pub daily: Vec<HormuzDailyPriceSummary>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HormuzMonthAheadSimulationReport {
    pub analysis_date: String,
    pub latest_observed_close_date: String,
    pub latest_observed_brent_price_usd_per_bbl: f64,
    pub latest_observed_wti_price_usd_per_bbl: f64,
    pub eia_next_two_month_floor_brent_usd_per_bbl: f64,
    pub eia_q2_2026_average_brent_usd_per_bbl: f64,
    pub days: usize,
    pub paths: usize,
    pub scenarios: Vec<HormuzScenarioSimulationSummary>,
    pub notes: Vec<String>,
}
