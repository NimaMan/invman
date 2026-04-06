use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

use crate::problems::hormuz_strait::references::{
    current_market_anchors, HORMUZ_2024_AVAILABLE_BYPASS_CAPACITY_MILLION_BPD,
    HORMUZ_2024_TOTAL_OIL_FLOW_MILLION_BPD,
};

use super::presets::{month_ahead_scenario_presets, HormuzPriceScenarioPreset};
use super::report::{
    HormuzDailyPriceSummary, HormuzMonthAheadSimulationReport, HormuzScenarioAssumption,
    HormuzScenarioSimulationSummary,
};

const MAX_SCARCITY_PREMIUM_USD_PER_BBL: f64 = 72.0;
const SCARCITY_TIGHTNESS_SCALE_MILLION_BPD: f64 = 7.0;
const SUPPLY_RESPONSE_RAMP_DAYS: usize = 14;
const RESERVE_RESPONSE_RAMP_DAYS: usize = 7;
const FLOATING_STORAGE_RESPONSE_RAMP_DAYS: usize = 10;

#[derive(Clone, Copy, Debug)]
struct DailyMarketState {
    closure_fraction: f64,
    blocked_flow_million_bpd: f64,
    rerouted_flow_million_bpd: f64,
    reserve_release_million_bpd: f64,
    floating_storage_release_million_bpd: f64,
    non_hormuz_supply_response_million_bpd: f64,
    inventory_buffer_draw_million_bpd: f64,
    effective_tightness_million_bpd: f64,
    target_price_usd_per_bbl: f64,
}

fn clamp_probability(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}

fn linear_progress(day_index: usize, days: usize) -> f64 {
    if days <= 1 {
        1.0
    } else {
        day_index as f64 / (days - 1) as f64
    }
}

fn activation(day_index: usize, ramp_days: usize, initial_share: f64) -> f64 {
    if ramp_days == 0 {
        1.0
    } else {
        let progress = ((day_index + 1) as f64 / ramp_days as f64).min(1.0);
        initial_share + (1.0 - initial_share) * progress
    }
}

fn percentile(sorted_values: &[f64], percentile: f64) -> f64 {
    if sorted_values.is_empty() {
        return 0.0;
    }
    if sorted_values.len() == 1 {
        return sorted_values[0];
    }
    let capped = percentile.clamp(0.0, 1.0);
    let position = capped * (sorted_values.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        sorted_values[lower]
    } else {
        let weight = position - lower as f64;
        sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
    }
}

fn daily_market_state(
    preset: HormuzPriceScenarioPreset,
    day_index: usize,
    days: usize,
) -> DailyMarketState {
    let anchors = current_market_anchors();
    let progress = linear_progress(day_index, days);
    let closure_fraction = preset.closure_fraction_day_0
        + (preset.closure_fraction_day_n - preset.closure_fraction_day_0) * progress;
    let blocked_flow_million_bpd = HORMUZ_2024_TOTAL_OIL_FLOW_MILLION_BPD * closure_fraction;
    let rerouted_flow_million_bpd = blocked_flow_million_bpd.min(
        HORMUZ_2024_AVAILABLE_BYPASS_CAPACITY_MILLION_BPD
            * clamp_probability(preset.bypass_utilization),
    );
    let reserve_release_million_bpd = preset.strategic_reserve_release_million_bpd
        * activation(day_index, RESERVE_RESPONSE_RAMP_DAYS, 0.45);
    let floating_storage_release_million_bpd = preset.floating_storage_release_million_bpd
        * activation(day_index, FLOATING_STORAGE_RESPONSE_RAMP_DAYS, 0.30);
    let non_hormuz_supply_response_million_bpd = (anchors.opec_may_2026_supply_adjustment_million_bpd
        + preset.additional_non_hormuz_supply_response_million_bpd)
        * activation(day_index, SUPPLY_RESPONSE_RAMP_DAYS, 0.15);
    let inventory_buffer_draw_million_bpd =
        preset.inventory_buffer_draw_million_bpd * (1.0 - 0.35 * progress);
    let uncovered_flow_loss_million_bpd = (blocked_flow_million_bpd
        - rerouted_flow_million_bpd
        - reserve_release_million_bpd
        - floating_storage_release_million_bpd
        - non_hormuz_supply_response_million_bpd)
        .max(0.0);
    let upstream_shut_in_million_bpd =
        uncovered_flow_loss_million_bpd * preset.upstream_shut_in_share_of_uncovered_loss;
    let effective_tightness_million_bpd = (uncovered_flow_loss_million_bpd
        + upstream_shut_in_million_bpd
        - inventory_buffer_draw_million_bpd)
        .max(0.0);
    let scarcity_premium_usd_per_bbl = MAX_SCARCITY_PREMIUM_USD_PER_BBL
        * (1.0
            - (-effective_tightness_million_bpd / SCARCITY_TIGHTNESS_SCALE_MILLION_BPD).exp());
    let target_price_usd_per_bbl = anchors.eia_next_two_month_floor_brent_usd_per_bbl
        + preset.persistent_risk_premium_usd_per_bbl
        + scarcity_premium_usd_per_bbl;

    DailyMarketState {
        closure_fraction,
        blocked_flow_million_bpd,
        rerouted_flow_million_bpd,
        reserve_release_million_bpd,
        floating_storage_release_million_bpd,
        non_hormuz_supply_response_million_bpd,
        inventory_buffer_draw_million_bpd,
        effective_tightness_million_bpd,
        target_price_usd_per_bbl,
    }
}

fn simulate_scenario(
    preset: HormuzPriceScenarioPreset,
    days: usize,
    paths: usize,
    seed: u64,
) -> HormuzScenarioSimulationSummary {
    let anchors = current_market_anchors();
    let daily_market: Vec<DailyMarketState> = (0..days)
        .map(|day_index| daily_market_state(preset, day_index, days))
        .collect();
    let normal = Normal::new(0.0, 1.0).expect("valid standard deviation");
    let mut rng = StdRng::seed_from_u64(seed);
    let mut price_paths = vec![vec![0.0; paths]; days];

    for path_index in 0..paths {
        let mut price = anchors.latest_observed_brent_usd_per_bbl;
        for (day_index, market_state) in daily_market.iter().enumerate() {
            let noise = normal.sample(&mut rng) * preset.daily_volatility_usd_per_bbl;
            price = (price
                + preset.mean_reversion_to_target
                    * (market_state.target_price_usd_per_bbl - price)
                + noise)
                .max(40.0);
            price_paths[day_index][path_index] = price;
        }
    }

    let mut daily = Vec::with_capacity(days);
    let mut monthly_mean_accumulator = 0.0;
    let mut peak_mean_brent_price_usd_per_bbl = f64::NEG_INFINITY;
    let mut peak_mean_price_day = 0usize;
    let mut effective_tightness_sum = 0.0;
    let mut max_effective_tightness_million_bpd: f64 = 0.0;

    for (day_index, market_state) in daily_market.iter().enumerate() {
        let prices = &mut price_paths[day_index];
        prices.sort_by(|left, right| left.total_cmp(right));
        let mean_brent_price_usd_per_bbl =
            prices.iter().copied().sum::<f64>() / prices.len() as f64;
        monthly_mean_accumulator += mean_brent_price_usd_per_bbl;
        if mean_brent_price_usd_per_bbl > peak_mean_brent_price_usd_per_bbl {
            peak_mean_brent_price_usd_per_bbl = mean_brent_price_usd_per_bbl;
            peak_mean_price_day = day_index + 1;
        }
        effective_tightness_sum += market_state.effective_tightness_million_bpd;
        max_effective_tightness_million_bpd = max_effective_tightness_million_bpd
            .max(market_state.effective_tightness_million_bpd);

        daily.push(HormuzDailyPriceSummary {
            day_index: day_index + 1,
            mean_brent_price_usd_per_bbl,
            p10_brent_price_usd_per_bbl: percentile(prices, 0.10),
            p50_brent_price_usd_per_bbl: percentile(prices, 0.50),
            p90_brent_price_usd_per_bbl: percentile(prices, 0.90),
            closure_fraction: market_state.closure_fraction,
            blocked_flow_million_bpd: market_state.blocked_flow_million_bpd,
            rerouted_flow_million_bpd: market_state.rerouted_flow_million_bpd,
            reserve_release_million_bpd: market_state.reserve_release_million_bpd,
            floating_storage_release_million_bpd: market_state.floating_storage_release_million_bpd,
            non_hormuz_supply_response_million_bpd: market_state
                .non_hormuz_supply_response_million_bpd,
            inventory_buffer_draw_million_bpd: market_state.inventory_buffer_draw_million_bpd,
            effective_tightness_million_bpd: market_state.effective_tightness_million_bpd,
            target_price_usd_per_bbl: market_state.target_price_usd_per_bbl,
        });
    }

    let day_30 = daily
        .last()
        .cloned()
        .expect("month-ahead simulation requires at least one day");

    HormuzScenarioSimulationSummary {
        scenario_id: String::from(preset.scenario_id),
        label: String::from(preset.label),
        description: String::from(preset.description),
        day_30_mean_brent_price_usd_per_bbl: day_30.mean_brent_price_usd_per_bbl,
        day_30_p10_brent_price_usd_per_bbl: day_30.p10_brent_price_usd_per_bbl,
        day_30_p50_brent_price_usd_per_bbl: day_30.p50_brent_price_usd_per_bbl,
        day_30_p90_brent_price_usd_per_bbl: day_30.p90_brent_price_usd_per_bbl,
        monthly_average_mean_brent_price_usd_per_bbl: monthly_mean_accumulator / days as f64,
        peak_mean_brent_price_usd_per_bbl,
        peak_mean_price_day,
        mean_effective_tightness_million_bpd: effective_tightness_sum / days as f64,
        max_effective_tightness_million_bpd,
        assumptions: vec![
            HormuzScenarioAssumption {
                name: String::from("closure_fraction_day_0"),
                value: preset.closure_fraction_day_0,
                units: String::from("share"),
            },
            HormuzScenarioAssumption {
                name: String::from("closure_fraction_day_30"),
                value: preset.closure_fraction_day_n,
                units: String::from("share"),
            },
            HormuzScenarioAssumption {
                name: String::from("bypass_utilization"),
                value: preset.bypass_utilization,
                units: String::from("share_of_2.6_mbd_capacity"),
            },
            HormuzScenarioAssumption {
                name: String::from("strategic_reserve_release"),
                value: preset.strategic_reserve_release_million_bpd,
                units: String::from("million_bpd"),
            },
            HormuzScenarioAssumption {
                name: String::from("floating_storage_release"),
                value: preset.floating_storage_release_million_bpd,
                units: String::from("million_bpd"),
            },
            HormuzScenarioAssumption {
                name: String::from("additional_non_hormuz_supply_response"),
                value: preset.additional_non_hormuz_supply_response_million_bpd,
                units: String::from("million_bpd"),
            },
            HormuzScenarioAssumption {
                name: String::from("inventory_buffer_draw"),
                value: preset.inventory_buffer_draw_million_bpd,
                units: String::from("million_bpd"),
            },
            HormuzScenarioAssumption {
                name: String::from("upstream_shut_in_share_of_uncovered_loss"),
                value: preset.upstream_shut_in_share_of_uncovered_loss,
                units: String::from("share"),
            },
            HormuzScenarioAssumption {
                name: String::from("persistent_risk_premium"),
                value: preset.persistent_risk_premium_usd_per_bbl,
                units: String::from("usd_per_bbl"),
            },
        ],
        daily,
    }
}

pub fn simulate_month_ahead_price_scenarios(
    days: usize,
    paths: usize,
    seed: u64,
) -> HormuzMonthAheadSimulationReport {
    let anchors = current_market_anchors();
    let scenario_seeds = month_ahead_scenario_presets()
        .iter()
        .enumerate()
        .map(|(index, preset)| simulate_scenario(*preset, days, paths, seed + index as u64 * 9973))
        .collect();

    HormuzMonthAheadSimulationReport {
        analysis_date: String::from(anchors.analysis_date),
        latest_observed_close_date: String::from(anchors.latest_observed_close_date),
        latest_observed_brent_price_usd_per_bbl: anchors.latest_observed_brent_usd_per_bbl,
        latest_observed_wti_price_usd_per_bbl: anchors.latest_observed_wti_usd_per_bbl,
        eia_next_two_month_floor_brent_usd_per_bbl: anchors
            .eia_next_two_month_floor_brent_usd_per_bbl,
        eia_q2_2026_average_brent_usd_per_bbl: anchors.eia_q2_2026_average_brent_usd_per_bbl,
        days,
        paths,
        scenarios: scenario_seeds,
        notes: vec![
            String::from(
                "This is a scenario engine, not a market-clearing futures model or a claim about realized prices.",
            ),
            String::from(
                "The model is anchored to official EIA and OPEC figures current as of 2026-04-06 and uses 2024 Hormuz flow weights for physical exposure.",
            ),
            String::from(
                "Brent prices evolve by daily mean reversion from the latest observed close toward a scenario-specific target built from net effective tightness and a persistent risk premium.",
            ),
        ],
    }
}
