// ============================================================================
// lp_dataset_loader.rs
//
// PURPOSE
// -------
// Parse the checked-in perfect-information LP datasets that mirror the Pahr &
// Grunow (2025) companion repository's per-instance `config.json` /
// `expected_revenue.json` / `upper_bound.json`. Each dataset is a small
// whitespace-delimited key/value text file (see
// `practical/datasets/<instance>_perfect_information_lp.txt`) so the crate
// stays dependency-free of any JSON parser.
//
// FORMAT (one entry per non-comment line, `#` starts a comment line):
//   key = v1 v2 ...            scalar or whitespace-separated f64 list
//   expected_revenue[p] = ...  per-product revenue grid (aligned to step)
//   slope[p] = ...             per-product slope grid (aligned to revenue)
//   allowBlending = 0|1
//   blendingRange = none | <usize>
//   ageRange = none | <usize ...>   (flattened; not used by carried instances)
//   published_max_reward = <f64>     the literature anchor to reproduce
//
// The loader returns both the LP inputs and the published anchor values so the
// verification test can re-run the LP and assert reproduction.
// ============================================================================

use crate::problems::ameliorating_inventory::perfect_information_lp::PerfectInformationLpInputs;

/// Published companion-repo anchor values for one instance.
#[derive(Clone, Debug)]
pub struct PublishedLpAnchor {
    pub instance: String,
    pub max_reward: f64,
    pub purchasing: f64,
    pub production: Vec<f64>,
    pub inventory_position: Vec<f64>,
}

/// LP inputs plus the published anchor parsed from one dataset file.
#[derive(Clone, Debug)]
pub struct LoadedLpDataset {
    pub inputs: PerfectInformationLpInputs,
    pub anchor: PublishedLpAnchor,
}

fn parse_f64_list(value: &str) -> Vec<f64> {
    value
        .split_whitespace()
        .map(|t| t.parse::<f64>().expect("dataset value must parse as f64"))
        .collect()
}

/// Parse one dataset text blob into LP inputs + published anchor.
pub fn parse_lp_dataset(text: &str) -> LoadedLpDataset {
    let mut instance = String::new();
    let mut num_ages = 0usize;
    let mut num_products = 0usize;
    let mut target_ages: Vec<usize> = Vec::new();
    let mut max_inventory = 0.0f64;
    let mut evaporation = 0.0f64;
    let mut decay_mean: Vec<f64> = Vec::new();
    let mut holding_costs = 0.0f64;
    let mut outdating_costs = 0.0f64;
    let mut decay_salvage: Vec<f64> = Vec::new();
    let mut allow_blending = false;
    let mut blending_range: Option<usize> = None;
    let mut age_range: Option<Vec<Vec<usize>>> = None;
    let mut price_mean = 0.0f64;
    let mut price_std = 0.0f64;
    let mut price_truncation = 0.0f64;
    let mut production_step_size = 0.0f64;
    let mut sales_bound: Vec<f64> = Vec::new();
    let expected_revenue: Vec<Vec<f64>>;
    let slope: Vec<Vec<f64>>;
    let mut published_max_reward = 0.0f64;
    let mut published_purchasing = 0.0f64;
    let mut published_production: Vec<f64> = Vec::new();
    let mut published_inventory_position: Vec<f64> = Vec::new();

    // collect indexed entries first, then assemble in order
    let mut er_indexed: Vec<(usize, Vec<f64>)> = Vec::new();
    let mut slope_indexed: Vec<(usize, Vec<f64>)> = Vec::new();

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (key, value) = match line.split_once('=') {
            Some((k, v)) => (k.trim(), v.trim()),
            None => continue,
        };
        match key {
            "instance" => instance = value.to_string(),
            "numAges" => num_ages = value.parse().unwrap(),
            "nProducts" => num_products = value.parse().unwrap(),
            "targetAges" => {
                target_ages = parse_f64_list(value).iter().map(|x| x.round() as usize).collect()
            }
            "maxInventory" => max_inventory = value.parse().unwrap(),
            "evaporation" => evaporation = value.parse().unwrap(),
            "decay_mean" => decay_mean = parse_f64_list(value),
            "holdingCosts" => holding_costs = value.parse().unwrap(),
            "outdatingCosts" => outdating_costs = value.parse().unwrap(),
            "decaySalvage" => decay_salvage = parse_f64_list(value),
            "allowBlending" => allow_blending = value == "1",
            "blendingRange" => {
                blending_range = if value == "none" {
                    None
                } else {
                    Some(value.parse().unwrap())
                }
            }
            "ageRange" => {
                age_range = if value == "none" {
                    None
                } else {
                    // single-product flattened fallback (unused by carried instances)
                    Some(vec![parse_f64_list(value)
                        .iter()
                        .map(|x| x.round() as usize)
                        .collect()])
                }
            }
            "price_mean" => price_mean = value.parse().unwrap(),
            "price_std" => price_std = value.parse().unwrap(),
            "price_truncation" => price_truncation = value.parse().unwrap(),
            "production_step_size" => production_step_size = value.parse().unwrap(),
            "sales_bound" => sales_bound = parse_f64_list(value),
            "published_max_reward" => published_max_reward = value.parse().unwrap(),
            "published_purchasing" => published_purchasing = value.parse().unwrap(),
            "published_production" => published_production = parse_f64_list(value),
            "published_inventory_position" => {
                published_inventory_position = parse_f64_list(value)
            }
            other => {
                if let Some(idx) = parse_indexed(other, "expected_revenue") {
                    er_indexed.push((idx, parse_f64_list(value)));
                } else if let Some(idx) = parse_indexed(other, "slope") {
                    slope_indexed.push((idx, parse_f64_list(value)));
                }
            }
        }
    }

    er_indexed.sort_by_key(|(i, _)| *i);
    slope_indexed.sort_by_key(|(i, _)| *i);
    expected_revenue = er_indexed.into_iter().map(|(_, v)| v).collect();
    slope = slope_indexed.into_iter().map(|(_, v)| v).collect();

    let inputs = PerfectInformationLpInputs {
        instance: instance.clone(),
        num_ages,
        num_products,
        target_ages,
        max_inventory,
        evaporation,
        decay_mean,
        holding_costs,
        outdating_costs,
        decay_salvage,
        allow_blending,
        blending_range,
        age_range,
        price_mean,
        price_std,
        price_truncation,
        production_step_size,
        sales_bound,
        expected_revenue,
        slope,
    };
    let anchor = PublishedLpAnchor {
        instance,
        max_reward: published_max_reward,
        purchasing: published_purchasing,
        production: published_production,
        inventory_position: published_inventory_position,
    };
    LoadedLpDataset { inputs, anchor }
}

/// Match a key like `expected_revenue[2]` and return its index `2`.
fn parse_indexed(key: &str, prefix: &str) -> Option<usize> {
    let rest = key.strip_prefix(prefix)?;
    let inner = rest.strip_prefix('[')?.strip_suffix(']')?;
    inner.parse().ok()
}

/// Embedded spirits_0001 dataset (10 ages, 3 products, no blending).
pub const SPIRITS_0001_DATASET: &str =
    include_str!("practical/datasets/spirits_0001_perfect_information_lp.txt");

/// Embedded port_wine dataset (25 ages, 2 products, blending enabled).
pub const PORT_WINE_DATASET: &str =
    include_str!("practical/datasets/port_wine_perfect_information_lp.txt");

/// Embedded spirits_0002 dataset (= spirits_0001 with blending ENABLED).
/// Price/demand/sales params are identical to spirits_0001, so the
/// expected_revenue/slope tables are byte-identical (companion
/// expected_revenue.json md5 matches across the three spirits instances).
pub const SPIRITS_0002_DATASET: &str =
    include_str!("practical/datasets/spirits_0002_perfect_information_lp.txt");

/// Embedded spirits_1002 dataset (= spirits_0002 with maxInventory 30, the
/// processing-capacity-constrained companion variant).
pub const SPIRITS_1002_DATASET: &str =
    include_str!("practical/datasets/spirits_1002_perfect_information_lp.txt");

/// Convenience: parse the spirits_0001 verification anchor.
pub fn load_spirits_0001() -> LoadedLpDataset {
    parse_lp_dataset(SPIRITS_0001_DATASET)
}

/// Convenience: parse the port_wine verification anchor.
pub fn load_port_wine() -> LoadedLpDataset {
    parse_lp_dataset(PORT_WINE_DATASET)
}

/// Convenience: parse the spirits_0002 (blending ON) verification anchor.
pub fn load_spirits_0002() -> LoadedLpDataset {
    parse_lp_dataset(SPIRITS_0002_DATASET)
}

/// Convenience: parse the spirits_1002 (capacity-constrained) verification anchor.
pub fn load_spirits_1002() -> LoadedLpDataset {
    parse_lp_dataset(SPIRITS_1002_DATASET)
}
