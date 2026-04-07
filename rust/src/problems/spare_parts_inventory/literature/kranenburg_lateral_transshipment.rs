use crate::problems::spare_parts_inventory::references::KranenburgLateralTransshipmentReferenceInstance;

pub const KRANENBURG_TABLE_ROUNDING_TOLERANCE: f64 = 0.02;
const MAX_SITUATION12_STOCK: usize = 1024;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KranenburgSituationSummary {
    pub optimal_r: f64,
    pub emergency_probability: f64,
    pub mean_waiting_time: f64,
    pub transport_cost_per_request: f64,
    pub total_cost: f64,
    pub waiting_constraint_binding: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KranenburgBenchmarkEvaluation {
    pub situation1: KranenburgSituationSummary,
    pub situation2: Option<KranenburgSituationSummary>,
    pub situation3: KranenburgSituationSummary,
    pub cost_ratio_situation1_over_situation3: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KranenburgPublishedComparison {
    pub tolerance: f64,
    pub situation1_optimal_r_abs_diff: f64,
    pub situation1_cost_abs_diff: f64,
    pub situation3_optimal_r_abs_diff: f64,
    pub situation3_cost_abs_diff: f64,
    pub cost_ratio_abs_diff: f64,
    pub matches_situation1_optimal_r: bool,
    pub matches_situation1_cost: bool,
    pub matches_situation3_optimal_r: bool,
    pub matches_situation3_cost: bool,
    pub matches_cost_ratio: bool,
    pub all_within_tolerance: bool,
}

pub fn evaluate_reference_instance(
    reference: &KranenburgLateralTransshipmentReferenceInstance,
) -> Result<KranenburgBenchmarkEvaluation, String> {
    let situation1 = solve_situation1(reference)?;
    let situation2 = solve_situation2(reference).ok();
    let situation3 = solve_situation3(reference)?;
    Ok(KranenburgBenchmarkEvaluation {
        situation1,
        situation2,
        situation3,
        cost_ratio_situation1_over_situation3: situation1.total_cost / situation3.total_cost,
    })
}

pub fn compare_to_published_table(
    reference: &KranenburgLateralTransshipmentReferenceInstance,
    evaluation: &KranenburgBenchmarkEvaluation,
    tolerance: f64,
) -> KranenburgPublishedComparison {
    let situation1_optimal_r_abs_diff =
        (evaluation.situation1.optimal_r - reference.published_situation1_optimal_r).abs();
    let situation1_cost_abs_diff =
        (evaluation.situation1.total_cost - reference.published_situation1_cost).abs();
    let situation3_optimal_r_abs_diff =
        (evaluation.situation3.optimal_r - reference.published_situation3_optimal_r).abs();
    let situation3_cost_abs_diff =
        (evaluation.situation3.total_cost - reference.published_situation3_cost).abs();
    let cost_ratio_abs_diff = (evaluation.cost_ratio_situation1_over_situation3
        - reference.published_cost_ratio_situation1_over_situation3)
        .abs();
    let matches_situation1_optimal_r = situation1_optimal_r_abs_diff <= tolerance;
    let matches_situation1_cost = situation1_cost_abs_diff <= tolerance;
    let matches_situation3_optimal_r = situation3_optimal_r_abs_diff <= tolerance;
    let matches_situation3_cost = situation3_cost_abs_diff <= tolerance;
    let matches_cost_ratio = cost_ratio_abs_diff <= tolerance;
    KranenburgPublishedComparison {
        tolerance,
        situation1_optimal_r_abs_diff,
        situation1_cost_abs_diff,
        situation3_optimal_r_abs_diff,
        situation3_cost_abs_diff,
        cost_ratio_abs_diff,
        matches_situation1_optimal_r,
        matches_situation1_cost,
        matches_situation3_optimal_r,
        matches_situation3_cost,
        matches_cost_ratio,
        all_within_tolerance: matches_situation1_optimal_r
            && matches_situation1_cost
            && matches_situation3_optimal_r
            && matches_situation3_cost
            && matches_cost_ratio,
    }
}

fn solve_situation1(
    reference: &KranenburgLateralTransshipmentReferenceInstance,
) -> Result<KranenburgSituationSummary, String> {
    let local_demand_work =
        reference.demand_rate_per_local_warehouse * reference.regular_replenishment_time;
    let total_demand_rate = total_demand_rate(reference);
    let mut best: Option<KranenburgSituationSummary> = None;

    for local_stock in 0..MAX_SITUATION12_STOCK {
        let low = local_stock as f64;
        let high = low + 1.0;
        let emergency_probability_low =
            poisson_lost_sales_probability(local_demand_work, local_stock);
        let emergency_probability_high =
            poisson_lost_sales_probability(local_demand_work, local_stock + 1);
        let waiting_time_low = reference.emergency_time * emergency_probability_low;
        let waiting_time_high = reference.emergency_time * emergency_probability_high;
        if waiting_time_high > reference.waiting_time_target {
            continue;
        }

        let feasible_start = if waiting_time_low <= reference.waiting_time_target {
            low
        } else {
            linear_boundary(
                low,
                waiting_time_low,
                high,
                waiting_time_high,
                reference.waiting_time_target,
            )?
        };
        let slope = reference.num_local_warehouses as f64 * reference.holding_cost
            + reference.emergency_cost
                * total_demand_rate
                * (emergency_probability_high - emergency_probability_low);
        let chosen_local_stock = if slope >= 0.0 { feasible_start } else { high };
        let summary = summarize_situation1(
            reference,
            chosen_local_stock,
            emergency_probability_low,
            emergency_probability_high,
            low,
        );
        update_best(&mut best, summary);
        if slope >= 0.0 {
            break;
        }
    }

    best.ok_or_else(|| {
        format!(
            "no feasible situation-1 policy found for Kranenburg instance '{}'",
            reference.name
        )
    })
}

fn solve_situation2(
    reference: &KranenburgLateralTransshipmentReferenceInstance,
) -> Result<KranenburgSituationSummary, String> {
    if reference.waiting_time_target <= reference.joint_warehouse_time {
        return Err(format!(
            "situation 2 is infeasible for '{}': waiting target {:.4} is below joint-warehouse time {:.4}",
            reference.name, reference.waiting_time_target, reference.joint_warehouse_time
        ));
    }

    let pooled_demand_work = total_demand_rate(reference) * reference.regular_replenishment_time;
    let total_demand_rate = total_demand_rate(reference);
    let mut best: Option<KranenburgSituationSummary> = None;

    for stock in 0..MAX_SITUATION12_STOCK {
        let low = stock as f64;
        let high = low + 1.0;
        let emergency_probability_low = poisson_lost_sales_probability(pooled_demand_work, stock);
        let emergency_probability_high =
            poisson_lost_sales_probability(pooled_demand_work, stock + 1);
        let waiting_time_low = situation2_waiting_time(reference, low, emergency_probability_low);
        let waiting_time_high =
            situation2_waiting_time(reference, high, emergency_probability_high);
        if waiting_time_high > reference.waiting_time_target {
            continue;
        }

        let feasible_start = if waiting_time_low <= reference.waiting_time_target {
            low
        } else {
            linear_boundary(
                low,
                waiting_time_low,
                high,
                waiting_time_high,
                reference.waiting_time_target,
            )?
        };
        let slope = reference.holding_cost
            + total_demand_rate
                * (reference.emergency_cost - reference.joint_warehouse_cost)
                * (emergency_probability_high - emergency_probability_low);
        let chosen_stock = if slope >= 0.0 { feasible_start } else { high };
        let summary = summarize_situation2(
            reference,
            chosen_stock,
            emergency_probability_low,
            emergency_probability_high,
            low,
        );
        update_best(&mut best, summary);
        if slope >= 0.0 {
            break;
        }
    }

    best.ok_or_else(|| {
        format!(
            "no feasible situation-2 policy found for Kranenburg instance '{}'",
            reference.name
        )
    })
}

fn solve_situation3(
    reference: &KranenburgLateralTransshipmentReferenceInstance,
) -> Result<KranenburgSituationSummary, String> {
    let pooled_demand_work = total_demand_rate(reference) * reference.regular_replenishment_time;
    let total_demand_rate = total_demand_rate(reference);
    let mut best: Option<KranenburgSituationSummary> = None;

    for stock in 0..reference.num_local_warehouses {
        let low = stock as f64;
        let high = low + 1.0;
        let emergency_probability_low = poisson_lost_sales_probability(pooled_demand_work, stock);
        let emergency_probability_high =
            poisson_lost_sales_probability(pooled_demand_work, stock + 1);
        let waiting_time_low = situation3_waiting_time(reference, low, emergency_probability_low);
        let waiting_time_high =
            situation3_waiting_time(reference, high, emergency_probability_high);
        if waiting_time_high > reference.waiting_time_target {
            continue;
        }

        let feasible_start = if waiting_time_low <= reference.waiting_time_target {
            low
        } else {
            linear_boundary(
                low,
                waiting_time_low,
                high,
                waiting_time_high,
                reference.waiting_time_target,
            )?
        };
        let slope = reference.holding_cost
            - total_demand_rate * reference.lateral_transshipment_cost
                / reference.num_local_warehouses as f64
            + total_demand_rate
                * (reference.emergency_cost
                    - (1.0
                        + reference.demand_rate_per_local_warehouse
                            * reference.regular_replenishment_time)
                        * reference.lateral_transshipment_cost)
                * (emergency_probability_high - emergency_probability_low);
        let chosen_stock = if slope >= 0.0 { feasible_start } else { high };
        let summary = summarize_situation3(
            reference,
            chosen_stock,
            emergency_probability_low,
            emergency_probability_high,
            low,
        );
        update_best(&mut best, summary);
        if slope >= 0.0 {
            break;
        }
    }

    best.ok_or_else(|| {
        format!(
            "no feasible situation-3 policy found for Kranenburg instance '{}'; the published Chapter 5 model only covers 0 <= R <= |J|",
            reference.name
        )
    })
}

fn summarize_situation1(
    reference: &KranenburgLateralTransshipmentReferenceInstance,
    chosen_local_stock: f64,
    emergency_probability_low: f64,
    emergency_probability_high: f64,
    interval_low: f64,
) -> KranenburgSituationSummary {
    let interpolation_weight = chosen_local_stock - interval_low;
    let emergency_probability = interpolate(
        emergency_probability_low,
        emergency_probability_high,
        interpolation_weight,
    );
    let optimal_r = chosen_local_stock * reference.num_local_warehouses as f64;
    let mean_waiting_time = reference.emergency_time * emergency_probability;
    let transport_cost_per_request = reference.emergency_cost * emergency_probability;
    KranenburgSituationSummary {
        optimal_r,
        emergency_probability,
        mean_waiting_time,
        transport_cost_per_request,
        total_cost: reference.holding_cost * optimal_r
            + transport_cost_per_request * total_demand_rate(reference),
        waiting_constraint_binding: is_binding(mean_waiting_time, reference.waiting_time_target),
    }
}

fn summarize_situation2(
    reference: &KranenburgLateralTransshipmentReferenceInstance,
    chosen_stock: f64,
    emergency_probability_low: f64,
    emergency_probability_high: f64,
    interval_low: f64,
) -> KranenburgSituationSummary {
    let interpolation_weight = chosen_stock - interval_low;
    let emergency_probability = interpolate(
        emergency_probability_low,
        emergency_probability_high,
        interpolation_weight,
    );
    let mean_waiting_time = situation2_waiting_time(reference, chosen_stock, emergency_probability);
    let transport_cost_per_request = reference.joint_warehouse_cost
        + emergency_probability * (reference.emergency_cost - reference.joint_warehouse_cost);
    KranenburgSituationSummary {
        optimal_r: chosen_stock,
        emergency_probability,
        mean_waiting_time,
        transport_cost_per_request,
        total_cost: reference.holding_cost * chosen_stock
            + transport_cost_per_request * total_demand_rate(reference),
        waiting_constraint_binding: is_binding(mean_waiting_time, reference.waiting_time_target),
    }
}

fn summarize_situation3(
    reference: &KranenburgLateralTransshipmentReferenceInstance,
    chosen_stock: f64,
    emergency_probability_low: f64,
    emergency_probability_high: f64,
    interval_low: f64,
) -> KranenburgSituationSummary {
    let interpolation_weight = chosen_stock - interval_low;
    let emergency_probability = interpolate(
        emergency_probability_low,
        emergency_probability_high,
        interpolation_weight,
    );
    let mean_waiting_time = situation3_waiting_time(reference, chosen_stock, emergency_probability);
    let transport_cost_per_request = (1.0
        + reference.demand_rate_per_local_warehouse * reference.regular_replenishment_time)
        * reference.lateral_transshipment_cost
        - chosen_stock * reference.lateral_transshipment_cost
            / reference.num_local_warehouses as f64
        + emergency_probability
            * (reference.emergency_cost
                - (1.0
                    + reference.demand_rate_per_local_warehouse
                        * reference.regular_replenishment_time)
                    * reference.lateral_transshipment_cost);
    KranenburgSituationSummary {
        optimal_r: chosen_stock,
        emergency_probability,
        mean_waiting_time,
        transport_cost_per_request,
        total_cost: reference.holding_cost * chosen_stock
            + transport_cost_per_request * total_demand_rate(reference),
        waiting_constraint_binding: is_binding(mean_waiting_time, reference.waiting_time_target),
    }
}

fn situation2_waiting_time(
    reference: &KranenburgLateralTransshipmentReferenceInstance,
    _chosen_stock: f64,
    emergency_probability: f64,
) -> f64 {
    reference.joint_warehouse_time
        + emergency_probability * (reference.emergency_time - reference.joint_warehouse_time)
}

fn situation3_waiting_time(
    reference: &KranenburgLateralTransshipmentReferenceInstance,
    chosen_stock: f64,
    emergency_probability: f64,
) -> f64 {
    (1.0 + reference.demand_rate_per_local_warehouse * reference.regular_replenishment_time)
        * reference.lateral_transshipment_time
        - chosen_stock * reference.lateral_transshipment_time
            / reference.num_local_warehouses as f64
        + emergency_probability
            * (reference.emergency_time
                - (1.0
                    + reference.demand_rate_per_local_warehouse
                        * reference.regular_replenishment_time)
                    * reference.lateral_transshipment_time)
}

fn total_demand_rate(reference: &KranenburgLateralTransshipmentReferenceInstance) -> f64 {
    reference.demand_rate_per_local_warehouse * reference.num_local_warehouses as f64
}

fn poisson_lost_sales_probability(workload: f64, stock: usize) -> f64 {
    let mut denominator = 1.0;
    let mut term = 1.0;
    for k in 1..=stock {
        term *= workload / k as f64;
        denominator += term;
    }
    term / denominator
}

fn interpolate(low: f64, high: f64, weight: f64) -> f64 {
    low + weight * (high - low)
}

fn linear_boundary(
    low_x: f64,
    low_y: f64,
    high_x: f64,
    high_y: f64,
    target_y: f64,
) -> Result<f64, String> {
    let denominator = low_y - high_y;
    if denominator.abs() < 1e-12 {
        return Err("degenerate linear interval while solving Kranenburg boundary".to_string());
    }
    Ok(low_x + (low_y - target_y) * (high_x - low_x) / denominator)
}

fn update_best(
    best: &mut Option<KranenburgSituationSummary>,
    candidate: KranenburgSituationSummary,
) {
    match best {
        Some(current) if current.total_cost <= candidate.total_cost => {}
        _ => *best = Some(candidate),
    }
}

fn is_binding(value: f64, target: f64) -> bool {
    (value - target).abs() <= 1e-9
}
