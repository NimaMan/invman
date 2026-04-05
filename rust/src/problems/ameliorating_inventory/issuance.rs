#[derive(Clone, Debug, PartialEq)]
pub struct IssuancePlan {
    pub shipments_by_product_age: Vec<Vec<usize>>,
    pub shipped_by_product: Vec<usize>,
    pub lost_sales_by_product: Vec<usize>,
    pub revenue: f64,
    pub age_excess: usize,
}

#[derive(Clone, Debug, PartialEq)]
struct CandidatePlan {
    shipments_by_product_age: Vec<Vec<usize>>,
    revenue: f64,
    shipped_units: usize,
    age_excess: usize,
}

fn enumerate_product_allocations(
    remaining_inventory: &[usize],
    demand: usize,
    target_age: usize,
    age_idx: usize,
    current_allocation: &mut Vec<usize>,
    current_units: usize,
    current_weighted_age: usize,
    allocations: &mut Vec<Vec<usize>>,
) {
    if age_idx == remaining_inventory.len() {
        if current_units == 0 || current_weighted_age >= target_age * current_units {
            allocations.push(current_allocation.clone());
        }
        return;
    }

    let max_quantity = remaining_inventory[age_idx].min(demand.saturating_sub(current_units));
    for quantity in 0..=max_quantity {
        current_allocation[age_idx] = quantity;
        enumerate_product_allocations(
            remaining_inventory,
            demand,
            target_age,
            age_idx + 1,
            current_allocation,
            current_units + quantity,
            current_weighted_age + age_idx * quantity,
            allocations,
        );
    }
    current_allocation[age_idx] = 0;
}

fn feasible_allocations_for_product(
    remaining_inventory: &[usize],
    demand: usize,
    target_age: usize,
) -> Vec<Vec<usize>> {
    let mut allocations = Vec::new();
    let mut current_allocation = vec![0usize; remaining_inventory.len()];
    enumerate_product_allocations(
        remaining_inventory,
        demand,
        target_age,
        0,
        &mut current_allocation,
        0,
        0,
        &mut allocations,
    );
    allocations
}

fn age_excess(allocation: &[usize], target_age: usize) -> usize {
    allocation
        .iter()
        .enumerate()
        .map(|(age, quantity)| age.saturating_sub(target_age) * quantity)
        .sum()
}

fn better_candidate(candidate: &CandidatePlan, incumbent: Option<&CandidatePlan>) -> bool {
    match incumbent {
        None => true,
        Some(best) => {
            if candidate.revenue > best.revenue + 1e-12 {
                return true;
            }
            if (candidate.revenue - best.revenue).abs() <= 1e-12
                && candidate.shipped_units > best.shipped_units
            {
                return true;
            }
            (candidate.revenue - best.revenue).abs() <= 1e-12
                && candidate.shipped_units == best.shipped_units
                && candidate.age_excess < best.age_excess
        }
    }
}

fn search_best_plan(
    product_idx: usize,
    remaining_inventory: &[usize],
    realized_demands: &[usize],
    target_ages: &[usize],
    product_prices: &[f64],
) -> CandidatePlan {
    if product_idx == realized_demands.len() {
        return CandidatePlan {
            shipments_by_product_age: Vec::new(),
            revenue: 0.0,
            shipped_units: 0,
            age_excess: 0,
        };
    }

    let mut best_plan: Option<CandidatePlan> = None;
    for allocation in feasible_allocations_for_product(
        remaining_inventory,
        realized_demands[product_idx],
        target_ages[product_idx],
    ) {
        let mut next_inventory = remaining_inventory.to_vec();
        for age in 0..remaining_inventory.len() {
            next_inventory[age] -= allocation[age];
        }
        let suffix_plan = search_best_plan(
            product_idx + 1,
            &next_inventory,
            realized_demands,
            target_ages,
            product_prices,
        );
        let shipped_units = allocation.iter().sum::<usize>();
        let mut shipments_by_product_age = Vec::with_capacity(suffix_plan.shipments_by_product_age.len() + 1);
        shipments_by_product_age.push(allocation.clone());
        shipments_by_product_age.extend(suffix_plan.shipments_by_product_age.iter().cloned());
        let candidate = CandidatePlan {
            shipments_by_product_age,
            revenue: shipped_units as f64 * product_prices[product_idx] + suffix_plan.revenue,
            shipped_units: shipped_units + suffix_plan.shipped_units,
            age_excess: age_excess(&allocation, target_ages[product_idx]) + suffix_plan.age_excess,
        };
        if better_candidate(&candidate, best_plan.as_ref()) {
            best_plan = Some(candidate);
        }
    }

    best_plan.expect("the zero-allocation candidate always exists")
}

pub fn optimize_average_age_blending(
    inventory_by_age: &[usize],
    realized_demands: &[usize],
    target_ages: &[usize],
    product_prices: &[f64],
) -> IssuancePlan {
    let candidate =
        search_best_plan(0, inventory_by_age, realized_demands, target_ages, product_prices);
    let shipped_by_product = candidate
        .shipments_by_product_age
        .iter()
        .map(|allocations| allocations.iter().sum::<usize>())
        .collect::<Vec<_>>();
    let lost_sales_by_product = realized_demands
        .iter()
        .zip(shipped_by_product.iter())
        .map(|(demand, shipped)| demand.saturating_sub(*shipped))
        .collect::<Vec<_>>();

    IssuancePlan {
        shipments_by_product_age: candidate.shipments_by_product_age,
        shipped_by_product,
        lost_sales_by_product,
        revenue: candidate.revenue,
        age_excess: candidate.age_excess,
    }
}
