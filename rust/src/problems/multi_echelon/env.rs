use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

#[derive(Clone, Debug)]
pub struct MultiEchelonState {
    pub warehouse_inventory: i64,
    pub warehouse_pipeline: Vec<usize>,
    pub retailer_inventory: Vec<i64>,
    pub retailer_pipeline: Vec<Vec<usize>>,
}

pub fn initialize_state(
    warehouse_lead_time: usize,
    retailer_lead_time: usize,
    num_retailers: usize,
    warehouse_levels: &[usize],
    retailer_levels: &[usize],
    demand_mean: f64,
    seed: u64,
) -> MultiEchelonState {
    let mut rng = StdRng::seed_from_u64(seed);
    let warehouse_level = *warehouse_levels.last().unwrap_or(&100);
    let retailer_level = *retailer_levels.last().unwrap_or(&40);
    let warehouse_pipeline = (0..warehouse_lead_time)
        .map(|_| {
            *warehouse_levels
                .get(rng.gen_range(0..warehouse_levels.len()))
                .unwrap_or(&warehouse_level)
        })
        .collect::<Vec<_>>();
    let retailer_pipeline = (0..num_retailers)
        .map(|_| {
            (0..retailer_lead_time)
                .map(|_| {
                    *retailer_levels
                        .get(rng.gen_range(0..retailer_levels.len()))
                        .unwrap_or(&retailer_level)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    MultiEchelonState {
        warehouse_inventory: (num_retailers as f64 * demand_mean.max(1.0)).round() as i64,
        warehouse_pipeline,
        retailer_inventory: vec![demand_mean.max(1.0).round() as i64; num_retailers],
        retailer_pipeline,
    }
}

pub fn flattened_policy_state(
    state: &MultiEchelonState,
    warehouse_inventory_cap: usize,
    retailer_inventory_cap: usize,
) -> Vec<f32> {
    let warehouse_available = state.warehouse_inventory + state.warehouse_pipeline[0] as i64;
    let mut output = vec![warehouse_available as f32 / warehouse_inventory_cap.max(1) as f32];
    for value in state.warehouse_pipeline.iter().copied().skip(1) {
        output.push(value as f32 / warehouse_inventory_cap.max(1) as f32);
    }
    for (retailer_idx, inventory) in state.retailer_inventory.iter().copied().enumerate() {
        let available = inventory + state.retailer_pipeline[retailer_idx][0] as i64;
        output.push(available as f32 / retailer_inventory_cap.max(1) as f32);
    }
    for retailer_idx in 0..state.retailer_inventory.len() {
        for value in state.retailer_pipeline[retailer_idx]
            .iter()
            .copied()
            .skip(1)
        {
            output.push(value as f32 / retailer_inventory_cap.max(1) as f32);
        }
    }
    output
}
