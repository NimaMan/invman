use crate::problems::perishable_inventory::env::{IssuingPolicy, PerishableState};
use crate::problems::perishable_inventory::heuristics::bsp_low_ew_order_quantity;

#[test]
fn bsp_low_ew_low_branch_uses_alpha_weighting() {
    let state = PerishableState {
        on_hand: vec![0, 0, 0],
        pipeline_orders: vec![3],
    };

    let order_quantity =
        bsp_low_ew_order_quantity(&state, 2, 5, 6, 10, 2, 4.0, IssuingPolicy::Fifo);

    assert_eq!(order_quantity, 1);
}
