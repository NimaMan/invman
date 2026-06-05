#![allow(dead_code)]

use crate::problems::perishable_inventory::env::{
    step_state, IssuingPolicy, PerishableState, PerishableStepOutcome,
};

fn worked_state() -> PerishableState {
    PerishableState {
        on_hand: vec![2, 1, 3],
        pipeline_orders: vec![4],
    }
}

fn fifo_outcome() -> PerishableStepOutcome {
    step_state(
        &worked_state(),
        5,
        4,
        1.0,
        5.0,
        7.0,
        3.0,
        IssuingPolicy::Fifo,
    )
}

fn lifo_outcome() -> PerishableStepOutcome {
    step_state(
        &worked_state(),
        5,
        4,
        1.0,
        5.0,
        7.0,
        3.0,
        IssuingPolicy::Lifo,
    )
}

pub fn verify_fifo_lifo_step_semantics() -> bool {
    let fifo = fifo_outcome();
    let lifo = lifo_outcome();

    fifo.shortage == 0
        && lifo.shortage == 0
        && fifo.waste == 0
        && lifo.waste == 2
        && fifo.holding_inventory == 2
        && lifo.holding_inventory == 0
        && fifo.next_state.on_hand == vec![4, 2, 0]
        && lifo.next_state.on_hand == vec![4, 0, 0]
        && fifo.next_state.pipeline_orders == vec![5]
        && lifo.next_state.pipeline_orders == vec![5]
        && (fifo.cost - 17.0).abs() <= 1e-9
        && (lifo.cost - 29.0).abs() <= 1e-9
}

#[cfg(test)]
mod tests {
    use super::{fifo_outcome, lifo_outcome, verify_fifo_lifo_step_semantics};

    #[test]
    fn fifo_and_lifo_step_semantics_match_expected_outcomes() {
        assert!(verify_fifo_lifo_step_semantics());
    }

    #[test]
    fn fifo_and_lifo_have_different_waste_paths() {
        let fifo = fifo_outcome();
        let lifo = lifo_outcome();

        assert_eq!(fifo.waste, 0);
        assert_eq!(lifo.waste, 2);
        assert_ne!(fifo.next_state.on_hand, lifo.next_state.on_hand);
    }
}
