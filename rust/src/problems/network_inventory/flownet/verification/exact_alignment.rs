#[cfg(test)]
mod tests {
    use crate::problems::network_inventory::finite_horizon_dp::{
        evaluate_named_heuristic, solve_optimal_policy,
    };
    use crate::problems::network_inventory::references::VERIFICATION_PROBLEM_INSTANCE;

    #[test]
    fn exact_dp_and_node_base_stock_match_reference_numbers() {
        let optimal = solve_optimal_policy(&VERIFICATION_PROBLEM_INSTANCE)
            .expect("exact optimal policy must solve");
        let base_stock =
            evaluate_named_heuristic(&VERIFICATION_PROBLEM_INSTANCE, "node_base_stock")
                .expect("node-base-stock evaluation must solve");

        assert!(
            (optimal.discounted_cost
                - VERIFICATION_PROBLEM_INSTANCE.expected_optimal_discounted_cost)
                .abs()
                < 1e-9
        );
        assert_eq!(
            optimal.first_action,
            VERIFICATION_PROBLEM_INSTANCE
                .expected_optimal_first_action
                .to_vec()
        );
        assert!(
            (base_stock.discounted_cost
                - VERIFICATION_PROBLEM_INSTANCE.expected_base_stock_discounted_cost)
                .abs()
                < 1e-9
        );
        assert_eq!(
            base_stock.first_action,
            VERIFICATION_PROBLEM_INSTANCE
                .expected_base_stock_first_action
                .to_vec()
        );
    }
}
