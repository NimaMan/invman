use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::problems::multi_echelon::general_backorder_fixed_cost::env::{
    advance_to_decision_state, apply_next_orders, build_raw_state, incoming_retail_edge_indices,
    initialize_zero_state, retailer_total_inventory_positions, warehouse_inventory_positions,
    GeneralBackorderFixedCostNetwork,
};
use crate::problems::multi_echelon::general_backorder_fixed_cost::heuristics::{
    node_base_stock_orders, parse_benchmark_order_routing_mode, simulate_node_base_stock_policy,
};
use crate::problems::multi_echelon::general_backorder_fixed_cost::references::{
    CBC_SUPPLIER_LEAD_TIMES, GEEVERS_SET1_BASE_STOCK_LEVELS, GEEVERS_SET23_BASE_STOCK_LEVELS,
    LITERATURE_REFERENCE_INSTANCES, PRIMARY_REFERENCE_INSTANCE,
};

fn benchmark_network() -> GeneralBackorderFixedCostNetwork {
    let reference = PRIMARY_REFERENCE_INSTANCE;
    GeneralBackorderFixedCostNetwork {
        num_suppliers: reference.num_suppliers,
        num_warehouses: reference.num_warehouses,
        num_retailers: reference.num_retailers,
        supplier_lead_times: reference.supplier_lead_times.to_vec(),
        retail_edges: reference.retail_edges.to_vec(),
    }
}

#[test]
fn literature_catalog_matches_paper_rows() {
    assert_eq!(LITERATURE_REFERENCE_INSTANCES.len(), 3);
    assert_eq!(
        LITERATURE_REFERENCE_INSTANCES[0].published_benchmark_cost,
        10_467.0
    );
    assert_eq!(
        LITERATURE_REFERENCE_INSTANCES[1].published_benchmark_cost,
        4_797.0
    );
    assert_eq!(
        LITERATURE_REFERENCE_INSTANCES[2].published_benchmark_cost,
        4_797.0
    );
    assert_eq!(
        LITERATURE_REFERENCE_INSTANCES[0].benchmark_base_stock_levels,
        GEEVERS_SET1_BASE_STOCK_LEVELS
    );
    assert_eq!(
        LITERATURE_REFERENCE_INSTANCES[1].benchmark_base_stock_levels,
        GEEVERS_SET23_BASE_STOCK_LEVELS
    );
}

#[test]
fn zero_state_has_expected_dimensions() {
    let network = benchmark_network();
    let state = initialize_zero_state(&network).unwrap();
    let raw = build_raw_state(&network, &state).unwrap();
    assert_eq!(state.warehouse_inventory.len(), network.num_warehouses);
    assert_eq!(state.retailer_inventory.len(), network.num_retailers);
    assert_eq!(state.retailer_orders_due.len(), network.retail_edges.len());
    // raw layout (build_raw_state): 4*W (inventory, supplier_orders_due, supplier_deliveries_due,
    // supplier_in_transit) + 2*R (inventory, customer_backorders) + 4*E (retailer orders_due,
    // deliveries_due, in_transit, backorders). The benchmark network's audit-metric fields make
    // this 82 for the CardBoard Company instance; assert the layout formula so it cannot go stale.
    assert_eq!(
        raw.len(),
        4 * network.num_warehouses + 2 * network.num_retailers + 4 * network.retail_edges.len()
    );
}

#[test]
fn benchmark_edges_cover_each_retailer_with_unit_weight_sum() {
    let network = benchmark_network();
    for retailer_idx in 0..network.num_retailers {
        let incoming = incoming_retail_edge_indices(&network, retailer_idx);
        let total_weight = incoming
            .iter()
            .map(|edge_idx| network.retail_edges[*edge_idx].connection_weight)
            .sum::<f64>();
        assert!((total_weight - 1.0).abs() < 1e-12);
    }
    assert_eq!(CBC_SUPPLIER_LEAD_TIMES, &[1, 1, 1, 1]);
}

#[test]
fn one_step_dynamics_receive_shipments_before_new_orders() {
    let network = benchmark_network();
    let mut state = initialize_zero_state(&network).unwrap();
    state.supplier_deliveries_due[0] = 5;
    state.supplier_in_transit[0] = 5;
    state.retailer_deliveries_due[0] = 3;
    state.retailer_in_transit[0] = 3;
    state.supplier_orders_due[0] = 7;
    state.customer_backorders[0] = 2;
    let decision = advance_to_decision_state(
        &network,
        &state,
        &[4, 0, 0, 0, 0],
        PRIMARY_REFERENCE_INSTANCE.warehouse_holding_costs,
        PRIMARY_REFERENCE_INSTANCE.retailer_holding_costs,
        PRIMARY_REFERENCE_INSTANCE.warehouse_backorder_costs,
        PRIMARY_REFERENCE_INSTANCE.retailer_backorder_costs,
    )
    .unwrap();
    assert_eq!(decision.received_supplier_deliveries[0], 5);
    assert_eq!(decision.received_retail_deliveries[0], 3);
    assert_eq!(decision.decision_state.warehouse_inventory[0], 5);
    assert_eq!(decision.decision_state.retailer_inventory[0], 0);
    assert_eq!(decision.decision_state.supplier_orders_due[0], 0);
    assert_eq!(decision.decision_state.supplier_in_transit[0], 7);
    let next_state = apply_next_orders(
        &network,
        &decision.decision_state,
        &[1, 2, 3, 4],
        &vec![0usize; network.retail_edges.len()],
    )
    .unwrap();
    assert_eq!(next_state.period, 1);
    assert_eq!(next_state.supplier_orders_due, vec![1, 2, 3, 4]);
}

#[test]
fn base_stock_orders_use_reference_modes() {
    let network = benchmark_network();
    let state = initialize_zero_state(&network).unwrap();
    let mut rng = StdRng::seed_from_u64(123);
    let (warehouse_orders_set1, retailer_orders_set1) = node_base_stock_orders(
        &network,
        &state,
        GEEVERS_SET1_BASE_STOCK_LEVELS,
        parse_benchmark_order_routing_mode("random_single_connection_by_weight").unwrap(),
        &mut rng,
    )
    .unwrap();
    assert_eq!(warehouse_orders_set1, vec![82, 100, 64, 83]);
    assert_eq!(retailer_orders_set1.iter().sum::<usize>(), 35 * 5);

    let mut rng = StdRng::seed_from_u64(123);
    let (_, retailer_orders_set2) = node_base_stock_orders(
        &network,
        &state,
        GEEVERS_SET23_BASE_STOCK_LEVELS,
        parse_benchmark_order_routing_mode("split_across_all_connections_by_weight").unwrap(),
        &mut rng,
    )
    .unwrap();
    assert_eq!(retailer_orders_set2.iter().sum::<usize>(), 30 * 5);
}

#[test]
fn compact_positions_are_defined_on_zero_state() {
    let network = benchmark_network();
    let state = initialize_zero_state(&network).unwrap();
    let warehouse_positions = warehouse_inventory_positions(&network, &state).unwrap();
    let retailer_positions = retailer_total_inventory_positions(&network, &state).unwrap();
    assert_eq!(warehouse_positions, vec![0, 0, 0, 0]);
    assert_eq!(retailer_positions, vec![0, 0, 0, 0, 0]);
}

#[test]
fn set3_benchmark_smoke_runs() {
    let costs = simulate_node_base_stock_policy(
        PRIMARY_REFERENCE_INSTANCE,
        PRIMARY_REFERENCE_INSTANCE.benchmark_base_stock_levels,
        8,
        123,
    )
    .unwrap();
    assert_eq!(costs.len(), 8);
    assert!(costs.iter().all(|cost| cost.is_finite() && *cost >= 0.0));
}
