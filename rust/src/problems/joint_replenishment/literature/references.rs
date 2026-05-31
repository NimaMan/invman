#![allow(dead_code)]

use crate::problems::joint_replenishment::demand::DemandRange;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PublishedBenchmarkReference {
    pub source: &'static str,
    pub url: &'static str,
    pub benchmark_policies: &'static [&'static str],
    pub reported_numbers_available: bool,
    pub numbers_anchor_repo_assertions: bool,
    pub notes: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointReplenishmentReferenceInstance {
    pub name: &'static str,
    pub source: &'static str,
    pub url: &'static str,
    pub num_items: usize,
    pub truck_capacity: usize,
    pub major_order_cost: f64,
    pub minor_order_costs: &'static [f64],
    pub holding_costs: &'static [f64],
    pub shortage_costs: &'static [f64],
    pub demand_ranges: &'static [DemandRange],
    pub notes: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExactVerificationReference {
    pub source: &'static str,
    pub url: &'static str,
    pub literature_verified: bool,
    pub verification_source: &'static str,
    pub periods: usize,
    pub discount_factor: f64,
    pub truck_capacity: usize,
    pub max_order_quantities: &'static [usize],
    pub initial_inventory_levels: &'static [i32],
    pub major_order_cost: f64,
    pub minor_order_costs: &'static [f64],
    pub holding_costs: &'static [f64],
    pub shortage_costs: &'static [f64],
    pub demand_ranges: &'static [DemandRange],
    pub moq_item_targets: &'static [usize],
    pub moq_review_period: usize,
    pub moq_rounding_threshold: f64,
    pub dynout_item_targets: &'static [usize],
    pub notes: &'static str,
}

pub const VANVUCHELEN_2020_REFERENCE: PublishedBenchmarkReference = PublishedBenchmarkReference {
    source: "Vanvuchelen, Gijsbrechts & Boute (2020), Computers in Industry 119, 103239",
    url: "https://lirias.kuleuven.be/retrieve/badd4d5b-5bfc-44e4-84f1-b98fd113143d",
    benchmark_policies: &["ppo", "(Q,S|T)_moq", "dyn-out"],
    reported_numbers_available: true,
    numbers_anchor_repo_assertions: true,
    notes: "The paper exposes all 16 small-scale setting definitions in Table 2 (verbatim here) and \
            reports per-setting optimality gaps only as a figure (Figure 2): the (Q,S|T) and DYN-OUT \
            heuristics lie 4-25% above the optimal policy. The one EXACT, executable anchor the paper \
            states in prose (Section 6.2, around Figure 3, for setting 5) is the optimal-policy action: \
            in state (I1,I2)=(5,0) the optimal policy ships exactly one full truckload to shipper 2, \
            q=(0,6), while both heuristics order q=(2,4). That optimal action is carried in \
            VANVUCHELEN_2020_FIGURE3_ANCHOR and is reproduced by an independent infinite-horizon value \
            iteration over the repo cost/transition (scripts/joint_replenishment/benchmark_vanvuchelen_settings.py).",
};

/// Published, executable anchor from Vanvuchelen et al. (2020), Section 6.2 / Figure 3.
///
/// Figure 3 and Figure 4 visualise SETTING 5 (h=[1,1], b=[19,19], k=[40,10], K=75, V=6,
/// d1~U[0,5], d2~U[0,3], gamma=0.99). The authors state in prose:
///   "Suppose for instance that the system is in state (I1, I2) = (5, 0). Under the optimal
///    policy, only shipper 2 orders q2 = 6 units, while q1 = 0. ... The (Q,S|T) and DYN-OUT
///    policies, in contrast, both order q1 = 2 and q2 = 4 units."
///
/// The optimal action is an INFINITE-HORIZON discounted (gamma=0.99) stationary action obtained
/// by value iteration (paper Section 6, "value iteration (Puterman, 1994) with discount factor
/// gamma=0.99"). An independent value-iteration solver that mirrors the repo env cost (Eq. 2) and
/// balance (Eq. 4) reproduces this action exactly; see the benchmark script. The repo's bundled
/// finite-horizon DP (`finite_horizon_dp.rs`) is a SELF-CONSISTENCY comparator on a short horizon
/// and is not expected to reproduce the infinite-horizon stationary action directly.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PublishedActionAnchor {
    pub source: &'static str,
    pub url: &'static str,
    pub setting_name: &'static str,
    pub state_inventory_levels: &'static [i32],
    pub optimal_action: &'static [usize],
    pub heuristic_action: &'static [usize],
    pub discount_factor: f64,
    pub horizon: &'static str,
    pub notes: &'static str,
}

pub const VANVUCHELEN_2020_FIGURE3_ANCHOR: PublishedActionAnchor = PublishedActionAnchor {
    source: VANVUCHELEN_2020_REFERENCE.source,
    url: VANVUCHELEN_2020_REFERENCE.url,
    setting_name: "vanvuchelen2020_small_scale_setting_5",
    state_inventory_levels: &[5, 0],
    optimal_action: &[0, 6],
    heuristic_action: &[2, 4],
    discount_factor: 0.99,
    horizon: "infinite-horizon discounted (value iteration)",
    notes: "Setting 5 Figure 3 prose anchor: optimal ships one FTL to shipper 2 only, q=(0,6); both \
            paper heuristics order q=(2,4). Reproduced by independent infinite-horizon value iteration \
            over the repo env cost/transition. The repo's reduced finite-horizon DP is a separate \
            self-consistency comparator and is not asserted against this action.",
};

pub const SMALL_SCALE_DEMAND_RANGES_A: &[DemandRange] = &[
    DemandRange { low: 0, high: 5 },
    DemandRange { low: 0, high: 3 },
];

pub const SMALL_SCALE_DEMAND_RANGES_B: &[DemandRange] = &[
    DemandRange { low: 0, high: 6 },
    DemandRange { low: 0, high: 2 },
];

pub const SMALL_SCALE_MINOR_COSTS_A: &[f64] = &[10.0, 10.0];
pub const SMALL_SCALE_MINOR_COSTS_B: &[f64] = &[40.0, 10.0];

pub const SMALL_SCALE_SETTINGS: [JointReplenishmentReferenceInstance; 16] = [
    JointReplenishmentReferenceInstance {
        name: "vanvuchelen2020_small_scale_setting_1",
        source: VANVUCHELEN_2020_REFERENCE.source,
        url: VANVUCHELEN_2020_REFERENCE.url,
        num_items: 2,
        truck_capacity: 6,
        major_order_cost: 75.0,
        minor_order_costs: SMALL_SCALE_MINOR_COSTS_A,
        holding_costs: &[1.0, 1.0],
        shortage_costs: &[19.0, 19.0],
        demand_ranges: SMALL_SCALE_DEMAND_RANGES_A,
        notes: "Table 2 setting 1: low-low holding costs, symmetric minor ordering costs, and demand pair U[0,5] / U[0,3].",
    },
    JointReplenishmentReferenceInstance {
        name: "vanvuchelen2020_small_scale_setting_2",
        source: VANVUCHELEN_2020_REFERENCE.source,
        url: VANVUCHELEN_2020_REFERENCE.url,
        num_items: 2,
        truck_capacity: 6,
        major_order_cost: 75.0,
        minor_order_costs: SMALL_SCALE_MINOR_COSTS_A,
        holding_costs: &[1.0, 5.0],
        shortage_costs: &[19.0, 95.0],
        demand_ranges: SMALL_SCALE_DEMAND_RANGES_A,
        notes: "Table 2 setting 2: low-high holding costs, symmetric minor ordering costs, and demand pair U[0,5] / U[0,3].",
    },
    JointReplenishmentReferenceInstance {
        name: "vanvuchelen2020_small_scale_setting_3",
        source: VANVUCHELEN_2020_REFERENCE.source,
        url: VANVUCHELEN_2020_REFERENCE.url,
        num_items: 2,
        truck_capacity: 6,
        major_order_cost: 75.0,
        minor_order_costs: SMALL_SCALE_MINOR_COSTS_A,
        holding_costs: &[5.0, 1.0],
        shortage_costs: &[95.0, 19.0],
        demand_ranges: SMALL_SCALE_DEMAND_RANGES_A,
        notes: "Table 2 setting 3: high-low holding costs, symmetric minor ordering costs, and demand pair U[0,5] / U[0,3].",
    },
    JointReplenishmentReferenceInstance {
        name: "vanvuchelen2020_small_scale_setting_4",
        source: VANVUCHELEN_2020_REFERENCE.source,
        url: VANVUCHELEN_2020_REFERENCE.url,
        num_items: 2,
        truck_capacity: 6,
        major_order_cost: 75.0,
        minor_order_costs: SMALL_SCALE_MINOR_COSTS_A,
        holding_costs: &[5.0, 5.0],
        shortage_costs: &[95.0, 95.0],
        demand_ranges: SMALL_SCALE_DEMAND_RANGES_A,
        notes: "Table 2 setting 4: high-high holding costs, symmetric minor ordering costs, and demand pair U[0,5] / U[0,3].",
    },
    JointReplenishmentReferenceInstance {
        name: "vanvuchelen2020_small_scale_setting_5",
        source: VANVUCHELEN_2020_REFERENCE.source,
        url: VANVUCHELEN_2020_REFERENCE.url,
        num_items: 2,
        truck_capacity: 6,
        major_order_cost: 75.0,
        minor_order_costs: SMALL_SCALE_MINOR_COSTS_B,
        holding_costs: &[1.0, 1.0],
        shortage_costs: &[19.0, 19.0],
        demand_ranges: SMALL_SCALE_DEMAND_RANGES_A,
        notes: "Table 2 setting 5: low-low holding costs, asymmetric minor ordering costs, and demand pair U[0,5] / U[0,3]. Figure 3 and Figure 4 focus on this family.",
    },
    JointReplenishmentReferenceInstance {
        name: "vanvuchelen2020_small_scale_setting_6",
        source: VANVUCHELEN_2020_REFERENCE.source,
        url: VANVUCHELEN_2020_REFERENCE.url,
        num_items: 2,
        truck_capacity: 6,
        major_order_cost: 75.0,
        minor_order_costs: SMALL_SCALE_MINOR_COSTS_B,
        holding_costs: &[1.0, 5.0],
        shortage_costs: &[19.0, 95.0],
        demand_ranges: SMALL_SCALE_DEMAND_RANGES_A,
        notes: "Table 2 setting 6: low-high holding costs, asymmetric minor ordering costs, and demand pair U[0,5] / U[0,3].",
    },
    JointReplenishmentReferenceInstance {
        name: "vanvuchelen2020_small_scale_setting_7",
        source: VANVUCHELEN_2020_REFERENCE.source,
        url: VANVUCHELEN_2020_REFERENCE.url,
        num_items: 2,
        truck_capacity: 6,
        major_order_cost: 75.0,
        minor_order_costs: SMALL_SCALE_MINOR_COSTS_B,
        holding_costs: &[5.0, 1.0],
        shortage_costs: &[95.0, 19.0],
        demand_ranges: SMALL_SCALE_DEMAND_RANGES_A,
        notes: "Table 2 setting 7: high-low holding costs, asymmetric minor ordering costs, and demand pair U[0,5] / U[0,3].",
    },
    JointReplenishmentReferenceInstance {
        name: "vanvuchelen2020_small_scale_setting_8",
        source: VANVUCHELEN_2020_REFERENCE.source,
        url: VANVUCHELEN_2020_REFERENCE.url,
        num_items: 2,
        truck_capacity: 6,
        major_order_cost: 75.0,
        minor_order_costs: SMALL_SCALE_MINOR_COSTS_B,
        holding_costs: &[5.0, 5.0],
        shortage_costs: &[95.0, 95.0],
        demand_ranges: SMALL_SCALE_DEMAND_RANGES_A,
        notes: "Table 2 setting 8: high-high holding costs, asymmetric minor ordering costs, and demand pair U[0,5] / U[0,3].",
    },
    JointReplenishmentReferenceInstance {
        name: "vanvuchelen2020_small_scale_setting_9",
        source: VANVUCHELEN_2020_REFERENCE.source,
        url: VANVUCHELEN_2020_REFERENCE.url,
        num_items: 2,
        truck_capacity: 6,
        major_order_cost: 75.0,
        minor_order_costs: SMALL_SCALE_MINOR_COSTS_A,
        holding_costs: &[1.0, 1.0],
        shortage_costs: &[19.0, 19.0],
        demand_ranges: SMALL_SCALE_DEMAND_RANGES_B,
        notes: "Table 2 setting 9: low-low holding costs, symmetric minor ordering costs, and demand pair U[0,6] / U[0,2].",
    },
    JointReplenishmentReferenceInstance {
        name: "vanvuchelen2020_small_scale_setting_10",
        source: VANVUCHELEN_2020_REFERENCE.source,
        url: VANVUCHELEN_2020_REFERENCE.url,
        num_items: 2,
        truck_capacity: 6,
        major_order_cost: 75.0,
        minor_order_costs: SMALL_SCALE_MINOR_COSTS_A,
        holding_costs: &[1.0, 5.0],
        shortage_costs: &[19.0, 95.0],
        demand_ranges: SMALL_SCALE_DEMAND_RANGES_B,
        notes: "Table 2 setting 10: low-high holding costs, symmetric minor ordering costs, and demand pair U[0,6] / U[0,2].",
    },
    JointReplenishmentReferenceInstance {
        name: "vanvuchelen2020_small_scale_setting_11",
        source: VANVUCHELEN_2020_REFERENCE.source,
        url: VANVUCHELEN_2020_REFERENCE.url,
        num_items: 2,
        truck_capacity: 6,
        major_order_cost: 75.0,
        minor_order_costs: SMALL_SCALE_MINOR_COSTS_A,
        holding_costs: &[5.0, 1.0],
        shortage_costs: &[95.0, 19.0],
        demand_ranges: SMALL_SCALE_DEMAND_RANGES_B,
        notes: "Table 2 setting 11: high-low holding costs, symmetric minor ordering costs, and demand pair U[0,6] / U[0,2].",
    },
    JointReplenishmentReferenceInstance {
        name: "vanvuchelen2020_small_scale_setting_12",
        source: VANVUCHELEN_2020_REFERENCE.source,
        url: VANVUCHELEN_2020_REFERENCE.url,
        num_items: 2,
        truck_capacity: 6,
        major_order_cost: 75.0,
        minor_order_costs: SMALL_SCALE_MINOR_COSTS_A,
        holding_costs: &[5.0, 5.0],
        shortage_costs: &[95.0, 95.0],
        demand_ranges: SMALL_SCALE_DEMAND_RANGES_B,
        notes: "Table 2 setting 12: high-high holding costs, symmetric minor ordering costs, and demand pair U[0,6] / U[0,2].",
    },
    JointReplenishmentReferenceInstance {
        name: "vanvuchelen2020_small_scale_setting_13",
        source: VANVUCHELEN_2020_REFERENCE.source,
        url: VANVUCHELEN_2020_REFERENCE.url,
        num_items: 2,
        truck_capacity: 6,
        major_order_cost: 75.0,
        minor_order_costs: SMALL_SCALE_MINOR_COSTS_B,
        holding_costs: &[1.0, 1.0],
        shortage_costs: &[19.0, 19.0],
        demand_ranges: SMALL_SCALE_DEMAND_RANGES_B,
        notes: "Table 2 setting 13: low-low holding costs, asymmetric minor ordering costs, and demand pair U[0,6] / U[0,2].",
    },
    JointReplenishmentReferenceInstance {
        name: "vanvuchelen2020_small_scale_setting_14",
        source: VANVUCHELEN_2020_REFERENCE.source,
        url: VANVUCHELEN_2020_REFERENCE.url,
        num_items: 2,
        truck_capacity: 6,
        major_order_cost: 75.0,
        minor_order_costs: SMALL_SCALE_MINOR_COSTS_B,
        holding_costs: &[1.0, 5.0],
        shortage_costs: &[19.0, 95.0],
        demand_ranges: SMALL_SCALE_DEMAND_RANGES_B,
        notes: "Table 2 setting 14: low-high holding costs, asymmetric minor ordering costs, and demand pair U[0,6] / U[0,2].",
    },
    JointReplenishmentReferenceInstance {
        name: "vanvuchelen2020_small_scale_setting_15",
        source: VANVUCHELEN_2020_REFERENCE.source,
        url: VANVUCHELEN_2020_REFERENCE.url,
        num_items: 2,
        truck_capacity: 6,
        major_order_cost: 75.0,
        minor_order_costs: SMALL_SCALE_MINOR_COSTS_B,
        holding_costs: &[5.0, 1.0],
        shortage_costs: &[95.0, 19.0],
        demand_ranges: SMALL_SCALE_DEMAND_RANGES_B,
        notes: "Table 2 setting 15: high-low holding costs, asymmetric minor ordering costs, and demand pair U[0,6] / U[0,2].",
    },
    JointReplenishmentReferenceInstance {
        name: "vanvuchelen2020_small_scale_setting_16",
        source: VANVUCHELEN_2020_REFERENCE.source,
        url: VANVUCHELEN_2020_REFERENCE.url,
        num_items: 2,
        truck_capacity: 6,
        major_order_cost: 75.0,
        minor_order_costs: SMALL_SCALE_MINOR_COSTS_B,
        holding_costs: &[5.0, 5.0],
        shortage_costs: &[95.0, 95.0],
        demand_ranges: SMALL_SCALE_DEMAND_RANGES_B,
        notes: "Table 2 setting 16: high-high holding costs, asymmetric minor ordering costs, and demand pair U[0,6] / U[0,2].",
    },
];

pub const PRIMARY_REFERENCE_INSTANCE: JointReplenishmentReferenceInstance = SMALL_SCALE_SETTINGS[4];

pub const VERIFICATION_PROBLEM_INSTANCE: ExactVerificationReference = ExactVerificationReference {
    source: VANVUCHELEN_2020_REFERENCE.source,
    url: VANVUCHELEN_2020_REFERENCE.url,
    literature_verified: false,
    verification_source: "repo_finite_horizon_self_consistency_comparator",
    periods: 4,
    discount_factor: 0.99,
    truck_capacity: 6,
    max_order_quantities: &[12, 12],
    initial_inventory_levels: &[2, 0],
    major_order_cost: 75.0,
    minor_order_costs: SMALL_SCALE_MINOR_COSTS_A,
    holding_costs: &[1.0, 1.0],
    shortage_costs: &[19.0, 19.0],
    demand_ranges: SMALL_SCALE_DEMAND_RANGES_A,
    moq_item_targets: &[8, 5],
    moq_review_period: 1,
    moq_rounding_threshold: 2.0,
    dynout_item_targets: &[8, 5],
    notes: "Reduced finite-horizon (4-period, discounted) self-consistency comparator on the setting-1 \
            model family (k=[10,10], h=[1,1], b=[19,19]). It checks that the repo's exact DP dominates \
            the carried heuristics on a short horizon; it does NOT reproduce the paper's infinite-horizon \
            optimal action and is not a literature anchor on its own. The paper's executable literature \
            anchor is VANVUCHELEN_2020_FIGURE3_ANCHOR (setting 5 optimal action q=(0,6) at state (5,0)), \
            which the env reproduces under an independent infinite-horizon value iteration.",
};
