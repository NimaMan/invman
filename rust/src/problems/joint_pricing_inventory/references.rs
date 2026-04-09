#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PublishedBenchmarkReference {
    pub source: &'static str,
    pub url: &'static str,
    pub benchmark_policies: &'static [&'static str],
    pub notes: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointPricingInventoryReferenceInstance {
    pub name: &'static str,
    pub source: &'static str,
    pub url: &'static str,
    pub periods: usize,
    pub demand_distribution_kind: &'static str,
    pub price_levels: &'static [f64],
    pub price_demand_means: &'static [f64],
    pub initial_inventory_level: usize,
    pub procurement_cost_per_unit: f64,
    pub holding_cost_per_unit: f64,
    pub stockout_cost_per_unit: f64,
    pub salvage_value_per_unit: f64,
    pub max_order_quantity: usize,
    pub benchmark_static_order_up_to: usize,
    pub benchmark_static_price_index: usize,
    pub benchmark_inventory_sensitive_order_up_to: usize,
    pub benchmark_markdown_threshold: usize,
    pub benchmark_high_price_index: usize,
    pub benchmark_low_price_index: usize,
    pub notes: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorkedTransitionReference {
    pub source: &'static str,
    pub url: &'static str,
    pub price_levels: &'static [f64],
    pub initial_inventory_level: usize,
    pub order_quantity: usize,
    pub price_index: usize,
    pub realized_demand: usize,
    pub procurement_cost_per_unit: f64,
    pub holding_cost_per_unit: f64,
    pub stockout_cost_per_unit: f64,
    pub expected_sales: usize,
    pub expected_lost_sales: usize,
    pub expected_next_inventory_level: usize,
    pub expected_period_cost: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExactVerificationReference {
    pub source: &'static str,
    pub url: &'static str,
    pub periods: usize,
    pub discount_factor: f64,
    pub price_levels: &'static [f64],
    pub price_demand_supports: &'static [&'static [u32]],
    pub price_demand_probabilities: &'static [&'static [f64]],
    pub initial_inventory_level: usize,
    pub procurement_cost_per_unit: f64,
    pub holding_cost_per_unit: f64,
    pub stockout_cost_per_unit: f64,
    pub salvage_value_per_unit: f64,
    pub max_order_quantity: usize,
    pub static_order_up_to: usize,
    pub static_price_index: usize,
    pub inventory_sensitive_order_up_to: usize,
    pub markdown_threshold: usize,
    pub high_price_index: usize,
    pub low_price_index: usize,
    pub notes: &'static str,
}

pub const ZHOU_2022_REFERENCE: PublishedBenchmarkReference = PublishedBenchmarkReference {
    source: "Zhou et al. (2022), Deep reinforcement learning approach for solving joint pricing and inventory problem with reference price effects",
    url: "https://doi.org/10.1016/j.eswa.2022.116564",
    benchmark_policies: &[
        "ddqn_joint_price_inventory",
        "value_iteration_baseline",
        "q_learning_baseline",
    ],
    notes: "The paper formulates joint pricing and inventory as an MDP and solves it with DRL under reference-price effects. The repo interpretation removes the reference-price state and keeps the core coupled price-and-order control problem with price-sensitive stochastic demand.",
};

pub const QIN_2022_REFERENCE: PublishedBenchmarkReference = PublishedBenchmarkReference {
    source: "Qin, Simchi-Levi, and Wang (2022), Data-Driven Approximation Schemes for Joint Pricing and Inventory Control Models",
    url: "https://doi.org/10.1287/mnsc.2021.4212",
    benchmark_policies: &[
        "data_driven_approximation",
        "deterministic_baseline",
        "random_baseline",
    ],
    notes: "This paper anchors joint pricing-inventory as a classic finite-horizon control problem with price-dependent random demand. It motivates keeping a clean reduced family for policy design even when the learning signal comes from data.",
};

pub const PRIMARY_PRICE_LEVELS: &[f64] = &[8.0, 10.0, 12.0];
pub const PRIMARY_DEMAND_MEANS: &[f64] = &[4.0, 2.6, 1.6];

pub const PRIMARY_REFERENCE_INSTANCE: JointPricingInventoryReferenceInstance =
    JointPricingInventoryReferenceInstance {
        name: "zhou2022_style_price_ladder",
        source: ZHOU_2022_REFERENCE.source,
        url: ZHOU_2022_REFERENCE.url,
        periods: 18,
        demand_distribution_kind: "poisson",
        price_levels: PRIMARY_PRICE_LEVELS,
        price_demand_means: PRIMARY_DEMAND_MEANS,
        initial_inventory_level: 2,
        procurement_cost_per_unit: 4.0,
        holding_cost_per_unit: 0.5,
        stockout_cost_per_unit: 5.0,
        salvage_value_per_unit: 1.0,
        max_order_quantity: 6,
        benchmark_static_order_up_to: 3,
        benchmark_static_price_index: 1,
        benchmark_inventory_sensitive_order_up_to: 4,
        benchmark_markdown_threshold: 3,
        benchmark_high_price_index: 2,
        benchmark_low_price_index: 0,
        notes: "Canonical repo interpretation of joint_pricing_inventory: one item, one discrete price ladder, periodic ordering, and price-dependent stochastic lost-sales demand. This strips away reference-price memory while keeping the coupled price and inventory decisions.",
    };

pub const WORKED_TRANSITION_REFERENCE: WorkedTransitionReference = WorkedTransitionReference {
    source: PRIMARY_REFERENCE_INSTANCE.source,
    url: PRIMARY_REFERENCE_INSTANCE.url,
    price_levels: PRIMARY_PRICE_LEVELS,
    initial_inventory_level: 1,
    order_quantity: 2,
    price_index: 1,
    realized_demand: 4,
    procurement_cost_per_unit: 4.0,
    holding_cost_per_unit: 0.5,
    stockout_cost_per_unit: 5.0,
    expected_sales: 3,
    expected_lost_sales: 1,
    expected_next_inventory_level: 0,
    expected_period_cost: -17.0,
};

pub const VERIFICATION_PRICE_LEVELS: &[f64] = &[7.0, 9.0, 11.0];
pub const VERIFICATION_SUPPORT_CHEAP: &[u32] = &[0, 1, 2, 3];
pub const VERIFICATION_PROBS_CHEAP: &[f64] = &[0.1, 0.2, 0.3, 0.4];
pub const VERIFICATION_SUPPORT_MID: &[u32] = &[0, 1, 2, 3];
pub const VERIFICATION_PROBS_MID: &[f64] = &[0.2, 0.3, 0.3, 0.2];
pub const VERIFICATION_SUPPORT_EXPENSIVE: &[u32] = &[0, 1, 2, 3];
pub const VERIFICATION_PROBS_EXPENSIVE: &[f64] = &[0.4, 0.3, 0.2, 0.1];

pub const VERIFICATION_PROBLEM_INSTANCE: ExactVerificationReference = ExactVerificationReference {
    source: QIN_2022_REFERENCE.source,
    url: QIN_2022_REFERENCE.url,
    periods: 5,
    discount_factor: 0.99,
    price_levels: VERIFICATION_PRICE_LEVELS,
    price_demand_supports: &[
        VERIFICATION_SUPPORT_CHEAP,
        VERIFICATION_SUPPORT_MID,
        VERIFICATION_SUPPORT_EXPENSIVE,
    ],
    price_demand_probabilities: &[
        VERIFICATION_PROBS_CHEAP,
        VERIFICATION_PROBS_MID,
        VERIFICATION_PROBS_EXPENSIVE,
    ],
    initial_inventory_level: 1,
    procurement_cost_per_unit: 4.0,
    holding_cost_per_unit: 0.5,
    stockout_cost_per_unit: 5.0,
    salvage_value_per_unit: 1.0,
    max_order_quantity: 4,
    static_order_up_to: 3,
    static_price_index: 1,
    inventory_sensitive_order_up_to: 3,
    markdown_threshold: 3,
    high_price_index: 2,
    low_price_index: 0,
    notes: "Repo-native exact verifier on a reduced joint pricing-inventory instance with a small discrete price ladder and price-specific demand distributions. This keeps the coupled order-price action while making exact finite-horizon DP feasible for regression tests.",
};
