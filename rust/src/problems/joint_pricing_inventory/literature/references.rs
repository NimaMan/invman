#![allow(dead_code)]

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
    pub literature_verified: bool,
    pub verification_source: &'static str,
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
pub struct ExactVerificationReference {
    pub source: &'static str,
    pub url: &'static str,
    pub literature_verified: bool,
    pub verification_source: &'static str,
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

/// Classical formulation anchors for the *executable* model in this package.
///
/// The env implemented in `env.rs` (zero lead time, price-dependent stochastic demand, lost sales,
/// holding cost on ending inventory, profit objective) reduces at T = 1 to the textbook
/// price-setting newsvendor: overage cost `Co = c + h`, underage cost `Cu = p + s - c`, optimal
/// order-up-to = smallest `y` with `F(y) >= Cu / (Cu + Co)`. The finite-horizon multi-period version
/// is the classic finite-horizon joint pricing-and-inventory control problem. These classical results
/// are the formulation the package actually implements, and the analytical critical-fractile check in
/// `verification/tests.rs` validates the env transition + cost against the closed form independently of
/// the repo's own DP.
pub const PRICE_SETTING_NEWSVENDOR_ANCHOR: PublishedBenchmarkReference = PublishedBenchmarkReference {
    source: "Price-setting newsvendor / finite-horizon joint pricing-inventory: Whitin (1955) Management Science 2(1):61-68 doi:10.1287/mnsc.2.1.61; Petruzzi & Dada (1999) Operations Research 47(2):183-194 doi:10.1287/opre.47.2.183; Federgruen & Heching (1999) Operations Research 47(3):454-475 doi:10.1287/opre.47.3.454",
    url: "https://doi.org/10.1287/opre.47.2.183",
    benchmark_policies: &[
        "critical_fractile_newsvendor",
        "finite_horizon_exact_dp",
    ],
    notes: "The single-period reduction of this env is the price-setting newsvendor with overage Co = c + h and underage Cu = p + s - c; the optimal order-up-to is the critical-fractile quantile of price-dependent demand. The finite-horizon multi-period version is the classic joint pricing-and-inventory control problem (Federgruen & Heching 1999). The `url` field stores the Petruzzi & Dada (1999) DOI (10.1287/opre.47.2.183); the Federgruen & Heching (1999) DOI is 10.1287/opre.47.3.454 and the Whitin (1955) DOI is 10.1287/mnsc.2.1.61 (all listed in `source`). verification/tests.rs checks the env's T=1 optimum equals the closed-form critical fractile for every price on VERIFICATION_PROBLEM_INSTANCE. This is an analytical (classical-literature) anchor for the env transition + cost; it is NOT a reproduction of a published per-instance optimal-profit table, so the package remains literature_verified = false.",
};

pub const PRIMARY_PRICE_LEVELS: &[f64] = &[8.0, 10.0, 12.0];
pub const PRIMARY_DEMAND_MEANS: &[f64] = &[4.0, 2.6, 1.6];

pub const PRIMARY_REFERENCE_INSTANCE: JointPricingInventoryReferenceInstance =
    JointPricingInventoryReferenceInstance {
        name: "zhou2022_style_price_ladder",
        source: ZHOU_2022_REFERENCE.source,
        url: ZHOU_2022_REFERENCE.url,
        literature_verified: false,
        verification_source: "repo_exact_solver_not_verified_against_literature",
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
    literature_verified: false,
    verification_source: "repo_exact_solver_not_verified_against_literature",
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
