// ============================================================================
// references.rs  (canonical, faithful model)
//
// Source of truth for the FAITHFUL ameliorating-inventory literature instances
// and the published companion-repo anchors that the executing verification test
// reproduces.
//
// Pahr & Grunow (2025), "The Value of Blending — Managing Ameliorating
// Inventory Using Deep Reinforcement Learning", Production and Operations
// Management 35(5). DOI 10.1177/10591478251387795.
// Companion code & per-instance configurations / perfect-information upper
// bounds: https://github.com/amelioratinginventory/ameliorating_inventory
//
// Distinction enforced by this file (per repo references.rs rules):
//   - exact published values         -> `PublishedUpperBoundAnchor.published_max_reward`
//                                       (printed in companion upper_bound.json)
//   - repo-native benchmark values   -> the freshly re-solved LP value asserted
//                                       in tests/verification.rs (not stored)
//   - deterministic worked-example   -> none needed; the LP itself is the exact
//                                       reproduction.
// ============================================================================

/// A published companion-repo reference for one instance: everything needed to
/// locate the parameters, the dataset, and the number to reproduce.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AmelioratingInventoryReferenceInstance {
    pub name: &'static str,
    pub source: &'static str,
    pub url: &'static str,
    /// Relative path (under this module's `practical/datasets/`) of the dataset.
    pub dataset_file: &'static str,
    pub num_ages: usize,
    pub num_products: usize,
    pub target_ages: &'static [usize],
    pub max_inventory: f64,
    pub evaporation: f64,
    pub holding_cost: f64,
    pub allow_blending: bool,
    /// Published long-run AVERAGE-PROFIT upper bound (`max_reward`).
    pub published_max_reward: f64,
    /// `true` only when an executing test re-solves the LP and reproduces
    /// `published_max_reward` within tolerance.
    pub literature_verified: bool,
    pub notes: &'static str,
}

/// The published perfect-information upper-bound anchor for the verification
/// instance, plus the tolerance the executing test must meet.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PublishedUpperBoundAnchor {
    pub instance_name: &'static str,
    pub source: &'static str,
    pub url: &'static str,
    pub dataset_file: &'static str,
    /// The exact value printed in the companion `upper_bound.json`.
    pub published_max_reward: f64,
    /// Absolute tolerance for the re-solved LP value.
    pub max_reward_tolerance: f64,
    pub literature_verified: bool,
    pub notes: &'static str,
}

pub const PAHR_GRUNOW_2025_SOURCE: &str =
    "Pahr and Grunow (2025), Production and Operations Management 35(5), DOI 10.1177/10591478251387795";
pub const PAHR_GRUNOW_2025_PAPER_URL: &str =
    "https://journals.sagepub.com/doi/10.1177/10591478251387795";
pub const PAHR_GRUNOW_2025_REPO_URL: &str =
    "https://github.com/amelioratinginventory/ameliorating_inventory";

/// PRIMARY reference instance: the companion default spirits configuration
/// `spirits_0001` (10 age classes, 3 products, target ages 2/4/6, capacity 50,
/// holding 2.5, no blending). Its perfect-information upper bound is
/// reproduced by the executing verification test.
pub const PRIMARY_REFERENCE_INSTANCE: AmelioratingInventoryReferenceInstance =
    AmelioratingInventoryReferenceInstance {
        name: "pahr_grunow2025_spirits_0001",
        source: PAHR_GRUNOW_2025_SOURCE,
        url: PAHR_GRUNOW_2025_REPO_URL,
        dataset_file: "spirits_0001_perfect_information_lp.txt",
        num_ages: 10,
        num_products: 3,
        target_ages: &[2, 4, 6],
        max_inventory: 50.0,
        evaporation: 0.03,
        holding_cost: 2.5,
        allow_blending: false,
        published_max_reward: 1991.9344293376805,
        literature_verified: true,
        notes: "Companion default spirits instance. The perfect-information LP \
                average-profit upper bound printed in problem_configurations/\
                spirits_0001/upper_bound.json is reproduced by tests/verification.rs.",
    };

/// SECONDARY reference instance: the industry `port_wine` case study (25 age
/// classes, 2 products, target ages 9/19, blending enabled). Also reproduced.
pub const PORT_WINE_REFERENCE_INSTANCE: AmelioratingInventoryReferenceInstance =
    AmelioratingInventoryReferenceInstance {
        name: "pahr_grunow2025_port_wine",
        source: PAHR_GRUNOW_2025_SOURCE,
        url: PAHR_GRUNOW_2025_REPO_URL,
        dataset_file: "port_wine_perfect_information_lp.txt",
        num_ages: 25,
        num_products: 2,
        target_ages: &[9, 19],
        max_inventory: 50.0,
        evaporation: 0.02,
        holding_cost: 1.0,
        allow_blending: true,
        published_max_reward: 2444.8010643781136,
        literature_verified: true,
        notes: "Port-wine industry case study. Perfect-information upper bound \
                from problem_configurations/port_wine/upper_bound.json reproduced \
                by tests/verification.rs.",
    };

pub const REFERENCE_INSTANCES: [AmelioratingInventoryReferenceInstance; 2] =
    [PRIMARY_REFERENCE_INSTANCE, PORT_WINE_REFERENCE_INSTANCE];

/// The verification anchor: the spirits_0001 perfect-information upper bound.
pub const VERIFICATION_PROBLEM_INSTANCE: PublishedUpperBoundAnchor = PublishedUpperBoundAnchor {
    instance_name: "pahr_grunow2025_spirits_0001",
    source: PAHR_GRUNOW_2025_SOURCE,
    url: PAHR_GRUNOW_2025_REPO_URL,
    dataset_file: "spirits_0001_perfect_information_lp.txt",
    published_max_reward: 1991.9344293376805,
    max_reward_tolerance: 1.0e-3,
    literature_verified: true,
    notes: "Re-solving the perfect-information LP with the in-crate microlp \
            simplex reproduces the companion-published max_reward to < 1e-3.",
};

/// Secondary verification anchor: the port_wine perfect-information upper bound.
pub const PORT_WINE_VERIFICATION_ANCHOR: PublishedUpperBoundAnchor = PublishedUpperBoundAnchor {
    instance_name: "pahr_grunow2025_port_wine",
    source: PAHR_GRUNOW_2025_SOURCE,
    url: PAHR_GRUNOW_2025_REPO_URL,
    dataset_file: "port_wine_perfect_information_lp.txt",
    published_max_reward: 2444.8010643781136,
    max_reward_tolerance: 1.0e-3,
    literature_verified: true,
    notes: "Re-solving the perfect-information LP reproduces the companion-\
            published port_wine max_reward to < 1e-3.",
};
