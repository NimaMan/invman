#![allow(dead_code)]

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PublishedBenchmarkReference {
    pub source: &'static str,
    pub url: &'static str,
    pub benchmark_policies: &'static [&'static str],
    pub notes: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DualSourcingReferenceInstance {
    pub name: &'static str,
    pub source: &'static str,
    pub url: &'static str,
    pub regular_lead_time: usize,
    pub expedited_lead_time: usize,
    pub regular_order_cost: f64,
    pub expedited_order_cost: f64,
    pub holding_cost: f64,
    pub shortage_cost: f64,
    pub regular_max_order_size: usize,
    pub expedited_max_order_size: usize,
    pub demand_low: usize,
    pub demand_high: usize,
    pub notes: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PublishedOptimalityGapReference {
    pub source: &'static str,
    pub url: &'static str,
    pub instance_name: &'static str,
    pub capped_dual_index_gap_pct: f64,
    pub dual_index_gap_pct: f64,
    pub single_index_gap_pct: f64,
    pub tailored_base_surge_gap_pct: f64,
    pub a3c_gap_pct: f64,
    pub notes: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorkedTransitionReference {
    pub source: &'static str,
    pub url: &'static str,
    pub initial_reduced_state: &'static [i64],
    pub regular_order: usize,
    pub expedited_order: usize,
    pub realized_demand: usize,
    pub regular_order_cost: f64,
    pub expedited_order_cost: f64,
    pub holding_cost: f64,
    pub shortage_cost: f64,
    pub expected_next_reduced_state: &'static [i64],
    pub expected_period_cost: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VerificationProblemInstance {
    pub name: &'static str,
    pub reference_instance_name: &'static str,
    pub inventory_lower: i64,
    pub inventory_upper: i64,
    pub solver_tolerance: f64,
    pub max_iterations: usize,
    pub search_seed: u64,
    pub search_horizon: usize,
    pub warm_up_periods_ratio: f64,
    pub exact_abs_tolerance: f64,
    pub literature_gap_abs_tolerance_pct: f64,
}

pub const BENCHMARK_POLICIES: &[&str] = &[
    "optimal_dp",
    "single_index",
    "dual_index",
    "capped_dual_index",
    "tailored_base_surge",
    "lp_adp",
    "a3c",
];

pub const GIJSBRECHTS_2022_REFERENCE: PublishedBenchmarkReference = PublishedBenchmarkReference {
    source: "Gijsbrechts et al. (2022), Section 6.2 / Figure 9",
    url: "https://doi.org/10.1287/msom.2021.1064",
    benchmark_policies: BENCHMARK_POLICIES,
    notes: "Section 6.2 defines the six small-scale dual-sourcing instances with l_e = 0, l_r in {2,3,4}, c_r = 100, c_e in {105,110}, h = 5, b = 495, and demand uniform on {0,1,2,3,4}. Figure 9 prints per-instance optimality-gap labels for capped dual-index, dual-index, single-index, tailored base-surge, and A3C, but not a table of absolute costs.",
};

pub const VEERARAGHAVAN_2008_REFERENCE: PublishedBenchmarkReference = PublishedBenchmarkReference {
    source: "Veeraraghavan and Scheller-Wolf (2008), Operations Research 56(4):850-864",
    url: "https://repository.upenn.edu/bitstreams/50f320cb-e610-4a2b-87c9-17e86061f845/download",
    benchmark_policies: &["optimal_dp", "dual_index", "single_sourcing"],
    notes: "Open repository copy of the dual-index paper. The experiments include U[0,4] demand with h = 5, c_r = 100, p = 495, and lr in {2,3}, but they are sensitivity curves over expediting cost and service level, not the six fixed Gijsbrechts benchmark rows and not the later capped-dual-index or tailored-base-surge comparisons.",
};

pub const SHEOPURI_2010_REFERENCE: PublishedBenchmarkReference = PublishedBenchmarkReference {
    source: "Sheopuri et al. (2010), Operations Research 58(3):734-745",
    url: "https://doi.org/10.1287/opre.1090.0799",
    benchmark_policies: &[
        "single_index",
        "dual_index",
        "best_weighted_bounds",
        "tailored_base_surge",
    ],
    notes: "This paper extends the classical dual-sourcing policy family beyond the original dual-index rule. It is the right policy-family source for capped or weighted dual-sourcing heuristics, but it is not the source of the six exact Figure 9 benchmark gap labels used by Gijsbrechts et al. (2022).",
};

// Three-tier CDI-optimality taxonomy (docs/benchmarks/DUAL_SOURCING_INSTANCE_TAXONOMY_2026_06_07.md).
// The six Gijsbrechts rows above are Tier A (CDI-optimal, gap <= 0.12% to the bounded-DP optimum).
// The two synthetic rows below are ADDITIVE explorations of the harder end of the *reachable*
// regime, found by a validated-box bounded-DP sweep over expedite premium / penalty / demand
// variability at l_r=2 (the only lead time where the bounded DP is validatable in budget):
//   - Tier B (moderate): largest U[0,4] gap found, +0.188% single-path.
//   - Tier C (hardest demonstrable): largest gap in the whole sweep, +0.305% single-path /
//     +0.160% out-of-sample; the lever is high demand variability (U[0,8]).
// HONEST FRAMING: no genuinely hard (>=5%) regime exists. CDI is heuristics-excellent across the
// reachable space; these rows document degrees of excellence, not a CDI-failure cliff. They are
// NOT Gijsbrechts published rows (no published cost/gap exists for them) -- they are repo-native
// taxonomy probes with a bounded-DP optimum as the denominator.
pub const DUAL_SOURCING_TAXONOMY_SOURCE: &str =
    "invman repo-native CDI-optimality taxonomy (2026-06-07), bounded-DP optimum denominator";
pub const DUAL_SOURCING_TAXONOMY_URL: &str =
    "docs/benchmarks/DUAL_SOURCING_INSTANCE_TAXONOMY_2026_06_07.md";

pub const DUAL_SOURCING_TAXONOMY_INSTANCES: [DualSourcingReferenceInstance; 2] = [
    DualSourcingReferenceInstance {
        name: "dual_l2_ce110_b50_u04_catB",
        source: DUAL_SOURCING_TAXONOMY_SOURCE,
        url: DUAL_SOURCING_TAXONOMY_URL,
        regular_lead_time: 2,
        expedited_lead_time: 0,
        regular_order_cost: 100.0,
        expedited_order_cost: 110.0,
        holding_cost: 5.0,
        shortage_cost: 50.0,
        regular_max_order_size: 12,
        expedited_max_order_size: 12,
        demand_low: 0,
        demand_high: 4,
        notes: "category:B(moderate); CDI gap-to-bounded-DP-optimum +0.188% single-path (largest U[0,4] cell in the 2026-06-07 sweep); validated box (-24,48), DP_opt 219.173; not a published row.",
    },
    DualSourcingReferenceInstance {
        name: "dual_l2_ce110_b50_u08_catC",
        source: DUAL_SOURCING_TAXONOMY_SOURCE,
        url: DUAL_SOURCING_TAXONOMY_URL,
        regular_lead_time: 2,
        expedited_lead_time: 0,
        regular_order_cost: 100.0,
        expedited_order_cost: 110.0,
        holding_cost: 5.0,
        shortage_cost: 50.0,
        regular_max_order_size: 12,
        expedited_max_order_size: 12,
        demand_low: 0,
        demand_high: 8,
        notes: "category:C(hardest demonstrable); CDI gap-to-bounded-DP-optimum +0.305% single-path / +0.160% out-of-sample (largest in the 2026-06-07 sweep, lever = demand variability U[0,8]); validated box (-40,72), DP_opt 435.217; gap < path noise, CDI ~ optimum; not a published row.",
    },
];

pub const DUAL_SOURCING_REFERENCE_INSTANCES: [DualSourcingReferenceInstance; 6] = [
    DualSourcingReferenceInstance {
        name: "dual_l2_ce105",
        source: GIJSBRECHTS_2022_REFERENCE.source,
        url: GIJSBRECHTS_2022_REFERENCE.url,
        regular_lead_time: 2,
        expedited_lead_time: 0,
        regular_order_cost: 100.0,
        expedited_order_cost: 105.0,
        holding_cost: 5.0,
        shortage_cost: 495.0,
        regular_max_order_size: 12,
        expedited_max_order_size: 12,
        demand_low: 0,
        demand_high: 4,
        notes: "Small-scale linear-cost benchmark row from Gijsbrechts et al. (2022), Section 6.2.",
    },
    DualSourcingReferenceInstance {
        name: "dual_l2_ce110",
        source: GIJSBRECHTS_2022_REFERENCE.source,
        url: GIJSBRECHTS_2022_REFERENCE.url,
        regular_lead_time: 2,
        expedited_lead_time: 0,
        regular_order_cost: 100.0,
        expedited_order_cost: 110.0,
        holding_cost: 5.0,
        shortage_cost: 495.0,
        regular_max_order_size: 12,
        expedited_max_order_size: 12,
        demand_low: 0,
        demand_high: 4,
        notes: "Small-scale linear-cost benchmark row from Gijsbrechts et al. (2022), Section 6.2.",
    },
    DualSourcingReferenceInstance {
        name: "dual_l3_ce105",
        source: GIJSBRECHTS_2022_REFERENCE.source,
        url: GIJSBRECHTS_2022_REFERENCE.url,
        regular_lead_time: 3,
        expedited_lead_time: 0,
        regular_order_cost: 100.0,
        expedited_order_cost: 105.0,
        holding_cost: 5.0,
        shortage_cost: 495.0,
        regular_max_order_size: 12,
        expedited_max_order_size: 12,
        demand_low: 0,
        demand_high: 4,
        notes: "Small-scale linear-cost benchmark row from Gijsbrechts et al. (2022), Section 6.2.",
    },
    DualSourcingReferenceInstance {
        name: "dual_l3_ce110",
        source: GIJSBRECHTS_2022_REFERENCE.source,
        url: GIJSBRECHTS_2022_REFERENCE.url,
        regular_lead_time: 3,
        expedited_lead_time: 0,
        regular_order_cost: 100.0,
        expedited_order_cost: 110.0,
        holding_cost: 5.0,
        shortage_cost: 495.0,
        regular_max_order_size: 12,
        expedited_max_order_size: 12,
        demand_low: 0,
        demand_high: 4,
        notes: "Small-scale linear-cost benchmark row from Gijsbrechts et al. (2022), Section 6.2.",
    },
    DualSourcingReferenceInstance {
        name: "dual_l4_ce105",
        source: GIJSBRECHTS_2022_REFERENCE.source,
        url: GIJSBRECHTS_2022_REFERENCE.url,
        regular_lead_time: 4,
        expedited_lead_time: 0,
        regular_order_cost: 100.0,
        expedited_order_cost: 105.0,
        holding_cost: 5.0,
        shortage_cost: 495.0,
        regular_max_order_size: 12,
        expedited_max_order_size: 12,
        demand_low: 0,
        demand_high: 4,
        notes: "Small-scale linear-cost benchmark row from Gijsbrechts et al. (2022), Section 6.2.",
    },
    DualSourcingReferenceInstance {
        name: "dual_l4_ce110",
        source: GIJSBRECHTS_2022_REFERENCE.source,
        url: GIJSBRECHTS_2022_REFERENCE.url,
        regular_lead_time: 4,
        expedited_lead_time: 0,
        regular_order_cost: 100.0,
        expedited_order_cost: 110.0,
        holding_cost: 5.0,
        shortage_cost: 495.0,
        regular_max_order_size: 12,
        expedited_max_order_size: 12,
        demand_low: 0,
        demand_high: 4,
        notes: "Small-scale linear-cost benchmark row from Gijsbrechts et al. (2022), Section 6.2.",
    },
];

pub const PRIMARY_REFERENCE_INSTANCE: DualSourcingReferenceInstance =
    DUAL_SOURCING_REFERENCE_INSTANCES[5];

pub const FIGURE_9_GAP_REFERENCES: [PublishedOptimalityGapReference; 6] = [
    PublishedOptimalityGapReference {
        source: GIJSBRECHTS_2022_REFERENCE.source,
        url: GIJSBRECHTS_2022_REFERENCE.url,
        instance_name: "dual_l2_ce105",
        capped_dual_index_gap_pct: 0.00,
        dual_index_gap_pct: 0.11,
        single_index_gap_pct: 0.56,
        tailored_base_surge_gap_pct: 0.06,
        a3c_gap_pct: 0.52,
        notes: "Bar labels transcribed from Figure 9.",
    },
    PublishedOptimalityGapReference {
        source: GIJSBRECHTS_2022_REFERENCE.source,
        url: GIJSBRECHTS_2022_REFERENCE.url,
        instance_name: "dual_l2_ce110",
        capped_dual_index_gap_pct: 0.03,
        dual_index_gap_pct: 0.18,
        single_index_gap_pct: 1.03,
        tailored_base_surge_gap_pct: 0.99,
        a3c_gap_pct: 0.80,
        notes: "Bar labels transcribed from Figure 9.",
    },
    PublishedOptimalityGapReference {
        source: GIJSBRECHTS_2022_REFERENCE.source,
        url: GIJSBRECHTS_2022_REFERENCE.url,
        instance_name: "dual_l3_ce105",
        capped_dual_index_gap_pct: 0.00,
        dual_index_gap_pct: 0.27,
        single_index_gap_pct: 0.98,
        tailored_base_surge_gap_pct: 0.01,
        a3c_gap_pct: 0.82,
        notes: "Bar labels transcribed from Figure 9.",
    },
    PublishedOptimalityGapReference {
        source: GIJSBRECHTS_2022_REFERENCE.source,
        url: GIJSBRECHTS_2022_REFERENCE.url,
        instance_name: "dual_l3_ce110",
        capped_dual_index_gap_pct: 0.06,
        dual_index_gap_pct: 0.36,
        single_index_gap_pct: 2.11,
        tailored_base_surge_gap_pct: 0.71,
        a3c_gap_pct: 0.51,
        notes: "Bar labels transcribed from Figure 9.",
    },
    PublishedOptimalityGapReference {
        source: GIJSBRECHTS_2022_REFERENCE.source,
        url: GIJSBRECHTS_2022_REFERENCE.url,
        instance_name: "dual_l4_ce105",
        capped_dual_index_gap_pct: 0.00,
        dual_index_gap_pct: 0.36,
        single_index_gap_pct: 1.43,
        tailored_base_surge_gap_pct: 0.00,
        a3c_gap_pct: 1.85,
        notes: "Bar labels transcribed from Figure 9.",
    },
    PublishedOptimalityGapReference {
        source: GIJSBRECHTS_2022_REFERENCE.source,
        url: GIJSBRECHTS_2022_REFERENCE.url,
        instance_name: "dual_l4_ce110",
        capped_dual_index_gap_pct: 0.11,
        dual_index_gap_pct: 0.49,
        single_index_gap_pct: 2.44,
        tailored_base_surge_gap_pct: 0.58,
        a3c_gap_pct: 1.33,
        notes: "Bar labels transcribed from Figure 9.",
    },
];

pub const WORKED_TRANSITION_REFERENCE: WorkedTransitionReference = WorkedTransitionReference {
    source: GIJSBRECHTS_2022_REFERENCE.source,
    url: GIJSBRECHTS_2022_REFERENCE.url,
    initial_reduced_state: &[8, 3, 1],
    regular_order: 2,
    expedited_order: 1,
    realized_demand: 4,
    regular_order_cost: 100.0,
    expedited_order_cost: 105.0,
    holding_cost: 5.0,
    shortage_cost: 495.0,
    expected_next_reduced_state: &[8, 1, 2],
    expected_period_cost: 330.0,
};

pub const VERIFICATION_PROBLEM_INSTANCE: VerificationProblemInstance =
    VerificationProblemInstance {
        name: "dual_sourcing_l2_ce105_rust_benchmark",
        reference_instance_name: "dual_l2_ce105",
        inventory_lower: -12,
        inventory_upper: 24,
        solver_tolerance: 1e-8,
        max_iterations: 250,
        search_seed: 123,
        search_horizon: 6000,
        warm_up_periods_ratio: 0.2,
        exact_abs_tolerance: 1e-6,
        literature_gap_abs_tolerance_pct: 0.01,
    };

pub fn list_reference_instances() -> &'static [DualSourcingReferenceInstance] {
    &DUAL_SOURCING_REFERENCE_INSTANCES
}

/// The two additive CDI-optimality-taxonomy probe instances (Tier B / Tier C).
/// Kept separate from the six published Gijsbrechts rows so the published-grid
/// validators and Figure-9 drift guards see exactly the six rows they expect.
pub fn list_taxonomy_instances() -> &'static [DualSourcingReferenceInstance] {
    &DUAL_SOURCING_TAXONOMY_INSTANCES
}

pub fn get_reference_instance(name: &str) -> Option<&'static DualSourcingReferenceInstance> {
    DUAL_SOURCING_REFERENCE_INSTANCES
        .iter()
        .chain(DUAL_SOURCING_TAXONOMY_INSTANCES.iter())
        .find(|instance| instance.name == name)
}

pub fn get_primary_reference_instance() -> &'static DualSourcingReferenceInstance {
    &PRIMARY_REFERENCE_INSTANCE
}

pub fn get_figure_9_gap_reference(name: &str) -> Option<&'static PublishedOptimalityGapReference> {
    FIGURE_9_GAP_REFERENCES
        .iter()
        .find(|reference| reference.instance_name == name)
}
