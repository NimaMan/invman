#![allow(dead_code)]

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
pub struct RandomYieldReferenceInstance {
    pub name: &'static str,
    pub source: &'static str,
    pub url: &'static str,
    pub literature_verified: bool,
    pub verification_source: &'static str,
    pub periods: usize,
    pub lead_time: usize,
    pub demand_mean: f64,
    pub success_probability: f64,
    pub holding_cost: f64,
    pub shortage_cost: f64,
    pub procurement_cost: f64,
    pub discount_factor: f64,
    pub initial_inventory_level: f64,
    pub initial_pipeline_orders: &'static [f64],
    pub notes: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExactVerificationReference {
    pub source: &'static str,
    pub url: &'static str,
    pub literature_verified: bool,
    pub verification_source: &'static str,
    pub periods: usize,
    pub lead_time: usize,
    pub success_probability: f64,
    pub holding_cost: f64,
    pub shortage_cost: f64,
    pub procurement_cost: f64,
    pub discount_factor: f64,
    pub initial_inventory_level: i32,
    pub initial_pipeline_orders: &'static [u32],
    pub demand_support: &'static [u32],
    pub demand_probabilities: &'static [f64],
    pub max_order_quantity: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LiteratureBenchmarkFamily {
    pub name: &'static str,
    pub source: &'static str,
    pub url: &'static str,
    pub horizon_type: &'static str,
    pub demand_family: &'static str,
    pub yield_model: &'static str,
    pub model_match: &'static str,
    pub access_level: &'static str,
    pub reported_numbers_available: bool,
    pub repo_assertion_basis: &'static str,
    pub benchmark_policies: &'static [&'static str],
    pub lead_times: &'static [usize],
    pub demand_means: &'static [f64],
    pub demand_cvs: &'static [f64],
    pub success_probabilities: &'static [f64],
    pub critical_ratios: &'static [f64],
    pub yield_rate_mean_cv_pairs: &'static [(f64, f64)],
    pub notes: &'static str,
}

pub const EMPTY_USIZE_SLICE: &[usize] = &[];
pub const EMPTY_F64_SLICE: &[f64] = &[];
pub const EMPTY_MEAN_CV_PAIR_SLICE: &[(f64, f64)] = &[];

pub const YAN_2026_REFERENCE: PublishedBenchmarkReference = PublishedBenchmarkReference {
    source: "Yan et al. (2026), Computers & Operations Research 186, 107305",
    url: "https://doi.org/10.1016/j.cor.2025.107305",
    benchmark_policies: &["optimal_dp", "linear_inflation", "weighted_newsvendor", "gdsh", "drl"],
    reported_numbers_available: false,
    numbers_anchor_repo_assertions: false,
    notes: "This paper defines the finite-horizon discounted all-or-nothing random-yield problem with positive lead time and backlogging. The accessible record exposes the model and benchmark policy families, but not a public table of exact per-instance benchmark numbers that the repo can assert against.",
};

pub const INDERFURTH_2015_REFERENCE: PublishedBenchmarkReference = PublishedBenchmarkReference {
    source: "Inderfurth and Kiesmuller (2015), Exact and heuristic linear-inflation policies for an inventory model with random yield and arbitrary lead times, European Journal of Operational Research 245(1), 109-120",
    url: "https://www.fww.ovgu.de/fww_media/femm/femm_2013/2013_07.pdf",
    benchmark_policies: &["linear_inflation"],
    reported_numbers_available: true,
    numbers_anchor_repo_assertions: false,
    notes: "This paper gives the canonical linear-inflation rule q = F * (S - X)^+ with inflation factor F = 1/E[yield rate] (so F = 1/p for binomial yield). VERIFIED 2026-05 from the open working-paper copy: it is an INFINITE-HORIZON AVERAGE-COST model with PER-UNIT BINOMIAL yield (Y(Q) ~ Binomial(Q,p), each unit independently good) or STOCHASTICALLY PROPORTIONAL yield (Y(Q) = Z*Q). This is a DIFFERENT yield model than the repo's finite-horizon ALL-OR-NOTHING batch yield (whole batch arrives with prob p or not at all), so its published numbers cannot anchor repo assertions; only the F = 1/p inflation factor carries over.",
};

pub const CHEN_2018_REFERENCE: PublishedBenchmarkReference = PublishedBenchmarkReference {
    source: "Chen et al. (2018), Heuristics and Bounds for an Inventory System with an All-or-Nothing Yield Pattern and Lead-times",
    url: "https://dblp.org/rec/conf/soli/ChenHYY18",
    benchmark_policies: &["weighted_newsvendor"],
    reported_numbers_available: false,
    numbers_anchor_repo_assertions: false,
    notes: "The later Yan et al. (2026) paper identifies this weighted newsvendor heuristic as the main all-or-nothing benchmark policy and describes it as a sample-path weighted-average gap rule over pipeline-yield realizations. We have not recovered a public table of benchmark numbers from this source.",
};

pub const INDERFURTH_2015_DEMAND_MEANS: &[f64] = &[20.0];
pub const INDERFURTH_2015_DEMAND_CVS: &[f64] = &[0.1, 0.2, 0.3, 0.5, 0.75];
pub const INDERFURTH_2015_CRITICAL_RATIOS: &[f64] = &[0.85, 0.90, 0.95, 0.97, 0.99, 0.995];
pub const INDERFURTH_2015_BINOMIAL_SUCCESS_PROBABILITIES: &[f64] = &[0.5, 0.7, 0.9];
pub const INDERFURTH_2015_ZERO_LEAD_TIME: &[usize] = &[0];
pub const INDERFURTH_2015_POSITIVE_LEAD_TIMES: &[usize] = &[2, 5, 10];
pub const INDERFURTH_2015_PROPORTIONAL_YIELD_PAIRS: &[(f64, f64)] = &[
    (0.85, 0.1),
    (0.85, 0.2),
    (0.75, 0.2),
    (0.5, 0.2),
    (0.5, 0.4),
    (0.5, 0.5774),
];

pub const YAN_2026_SMALL_SCALE_FAMILY: LiteratureBenchmarkFamily = LiteratureBenchmarkFamily {
    name: "yan2026_small_scale_exact_dp_family",
    source: YAN_2026_REFERENCE.source,
    url: YAN_2026_REFERENCE.url,
    horizon_type: "finite_horizon_discounted",
    demand_family: "data_driven_or_sampled_distribution",
    yield_model: "all_or_nothing",
    model_match: "exact_model_match",
    access_level: "preview_only",
    reported_numbers_available: false,
    repo_assertion_basis: "do_not_use_for_repo_assertions",
    benchmark_policies: YAN_2026_REFERENCE.benchmark_policies,
    lead_times: EMPTY_USIZE_SLICE,
    demand_means: EMPTY_F64_SLICE,
    demand_cvs: EMPTY_F64_SLICE,
    success_probabilities: EMPTY_F64_SLICE,
    critical_ratios: EMPTY_F64_SLICE,
    yield_rate_mean_cv_pairs: EMPTY_MEAN_CV_PAIR_SLICE,
    notes: "The accessible record confirms a small-scale exact-DP experiment family and a larger heuristic/DRL experiment family for the all-or-nothing problem, but it does not expose public benchmark numbers that can be copied into repo assertions.",
};

pub const CHEN_2018_WNH_FAMILY: LiteratureBenchmarkFamily = LiteratureBenchmarkFamily {
    name: "chen2018_weighted_newsvendor_family",
    source: CHEN_2018_REFERENCE.source,
    url: CHEN_2018_REFERENCE.url,
    horizon_type: "finite_horizon_discounted_or_simulated",
    demand_family: "all_or_nothing_benchmark_family_not_publicly_recovered",
    yield_model: "all_or_nothing",
    model_match: "exact_model_match",
    access_level: "bibliographic_only",
    reported_numbers_available: false,
    repo_assertion_basis: "do_not_use_for_repo_assertions",
    benchmark_policies: CHEN_2018_REFERENCE.benchmark_policies,
    lead_times: EMPTY_USIZE_SLICE,
    demand_means: EMPTY_F64_SLICE,
    demand_cvs: EMPTY_F64_SLICE,
    success_probabilities: EMPTY_F64_SLICE,
    critical_ratios: EMPTY_F64_SLICE,
    yield_rate_mean_cv_pairs: EMPTY_MEAN_CV_PAIR_SLICE,
    notes: "This source is the main published anchor for the weighted-newsvendor heuristic in the all-or-nothing setting, but the accessible bibliographic record does not expose reusable benchmark numbers.",
};

pub const INDERFURTH_2015_ZERO_LT_BINOMIAL_FAMILY: LiteratureBenchmarkFamily =
    LiteratureBenchmarkFamily {
        name: "inderfurth2015_zero_lt_binomial_grid",
        source: INDERFURTH_2015_REFERENCE.source,
        url: INDERFURTH_2015_REFERENCE.url,
        horizon_type: "infinite_horizon_average_cost",
        demand_family: "discretized_normal_or_gamma",
        yield_model: "binomial",
        model_match: "partial_match_general_random_yield",
        access_level: "grid_summary_accessible",
        reported_numbers_available: true,
        repo_assertion_basis: "related_model_aggregate_only",
        benchmark_policies: &["optimal_linear_inflation", "markov_chain", "steady_state"],
        lead_times: INDERFURTH_2015_ZERO_LEAD_TIME,
        demand_means: INDERFURTH_2015_DEMAND_MEANS,
        demand_cvs: INDERFURTH_2015_DEMAND_CVS,
        success_probabilities: INDERFURTH_2015_BINOMIAL_SUCCESS_PROBABILITIES,
        critical_ratios: INDERFURTH_2015_CRITICAL_RATIOS,
        yield_rate_mean_cv_pairs: EMPTY_MEAN_CV_PAIR_SLICE,
        notes: "Section 3.3 uses mean demand 20, holding cost 1, demand CV values {0.1, 0.2, 0.3, 0.5, 0.75}, critical ratios {0.85, 0.90, 0.95, 0.97, 0.99, 0.995}, and binomial-yield probabilities {0.5, 0.7, 0.9}. Public numeric results exist, but this is not the same yield model as the repo's all-or-nothing environment.",
    };

pub const INDERFURTH_2015_POSITIVE_LT_BINOMIAL_FAMILY: LiteratureBenchmarkFamily =
    LiteratureBenchmarkFamily {
        name: "inderfurth2015_positive_lt_binomial_grid",
        source: INDERFURTH_2015_REFERENCE.source,
        url: INDERFURTH_2015_REFERENCE.url,
        horizon_type: "infinite_horizon_average_cost",
        demand_family: "discretized_normal_or_gamma",
        yield_model: "binomial",
        model_match: "partial_match_general_random_yield",
        access_level: "grid_summary_accessible",
        reported_numbers_available: true,
        repo_assertion_basis: "related_model_aggregate_only",
        benchmark_policies: &["optimal_linear_inflation", "markov_chain", "steady_state"],
        lead_times: INDERFURTH_2015_POSITIVE_LEAD_TIMES,
        demand_means: INDERFURTH_2015_DEMAND_MEANS,
        demand_cvs: INDERFURTH_2015_DEMAND_CVS,
        success_probabilities: INDERFURTH_2015_BINOMIAL_SUCCESS_PROBABILITIES,
        critical_ratios: INDERFURTH_2015_CRITICAL_RATIOS,
        yield_rate_mean_cv_pairs: EMPTY_MEAN_CV_PAIR_SLICE,
        notes: "Section 4.3 reuses the Section 3.3 grid and evaluates lead times {2, 5, 10} by simulation. Public numeric results exist, but this benchmark family is not directly executable in the current all-or-nothing model.",
    };

pub const INDERFURTH_2015_ZERO_LT_PROPORTIONAL_FAMILY: LiteratureBenchmarkFamily =
    LiteratureBenchmarkFamily {
        name: "inderfurth2015_zero_lt_proportional_grid",
        source: INDERFURTH_2015_REFERENCE.source,
        url: INDERFURTH_2015_REFERENCE.url,
        horizon_type: "infinite_horizon_average_cost",
        demand_family: "discretized_normal_or_gamma",
        yield_model: "stochastically_proportional",
        model_match: "special_case_related_to_all_or_nothing",
        access_level: "grid_summary_accessible",
        reported_numbers_available: true,
        repo_assertion_basis: "related_model_aggregate_only",
        benchmark_policies: &["optimal_linear_inflation", "markov_chain", "steady_state"],
        lead_times: INDERFURTH_2015_ZERO_LEAD_TIME,
        demand_means: INDERFURTH_2015_DEMAND_MEANS,
        demand_cvs: INDERFURTH_2015_DEMAND_CVS,
        success_probabilities: EMPTY_F64_SLICE,
        critical_ratios: INDERFURTH_2015_CRITICAL_RATIOS,
        yield_rate_mean_cv_pairs: INDERFURTH_2015_PROPORTIONAL_YIELD_PAIRS,
        notes: "The all-or-nothing model is a Bernoulli special case of proportional yield, but the published proportional-yield grid uses more general mean/CV pairs than order-level Bernoulli success alone can represent. Public numeric results exist only for the broader related model.",
    };

pub const INDERFURTH_2015_POSITIVE_LT_PROPORTIONAL_FAMILY: LiteratureBenchmarkFamily =
    LiteratureBenchmarkFamily {
        name: "inderfurth2015_positive_lt_proportional_grid",
        source: INDERFURTH_2015_REFERENCE.source,
        url: INDERFURTH_2015_REFERENCE.url,
        horizon_type: "infinite_horizon_average_cost",
        demand_family: "discretized_normal_or_gamma",
        yield_model: "stochastically_proportional",
        model_match: "special_case_related_to_all_or_nothing",
        access_level: "grid_summary_accessible",
        reported_numbers_available: true,
        repo_assertion_basis: "related_model_aggregate_only",
        benchmark_policies: &["optimal_linear_inflation", "markov_chain", "steady_state"],
        lead_times: INDERFURTH_2015_POSITIVE_LEAD_TIMES,
        demand_means: INDERFURTH_2015_DEMAND_MEANS,
        demand_cvs: INDERFURTH_2015_DEMAND_CVS,
        success_probabilities: EMPTY_F64_SLICE,
        critical_ratios: INDERFURTH_2015_CRITICAL_RATIOS,
        yield_rate_mean_cv_pairs: INDERFURTH_2015_PROPORTIONAL_YIELD_PAIRS,
        notes: "This family is the closest literature grid we currently have to the all-or-nothing setting, but it is still only a partial match because the published proportional-yield pairs are more general than Bernoulli shipment success. Public numeric results exist only for the broader related model.",
    };

pub const LITERATURE_BENCHMARK_FAMILIES: &[LiteratureBenchmarkFamily] = &[
    YAN_2026_SMALL_SCALE_FAMILY,
    CHEN_2018_WNH_FAMILY,
    INDERFURTH_2015_ZERO_LT_BINOMIAL_FAMILY,
    INDERFURTH_2015_POSITIVE_LT_BINOMIAL_FAMILY,
    INDERFURTH_2015_ZERO_LT_PROPORTIONAL_FAMILY,
    INDERFURTH_2015_POSITIVE_LT_PROPORTIONAL_FAMILY,
];

pub const PRIMARY_REFERENCE_INSTANCE: RandomYieldReferenceInstance = RandomYieldReferenceInstance {
    name: "yan2026_style_lt2_p075_discounted",
    source: YAN_2026_REFERENCE.source,
    url: YAN_2026_REFERENCE.url,
    literature_verified: false,
    verification_source: "repo_exact_solver_not_verified_against_literature",
    periods: 12,
    lead_time: 2,
    demand_mean: 4.0,
    success_probability: 0.75,
    holding_cost: 1.0,
    shortage_cost: 9.0,
    procurement_cost: 1.0,
    discount_factor: 0.99,
    initial_inventory_level: 6.0,
    initial_pipeline_orders: &[4.0, 3.0],
    notes: "Canonical first learned-policy benchmark for the repo. This is a repo-native instance shaped after the small-lead-time discounted setting emphasized by Yan et al. (2026), not a verbatim literature table row. We use it because the paper does not expose public benchmark numbers for direct verification.",
};

pub const VERIFICATION_DEMAND_SUPPORT: &[u32] = &[0, 1, 2, 3, 4, 5];
pub const VERIFICATION_DEMAND_PROBABILITIES: &[f64] = &[0.05, 0.15, 0.30, 0.25, 0.15, 0.10];

pub const VERIFICATION_PROBLEM_INSTANCE: ExactVerificationReference = ExactVerificationReference {
    source: YAN_2026_REFERENCE.source,
    url: YAN_2026_REFERENCE.url,
    literature_verified: false,
    verification_source: "repo_exact_solver_not_verified_against_literature",
    periods: 5,
    lead_time: 2,
    success_probability: 0.75,
    holding_cost: 1.0,
    shortage_cost: 9.0,
    procurement_cost: 1.0,
    discount_factor: 0.99,
    initial_inventory_level: 4,
    initial_pipeline_orders: &[3, 2],
    demand_support: VERIFICATION_DEMAND_SUPPORT,
    demand_probabilities: VERIFICATION_DEMAND_PROBABILITIES,
    max_order_quantity: 8,
};
