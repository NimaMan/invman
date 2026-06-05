#![allow(dead_code)]

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PublishedBenchmarkReference {
    pub source: &'static str,
    pub url: &'static str,
    pub benchmark_policies: &'static [&'static str],
    pub notes: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PublishedPolicyBenchmark {
    pub source: &'static str,
    pub url: &'static str,
    pub policy_name: &'static str,
    pub per_agent_mean_costs: &'static [f64],
    pub total_mean_cost: f64,
    pub notes: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecentralizedInventoryReferenceInstance {
    pub name: &'static str,
    pub source: &'static str,
    pub url: &'static str,
    pub num_agents: usize,
    pub benchmark_customer_demands: Option<&'static [usize]>,
    pub shipment_lead_times: &'static [usize],
    pub order_lead_times: &'static [usize],
    pub initial_on_hand_inventory: &'static [usize],
    pub initial_backlog: &'static [usize],
    pub initial_shipment_pipelines: &'static [&'static [usize]],
    pub initial_order_pipelines: &'static [&'static [usize]],
    pub initial_last_received_shipments: &'static [usize],
    pub initial_last_received_orders: &'static [usize],
    pub initial_forecast_orders: &'static [f64],
    pub initial_last_actions: &'static [usize],
    pub holding_costs: &'static [f64],
    pub backlog_costs: &'static [f64],
    pub sterman_smoothing_factors: &'static [f64],
    pub sterman_target_positions: &'static [f64],
    pub sterman_adjustment_times: &'static [f64],
    pub sterman_supply_line_weights: &'static [f64],
    pub published_sterman_benchmark: Option<PublishedPolicyBenchmark>,
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
    pub initial_on_hand_inventory: &'static [usize],
    pub initial_backlog: &'static [usize],
    pub initial_shipment_pipelines: &'static [&'static [usize]],
    pub initial_order_pipelines: &'static [&'static [usize]],
    pub initial_last_received_shipments: &'static [usize],
    pub initial_last_received_orders: &'static [usize],
    pub initial_forecast_orders: &'static [f64],
    pub initial_last_actions: &'static [usize],
    pub demand_smoothing_factors: &'static [f64],
    pub holding_costs: &'static [f64],
    pub backlog_costs: &'static [f64],
    pub customer_demand_support: &'static [u32],
    pub customer_demand_probabilities: &'static [f64],
    pub max_order_quantities: &'static [usize],
    pub base_stock_levels: &'static [usize],
    pub sterman_target_positions: &'static [f64],
    pub sterman_adjustment_times: &'static [f64],
    pub sterman_supply_line_weights: &'static [f64],
    pub notes: &'static str,
}

pub const CLASSIC_BEER_GAME_CUSTOMER_DEMANDS: &[usize] = &[
    4, 4, 4, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8,
];

pub const STERMAN_1989_REFERENCE: PublishedBenchmarkReference = PublishedBenchmarkReference {
    source: "Sterman, J. D. (1989), Modeling Managerial Behavior: Misperceptions of Feedback in a Dynamic Decision Making Experiment, Management Science 35(3):321-339",
    url: "https://doi.org/10.1287/mnsc.35.3.321",
    benchmark_policies: &["sterman_anchor_adjust"],
    notes: "Classic four-stage Beer Game benchmark. Citation verified against Crossref (DOI 10.1287/mnsc.35.3.321): Management Science vol 35 iss 3 pp 321-339, March 1989, sole author John D. Sterman (MIT Sloan). The paper proposes the anchor-and-adjust ordering heuristic and reports a benchmark cost for the optimized policy; the exact board-game equations are carried and replicated by Edali & Yasarcan (2014). NOTE: the per-stage [46,50,54,54]/total 204 figures stored here are the values produced by the repo's port of the Edali & Yasarcan equations (verification/classic_board_game.rs) under the published parameters; they were NOT re-confirmed against a freely-quotable line of Sterman (1989) during the 2026-05-31 citation audit (the only open full text is an image-only scan), and they are reproduced ONLY by the closed-form board-game simulator, not by the reusable env.rs transition that the heuristics and learned soft-tree run on.",
};

pub const OROOJLOYJADID_2021_REFERENCE: PublishedBenchmarkReference = PublishedBenchmarkReference {
    source: "Oroojlooyjadid, Nazari, Snyder & Takac, A Deep Q-Network for the Beer Game: Deep Reinforcement Learning for Inventory Optimization, Manufacturing & Service Operations Management 24(1):285-304 (issue year 2022; Articles-in-Advance online 2021)",
    url: "https://doi.org/10.1287/msom.2020.0939",
    benchmark_policies: &["base_stock", "sterman_anchor_adjust", "dqn"],
    notes: "Background RL paper on decentralized Beer-Game control with local observations only. Citation verified against Crossref (DOI 10.1287/msom.2020.0939): MSOM vol 24 iss 1 pp 285-304, bound-issue year 2022, posted Articles-in-Advance in 2021 (the constant name keeps the 2021 online-first year). Authors: Afshin Oroojlooyjadid, MohammadReza Nazari, Lawrence V. Snyder, Martin Takac. The paper reports a 100-period uniform-demand benchmark and a Sterman row, but the public paper formula, timing description, and released code do not line up tightly enough for the repo to carry that row as an executable verification anchor.",
};

// CANER_2014_REFERENCE: historical constant name kept stable for internal references.
// CITATION CORRECTION: the actual JASSS 2014 paper is by Edali & Yasarcan, not "Caner et al.".
// The author attribution that previously read "Caner et al." was wrong; the URL and content
// (the exact board-game R reconstruction) were always the Edali & Yasarcan paper.
pub const CANER_2014_REFERENCE: PublishedBenchmarkReference = PublishedBenchmarkReference {
    source: "Edali, M. & Yasarcan, H. (2014), A Mathematical Model of the Beer Game, Journal of Artificial Societies and Social Simulation (JASSS) 17(4):2, DOI 10.18564/jasss.2555",
    url: "https://www.jasss.org/17/4/2.html",
    benchmark_policies: &["sterman_anchor_adjust"],
    notes: "Edali & Yasarcan reconstruct the board-game Beer Game exactly and provide public R code for the verification benchmark. Citation verified against Crossref (DOI 10.18564/jasss.2555, resolves to jasss.org/17/4/2.html): authors Mert Edali & Hakan Yasarcan, JASSS vol 17 iss 4 article 2, 2014. The paper states (sec 5) that with theta=0, sat=1, wsl=1, S'=[28,28,28,20], h=0.5, p=1.0 it obtains 'the exact same benchmark cost values reported by Sterman' (those exact per-stage figures are not quoted in the open text). The repo port of that R code (verification/classic_board_game.rs) yields per-stage [46,50,54,54], total 204. (The previous 'Caner et al.' attribution was a citation error; constant name retained to avoid churn.)",
};

pub const MOUSA_2024_REFERENCE: PublishedBenchmarkReference = PublishedBenchmarkReference {
    source: "Mousa, van de Berg, Kotecha, del Rio-Chanona & Mowbray (2024), An analysis of multi-agent reinforcement learning for decentralized inventory control systems, Computers & Chemical Engineering 188, 108783",
    url: "https://doi.org/10.1016/j.compchemeng.2024.108783",
    benchmark_policies: &["marl", "decentralized local-information baselines"],
    notes: "Citation verified against Crossref (DOI 10.1016/j.compchemeng.2024.108783): Computers & Chemical Engineering vol 188 article 108783, 2024; authors Marwan Mousa, Damien van de Berg, Niki Kotecha, Ehecatl Antonio del Rio-Chanona, Max Mowbray. This paper broadens decentralized inventory control beyond the four-stage Beer Game and motivates local-information policy interfaces for serial and networked supply chains. Background context only; no benchmark number is carried as a repo verification anchor.",
};

pub const STERMAN_1989_CLASSIC_BENCHMARK: PublishedPolicyBenchmark = PublishedPolicyBenchmark {
    source: STERMAN_1989_REFERENCE.source,
    url: STERMAN_1989_REFERENCE.url,
    policy_name: "sterman_anchor_adjust",
    per_agent_mean_costs: &[46.0, 50.0, 54.0, 54.0],
    total_mean_cost: 204.0,
    notes: "Classic 36-week Beer Game benchmark costs for the optimized anchor-and-adjust policy. Edali & Yasarcan (2014) state that their exact board-game reconstruction reproduces these Sterman (1989) benchmark values exactly, and the repo port verification/classic_board_game.rs confirms [46,50,54,54]/204. IMPORTANT: this 204 is a property of the closed-form board-game bookkeeping ONLY. Running the repo's reusable env.rs anchor-and-adjust heuristic with these same parameters on PRIMARY_REFERENCE_INSTANCE yields 378 (measured via decentralized_inventory_control_policy_rollout_from_paths), so env.rs is NOT calibrated to this published anchor.",
};

pub const PRIMARY_REFERENCE_INSTANCE: DecentralizedInventoryReferenceInstance =
    DecentralizedInventoryReferenceInstance {
        name: "beer_game_classic_four_stage",
        source: CANER_2014_REFERENCE.source,
        url: CANER_2014_REFERENCE.url,
        num_agents: 4,
        benchmark_customer_demands: Some(CLASSIC_BEER_GAME_CUSTOMER_DEMANDS),
        shipment_lead_times: &[2, 2, 2, 2],
        order_lead_times: &[0, 1, 1, 1],
        initial_on_hand_inventory: &[12, 12, 12, 12],
        initial_backlog: &[0, 0, 0, 0],
        initial_shipment_pipelines: &[&[4, 4], &[4, 4], &[4, 4], &[4, 4]],
        initial_order_pipelines: &[&[], &[4], &[4], &[4]],
        initial_last_received_shipments: &[4, 4, 4, 4],
        initial_last_received_orders: &[4, 4, 4, 4],
        initial_forecast_orders: &[4.0, 4.0, 4.0, 4.0],
        initial_last_actions: &[4, 4, 4, 4],
        holding_costs: &[0.5, 0.5, 0.5, 0.5],
        backlog_costs: &[1.0, 1.0, 1.0, 1.0],
        sterman_smoothing_factors: &[0.0, 0.0, 0.0, 0.0],
        sterman_target_positions: &[28.0, 28.0, 28.0, 20.0],
        sterman_adjustment_times: &[1.0, 1.0, 1.0, 1.0],
        sterman_supply_line_weights: &[1.0, 1.0, 1.0, 1.0],
        published_sterman_benchmark: Some(STERMAN_1989_CLASSIC_BENCHMARK),
        notes: "Classic four-stage Beer Game state with the canonical 36-week demand path 4,4,4,4,8,...,8. The state and Sterman parameter values follow the exact board-game reconstruction reported by Edali & Yasarcan (2014), which in turn reproduces the benchmark costs from Sterman (1989). These parameters reproduce 204 ONLY inside verification/classic_board_game.rs; the reusable env.rs transition is a different (also-valid) decentralized serial MDP whose pipeline/supply-line bookkeeping differs, and it does NOT reproduce 204 under these parameters (anchor-and-adjust -> 378; best simple base-stock S=24 -> 278). Treat this instance as the literature anchor for the closed-form simulator, NOT as a verification target for env.rs.",
    };

pub const VERIFICATION_CUSTOMER_DEMAND_SUPPORT: &[u32] = &[0, 1];
pub const VERIFICATION_CUSTOMER_DEMAND_PROBABILITIES: &[f64] = &[0.5, 0.5];

pub const VERIFICATION_PROBLEM_INSTANCE: ExactVerificationReference = ExactVerificationReference {
    source: "Repo exact verification instance for decentralized inventory control",
    url: "",
    literature_verified: false,
    verification_source: "repo_exact_solver_not_verified_against_literature",
    periods: 3,
    discount_factor: 0.99,
    initial_on_hand_inventory: &[2, 1],
    initial_backlog: &[0, 0],
    initial_shipment_pipelines: &[&[1], &[0]],
    initial_order_pipelines: &[&[], &[1]],
    initial_last_received_shipments: &[1, 0],
    initial_last_received_orders: &[1, 1],
    initial_forecast_orders: &[1.0, 1.0],
    initial_last_actions: &[1, 1],
    demand_smoothing_factors: &[0.0, 0.0],
    holding_costs: &[0.5, 0.5],
    backlog_costs: &[1.0, 1.0],
    customer_demand_support: VERIFICATION_CUSTOMER_DEMAND_SUPPORT,
    customer_demand_probabilities: VERIFICATION_CUSTOMER_DEMAND_PROBABILITIES,
    max_order_quantities: &[4, 4],
    base_stock_levels: &[3, 3],
    sterman_target_positions: &[4.0, 4.0],
    sterman_adjustment_times: &[1.0, 1.0],
    sterman_supply_line_weights: &[1.0, 1.0],
    notes: "Repo-native exact verifier on a reduced two-agent Beer-Game-shaped serial chain. The instance keeps local forecasts and positive lead times but stays small enough for routine finite-horizon DP assertions.",
};
