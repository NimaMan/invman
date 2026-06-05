// ============================================================================
// vendor_managed_inventory / literature / references.rs
//
// OBJECTIVE
//   Hold the literature instances and published-value records for the
//   vendor-managed-inventory (VMI) family, and state HONESTLY which of them is
//   literature-verified per the repo rule (an in-crate test must re-run the
//   env/solver and reproduce a number PRINTED IN A PAPER within tolerance).
//
// PROVENANCE / HONESTY NOTE (corrected 2026-06-04)
//   The primary paper this family targets is:
//
//     Sui, Z., A. Gosavi, and L. Lin (2010). "A Reinforcement Learning Approach
//     for Inventory Replenishment in Vendor-Managed Inventory Systems With
//     Consignment Inventory." Engineering Management Journal 22(4): 44-53.
//     DOI: 10.1080/10429247.2010.11431878.
//
//   Earlier revisions of this file mis-attributed that DOI and title to
//   "Giannoccaro and Pontrandolfo (2010)". That attribution was WRONG: the DOI
//   10.1080/10429247.2010.11431878 and the exact title belong to Sui/Gosavi/Lin.
//   All symbols are renamed to SUI_GOSAVI_LIN_2010_* accordingly.
//
//   WHAT IS REPRODUCIBLE BY THIS FAMILY, AND FROM WHERE
//   - The only openly accessible artifact carrying concrete numbers is the
//     instructor TEACHING CASE STUDY:
//       Gosavi, A. "CASE STUDY FOR VENDOR-MANAGED INVENTORY (BASED ON SUI,
//       GOSAVI, & LIN, 2010)", Missouri University of Science and Technology,
//       Sept 7, 2010 (PDF marked "Copyrighted Material 2020").
//     It self-describes as a class case study ("As discussed in class ...") and
//     states "This case study is based on the journal article: Sui, Gosavi, and
//     Lin (2010)." Its "Worked out example with data from the paper" computes,
//     for the single-retailer single-product newsvendor:
//       mu = lambda(a+b)/2 = 0.25*3/2 = 0.375
//       sigma^2 = 0.25*[1/12 + 9/4] = 0.5833
//       mu_cycle = 0.375*40 = 15 ; sigma^2_cycle = 40*0.5833 + 0.375^2*50 = 30.36
//       MDH order-up-to S = 15 ; six-sigma S = 15+3*sqrt(30.36) = 31.53
//       newsvendor S = 15 + Phi^{-1}(4/4.06)*sqrt(30.36) = 26.96
//     evaluate_newsvendor_worked_case(...) re-derives these exactly; the
//     verification test asserts reproduction within tolerance.
//
//   WHY literature_verified = false FOR THE PEER-REVIEWED PAPER
//   - Those reproduced order-up-to numbers (15 / 31.53 / 26.96) are GOSAVI's own
//     worked computation in the instructor HANDOUT, not a results row PRINTED in
//     the peer-reviewed Sui/Gosavi/Lin (2010) EMJ paper. Per the repo rule, an
//     instructor/handout number is explicitly NOT literature verification.
//   - The peer-reviewed paper's experimental results table (RL vs. newsvendor
//     profit per case, pp. 44-53) is paywalled (Taylor & Francis) and is not
//     openly reproducible; no open source quotes its numeric rows. The
//     handout's input parameters (T-distribution {30,40,50} w.p. {.25,.5,.25},
//     d~UNIF(1,2), lambda=0.25, h=0.06, p=4.00) are described as "data from the
//     paper", but the published profit/cost rows themselves are not accessible.
//   - The repo's continuous-time 10-retailer/2-product truck-dispatch
//     PaperVendorManagedInventoryModel and its 8 case definitions are a
//     REPO-CONSTRUCTED interpretation of the paper's structure; their parameter
//     rows are not transcribed from a published table and their profit rows do
//     not reproduce the published table (the row-level gap is statistically
//     meaningful and the demand-signal process is not public enough to resolve).
//
//   STATUS SUMMARY
//   - literature_verified (peer-reviewed paper number): FALSE.
//   - The Gosavi instructor case-study worked example IS reproduced exactly by
//     an executing in-crate test; it is kept as a clearly-labeled instructor
//     worked example, NOT as peer-reviewed-paper verification.
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PublishedBenchmarkReference {
    pub source: &'static str,
    pub url: &'static str,
    /// True only if an in-crate test re-runs the env/solver and reproduces a
    /// number PRINTED IN THE PEER-REVIEWED PAPER within a stated tolerance.
    /// False here: only the Gosavi instructor case-study worked example is
    /// reproduced (a teaching handout, not the peer-reviewed paper).
    pub literature_verified: bool,
    /// Precisely what the reproduction test re-runs and against which artifact.
    pub verification_source: &'static str,
    pub benchmark_policies: &'static [&'static str],
    pub notes: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VendorManagedInventoryReferenceInstance {
    pub name: &'static str,
    pub source: &'static str,
    pub url: &'static str,
    pub periods: usize,
    pub demand_distribution_kind: &'static str,
    pub demand_mean: f64,
    pub initial_dc_on_hand: usize,
    pub initial_retailer_on_hand: usize,
    pub initial_retailer_pipeline: usize,
    pub dc_replenishment_quantity: usize,
    pub dc_capacity: usize,
    pub shipment_cost_per_unit: f64,
    pub dc_holding_cost_per_unit: f64,
    pub retailer_holding_cost_per_unit: f64,
    pub stockout_cost_per_unit: f64,
    pub salvage_value_per_unit: f64,
    pub max_shipment_quantity: usize,
    pub benchmark_retailer_base_stock_level: usize,
    pub benchmark_dc_reserve_base_stock_level: usize,
    pub benchmark_dc_reserve_quantity: usize,
    pub notes: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExactVerificationReference {
    pub source: &'static str,
    pub url: &'static str,
    pub periods: usize,
    pub discount_factor: f64,
    pub initial_dc_on_hand: usize,
    pub initial_retailer_on_hand: usize,
    pub initial_retailer_pipeline: usize,
    pub dc_replenishment_quantity: usize,
    pub dc_capacity: usize,
    pub shipment_cost_per_unit: f64,
    pub dc_holding_cost_per_unit: f64,
    pub retailer_holding_cost_per_unit: f64,
    pub stockout_cost_per_unit: f64,
    pub salvage_value_per_unit: f64,
    pub demand_support: &'static [u32],
    pub demand_probabilities: &'static [f64],
    pub max_shipment_quantity: usize,
    pub retailer_base_stock_level: usize,
    pub dc_reserve_base_stock_level: usize,
    pub dc_reserve_quantity: usize,
    pub notes: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NewsvendorWorkedCaseReference {
    pub source: &'static str,
    pub url: &'static str,
    pub matlab_code_url: &'static str,
    /// False: these displayed numbers are from the Gosavi INSTRUCTOR CASE STUDY
    /// (a teaching handout based on the paper), not a results row printed in the
    /// peer-reviewed Sui/Gosavi/Lin (2010) EMJ paper. Per the repo rule, an
    /// instructor/handout number is NOT literature verification.
    pub literature_verified: bool,
    /// What the reproduction asserts and against which artifact.
    pub verification_source: &'static str,
    pub notes: &'static str,
    pub customer_arrival_rate: f64,
    pub demand_size_low: f64,
    pub demand_size_high: f64,
    pub holding_cost_per_unit: f64,
    pub stockout_cost_per_unit: f64,
    pub cycle_time_support: &'static [f64],
    pub cycle_time_probabilities: &'static [f64],
    pub displayed_mean_demand_rate: f64,
    pub displayed_demand_variance: f64,
    pub displayed_cycle_time_mean: f64,
    pub displayed_cycle_time_variance: f64,
    pub displayed_cycle_demand_mean: f64,
    pub displayed_cycle_demand_variance: f64,
    pub displayed_mean_demand_heuristic_order_up_to: f64,
    pub displayed_six_sigma_order_up_to: f64,
    pub displayed_newsvendor_order_up_to: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PaperExperimentCaseDefinition {
    pub case_id: usize,
    pub retailer_penalty_level: i32,
    pub retailer_holding_level: i32,
    pub demand_rate_level: i32,
}

const PAPER_BASELINE_RETAILER_PRODUCT_PARAMS: [PaperRetailerProductParams; 20] = [
    PaperRetailerProductParams {
        retailer_index: 0,
        product_index: 0,
        arrival_rate: 0.25,
        demand_low: 1.0,
        demand_high: 2.0,
        retailer_holding_cost_per_unit_time: 0.06,
        retailer_stockout_cost_per_unit: 4.0,
        revenue_per_unit_sold: 5.0,
    },
    PaperRetailerProductParams {
        retailer_index: 1,
        product_index: 0,
        arrival_rate: 0.5,
        demand_low: 0.5,
        demand_high: 1.5,
        retailer_holding_cost_per_unit_time: 0.05,
        retailer_stockout_cost_per_unit: 4.0,
        revenue_per_unit_sold: 5.0,
    },
    PaperRetailerProductParams {
        retailer_index: 2,
        product_index: 0,
        arrival_rate: 0.3,
        demand_low: 1.0,
        demand_high: 2.0,
        retailer_holding_cost_per_unit_time: 0.03,
        retailer_stockout_cost_per_unit: 4.0,
        revenue_per_unit_sold: 5.0,
    },
    PaperRetailerProductParams {
        retailer_index: 3,
        product_index: 0,
        arrival_rate: 0.25,
        demand_low: 1.0,
        demand_high: 2.0,
        retailer_holding_cost_per_unit_time: 0.04,
        retailer_stockout_cost_per_unit: 4.0,
        revenue_per_unit_sold: 5.0,
    },
    PaperRetailerProductParams {
        retailer_index: 4,
        product_index: 0,
        arrival_rate: 0.1,
        demand_low: 2.0,
        demand_high: 4.0,
        retailer_holding_cost_per_unit_time: 0.03,
        retailer_stockout_cost_per_unit: 4.0,
        revenue_per_unit_sold: 5.0,
    },
    PaperRetailerProductParams {
        retailer_index: 5,
        product_index: 0,
        arrival_rate: 0.2,
        demand_low: 1.0,
        demand_high: 3.0,
        retailer_holding_cost_per_unit_time: 0.05,
        retailer_stockout_cost_per_unit: 4.0,
        revenue_per_unit_sold: 5.0,
    },
    PaperRetailerProductParams {
        retailer_index: 6,
        product_index: 0,
        arrival_rate: 0.3,
        demand_low: 1.0,
        demand_high: 1.5,
        retailer_holding_cost_per_unit_time: 0.03,
        retailer_stockout_cost_per_unit: 4.0,
        revenue_per_unit_sold: 5.0,
    },
    PaperRetailerProductParams {
        retailer_index: 7,
        product_index: 0,
        arrival_rate: 0.5,
        demand_low: 0.5,
        demand_high: 1.5,
        retailer_holding_cost_per_unit_time: 0.06,
        retailer_stockout_cost_per_unit: 4.0,
        revenue_per_unit_sold: 5.0,
    },
    PaperRetailerProductParams {
        retailer_index: 8,
        product_index: 0,
        arrival_rate: 0.15,
        demand_low: 2.0,
        demand_high: 3.0,
        retailer_holding_cost_per_unit_time: 0.04,
        retailer_stockout_cost_per_unit: 4.0,
        revenue_per_unit_sold: 5.0,
    },
    PaperRetailerProductParams {
        retailer_index: 9,
        product_index: 0,
        arrival_rate: 0.2,
        demand_low: 1.0,
        demand_high: 3.0,
        retailer_holding_cost_per_unit_time: 0.05,
        retailer_stockout_cost_per_unit: 4.0,
        revenue_per_unit_sold: 5.0,
    },
    PaperRetailerProductParams {
        retailer_index: 0,
        product_index: 1,
        arrival_rate: 0.5,
        demand_low: 0.5,
        demand_high: 1.5,
        retailer_holding_cost_per_unit_time: 0.04,
        retailer_stockout_cost_per_unit: 3.0,
        revenue_per_unit_sold: 5.0,
    },
    PaperRetailerProductParams {
        retailer_index: 1,
        product_index: 1,
        arrival_rate: 0.1,
        demand_low: 2.0,
        demand_high: 3.0,
        retailer_holding_cost_per_unit_time: 0.06,
        retailer_stockout_cost_per_unit: 3.0,
        revenue_per_unit_sold: 5.0,
    },
    PaperRetailerProductParams {
        retailer_index: 2,
        product_index: 1,
        arrival_rate: 0.15,
        demand_low: 1.0,
        demand_high: 3.0,
        retailer_holding_cost_per_unit_time: 0.04,
        retailer_stockout_cost_per_unit: 3.0,
        revenue_per_unit_sold: 5.0,
    },
    PaperRetailerProductParams {
        retailer_index: 3,
        product_index: 1,
        arrival_rate: 0.3,
        demand_low: 0.5,
        demand_high: 2.0,
        retailer_holding_cost_per_unit_time: 0.05,
        retailer_stockout_cost_per_unit: 3.0,
        revenue_per_unit_sold: 5.0,
    },
    PaperRetailerProductParams {
        retailer_index: 4,
        product_index: 1,
        arrival_rate: 0.35,
        demand_low: 0.5,
        demand_high: 1.5,
        retailer_holding_cost_per_unit_time: 0.03,
        retailer_stockout_cost_per_unit: 3.0,
        revenue_per_unit_sold: 5.0,
    },
    PaperRetailerProductParams {
        retailer_index: 5,
        product_index: 1,
        arrival_rate: 0.25,
        demand_low: 1.0,
        demand_high: 2.0,
        retailer_holding_cost_per_unit_time: 0.06,
        retailer_stockout_cost_per_unit: 3.0,
        revenue_per_unit_sold: 5.0,
    },
    PaperRetailerProductParams {
        retailer_index: 6,
        product_index: 1,
        arrival_rate: 0.4,
        demand_low: 0.5,
        demand_high: 1.5,
        retailer_holding_cost_per_unit_time: 0.04,
        retailer_stockout_cost_per_unit: 3.0,
        revenue_per_unit_sold: 5.0,
    },
    PaperRetailerProductParams {
        retailer_index: 7,
        product_index: 1,
        arrival_rate: 0.2,
        demand_low: 1.0,
        demand_high: 3.0,
        retailer_holding_cost_per_unit_time: 0.03,
        retailer_stockout_cost_per_unit: 3.0,
        revenue_per_unit_sold: 5.0,
    },
    PaperRetailerProductParams {
        retailer_index: 8,
        product_index: 1,
        arrival_rate: 0.15,
        demand_low: 1.5,
        demand_high: 2.5,
        retailer_holding_cost_per_unit_time: 0.05,
        retailer_stockout_cost_per_unit: 3.0,
        revenue_per_unit_sold: 5.0,
    },
    PaperRetailerProductParams {
        retailer_index: 9,
        product_index: 1,
        arrival_rate: 0.25,
        demand_low: 1.0,
        demand_high: 2.0,
        retailer_holding_cost_per_unit_time: 0.03,
        retailer_stockout_cost_per_unit: 3.0,
        revenue_per_unit_sold: 5.0,
    },
];

const PAPER_BASELINE_DC_PARAMS: [PaperDcProductParams; 2] = [
    PaperDcProductParams {
        product_index: 0,
        dc_holding_cost_per_unit_time: 0.005,
        dc_shortage_penalty_per_unit: 1.0,
        reorder_quantity: 500.0,
        reorder_point: 150.0,
        fixed_reorder_cost: 50.0,
    },
    PaperDcProductParams {
        product_index: 1,
        dc_holding_cost_per_unit_time: 0.005,
        dc_shortage_penalty_per_unit: 1.0,
        reorder_quantity: 500.0,
        reorder_point: 150.0,
        fixed_reorder_cost: 50.0,
    },
];

pub const SUI_GOSAVI_LIN_2010_REFERENCE: PublishedBenchmarkReference = PublishedBenchmarkReference {
    source: "Sui, Z., A. Gosavi, and L. Lin (2010), A Reinforcement Learning Approach for Inventory Replenishment in Vendor-Managed Inventory Systems With Consignment Inventory, Engineering Management Journal 22(4): 44-53",
    url: "https://doi.org/10.1080/10429247.2010.11431878",
    literature_verified: false,
    verification_source: "no_peer_reviewed_paper_number_reproduced; only the Gosavi instructor case-study worked example (a teaching handout based on this paper) is re-run and reproduced by an in-crate test",
    benchmark_policies: &["gosavi_instructor_case_study_worked_newsvendor_calculation"],
    notes: "The peer-reviewed paper studies a continuous-time multi-retailer VMI system with consignment inventory, truck-capacity transport decisions, a DC (Q,R) replenishment rule, and a newsvendor-based allocation heuristic. NOT literature-verified: the paper's results table (RL vs. newsvendor profit per case, pp.44-53) is paywalled and not openly reproducible, so no number printed in the peer-reviewed paper is re-run here. What IS reproduced is the Gosavi (2010) instructor TEACHING CASE STUDY worked newsvendor example (a handout based on this paper), kept separately below as a clearly-labeled instructor worked example. The repo-constructed 10-retailer/2-product truck-dispatch case definitions below are a structural interpretation, not transcriptions of a published table, and do not reproduce the published profit rows.",
};

pub const SUI_GOSAVI_LIN_2010_GOSAVI_CASE_STUDY_WORKED_EXAMPLE: NewsvendorWorkedCaseReference =
    NewsvendorWorkedCaseReference {
        source: "Gosavi, A. 'CASE STUDY FOR VENDOR-MANAGED INVENTORY (BASED ON SUI, GOSAVI, & LIN, 2010)', Missouri Univ. of Science and Technology, Sept 7, 2010 (instructor teaching case study / handout; PDF marked 'Copyrighted Material 2020')",
        url: "https://web.mst.edu/_disabled/gosavia/vmi_case_study.pdf",
        matlab_code_url: "https://web.mst.edu/_disabled/gosavia/vmi_newsvendor.m",
        literature_verified: false,
        verification_source: "instructor_case_study_worked_example_reproduced_exactly; NOT a number printed in the peer-reviewed Sui/Gosavi/Lin (2010) EMJ paper",
        notes: "Single-retailer single-product newsvendor worked example from Gosavi's instructor TEACHING CASE STUDY ('Worked out example with data from the paper'). The case study self-describes as class material ('As discussed in class ...') and states it is 'based on the journal article: Sui, Gosavi, and Lin (2010)'. The displayed order-up-to numbers (MDH=15, six-sigma=31.53, newsvendor=26.96) are Gosavi's own worked computation in the handout, NOT a results row printed in the peer-reviewed paper. This is the cleanest OPEN executable anchor available, but per the repo rule a handout number is NOT literature verification; it is kept as a labeled instructor worked example only.",
        customer_arrival_rate: 0.25,
        demand_size_low: 1.0,
        demand_size_high: 2.0,
        holding_cost_per_unit: 0.06,
        stockout_cost_per_unit: 4.0,
        cycle_time_support: &[30.0, 40.0, 50.0],
        cycle_time_probabilities: &[0.25, 0.5, 0.25],
        displayed_mean_demand_rate: 0.375,
        displayed_demand_variance: 0.5833,
        displayed_cycle_time_mean: 40.0,
        displayed_cycle_time_variance: 50.0,
        displayed_cycle_demand_mean: 15.0,
        displayed_cycle_demand_variance: 30.36,
        displayed_mean_demand_heuristic_order_up_to: 15.0,
        displayed_six_sigma_order_up_to: 31.53,
        displayed_newsvendor_order_up_to: 26.96,
    };

pub const SUI_GOSAVI_LIN_2010_CASE_DEFINITIONS: &[PaperExperimentCaseDefinition] = &[
    PaperExperimentCaseDefinition {
        case_id: 1,
        retailer_penalty_level: -1,
        retailer_holding_level: -1,
        demand_rate_level: -1,
    },
    PaperExperimentCaseDefinition {
        case_id: 2,
        retailer_penalty_level: -1,
        retailer_holding_level: -1,
        demand_rate_level: 1,
    },
    PaperExperimentCaseDefinition {
        case_id: 3,
        retailer_penalty_level: -1,
        retailer_holding_level: 1,
        demand_rate_level: -1,
    },
    PaperExperimentCaseDefinition {
        case_id: 4,
        retailer_penalty_level: -1,
        retailer_holding_level: 1,
        demand_rate_level: 1,
    },
    PaperExperimentCaseDefinition {
        case_id: 5,
        retailer_penalty_level: 1,
        retailer_holding_level: -1,
        demand_rate_level: -1,
    },
    PaperExperimentCaseDefinition {
        case_id: 6,
        retailer_penalty_level: 1,
        retailer_holding_level: -1,
        demand_rate_level: 1,
    },
    PaperExperimentCaseDefinition {
        case_id: 7,
        retailer_penalty_level: 1,
        retailer_holding_level: 1,
        demand_rate_level: -1,
    },
    PaperExperimentCaseDefinition {
        case_id: 8,
        retailer_penalty_level: 1,
        retailer_holding_level: 1,
        demand_rate_level: 1,
    },
];

pub fn paper_experiment_case_definition(
    case_id: usize,
) -> Option<&'static PaperExperimentCaseDefinition> {
    SUI_GOSAVI_LIN_2010_CASE_DEFINITIONS
        .iter()
        .find(|row| row.case_id == case_id)
}

/// Build a repo-constructed structural interpretation of one Sui/Gosavi/Lin
/// (2010) experiment case. NOTE: the 10-retailer/2-product parameter rows are a
/// repo interpretation of the paper's structure, not transcriptions of a
/// published table, and the resulting profit rows do not reproduce the paper's
/// (paywalled) results table. Not literature-verified.
pub fn build_sui_gosavi_lin_2010_case(case_id: usize) -> Option<PaperVendorManagedInventoryModel> {
    let row = paper_experiment_case_definition(case_id)?;
    let mut retailer_product_params = PAPER_BASELINE_RETAILER_PRODUCT_PARAMS.to_vec();
    for param in retailer_product_params.iter_mut() {
        if row.demand_rate_level > 0 {
            param.arrival_rate *= 1.5;
        }
        if row.retailer_holding_level > 0 {
            param.retailer_holding_cost_per_unit_time *= 2.0;
        }
        if row.retailer_penalty_level > 0 {
            param.retailer_stockout_cost_per_unit *= 1.5;
        }
    }

    Some(PaperVendorManagedInventoryModel {
        name: "sui_gosavi_lin_2010_truck_dispatch_repo_interpretation",
        source: SUI_GOSAVI_LIN_2010_REFERENCE.source,
        url: SUI_GOSAVI_LIN_2010_REFERENCE.url,
        num_retailers: 10,
        num_products: 2,
        retailer_product_params,
        dc_product_params: PAPER_BASELINE_DC_PARAMS.to_vec(),
        truck_capacity: 100.0,
        transport_cost_per_truck_per_unit_time: 10.0,
        dc_service_time: UniformTimeDistribution {
            low: 0.2,
            high: 0.3,
        },
        dc_to_first_retailer_time: UniformTimeDistribution {
            low: 2.0,
            high: 4.0,
        },
        retailer_to_retailer_time: UniformTimeDistribution {
            low: 0.5,
            high: 1.0,
        },
        retailer_service_time: UniformTimeDistribution {
            low: 0.01,
            high: 0.015,
        },
        last_retailer_to_dc_time: UniformTimeDistribution {
            low: 2.0,
            high: 4.0,
        },
        manufacturer_lead_time: UniformTimeDistribution {
            low: 30.0,
            high: 50.0,
        },
        max_trucks: 2,
        low_signal_multiplier: 0.5,
        high_signal_multiplier: 1.5,
        expected_signal_multiplier: 1.0,
        high_signal_probability: 0.5,
        initial_dc_inventory: vec![650.0, 650.0],
        initial_retailer_inventory: vec![vec![0.0, 0.0]; 10],
    })
}

pub const PRIMARY_REFERENCE_INSTANCE: VendorManagedInventoryReferenceInstance =
    VendorManagedInventoryReferenceInstance {
        name: "sui_gosavi_lin_2010_style_single_retailer",
        source: SUI_GOSAVI_LIN_2010_REFERENCE.source,
        url: SUI_GOSAVI_LIN_2010_REFERENCE.url,
        periods: 24,
        demand_distribution_kind: "poisson",
        demand_mean: 2.5,
        initial_dc_on_hand: 8,
        initial_retailer_on_hand: 2,
        initial_retailer_pipeline: 1,
        dc_replenishment_quantity: 3,
        dc_capacity: 10,
        shipment_cost_per_unit: 0.4,
        dc_holding_cost_per_unit: 0.25,
        retailer_holding_cost_per_unit: 0.6,
        stockout_cost_per_unit: 5.0,
        salvage_value_per_unit: 0.2,
        max_shipment_quantity: 5,
        benchmark_retailer_base_stock_level: 4,
        benchmark_dc_reserve_base_stock_level: 5,
        benchmark_dc_reserve_quantity: 2,
        notes: "Repo-native reduced single-retailer slice (the `name` keeps the legacy 'giannoccaro2010_style' tag for the Python benchmark script; the actual cited paper is Sui, Gosavi, and Lin 2010). A supplier DC chooses shipments into one retailer's consignment stock under a one-period transport lead time, deterministic upstream replenishment at the DC, and retailer lost-sales demand. These parameter values are a repo-chosen instance, NOT taken from any published table; there is no published benchmark number reproduced on this instance. This is not the full paper model.",
    };

pub const VERIFICATION_PROBLEM_INSTANCE: ExactVerificationReference = ExactVerificationReference {
    source: SUI_GOSAVI_LIN_2010_REFERENCE.source,
    url: SUI_GOSAVI_LIN_2010_REFERENCE.url,
    periods: 5,
    discount_factor: 0.99,
    initial_dc_on_hand: 4,
    initial_retailer_on_hand: 1,
    initial_retailer_pipeline: 1,
    dc_replenishment_quantity: 2,
    dc_capacity: 5,
    shipment_cost_per_unit: 0.4,
    dc_holding_cost_per_unit: 0.3,
    retailer_holding_cost_per_unit: 0.6,
    stockout_cost_per_unit: 4.0,
    salvage_value_per_unit: 0.2,
    demand_support: &[0, 1, 2, 3],
    demand_probabilities: &[0.15, 0.35, 0.3, 0.2],
    max_shipment_quantity: 4,
    retailer_base_stock_level: 3,
    dc_reserve_base_stock_level: 4,
    dc_reserve_quantity: 1,
    notes: "Repo-native exact verifier on a reduced single-retailer VMI instance with a small discrete demand support. It preserves the DC stock constraint, one-period shipment pipeline, and vendor-controlled replenishment action while keeping the finite-horizon DP small enough for exact regression tests.",
};
use crate::problems::vendor_managed_inventory::env::{
    PaperDcProductParams, PaperRetailerProductParams, PaperVendorManagedInventoryModel,
    UniformTimeDistribution,
};
