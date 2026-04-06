#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HormuzFlowRecord {
    pub entity: &'static str,
    pub flow_million_bpd_2024: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HormuzNodeRole {
    OriginExporter,
    TransitAsset,
    DemandMarket,
    RefiningStorageHub,
    ReserveBuffer,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HormuzNodeDefinition {
    pub node_id: &'static str,
    pub label: &'static str,
    pub role: HormuzNodeRole,
    pub baseline_flow_million_bpd_2024: f64,
    pub source_basis: &'static str,
    pub notes: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HormuzScenarioReference {
    pub name: &'static str,
    pub baseline_year: usize,
    pub node_count: usize,
    pub closure_fraction: f64,
    pub total_oil_flow_million_bpd_2024: f64,
    pub crude_and_condensate_flow_million_bpd_2024: f64,
    pub petroleum_products_flow_million_bpd_2024: f64,
    pub available_bypass_capacity_million_bpd: f64,
    pub asian_destination_share_of_crude_flows: f64,
    pub top_four_asian_destination_share_of_crude_flows: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HormuzMarketAnchorReference {
    pub analysis_date: &'static str,
    pub latest_observed_close_date: &'static str,
    pub latest_observed_brent_usd_per_bbl: f64,
    pub latest_observed_wti_usd_per_bbl: f64,
    pub eia_next_two_month_floor_brent_usd_per_bbl: f64,
    pub eia_q2_2026_average_brent_usd_per_bbl: f64,
    pub opec_may_2026_supply_adjustment_million_bpd: f64,
}

pub const EIA_HORMUZ_TIE_2025_URL: &str = "https://www.eia.gov/todayinenergy/detail.php?id=65504";
pub const EIA_HORMUZ_FIG1_2025_URL: &str =
    "https://www.eia.gov/todayinenergy/images/2025.06.16/fig1.xlsx";
pub const EIA_HORMUZ_FIG3_2025_URL: &str =
    "https://www.eia.gov/todayinenergy/images/2025.06.16/fig3.xlsx";
pub const EIA_DAILY_PRICES_URL: &str = "https://www.eia.gov/todayinenergy/prices.php";
pub const EIA_MARCH_2026_STEO_URL: &str = "https://www.eia.gov/outlooks/steo/pdf/steo_full.pdf";
pub const EIA_TOP_PRODUCERS_AND_CONSUMERS_FAQ_URL: &str =
    "https://www.eia.gov/tools/faqs/faq.php?id=709&t=6";
pub const OPEC_APRIL_5_2026_PRODUCTION_DECISION_URL: &str =
    "https://www.opec.org/pr-detail/597-5-april-2026.html?mod=livecoverage_web";
pub const EIA_WORLD_OIL_TRANSIT_CHOKEPOINTS_2024_URL: &str =
    "https://www.eia.gov/international/content/analysis/special_topics/World_Oil_Transit_Chokepoints/wotc.pdf";

pub const HORMUZ_2024_TOTAL_OIL_FLOW_MILLION_BPD: f64 = 20.261741721311477;
pub const HORMUZ_2024_CRUDE_AND_CONDENSATE_FLOW_MILLION_BPD: f64 = 14.318613808743168;
pub const HORMUZ_2024_PETROLEUM_PRODUCTS_FLOW_MILLION_BPD: f64 = 5.9431279125683094;
pub const HORMUZ_2024_AVAILABLE_BYPASS_CAPACITY_MILLION_BPD: f64 = 2.6;
pub const HORMUZ_2024_ASIAN_DESTINATION_SHARE_OF_CRUDE_FLOWS: f64 = 0.84;
pub const HORMUZ_2024_TOP_FOUR_ASIAN_DESTINATION_SHARE_OF_CRUDE_FLOWS: f64 = 0.69;
pub const BRENT_SPOT_PRICE_2026_04_02_USD_PER_BBL: f64 = 127.61;
pub const WTI_SPOT_PRICE_2026_04_02_USD_PER_BBL: f64 = 113.23;
pub const EIA_NEXT_TWO_MONTH_FLOOR_BRENT_USD_PER_BBL: f64 = 95.0;
pub const EIA_Q2_2026_AVERAGE_BRENT_USD_PER_BBL: f64 = 91.0;
pub const OPEC_MAY_2026_SUPPLY_ADJUSTMENT_MILLION_BPD: f64 = 0.206;

pub const HORMUZ_2024_ORIGIN_FLOWS: &[HormuzFlowRecord] = &[
    HormuzFlowRecord {
        entity: "Saudi Arabia",
        flow_million_bpd_2024: 5.477689579234973,
    },
    HormuzFlowRecord {
        entity: "Iraq",
        flow_million_bpd_2024: 3.223645852459016,
    },
    HormuzFlowRecord {
        entity: "United Arab Emirates",
        flow_million_bpd_2024: 1.8900929781420766,
    },
    HormuzFlowRecord {
        entity: "Iran",
        flow_million_bpd_2024: 1.3996683715846996,
    },
    HormuzFlowRecord {
        entity: "Kuwait",
        flow_million_bpd_2024: 1.3257541338797816,
    },
    HormuzFlowRecord {
        entity: "Qatar",
        flow_million_bpd_2024: 0.6490193579234973,
    },
    HormuzFlowRecord {
        entity: "Other Hormuz exporters",
        flow_million_bpd_2024: 0.35274353551912263,
    },
];

pub const HORMUZ_2024_DESTINATION_FLOWS: &[HormuzFlowRecord] = &[
    HormuzFlowRecord {
        entity: "China",
        flow_million_bpd_2024: 4.7846856393442625,
    },
    HormuzFlowRecord {
        entity: "Other Asia",
        flow_million_bpd_2024: 2.071853393442623,
    },
    HormuzFlowRecord {
        entity: "India",
        flow_million_bpd_2024: 1.8853302295081966,
    },
    HormuzFlowRecord {
        entity: "South Korea",
        flow_million_bpd_2024: 1.7266257540983607,
    },
    HormuzFlowRecord {
        entity: "Japan",
        flow_million_bpd_2024: 1.5153605710382514,
    },
    HormuzFlowRecord {
        entity: "Other destinations",
        flow_million_bpd_2024: 0.9017658169398892,
    },
    HormuzFlowRecord {
        entity: "Europe",
        flow_million_bpd_2024: 0.7206034344262295,
    },
    HormuzFlowRecord {
        entity: "United States",
        flow_million_bpd_2024: 0.4843527459016393,
    },
    HormuzFlowRecord {
        entity: "Saudi Arabia destination market",
        flow_million_bpd_2024: 0.22803622404371585,
    },
];

pub const HORMUZ_NODE_SET_V1: &[HormuzNodeDefinition] = &[
    HormuzNodeDefinition {
        node_id: "saudi_arabia_origin",
        label: "Saudi Arabia origin exports",
        role: HormuzNodeRole::OriginExporter,
        baseline_flow_million_bpd_2024: 5.477689579234973,
        source_basis: "eia_hormuz_fig3_origin_2024",
        notes: "Largest 2024 origin flow through Hormuz.",
    },
    HormuzNodeDefinition {
        node_id: "iraq_origin",
        label: "Iraq origin exports",
        role: HormuzNodeRole::OriginExporter,
        baseline_flow_million_bpd_2024: 3.223645852459016,
        source_basis: "eia_hormuz_fig3_origin_2024",
        notes: "Basra-centered export flow through Hormuz.",
    },
    HormuzNodeDefinition {
        node_id: "uae_origin",
        label: "United Arab Emirates origin exports",
        role: HormuzNodeRole::OriginExporter,
        baseline_flow_million_bpd_2024: 1.8900929781420766,
        source_basis: "eia_hormuz_fig3_origin_2024",
        notes: "Includes UAE-origin crude and condensate crossing Hormuz.",
    },
    HormuzNodeDefinition {
        node_id: "iran_origin",
        label: "Iran origin exports",
        role: HormuzNodeRole::OriginExporter,
        baseline_flow_million_bpd_2024: 1.3996683715846996,
        source_basis: "eia_hormuz_fig3_origin_2024",
        notes: "Primarily China-bound crude exports in the current trade pattern.",
    },
    HormuzNodeDefinition {
        node_id: "kuwait_origin",
        label: "Kuwait origin exports",
        role: HormuzNodeRole::OriginExporter,
        baseline_flow_million_bpd_2024: 1.3257541338797816,
        source_basis: "eia_hormuz_fig3_origin_2024",
        notes: "Kuwaiti export flow through Hormuz.",
    },
    HormuzNodeDefinition {
        node_id: "qatar_origin",
        label: "Qatar origin exports",
        role: HormuzNodeRole::OriginExporter,
        baseline_flow_million_bpd_2024: 0.6490193579234973,
        source_basis: "eia_hormuz_fig3_origin_2024",
        notes: "Oil and condensate only; LNG is tracked separately in the source article.",
    },
    HormuzNodeDefinition {
        node_id: "other_hormuz_origins",
        label: "Other Hormuz exporters",
        role: HormuzNodeRole::OriginExporter,
        baseline_flow_million_bpd_2024: 0.35274353551912263,
        source_basis: "eia_hormuz_fig3_origin_2024",
        notes: "Residual exporter bucket from the EIA figure data.",
    },
    HormuzNodeDefinition {
        node_id: "strait_of_hormuz",
        label: "Strait of Hormuz chokepoint",
        role: HormuzNodeRole::TransitAsset,
        baseline_flow_million_bpd_2024: HORMUZ_2024_TOTAL_OIL_FLOW_MILLION_BPD,
        source_basis: "eia_hormuz_fig1_2024",
        notes: "Main disrupted maritime chokepoint.",
    },
    HormuzNodeDefinition {
        node_id: "aggregate_bypass_capacity",
        label: "Aggregate Saudi and UAE bypass capacity",
        role: HormuzNodeRole::TransitAsset,
        baseline_flow_million_bpd_2024: HORMUZ_2024_AVAILABLE_BYPASS_CAPACITY_MILLION_BPD,
        source_basis: "eia_hormuz_tie_2025_alt_routes",
        notes: "Initial model uses the EIA estimate of effective unused bypass capacity as one aggregate mitigation asset.",
    },
    HormuzNodeDefinition {
        node_id: "china_market",
        label: "China destination market",
        role: HormuzNodeRole::DemandMarket,
        baseline_flow_million_bpd_2024: 4.7846856393442625,
        source_basis: "eia_hormuz_fig3_destination_2024",
        notes: "Largest 2024 destination market in the EIA figure data.",
    },
    HormuzNodeDefinition {
        node_id: "india_market",
        label: "India destination market",
        role: HormuzNodeRole::DemandMarket,
        baseline_flow_million_bpd_2024: 1.8853302295081966,
        source_basis: "eia_hormuz_fig3_destination_2024",
        notes: "Second-largest named Asian destination in the 2024 data.",
    },
    HormuzNodeDefinition {
        node_id: "south_korea_market",
        label: "South Korea destination market",
        role: HormuzNodeRole::DemandMarket,
        baseline_flow_million_bpd_2024: 1.7266257540983607,
        source_basis: "eia_hormuz_fig3_destination_2024",
        notes: "Top four Asian destination market.",
    },
    HormuzNodeDefinition {
        node_id: "japan_market",
        label: "Japan destination market",
        role: HormuzNodeRole::DemandMarket,
        baseline_flow_million_bpd_2024: 1.5153605710382514,
        source_basis: "eia_hormuz_fig3_destination_2024",
        notes: "Top four Asian destination market.",
    },
    HormuzNodeDefinition {
        node_id: "other_asia_market",
        label: "Other Asia destination market",
        role: HormuzNodeRole::DemandMarket,
        baseline_flow_million_bpd_2024: 2.071853393442623,
        source_basis: "eia_hormuz_fig3_destination_2024",
        notes: "Residual Asian destination bucket from the EIA figure data.",
    },
    HormuzNodeDefinition {
        node_id: "europe_market",
        label: "Europe destination market",
        role: HormuzNodeRole::DemandMarket,
        baseline_flow_million_bpd_2024: 0.7206034344262295,
        source_basis: "eia_hormuz_fig3_destination_2024",
        notes: "Europe bucket from the EIA figure data.",
    },
    HormuzNodeDefinition {
        node_id: "united_states_market",
        label: "United States destination market",
        role: HormuzNodeRole::DemandMarket,
        baseline_flow_million_bpd_2024: 0.4843527459016393,
        source_basis: "eia_hormuz_fig3_destination_2024",
        notes: "Smaller direct Hormuz destination with larger indirect price exposure.",
    },
    HormuzNodeDefinition {
        node_id: "saudi_arabia_market",
        label: "Saudi Arabia destination market",
        role: HormuzNodeRole::DemandMarket,
        baseline_flow_million_bpd_2024: 0.22803622404371585,
        source_basis: "eia_hormuz_fig3_destination_2024",
        notes: "Represents local Gulf demand captured as a destination in the figure data.",
    },
    HormuzNodeDefinition {
        node_id: "other_destinations_market",
        label: "Other destination markets",
        role: HormuzNodeRole::DemandMarket,
        baseline_flow_million_bpd_2024: 0.9017658169398892,
        source_basis: "eia_hormuz_fig3_destination_2024",
        notes: "Residual destination bucket from the EIA figure data.",
    },
    HormuzNodeDefinition {
        node_id: "gulf_refining_and_storage_hub",
        label: "Gulf refining and storage hub",
        role: HormuzNodeRole::RefiningStorageHub,
        baseline_flow_million_bpd_2024: 0.0,
        source_basis: "eia_hormuz_tie_2025_local_refining_note",
        notes: "Captures the documented shift of some flows to local refining and storage within the Gulf states.",
    },
    HormuzNodeDefinition {
        node_id: "strategic_reserve_and_floating_storage",
        label: "Strategic reserve and floating storage buffer",
        role: HormuzNodeRole::ReserveBuffer,
        baseline_flow_million_bpd_2024: 0.0,
        source_basis: "modeling_assumption_v1",
        notes: "Explicit inventory-response node added so reserve release can be a first-class control action.",
    },
];

pub const HORMUZ_FULL_CLOSURE_SCENARIO_V1: HormuzScenarioReference = HormuzScenarioReference {
    name: "hormuz_full_closure_2024_v1",
    baseline_year: 2024,
    node_count: 20,
    closure_fraction: 1.0,
    total_oil_flow_million_bpd_2024: HORMUZ_2024_TOTAL_OIL_FLOW_MILLION_BPD,
    crude_and_condensate_flow_million_bpd_2024: HORMUZ_2024_CRUDE_AND_CONDENSATE_FLOW_MILLION_BPD,
    petroleum_products_flow_million_bpd_2024: HORMUZ_2024_PETROLEUM_PRODUCTS_FLOW_MILLION_BPD,
    available_bypass_capacity_million_bpd: HORMUZ_2024_AVAILABLE_BYPASS_CAPACITY_MILLION_BPD,
    asian_destination_share_of_crude_flows: HORMUZ_2024_ASIAN_DESTINATION_SHARE_OF_CRUDE_FLOWS,
    top_four_asian_destination_share_of_crude_flows:
        HORMUZ_2024_TOP_FOUR_ASIAN_DESTINATION_SHARE_OF_CRUDE_FLOWS,
};

pub const HORMUZ_MARKET_ANCHORS_V1: HormuzMarketAnchorReference = HormuzMarketAnchorReference {
    analysis_date: "2026-04-06",
    latest_observed_close_date: "2026-04-02",
    latest_observed_brent_usd_per_bbl: BRENT_SPOT_PRICE_2026_04_02_USD_PER_BBL,
    latest_observed_wti_usd_per_bbl: WTI_SPOT_PRICE_2026_04_02_USD_PER_BBL,
    eia_next_two_month_floor_brent_usd_per_bbl: EIA_NEXT_TWO_MONTH_FLOOR_BRENT_USD_PER_BBL,
    eia_q2_2026_average_brent_usd_per_bbl: EIA_Q2_2026_AVERAGE_BRENT_USD_PER_BBL,
    opec_may_2026_supply_adjustment_million_bpd: OPEC_MAY_2026_SUPPLY_ADJUSTMENT_MILLION_BPD,
};

pub fn node_count_v1() -> usize {
    HORMUZ_NODE_SET_V1.len()
}

pub fn top_origin_2024() -> &'static HormuzFlowRecord {
    &HORMUZ_2024_ORIGIN_FLOWS[0]
}

pub fn top_destination_2024() -> &'static HormuzFlowRecord {
    &HORMUZ_2024_DESTINATION_FLOWS[0]
}

pub fn current_market_anchors_v1() -> &'static HormuzMarketAnchorReference {
    &HORMUZ_MARKET_ANCHORS_V1
}
