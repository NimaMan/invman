#![allow(dead_code)]

use crate::problems::core::flownet::{FlowNetInstance, FlowNetParameter};
use crate::problems::lost_sales::demand::{LostSalesDemandConfig, LostSalesDemandKind};
use crate::problems::lost_sales::vanilla::flownet::formulation::LOST_SALES_FLOWNET_NAME;
use crate::problems::lost_sales::vanilla::rollout::LostSalesRolloutConfig;

pub fn demand_model_description(config: &LostSalesDemandConfig) -> String {
    match config.kind {
        LostSalesDemandKind::Poisson => format!("Poisson(mean={:.3})", config.demand_rate),
        LostSalesDemandKind::Geometric => format!("Geometric(mean={:.3})", config.demand_rate),
        LostSalesDemandKind::MarkovModulatedPoisson2 => format!(
            "MarkovModulatedPoisson2(mean={:.3}, lambda_low={:.3}, lambda_high={:.3}, p00={:.3}, p11={:.3})",
            config.demand_rate,
            config.demand_lambda_low,
            config.demand_lambda_high,
            config.demand_p00,
            config.demand_p11
        ),
    }
}

pub fn instance_from_rollout_config(
    name: impl Into<String>,
    config: &LostSalesRolloutConfig,
) -> FlowNetInstance {
    FlowNetInstance {
        name: name.into(),
        flownet_name: String::from(LOST_SALES_FLOWNET_NAME),
        parameters: vec![
            FlowNetParameter {
                name: String::from("lead_time"),
                value: config.lead_time.to_string(),
            },
            FlowNetParameter {
                name: String::from("demand_model"),
                value: demand_model_description(&config.demand_config),
            },
            FlowNetParameter {
                name: String::from("holding_cost"),
                value: config.holding_cost.to_string(),
            },
            FlowNetParameter {
                name: String::from("shortage_cost"),
                value: config.shortage_cost.to_string(),
            },
            FlowNetParameter {
                name: String::from("procurement_cost"),
                value: config.procurement_cost.to_string(),
            },
            FlowNetParameter {
                name: String::from("fixed_order_cost"),
                value: config.fixed_order_cost.to_string(),
            },
        ],
        horizon_periods: Some(config.horizon),
        notes: vec![
            String::from(
                "policy architecture, input normalization, and tree-specific parameters are intentionally excluded from the problem-side FlowNet instance",
            ),
            String::from(
                "the current implementation exposes the pipeline-state observation rather than a separate observation-event object",
            ),
        ],
    }
}
