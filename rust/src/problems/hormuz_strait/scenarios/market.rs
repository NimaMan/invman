use crate::problems::hormuz_strait::references::{
    current_market_anchors, HormuzMarketAnchorReference,
};

pub fn current_market_context() -> &'static HormuzMarketAnchorReference {
    current_market_anchors()
}

pub fn baseline_rebalance_brent_price_usd_per_bbl() -> f64 {
    current_market_anchors().eia_next_two_month_floor_brent_usd_per_bbl
}
