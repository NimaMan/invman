# Vendor Managed Inventory

Rust-first problem home for `vendor_managed_inventory`.

Repo interpretation:

- one vendor DC manages replenishment into a retailer's consignment stock
- the retailer sees stochastic end-customer demand
- the shipment lead time from the DC to the retailer is one period
- the DC has limited on-hand stock and receives deterministic upstream replenishment

Code lives under `rust/src/problems/vendor_managed_inventory/`.
