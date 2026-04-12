# joint_pricing_inventory

Rust-first problem home for `joint_pricing_inventory`.

## Formulation

Repo interpretation:

- one item
- one periodic order quantity decision
- one discrete selling-price decision
- stochastic price-sensitive lost-sales demand
- finite planning horizon with terminal salvage value

At period `t`, the state is `(t, I_t)` where `I_t` is on-hand inventory. The action is a pair
`(q_t, p_t)`:

- `q_t` is the order quantity, bounded by `max_order_quantity`
- `p_t` is a discrete index into the available price ladder

Demand in each period is stochastic and price-dependent. Sales are capped by on-hand inventory, so
unmet demand is lost sales. The period objective combines:

- procurement cost
- holding cost on ending inventory
- stockout penalty on lost sales
- terminal salvage value at the horizon

Code lives under `rust/src/problems/joint_pricing_inventory/`.

## Layout

Literature and verification assets live in:

- `literature/references.rs`
- `verification/tests.rs`
- `literature/`
- `practical/`
- `experiments/`
- `verification/`

Core executable code remains at the package root:

- `env.rs`
- `demand.rs`
- `heuristics/`
- `finite_horizon_dp.rs`
- `rollout.rs`
- `bindings.rs`

## Verification Status

Current status: `joint_pricing_inventory` is not literature-verified.

Reason:

- Zhou et al. (2022) study an infinite-horizon joint pricing-and-inventory problem with reference
  price effects. That is not the same executable formulation as this repo package.
- Qin, Simchi-Levi, and Wang (2022) study a data-driven version of the classic joint
  pricing-inventory problem, but the publicly accessible article metadata does not provide a clean
  instance-level benchmark table for this repo package, and the replication files are not openly
  accessible from this environment without the separate INFORMS download flow.

So this package currently uses:

- literature references as formulation anchors
- a repo-native reduced exact verifier for implementation correctness

State interface:

- `env.rs` exposes raw state quantities only
- the current soft-tree benchmark keeps derived demand and price features in `rollout.rs`
- environment code must not hide learned-policy preprocessing
