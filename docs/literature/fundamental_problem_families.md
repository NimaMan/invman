# Fundamental Problem Families

This note fixes the expansion order for new `invman` problem families and the rule we use to decide
what counts as a good next addition.

## Goal

The objective is not to add every RL inventory paper. The objective is to cover the fundamental
inventory-control families in an implementation order that is both:

- literature-backed
- incremental relative to the current repo
- testable with a clear verification instance

Current implemented families:

- `lost_sales`
- `lost_sales_fixed_order_cost`
- `dual_sourcing`
- `multi_echelon`
- `perishable_inventory`
- `nonstationary_lot_sizing`
- `random_yield_inventory`
- `joint_replenishment`
- `one_warehouse_multi_retailer`
- `decentralized_inventory_control`
- `network_inventory`

## Decision Rule

New problem families should be prioritized when they satisfy all four conditions:

- they are a classical inventory-control family, not only a narrow application variant
- they already have at least one credible RL paper we can cite in the repo
- they add one new modeling axis that is not already covered in `invman`
- they admit at least one verification anchor: either exact published numbers, or a small
  deterministic worked instance if the paper does not publish exact costs

That rule changes the implementation order slightly from a pure OR taxonomy. We want to add one new
axis at a time.

## Implementation Order

### 1. `perishable_inventory`

Status: implemented

This is the next problem to implement.

Why first:

- it is the cleanest next classical family after vanilla lost sales
- it keeps the single-item, single-order-quantity structure
- it adds one important missing state variable: inventory age profile / shelf life
- the repo already identified perishables as the cleanest fifth family after the current four

Primary literature anchors:

- De Moor et al. (2022), reward shaping for perishable inventory:
  <https://doi.org/10.1016/j.ejor.2021.10.045>
- Jullien et al. (2022), `RetaiL` / waste-reduction RL environment for perishable restocking:
  <https://arxiv.org/abs/2205.15455>
- Maggiar et al. (2025), structure-informed inventory benchmarks including perishables:
  <https://arxiv.org/abs/2507.22040>

Initial heuristic set to support:

- base-stock benchmark
- the best transcribable teacher heuristic from De Moor et al. if the paper gives enough detail
- fixed issuance rule as part of the instance definition, for example FIFO or LIFO

### 2. `nonstationary_lot_sizing`

Status: implemented

Why second:

- it stays in the single-item setting
- it adds forecast-driven nonstationarity without immediately introducing multi-item coupling
- it is closer to real planning than stationary lost-sales benchmarks

Primary literature anchor:

- Dehaybe et al. (2024), DRL for nonstationary lot sizing with forecasts:
  <https://doi.org/10.1016/j.ejor.2023.10.007>

### 3. `random_yield_inventory`

Status: implemented

Why third:

- it is another one-axis extension of the single-item problem
- it adds supply-side uncertainty, which is not covered by the current repo
- it is still much simpler than moving directly to multi-item coupling or decentralized control

Primary literature anchor:

- Yan et al. (2026), all-or-nothing random yield with positive lead times:
  <https://doi.org/10.1016/j.cor.2025.107305>

### 4. `joint_replenishment`

Status: implemented

Why fourth:

- this is the first clean multi-item coupling problem to add
- it is a fundamental OR family, not just a retail-specific variant
- it requires a genuine vector-action benchmark because items share a major ordering cost

Primary literature anchor:

- Vanvuchelen et al. (2020), PPO for the joint replenishment problem:
  <https://www.sciencedirect.com/science/article/pii/S0166361519308218>

### 5. `one_warehouse_multi_retailer`

Status: implemented

Why fifth:

- this is the natural next step after the current two-echelon folder
- it separates the divergent allocation problem from the current simplified shared-base-stock setup
- it introduces allocation logic without yet moving to decentralized control

Primary literature anchor:

- Kaynov et al. (2024), deep reinforcement learning for one-warehouse multi-retailer inventory:
  <https://doi.org/10.1016/j.ijpe.2023.109088>

### 6. `decentralized_inventory_control`

Status: implemented

Why sixth:

- this is the right place to add local-information and multi-agent control
- it is a fundamental supply-chain control family, but it changes the learning/control interface more
  than any of the earlier additions

Primary literature anchors:

- Oroojlooyjadid et al. (2021), Beer Game DQN:
  <https://doi.org/10.1287/msom.2020.0939>
- Mousa et al. (2024), decentralized inventory control with MARL:
  <https://doi.org/10.1016/j.compchemeng.2024.108783>
- Kotecha and del Rio Chanona (2025), GNN + MARL for supply-chain inventory control:
  <https://doi.org/10.1016/j.compchemeng.2025.109111>

### 7. `network_inventory`

Status: implemented

Why later:

- it is a broad umbrella family rather than the cleanest next primitive
- most of its machinery will reuse ideas from `one_warehouse_multi_retailer` and
  `decentralized_inventory_control`

Primary literature anchor:

- Pirhooshyaran and Snyder (2021), neural policies for general stochastic multi-echelon networks:
  <https://arxiv.org/abs/2006.05608>

## Families To Defer

These are real inventory-control topics, but not first-wave additions for `invman`:

- `spare_parts_inventory`
- `ameliorating_inventory`
- `procurement_removal_inventory`
- `joint_pricing_inventory`
- `vendor_managed_inventory`

Reason:

- they are either more domain-specific, hybridized with other decision layers, or better treated
  after the core replenishment families above exist

## Immediate Decision

The first-wave expansion sequence is now implemented through `network_inventory`.

The next additions, if we continue expanding the family set, should come from the deferred list
rather than from the first-wave core families above.

## Review Anchor

For the broad landscape and terminology, keep using:

- Boute et al. (2022), *Deep reinforcement learning for inventory control: A roadmap*:
  <https://doi.org/10.1016/j.ejor.2021.07.016>
