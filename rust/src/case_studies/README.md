# Case Studies

`rust/src/case_studies/` is the home for source-backed real-world applications built with the
shared problem language and the reusable problem families.

These are not generic inventory problem families.

A case study should live here when it:

- represents a specific operational system or geopolitical setting
- depends on tracked external data and publisher sources
- uses the FlowNet language to describe stock-flow dynamics under supplied events and controls
- may later be paired with optimization, but is currently centered on simulation and verification

Current case studies:

- `hormuz_strait`

Design rule:

- `rust/src/problems/`
  - reusable problem families such as `lost_sales`, `perishable_inventory`, and
    `network_inventory`
- `rust/src/case_studies/`
  - concrete source-backed systems such as Hormuz oil disruption

That separation keeps the FlowNet backbone honest:

- the problem layer defines the reusable language and generic families
- the case-study layer uses that language to model a real system under exogenous event inputs and a
  supplied control regime
- optimization can then sit above the simulation later instead of being baked into the case-study
  structure
